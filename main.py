"""
===============================================================================
Autonomous Mobile Robot Navigation App/Pipeline
Author: Jonathan Loo
Version: 1.0
Date: October 2025
===============================================================================
Purpose
--------
Implements a synchronous Sense→Think→Act control loop for autonomous maze navigation.
Each loop iteration reads the robot pose and LiDAR scan, refines pose via
ICP scan matching, updates the occupancy grid map (OGM), computes or updates a path
(A* or frontier-based), generates a lookahead setpoint, applies it to the simulated
robot, and visualises/logs the result.

Core Concept
-------------
Demonstrates a compact “SLAM” pipeline:
    ICP-aided localisation + OGM mapping + goal/frontier navigation
executed in real time within a single blocking loop.

Simulation vs Real Operation
----------------------------
- **SIMULATION (default):** 
  `apply_setpoint()` advances robot pose internally via unicycle kinematics.
- **REAL MODE:** 
  `apply_setpoint()` transmits setpoints to hardware; display updates only from
  robot-reported pose/scan data. Loop remains synchronous and blocking.

Main Loop Sequence
------------------
SENSE → (ICP) → FUSE → MAP → PLAN → ACT → LOG/VIZ

1) Pose & LiDAR acquisition  
2) ICP alignment and gated fusion  
3) Occupancy grid update  
4) Path planning (`determine_navigation_path()`)  
5) Setpoint computation (`compute_setpoint()`)  
6) Motion update (`apply_setpoint()`)  
7) Visualisation and CSV logging  

Modes
-----
- **KNOWN:** Preplanned A* path to fixed goal.  
- **UNKNOWN:** Frontier-based exploration until goal discovered.  
- **GOALSEEKING:** Path-following using lookahead setpoints.  

Termination
------------
Loop ends when the robot reaches the goal (`arrival_tol_m`) or user quits ('q').

Notes
-----
- All localisation, mapping, and control logic run in one synchronous loop.
- For real-robot use, implement:
      get_pose(), get_scan(), apply_setpoint()
- Candidates only modify `determine_frontier_path()` for the unknown-world task.
"""

from util import *

# -----------------------------------------------------------------------------
# This is the main simulation configuration
# -----------------------------------------------------------------------------
DEFAULTS: Dict[str, Dict] = {
    "world": {
        "wall_half_thickness_m": 0.005,
        "border_thickness_m": 0.01,
    },
    "snake_maze": {
        "size_m": 1.80,
        "cell_size_m": 0.45,
        "num_walls": 4,
        "gap_cells": 1,
    },
    "random_maze": {
        "size_m": 1.80,
        "cell_size_m": 0.45,
        "random_wall_count": 5,
        "random_seed": None,
        "candidates_to_list": 3,
        "seed_scan_start": 0,
        "seed_scan_stride": 1,
        "max_attempts_per_page": 10000,
        "segment_len_cells_min": 1,
        "segment_len_cells_max": 2,
        "orientation_bias": 0.5,
    },
    "planning": {
        "sample_step_m": 0.03,
        "resample_ds_m": 0.05,
        "equal_eps": 1e-6,
        "seg_eps": 1e-9,
    },
    "lidar": {
        "num_rays": 360,
        "max_range_m": 3.0,
        "raycast_eps": 1e-6,
    },
    "ogm": {
        "xyreso_m": 0.03,
        "l_free": -0.4,
        "l_occ": 0.85,
        "l_min": -4.0,
        "l_max": 4.0,
        "hit_margin_m": 1e-3,
        "prob_free_max": 0.35,
        "prob_occ_min": 0.65,
        "size_eps": 1e-9,
        "gray_free": 0.9,
        "gray_occ": 0.0,
        "gray_unk": 1.0,
    },

    "icp_fusion": {
        "enabled": True,
        "alpha": 0.1,
        "max_trans_m": 0.20,
        "max_rot_deg": 20.0,
        "min_points": 50,
        "max_rmse_m": 0.05,
        "snap_trans_m": 0.02,
        "snap_rot_deg": 2.0,
    },
    "viz": {
        "main_figsize_in": (14, 10),
        "robot_arrow_len_m": 0.05,
        "robot_arrow_head_m": (0.03, 0.03),
        "ogm_arrow_len_m": 0.05,
        "ogm_arrow_head_m": (0.03, 0.03),
        "lidar_alpha": 0.2,
        "lidar_lw": 0.5,
        "thumb_size_in": (3, 3),
        "pause_s": 0.01,
    },
    "logging": {
        "level": logging.INFO,
        "format": "[%(levelname)s] %(message)s",
        "pose_csv": "pose.csv",
        "lidar_csv": "lidar.csv",
    },
    "app": {
        "arrival_tolerance_m": 0.1,
        "mode": "GOALSEEKING",  # fixed mode
        "map_type": "RANDOM",  # RANDOM | SNAKE
        "entrance_cell": (0, 0),
        "snake_goal_cell": (3, 3),
        "random_goal_cell": (3, 3),
    },
    "robot": {
        "robot_radius_m": 0.15,
        "turn_angle_rad": math.radians(36),
        "k_ang": 10,
        "v_max_mps": 1.0,  # may be 0.35 for real robot
        "dt_s": 0.1,
        "dt_guard_s": 1e-3,
    },
    "setpoing_cfg": {
        "lookahead_m": 0.3,
    },    
}

def install_key_to_viz(viz: Dict) -> None:
    """Attach keyboard listeners for the live plot window."""
    def _on_key(event):
        globals()["_LAST_KEY"] = event.key
    viz["fig"].canvas.mpl_connect("key_press_event", _on_key)

logging.basicConfig(level=DEFAULTS["logging"]["level"], format=DEFAULTS["logging"]["format"])
log = logging.getLogger("maze_app")

# -----------------------------------------------------------------------------
# This is the main application loop
# -----------------------------------------------------------------------------

def main() -> None:

# -----------------------------------------------------------------------------
# The following is the initial setup including user input, maze world generation, entrance and goal "cell" coordinates,
# initial path planning (mainly for the known maze), lidar, occupancy grid map (OGM), visualisation and logging setup. 
# -----------------------------------------------------------------------------
    settings = copy.deepcopy(DEFAULTS)
    app = ask_options(settings)
    nav_mode = choose_navigation_mode(settings)

    world, entrance, goal_cell = build_world(settings, app)
    planner = create_planner(world, settings["planning"]["sample_step_m"], settings["robot"]["robot_radius_m"])
    path = initialise_navigation_path(planner, entrance, goal_cell, settings, nav_mode)
    sensor = create_lidar(settings["lidar"])
    ogm = create_ogm(settings["ogm"], 0.0, 0.0, world["size_m"], world["size_m"])
    viz = create_viz(world["size_m"], world["cell_size_m"], settings["viz"], settings["robot"]["robot_radius_m"])
    logger_dict = create_logger(settings["lidar"]["num_rays"], settings["logging"])
    start_x, start_y = cell_center(entrance, world["cell_size_m"])
    start_heading = math.atan2(path[1][1] - start_y, path[1][0] - start_x) if len(path) >= 2 else 0.0
    astar_pts = planner["cspace"] if planner["cspace"] else planner["obstacles"]


    state = SimulationState(
        world=world,
        entrance=entrance,
        goal=make_goal(goal_cell),
        path=path,
        sensor=sensor,
        ogm=ogm,
        viz=viz,
        logger=logger_dict,
        pose=make_pose(start_x, start_y, start_heading),
        settings=settings,
        icp_prev_pts=None,
        icp_prev_pose=None,
        step=0,
        astar_pts=astar_pts,
        ctrl=settings["setpoing_cfg"].copy(),
        planner=planner,
    )
    state.robot_iface = load_robot_interface(state.settings)

    install_key_to_viz(state.viz)

    while True:
        key = globals().get("_LAST_KEY", None)
        globals()["_LAST_KEY"] = None
        if key == "q":
            print("Quit requested.")
            break
# -----------------------------------------------------------------------------
# Interface to simulated robot data or real robot data
# For real robot data, simply load the real robot data via the load_robot_interface()
# -----------------------------------------------------------------------------
        robot = state.robot_iface
        if robot is None:
            robot = state.robot_iface = load_robot_interface(state.settings)

# -----------------------------------------------------------------------------
# Main navigation pipeline
# read robot (pose, lidar) --> ICP matching (pose estimation) --> pose fusion --> update OGM --> path planning --> setpoint control --> apply to robot --> map visualisation
# -----------------------------------------------------------------------------
        pose = robot.get_pose(state)
        state.pose = pose
        scan_data = robot.get_scan(state, pose)
        curr_pts = icp_points(pose, scan_data, state.settings["lidar"])
        state.icp_prev_pts, state.icp_prev_pose = curr_pts, pose
        icp_pose, rmse, n_pts, tf_pts = icp_match_step(state.icp_prev_pts, curr_pts, state.icp_prev_pose)
        pose = fuse_icp_pose(state.settings, pose, icp_pose, rmse, n_pts)
        state.pose = pose
        update_ogm(state.ogm, scan_data, pose)
        determine_navigation_path(state)
        setpoint = compute_setpoint(state.ctrl, state.path, pose)

        new_pose = robot.apply_setpoint(state, pose, setpoint)
        state.pose = new_pose
        state.step += 1
# -----------------------------------------------------------------------------
# Visualisation and Logging
# -----------------------------------------------------------------------------
        render(state.viz, state.world, state.ogm, pose, scan_data, state.goal, state.step, state.path, state.entrance, state.icp_prev_pts, curr_pts, tf_pts, state.astar_pts, state.frontier_goal, state.frontier_candidates)

        with state.logger["pose"].open("a", newline="") as handle:
            csv.writer(handle).writerow([state.step, new_pose["x"], new_pose["y"], math.degrees(new_pose["theta"]), state.settings["app"]["mode"]])

        nav_mode = state.settings.get("navigation", {}).get("mode", "KNOWN")
        if state.frontier_goal:
            fgx, fgy = cell_center(state.frontier_goal, state.world["cell_size_m"])
            fg_dist = math.hypot(fgx - new_pose["x"], fgy - new_pose["y"])
        else:
            fgx = fgy = fg_dist = float("nan")
        frontier_cells = ";".join(f"{cell[0]}:{cell[1]}" for cell in state.frontier_candidates) if state.frontier_candidates else ""
        path_length = len(state.path)
        if state.path:
            path_first_x, path_first_y = state.path[0]
        else:
            path_first_x = path_first_y = float("nan")

        diag_icp_x = diag_icp_y = diag_icp_theta = float("nan")
        diag_rmse = float("nan")
        diag_pts = 0
        diag_icp_x = icp_pose["x"]
        diag_icp_y = icp_pose["y"]
        diag_icp_theta = math.degrees(icp_pose["theta"])
        diag_rmse = rmse if rmse is not None else float("nan")
        diag_pts = n_pts

        with state.logger["diag"].open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    state.step, nav_mode, new_pose["x"], new_pose["y"], math.degrees(new_pose["theta"]), fgx, fgy, fg_dist,
                    f"{state.frontier_goal[0]}:{state.frontier_goal[1]}" if state.frontier_goal else "", len(state.frontier_candidates),
                    frontier_cells, path_length, path_first_x, path_first_y, diag_icp_x, diag_icp_y, diag_icp_theta, diag_rmse, diag_pts,
                ]
            )
        row = [state.step]
        for angle, distance in zip(scan_data["angles"], scan_data["ranges"]):
            row.extend([math.degrees(angle), distance])

        with state.logger["lidar"].open("a", newline="") as handle:
            csv.writer(handle).writerow(row)

        icp_info = f" | icp_pose=({icp_pose['x']:.3f},{icp_pose['y']:.3f},{math.degrees(icp_pose['theta']):.1f}°)"
        
        log.info("Step %05d | Maze World = %s | pose=(%.2f,%.2f,%.1f°)%s | setpoint=(%.2f,%.2f,%.1f°)", state.step, state.settings.get("navigation", {}).get("mode", "KNOWN").upper(), new_pose["x"], new_pose["y"], math.degrees(new_pose["theta"]), icp_info, setpoint["x"], setpoint["y"], math.degrees(setpoint["theta"])) 

# -----------------------------------------------------------------------------
# Stopping condition
# -----------------------------------------------------------------------------
        goal_x, goal_y = cell_center(state.goal["cell"], state.world["cell_size_m"])
        if math.hypot(goal_x - pose["x"], goal_y - pose["y"]) <= state.settings["app"]["arrival_tolerance_m"]:
            print("Simulation complete: Robot reached the goal.")
            log.info("Reached goal; stopping.")
            plt.show(block=True)
            break

    print("Done.")
    plt.close("all")

# -----------------------------------------------------------------------------
# WinterHack 2025: Candidate Selection Challenge
# The following function is to be completed by candidates as part of the challenge.
# Candidates only modify the code within the designated section. Candidates should not
# change the function signature, docstring, or any code outside the designated section.
# -----------------------------------------------------------------------------

def determine_frontier_path(state: SimulationState) -> None:
    """
    Determines and sets the frontier path for robot navigation in an unknown environment.
    This function identifies the next frontier cell to explore and plans a path to it. If the current
    frontier goal matches the ultimate goal cell, it returns that cell. Otherwise, it detects new
    frontiers and selects the most promising one based on various criteria including heading alignment,
    forward progress towards the goal, and distances.

    Args:
        state (SimulationState): The current simulation state containing robot pose, world information,
                                goals, and other navigation parameters.
    Returns:
         None. Modifies the state in place by setting the `frontier_goal` and `path` attributes.
         `frontier_goal` is a cell representing the chosen frontier to explore.
         `path` is a list of cells representing the plan to reach the `frontier_goal`.

    The function is expected to perform the following key steps:
    1. Checks if current frontier matches the overall goal.
    2. If not, detects new frontiers and their distances
    3. Select a frontier based on:
       - Heading alignment with robot's current orientation
       - Forward progress towards goal
       - Distance from robot
       - Proximity to ultimate goal
    3. Plans a path to the selected frontier.
    4. Update:
        - state.frontier_goal with selected frontier
        - state.path with planned path to the selected frontier
    """

    # 如果当前已经锁定的frontier正好就是最终goal cell，那就直接保持这个目标
    if state.frontier_goal == state.goal["cell"]:
        print("Found matched frontier to goal")
        return state.frontier_goal

    else:
        # 1. 基于当前OGM做frontier检测
        frontiers, distances = detect_frontiers(state)
        goal_cell = state.goal["cell"]

        # 记录一下可见frontier和距离信息，用于渲染/日志
        state.frontier_candidates = frontiers
        state.frontier_distances = distances

        # -----START: To be completed by candidate-----
        #   1. 与当前朝向一致性（朝向差越小说明转头越少，代价越低）
        #   2. 向最终目标前进的趋势（走到这个frontier之后，到最终goal是否更近）
        #   3. 机器人到该frontier的预计路程（越近越好）
        #   4. 该frontier本身离最终目标有多近（越近越好）


        # 如果当前没有任何frontier（极端情况：地图几乎全已知），
        # 就直接把最终goal当成frontier目标，避免后续崩溃。
        if not frontiers:
            best_frontier_cell = goal_cell
        else:
            # 机器人当前状态
            rx = state.pose["x"]
            ry = state.pose["y"]
            rtheta = state.pose["theta"]

            cell_size_m = state.world["cell_size_m"]

            # 终极目标在世界坐标下的位置（米）
            gx, gy = cell_center(goal_cell, cell_size_m)

            # 机器人当前到最终目标的直线距离
            robot_goal_dist = math.hypot(gx - rx, gy - ry)

            best_score = float("inf")
            best_frontier_cell = None

            for cell in frontiers:
                # 优先级：如果某个frontier恰好就是最终目标单元格，直接选它
                if cell == goal_cell:
                    best_frontier_cell = cell
                    best_score = float("-inf")
                    break

                fx, fy = cell_center(cell, cell_size_m)

                # 1) 朝向一致性：机器人当前朝向 vs 指向该frontier的方向
                heading_to_frontier = math.atan2(fy - ry, fx - rx)
                heading_penalty = abs(ang_diff(heading_to_frontier, rtheta))
                # heading_penalty ∈ [0, π]，越小越好（转角越小）

                # 2) 机器人走到该frontier的“路程代价”
                # detect_frontiers() 返回的 distances 是 BFS 步数（4-连通步数）
                # 乘以 cell_size_m 就得到大概的行走距离（米）
                step_dist_m = distances.get(cell, float("inf")) * cell_size_m

                # 3) 该frontier到最终目标有多远（越近越好）
                frontier_goal_dist = math.hypot(gx - fx, gy - fy)

                # 4) 前进收益：到达该frontier之后，相比现在，离最终目标是否更近
                progress_towards_goal = robot_goal_dist - frontier_goal_dist
                # 正值 = “更接近终点”；负值 = “离终点更远了”

                # 组合代价函数（权重是经验启发式缩放到相似量级）
                # - heading_penalty      : 不想大幅掉头
                # - step_dist_m          : 不想选太远的frontier
                # - frontier_goal_dist   : 倾向于靠近最终目标的frontier
                # - progress_towards_goal: 如果能显著接近终点，则额外奖励（降低cost）
                score = (
                        2.0 * heading_penalty +
                        1.0 * step_dist_m +
                        0.5 * frontier_goal_dist -
                        0.5 * progress_towards_goal
                )

                if score < best_score:
                    best_score = score
                    best_frontier_cell = cell

            # 兜底：如果循环里没能成功赋值（非常极端），就拿第一个frontier
            if best_frontier_cell is None:
                best_frontier_cell = frontiers[0]
        #
        # -----END: To be completed by candidate-----

    # 选出的frontier写回全局状态
    state.frontier_goal = best_frontier_cell

    # 以当前栅格cell为起点，规划一条到 frontier_goal 的路径
    start_cell = pose_to_cell(state.world, state.pose)
    state.path = plan_unknown_world(state, start_cell, state.frontier_goal)

    return


def detect_frontiers(state: SimulationState) -> Tuple[List[Cell], Dict[Cell, int]]:
    """
    Detect frontier cells in an occupancy grid map using a breadth-first search from the robot pose.
    Parameters
    ----------
    state : SimulationState
        The simulation state object providing the world and map information required for frontier
        detection. Expected fields and structure:
          - state.settings: a dict; navigation mode is read from
            state.settings.get("navigation", {}).get("mode", "KNOWN"). Mode must be the string
            "UNKNOWN" (case-insensitive) for frontier detection to run; otherwise the function
            returns ([], {}).
          - state.ogm: a dict describing the occupancy grid map with keys:
              - "grid": 2D numpy array (float) of log-odds or similar values. The code converts this to
                probabilities using the logistic/sigmoid function: p = 1 / (1 + exp(-grid)).
              - "cfg": a dict of optional configuration thresholds:
                  - "prob_free_max" (float, default 0.35) — cells with p <= prob_free_max are treated as free.
                  - "prob_occ_min"  (float, default 0.65) — cells with p >= prob_occ_min are treated as occupied.
              - "minx", "miny" (float) — origin of the occupancy grid in world coordinates.
              - "res" (float) — grid resolution (meters per grid cell).
          - state.world: a dict with world/grid parameters:
              - "cell_size_m" (float) — cell size used by pose_to_cell / cell_center.
              - either "grid_size" (int) or "size_m" (float). If "grid_size" not present, an integer grid
                size is computed as round(size_m / cell_size_m). grid_size must be > 0.
          - state.pose: robot pose used as the BFS start, converted to a starting grid cell using
            pose_to_cell(state.world, state.pose).
    Returns
    -------
    Tuple[List[Cell], Dict[Cell, int]]
        - frontier_cells: list of Cell (tuples of ints, e.g. (cx, cy)) that are reachable free cells
          adjacent (4-connected) to at least one "unknown" cell. The list is sorted by descending
          distance (farthest reachable first) and then by the cell coordinates as a tie-breaker.
        - frontier_distances: dict mapping each returned frontier cell to its integer Manhattan-style
          distance (number of 4-connected steps) from the start cell discovered by the BFS.
    """

    from collections import deque

    # -------------------------
    # 内部函数: 把某个 maze cell 判定成 "free" / "occupied" / "unknown"
    # -------------------------
    def classify(cell: Cell) -> str:
        cx, cy = cell

        # 把 maze 网格 cell 的中心点换算成真实世界坐标 (米)
        wx, wy = cell_center(cell, cell_size)

        # 再把真实世界坐标 (wx, wy) 投到 OGM 网格坐标 (ix, iy)
        ix = int((wx - minx) / res)
        iy = int((wy - miny) / res)

        # 落在OGM范围外 = 当作占用，防止机器人往图外乱走
        if not (0 <= ix < width and 0 <= iy < height):
            return "occupied"

        # 把 log-odds 或类似存储的值sigmoid成概率
        p = prob[iy, ix]

        # 超过占用阈值 -> occupied
        if p >= occ_thresh:
            return "occupied"

        # 低于自由阈值 -> free
        if p <= free_thresh:
            return "free"

        # 介于两者之间 -> unknown
        return "unknown"

    # 1. 只有在 UNKNOWN 模式下才进行 frontier 探索
    mode = state.settings.get("navigation", {}).get("mode", "KNOWN").upper()
    if mode != "UNKNOWN":
        return [], {}

    # 2. 读取并检查 OGM
    ogm = state.ogm
    if not ogm or ogm["grid"].size == 0:
        return [], {}

    grid = ogm["grid"]
    cfg = ogm["cfg"]

    # 把 log-odds 转成概率
    prob = 1 / (1 + np.exp(-grid))
    free_thresh = cfg.get("prob_free_max", 0.35)
    occ_thresh = cfg.get("prob_occ_min", 0.65)

    # 世界网格信息
    cell_size = state.world["cell_size_m"]
    grid_size = state.world.get("grid_size", int(round(state.world["size_m"] / cell_size)))
    if grid_size <= 0:
        return [], {}

    # OGM 尺寸和坐标偏移
    width = grid.shape[1]
    height = grid.shape[0]
    minx = ogm["minx"]
    miny = ogm["miny"]
    res = ogm["res"]

    # 3. BFS 起点：机器人当前所在的maze cell
    start_cell = pose_to_cell(state.world, state.pose)

    # 如果起点本身在OGM里看上去是"occupied"，直接放弃
    if classify(start_cell) == "occupied":
        return [], {}

    # BFS 队列和距离映射
    queue: "deque[Cell]" = deque([start_cell])
    distances: Dict[Cell, int] = {start_cell: 0}

    # 4. BFS 探索所有“可走的free cell”
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            # 限制在 maze 的 grid_size 范围内
            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue

            cell = (nx, ny)

            # 已经访问过就跳过
            if cell in distances:
                continue

            # BFS 只能穿过"free"区域
            if classify(cell) != "free":
                continue

            distances[cell] = distances[(cx, cy)] + 1
            queue.append(cell)

    # 5. frontier 定义：可到达的free cell，且与至少一个"unknown" cell 4-邻接
    frontier_cells: List[Cell] = []
    for cell, dist in distances.items():
        cx, cy = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cx + dx, cy + dy)
            if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size):
                continue
            if classify(nb) == "unknown":
                frontier_cells.append(cell)
                break  # 这个cell已经是frontier了，不需要再检查其他邻居

    if not frontier_cells:
        return [], {}

    # 6. 排序策略：距离越远(步数越大)的frontier靠前
    #    （这样可以倾向于把探索边界往外推，而不是只在附近抖动）
    frontier_cells.sort(key=lambda cell: (-distances[cell], cell))

    # 只给这些frontier保存距离
    frontier_distances = {cell: distances[cell] for cell in frontier_cells}

    return frontier_cells, frontier_distances


def determine_navigation_path(state: SimulationState) -> None:
    """
    Determines the navigation path to the goal cell based on the current simulation state.
    If the navigation mode is set to "UNKNOWN", computes a path to the frontier using
    `determine_frontier_path`. Otherwise, assumes the world is known and the path to the
    goal cell has already been determined during initialization.
    Args:
        state (SimulationState): The current simulation state containing settings and goal information.
    Returns:
        None
    """

    mode = state.settings.get("navigation", {}).get("mode", "KNOWN").upper()

    if mode == "UNKNOWN":
        determine_frontier_path(state)
        return
    else:
        #------------------------------
        # Known world: path to the goal cell already determined at initialisation
        #------------------------------
        if not state.path:
            determine_goal_path(state)
        return

if __name__ == "__main__":
    main()
