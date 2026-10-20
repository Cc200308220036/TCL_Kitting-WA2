#!/usr/bin/env python3
"""对齐P_pre后施加共同XY偏置再寻槽；低位仅跳过Z移动，共同Z导纳及自动释放。"""
import argparse
import math

XY_OFFSET_M = [0.0, 0.0]  # base坐标系，单位m；到P_pre后左右TCP共同偏移。
SEARCH_PATH = "square"  # square回字形；spiral螺旋；circle同心圆。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--state-file", help="包含上一层top_tray_pose/layer_index的状态文件")
    parser.add_argument("--search-path", choices=("square", "spiral", "circle"), default=SEARCH_PATH,
                        help="XY寻槽路线，默认回字形")
    parser.add_argument("--square-radius", type=float, default=None,
                        help="回字形最远角点到中心的距离，单位m；默认读取YAML")
    parser.add_argument("--square-ring-step", type=float, default=None,
                        help="回字形各方环半边长的增量，单位m")
    parser.add_argument("--xy-offset", nargs=2, type=float, default=XY_OFFSET_M, metavar=("X_M", "Y_M"),
                        help="到P_pre后双臂共同XY偏置，单位m；默认0 0")
    options = parser.parse_args()
    if not all(math.isfinite(value) for value in options.xy_offset):
        parser.error("XY偏置必须为有限数值")
    if options.square_radius is not None and (not math.isfinite(options.square_radius) or options.square_radius <= 0):
        parser.error("回字形最大半径必须为有限正数")
    if options.square_ring_step is not None and (not math.isfinite(options.square_ring_step) or options.square_ring_step <= 0):
        parser.error("回字形方环间距必须为有限正数")
    if not options.execute:
        print("计划：识别手中tray → 规划P_pre并记录高度差 → 对齐P_pre的XY（低位只跳过Z移动）→ 双臂共同XY偏置 → Z下降/XY寻槽/Yaw搜索 → 张爪撤离 → 记录顶层；加--execute执行。")
        print("Stage3直接调用封装place()，寻槽导纳、Z速度和M/D/K均使用封装配置。")
        print("到P_pre后XY偏置 [m]：{}。".format(options.xy_offset))
        if options.search_path == "circle":
            print("XY路线：circle；半径和圈间距读取YAML，沿用配置XY速度并自动计算超时。")
        elif options.search_path == "square":
            print("XY路线：square；半径={}，方环间距={}；未指定时读取YAML。".format(
                options.square_radius, options.square_ring_step))
        else:
            print("XY路线：spiral；使用配置中的螺旋半径、螺距和速度。")
        print("XY寻槽固定共同Z导纳；直接使用持物基线，目标力读取配置。")
        return

    import rospy
    from tray_control import MultiLayerTray
    from tray_control.algorithms.moving_z_admittance import MovingZParameters
    from terminal import confirm, before_vision

    rospy.init_node("test_stage3_multi_z_search_place", anonymous=True)
    tray = MultiLayerTray(state_path=options.state_file, confirm=confirm, before_vision=before_vision)
    radius = options.square_radius if options.search_path == "square" else None
    ring_step = options.square_ring_step if options.search_path == "square" else None
    stage3_args = tray.config.stage3_args(True, radius=radius,
        search_path=options.search_path, ring_step=ring_step)
    print("XY寻槽固定共同Z；持物基线相对目标力={:.1f}N，不二次归零；common连续超目标{}周期同步上退{:.2f}mm后恢复下降导纳，重新触发迟滞={:.1f}N。".format(
        stage3_args.search_z_target_force,
        stage3_args.search_z_retreat_trigger_cycles, 1000 * stage3_args.search_z_retreat_distance,
        stage3_args.search_z_recovery_hysteresis), flush=True)
    print("XY寻槽路线={}，最大半径={:.1f}mm，XY速度={:.1f}mm/s，路线超时={:.1f}s".format(
        options.search_path, 1000 * stage3_args.search_max_radius,
        1000 * stage3_args.search_xy_speed, stage3_args.search_timeout), flush=True)
    mdk = MovingZParameters()  # 正式start_moving_z_search/step_radius_z_admittance也使用此模型。
    print("移动Z导纳模型：M={}kg，D={}N·s/m，K={}N/m，平衡点增益={}，力误差死区=±{}N".format(
        mdk.mass, mdk.damping, mdk.stiffness, mdk.equilibrium_gain, mdk.force_deadband))
    print("XY寻槽Z下降限速={:g}mm/s，加速度={:g}mm/s²；退让限速={:g}mm/s，加速度={:g}mm/s²".format(
        1000 * stage3_args.search_z_down_speed,
        1000 * stage3_args.search_z_down_acceleration,
        1000 * stage3_args.search_z_speed_limit, 1000 * stage3_args.search_z_acceleration))
    print("最低点回程/Yaw：Z下沉限速={:g}mm/s，退让限速={:g}mm/s".format(
        1000 * stage3_args.radius_z_down_speed, 1000 * stage3_args.radius_z_retreat_speed), flush=True)

    tray.move_to_pre(xy_offset=options.xy_offset, held_tray_pose_from_tcp=False)
    result = tray.place(search_path=options.search_path, search_max_radius=radius,
                        ring_step=ring_step)
    print("新顶层tray [x,y,z,qx,qy,qz,qw]：", result.measurements["top_tray_pose"])
    if result.measurements.get("force_csv_path"):
        print("Stage3力数据CSV：", result.measurements["force_csv_path"], flush=True)
    print("状态文件：", result.state_path)


if __name__ == "__main__":
    import sys
    if "--execute" in sys.argv:
        from tray_control.observability.timing import run_with_summary
        sys.exit(run_with_summary(main, "Stage3多层"))
    else:
        sys.exit(main())
