#!/usr/bin/env python3
"""三层搬运入口，只复用v2控制函数，不改四个独立测试。

python3 main.py --execute --lift-cm 1 1 7
--layers 1/2/3控制层数；--search-path square/spiral/circle，固定共同Z导纳。
首层Stage2后双臂下降5cm；放置采图后，顶面记录与返回取料姿态并行。
耗时汇总剔除人工确认；力CSV保留原始真实时间轴。
"""
import argparse
import math
from pathlib import Path

LIFT_HEIGHTS_CM = [1, 1, 7]


def restore_initial_position(robot):
    """按现场提供的五步指令返回取料姿态，每步失败立即停止。"""
    import rospy
    from naviai_controller import ArmGroup
    from tray_control.observability.timing import phase

    arms = [0.0034274884183105314, -0.6290759512376098, 0.27089142729892046,
            -0.10119479791683261, -0.6753672982254557, 0.3916085281298365,
            -0.7539795216426809, -0.2230962605717245, 0.2789568108846652,
            -0.3353186221829674, -0.28536837138744886, -0.42941875875840196,
            -1.0219505528428046, -0.24293222981668805, -0.6105576534904424,
            0.2714628731281313]
    robot.stop()
    robot.clear_servo_params()
    moves = [
        ("返回抬腰", lambda: robot.movej([0., 0., 1.5, .2], ArmGroup.WAIST, v=.8, acc=.2, is_async=False)),
        ("返回转腰", lambda: robot.movej([0., 0., 0., .2], ArmGroup.WAIST, v=.8, acc=.2, is_async=False)),
        ("返回双臂", lambda: robot.movejh(arms, mask=3, v=.4, acc=.15, is_async=False)),
        ("返回下腰", lambda: robot.movej([0., 0., 0., .5], ArmGroup.WAIST, v=.2, acc=.05, is_async=False)),
        ("返回头部", lambda: robot.movej([-.0024864300978223773, .37587057063808177], ArmGroup.NECK, v=.05, acc=.05, is_async=False)),
    ]
    try:
        for name, move in moves:
            with phase("返回取料", name):
                if rospy.is_shutdown() or not move():
                    raise RuntimeError(name + "失败")
    except BaseException:
        robot.stop()
        raise


def stage2_waist(robot):
    """复用主程序机器人，依次执行三步同步腰部运动。"""
    from tray_control.stage2 import turn_waist
    from tray_control.common.runtime import operation
    from terminal import confirm

    with operation(confirm):
        turn_waist(robot)


def record_top_during_return(record, config, robot):
    """先采图并固定TF；位姿记录在线程中处理，主线程返回取料位置。"""
    from concurrent.futures import ThreadPoolExecutor
    import rospy
    from tray_control.common import vision
    from terminal import confirm

    if not confirm("restore_initial_position", {"prompt": "放置撤离完成，Enter采图并并行记录顶面、返回取料姿态："}):
        raise PermissionError("用户取消返回")
    args = config.stage3_args(True)
    rospy.sleep(args.vision_settle)
    frame = vision.capture_record_frame(args)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(vision.record_from_frame, record, frame)
        restore_initial_position(robot)
        return result.result()


def stage3_single(tray, robot, record_top_runner=None):
    """首层沿用5cm预下降，通过正式接口完成放置及记录。"""
    return tray.place(robot=robot, initial_lower_distance=.05, record_top_runner=record_top_runner)


def stage3_multi(tray, options, record_top_runner=None, robot=None):
    """调用正式预放置与多层放置接口，保持主流程配置和运动顺序。"""
    tray.move_to_pre(robot=robot, xy_offset=getattr(options, "xy_offset", (0.0, 0.0)))
    result = tray.place(search_path=options.search_path, record_top_runner=record_top_runner)
    print("Stage3力数据CSV：", result.measurements["force_csv_path"], flush=True)


def record_top_after_place(options):
    """多层已撤离后仅补记视觉顶面，保留目标层号，不执行机器人运动。"""
    import rospy
    from tray_control.config import TrayConfig
    from tray_control.common import state
    from tray_control.stage3_multi import record_top_pose
    from tray_control.common.runtime import operation
    from terminal import confirm, before_vision

    rospy.init_node("transport_tray_v2_record_top", anonymous=True)
    config = TrayConfig(options.config, state_path=options.state_file)
    current = state.load_state(config.state_path)
    layer = current.get("target_layer_index")
    if current.get("stage3_release_and_retreat_complete") is not True:
        raise RuntimeError("状态中未记录多层张爪及撤离完成，禁止补记")
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 2:
        raise RuntimeError("缺少有效多层target_layer_index，禁止自动推算层号")
    args = config.stage3_args(True)
    args.layer_index = layer  # 重复补记同一层，不能让层号递增。
    target = current.get("place_tray_target")
    with operation(confirm, before_vision):
        if not confirm("record_top", {"prompt": "确认第{}层已放置、双臂已撤离，Enter仅补记顶面：".format(layer)}):
            raise PermissionError("用户取消补记")
        pose = record_top_pose(args, expected_pose=target)
    steps = dict(current.get("package_steps") or {})
    steps.update(record_top="SUCCEEDED", place="SUCCEEDED")
    values = {"package_steps": steps}
    if target is not None:
        values["stage3_placed_minus_p_place_z_mm"] = 1000 * (pose[2] - target[2])
    state.update_state(config.state_path, **values)
    print("第{}层顶面补记完成，状态文件：{}".format(layer, config.state_path), flush=True)


def run(options):
    import rospy
    from naviai_controller import NaviController, RobotModel
    from tray_control import SingleLayerTray, MultiLayerTray
    from tray_control.common import state
    from tray_control.observability import timing
    from terminal import confirm, before_vision

    rospy.init_node("transport_tray_v2_main", anonymous=True)
    tray = SingleLayerTray(options.config, state_path=options.state_file, confirm=confirm, before_vision=before_vision)
    robot = NaviController(model=RobotModel.WA2)
    rospy.sleep(1.)
    timer, previous_timer = timing.LayerStateTimer("流程"), timing._active
    timing._active = timer
    status = "异常/中断，已执行部分"
    try:
        print("共享状态文件：", tray.state_path, flush=True)
        for layer in range(1, options.layers + 1):
            if rospy.is_shutdown():
                raise RuntimeError("ROS已关闭")
            timer.layer = layer
            print("\n开始第{}层，抓取后抬升{}cm".format(layer, options.lift_cm[layer - 1]), flush=True)
            tray.prep(speed=.01, acceleration=.02)
            tray.grip(free_speed=.006, contact_speed=.001, acceleration=.01)
            tray.lift(speed=.01, acceleration=.02,
                      height_m=options.lift_cm[layer - 1] / 100., robot=robot)
            stage2_waist(robot)
            record_top_runner = (lambda record: record_top_during_return(record, tray.config, robot)
                                 ) if layer < options.layers else None
            if layer == 1:
                stage3_single(tray, robot, record_top_runner)
            else:
                multi = MultiLayerTray(options.config, state_path=tray.state_path, confirm=confirm, before_vision=before_vision)
                stage3_multi(multi, options, record_top_runner, robot=robot)
            placed = state.load_state(tray.state_path)
            if placed.get("layer_index") != layer or not placed.get("stage3_complete"):
                raise RuntimeError("本层顶面未成功记录，禁止进入下一层")
        status = "完成"
    except BaseException:
        robot.stop()
        robot.clear_servo_params()
        raise
    finally:
        try:
            timer.report(status)
        finally:
            timing._active = previous_timer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("multi",), help="兼容旧多层顶面补记指令")
    parser.add_argument("--record-top-after-place", action="store_true", help="仅补记已撤离的多层顶面，不执行运动")
    parser.add_argument("--layers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--lift-cm", type=float, nargs=3, default=LIFT_HEIGHTS_CM, metavar=("L1", "L2", "L3"))
    parser.add_argument("--state-file", help="单层、多层共用的状态文件")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "tray_control/config/transport_tray.yaml"))
    parser.add_argument("--search-path", choices=("square", "spiral", "circle"), default="square")
    parser.add_argument("--xy-offset", nargs=2, type=float, default=(0.0, 0.0),
                        metavar=("X_M", "Y_M"), help="多层到P_pre后的共同XY偏置，单位m")
    options = parser.parse_args()
    if not all(math.isfinite(value) for value in options.xy_offset):
        parser.error("XY偏置必须为有限数值")
    if options.mode and not options.record_top_after_place:
        parser.error("--mode仅用于--record-top-after-place；完整流程请用--layers")
    if options.record_top_after_place:
        if options.execute:
            record_top_after_place(options)
        else:
            print("计划：仅识别并补记已撤离的多层顶面，保持原目标层号，不执行运动；加--execute执行。")
        return
    if not all(math.isfinite(v) and 0 < v <= 15 for v in options.lift_cm):
        parser.error("三个抬升高度必须在(0,15]cm内")
    if not options.execute:
        print("计划：逐层Stage0预抓取→Stage1抓取及抬升{}cm→Stage2转腰→Stage3放置及记录；共{}层。".format(options.lift_cm, options.layers))
        print("首层转腰后先同步下降5cm；后续层使用P_pre及{}寻槽、共同Z导纳。".format(options.search_path))
        print("层间按给定五步关节指令返回取料姿态；耗时汇总排除人工确认。加--execute执行。")
        return
    run(options)


if __name__ == "__main__":
    main()
