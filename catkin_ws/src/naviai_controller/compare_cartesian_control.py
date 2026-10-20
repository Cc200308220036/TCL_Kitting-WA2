#!/usr/bin/env python3
"""在新机器人 Orin 上对比 MoveL、SpeedL 和 ServoL 的最小示例。

三种模式完成相同任务：让指定机械臂 TCP 沿基座 X 轴移动一小段距离。
一次运行只执行一种控制模式，避免不同控制接口争抢机器人控制权。

示例：
    python3 TCL_Orin/compare_cartesian_control.py --mode movel --execute
    python3 TCL_Orin/compare_cartesian_control.py --mode speedl --execute
    python3 TCL_Orin/compare_cartesian_control.py --mode servol --execute
"""

import argparse
import time

import rospy

from naviai_controller import ArmGroup, NaviController, RobotModel


SERVO_PERIOD = 0.02  # ServoL 已验证周期：0.02 s，即 50 Hz
SERVO_GAIN = 800     # ServoL 已验证位置跟踪增益


def wait_robot_ready(controller, arm, timeout=5.0):
    """等待 TCP 和上肢状态到达，并拒绝在奇异/保护状态下开始。"""
    deadline = time.monotonic() + timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        pose = controller.get_tcp_rt(arm)
        tcp_age = controller.get_tcp_age(arm)
        state_age = controller.get_uplimb_state_age()
        singular = controller.get_is_singular(arm)
        cmd_num = controller.get_cmd_num()

        fresh = (
            pose is not None
            and tcp_age is not None
            and state_age is not None
            and tcp_age <= 0.2
            and state_age <= 0.2
        )
        if fresh:
            if singular is None:
                raise RuntimeError("SDK 未提供机械臂奇异状态")
            if singular:
                raise RuntimeError("机械臂处于奇异状态，拒绝运动")
            if cmd_num == 15:
                raise RuntimeError("机器人处于 PROTECTED 状态，请先排查并解锁")
            return list(pose)
        rospy.sleep(0.02)
    raise RuntimeError("5 秒内没有收到新鲜的 TCP/UplimbState 数据")


def run_movel(controller, arm, start_pose, distance):
    """MoveL：一次发送最终绝对位姿，由底层规划直线轨迹。"""
    target = start_pose.copy()
    target[0] += distance

    rospy.loginfo("MoveL target=%s", target)
    try:
        ok = controller.movel(
            target,
            arm,
            v=0.02,       # 最大线速度，m/s
            acc=0.05,     # 加速度参数，具体单位以 SDK 为准
            is_async=False,
        )
    except BaseException:
        # 正常情况下同步 MoveL 会自行结束；仅在异常/Ctrl+C 时紧急停止。
        controller.stop()
        raise
    if not ok:
        controller.stop()
        raise RuntimeError("MoveL 服务执行失败")


def run_speedl(controller, arm, distance, duration):
    """SpeedL：连续发送 TCP 速度，时间到后必须主动发零并关闭。"""
    vx = distance / duration
    if abs(vx) > 0.02:
        raise ValueError("示例限制 SpeedL 线速度不超过 0.02 m/s")

    if not controller.enable_speedl(True):
        raise RuntimeError("SpeedL 使能失败")

    rate = rospy.Rate(50)
    started = time.monotonic()
    try:
        while not rospy.is_shutdown() and time.monotonic() - started < duration:
            # [vx, vy, vz, wx, wy, wz]，单位为 m/s 和 rad/s。
            controller.speedl([vx, 0, 0, 0, 0, 0], arm, acc=0.05)
            rate.sleep()
    finally:
        # SpeedL 会保持最近的速度意图，因此退出时必须显式发送零速度。
        controller.stop_speedl(arm)
        controller.enable_speedl(False)


def run_servol(controller, arm, start_pose, distance, duration):
    """ServoL：以 50 Hz 连续发送插值后的绝对 TCP 位姿。"""
    if not controller.set_servo_params(SERVO_PERIOD, SERVO_GAIN, arm=arm):
        raise RuntimeError("ServoL 参数设置失败")

    rate = rospy.Rate(1.0 / SERVO_PERIOD)
    started = time.monotonic()
    try:
        while not rospy.is_shutdown():
            elapsed = time.monotonic() - started
            ratio = min(elapsed / duration, 1.0)

            # ServoL 输入是绝对位姿，不是位姿增量。第一个目标从实测位姿开始，
            # 然后只修改基座 X 坐标；姿态四元数保持不变且已经归一化。
            target = start_pose.copy()
            target[0] += distance * ratio
            controller.servol(target, arm)

            if ratio >= 1.0:
                break
            rate.sleep()
    finally:
        # ServoL 结束、异常或 Ctrl+C 时，都必须停止运动并清除伺服参数。
        controller.stop()
        controller.clear_servo_params()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("movel", "speedl", "servol"), required=True,
        help="本次要测试的笛卡尔控制方式",
    )
    parser.add_argument(
        "--arm", choices=("left", "right"), default="left",
        help="测试左臂或右臂，默认 left",
    )
    parser.add_argument(
        "--distance", type=float, default=0.01,
        help="沿基座 X 轴移动距离，单位 m，默认 0.01",
    )
    parser.add_argument(
        "--duration", type=float, default=1.0,
        help="SpeedL/ServoL 运动时长，单位 s，默认 1.0",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="必须显式提供该参数才允许连接真机执行",
    )
    return parser.parse_args(rospy.myargv()[1:])


def main():
    args = parse_args()
    if not args.execute:
        raise RuntimeError("这是实机运动示例；确认安全后添加 --execute")
    if not 0.0 < abs(args.distance) <= 0.02:
        raise ValueError("示例要求 0 < |distance| <= 0.02 m")
    if not 0.5 <= args.duration <= 5.0:
        raise ValueError("示例要求 0.5 <= duration <= 5.0 s")

    rospy.init_node("compare_cartesian_control", anonymous=True)
    arm = ArmGroup.LEFT if args.arm == "left" else ArmGroup.RIGHT
    controller = NaviController(model=RobotModel.WA2, auto_spin=True)
    start_pose = wait_robot_ready(controller, arm)

    rospy.logwarn(
        "即将用 %s 控制 WA2 %s 臂沿基座 X 移动 %.1f mm",
        args.mode,
        args.arm,
        args.distance * 1000.0,
    )
    rospy.logwarn("起始 TCP=%s", start_pose)
    input("确认工作空间安全、急停可用后按 Enter；Ctrl+C 取消：")

    if args.mode == "movel":
        run_movel(controller, arm, start_pose, args.distance)
    elif args.mode == "speedl":
        run_speedl(controller, arm, args.distance, args.duration)
    else:
        run_servol(controller, arm, start_pose, args.distance, args.duration)

    rospy.loginfo("%s 示例执行结束", args.mode)


if __name__ == "__main__":
    main()
