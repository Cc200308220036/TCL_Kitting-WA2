#!/usr/bin/env python3
"""调用prep/grip，抓取后同步抬升，再记录视觉与TCP；不转腰、不张爪。

默认仅打印计划，不导入ROS。实机：
    python3 test_stage0_stage1_pose_record.py --execute
每轮默认使用独立状态文件，数据保存在pose_record_runs/<时间>/result.json。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
LIFT_HEIGHT_M = 0.01  # 测试抬升高度：5cm；仅覆盖当前测试实例，不修改YAML。
LIFT_LABEL = "{:g}mm".format(1000 * LIFT_HEIGHT_M)


def pose_delta(before, after):
    """base坐标下有符号位移和XY净位移，不代表累计路径长度。"""
    delta = [1000.0 * (after[i] - before[i]) for i in range(3)]
    return dict(delta_xyz_mm=delta, xy_net_distance_mm=math.hypot(*delta[:2]))


def pose_details(pose):
    from scipy.spatial.transform import Rotation
    return dict(xyzquat=list(pose),
                rpy_xyz_deg=Rotation.from_quat(pose[3:]).as_euler("xyz", degrees=True).tolist())


def print_pose(label, pose):
    detail = pose_details(pose)
    print(label)
    print("  xyzquat [m, xyzw] = " + json.dumps(detail["xyzquat"]))
    print("  R/P/Y [deg] = " + str([round(v, 6) for v in detail["rpy_xyz_deg"]]))


def print_report(report):
    print("\n========== Stage0 / Stage1 位姿记录 ==========")
    print("结果：" + ("全部采集完成" if report["completed"] else "采集未全部完成"))
    if report.get("error"):
        print("错误：" + report["error"])
    print("标定文件：" + report.get("calibration_file", "未读取"))
    print("\n两次视觉识别的原始XYZ / Roll / Pitch / Yaw（base坐标系，XYZ单位：m，角度单位：度）")
    raw_before = report.get("stage0", {}).get("vision_pose_before_symmetry_reconstructed")
    for label, pose in (("第一次：Stage0抓取前（由对称对齐结果逆变换恢复）", raw_before),
                        ("第二次：抓取并抬升{}后（原始视觉结果）".format(LIFT_LABEL), report.get("tray_after_grip"))):
        if pose is None:
            print("  {}：未取得数据".format(label))
            continue
        roll, pitch, yaw = pose_details(pose)["rpy_xyz_deg"]
        print("  {}：X={:+.6f}m  Y={:+.6f}m  Z={:+.6f}m  Roll={:+.6f}°  Pitch={:+.6f}°  Yaw={:+.6f}°".format(
            label, pose[0], pose[1], pose[2], roll, pitch, yaw))
    print()
    if "stage0" in report:
        s0 = report["stage0"]
        print_pose("Stage0用于预抓取规划的tray位姿（对称对齐后）", s0["tray_pose"])
        print("  180度对称翻转：" + str(s0["symmetry_flipped"]))
        print("  相对标定原始/对齐Yaw差 [deg]：{:.6f} / {:.6f}".format(
            math.degrees(s0["raw_yaw_delta"]), math.degrees(s0["aligned_yaw_delta"])))
        print_pose("Stage0视觉原始姿态对应位姿（从对称对齐结果逆变换恢复，四元数符号可能不同）",
                   s0["vision_pose_before_symmetry_reconstructed"])
        for side in ("left", "right"):
            print_pose("Stage0预抓取目标TCP " + side, s0["targets"][side])
            print("  相对标定TCP的变化：" + json.dumps(
                report["calibration_to_stage0_tcp_delta"][side], ensure_ascii=False))
        print("Stage0 tray相对标定XYZ变化：" + json.dumps(report["calibration_to_stage0_tray_delta"]))
    for key, label in (("tcp_before_grip", "Stage1靠拢前实测TCP"),
                       ("tcp_after_grip", "Stage1抓取完成后实测TCP"),
                       ("tcp_after_lift", "同步抬升{}后实测TCP".format(LIFT_LABEL)),
                       ("tcp_after_post_vision", "抓取后视觉采集结束时实测TCP")):
        if key in report:
            for side in ("left", "right"):
                print_pose(label + " " + side, report[key][side])
    if "stage1_tcp_delta" in report:
        print("Stage1实际位移：抓取完成TCP - 靠拢前TCP（base坐标系）")
        for side, delta in report["stage1_tcp_delta"].items():
            dx, dy, dz = delta["delta_xyz_mm"]
            print("  {}: ΔX={:+.6f}mm ΔY={:+.6f}mm ΔZ={:+.6f}mm XY净位移={:.6f}mm".format(
                side, dx, dy, dz, delta["xy_net_distance_mm"]))
    if "tray_after_grip" in report:
        print_pose("抓取并抬升{}后视觉tray位姿（原始视觉结果）".format(LIFT_LABEL), report["tray_after_grip"])
        print_pose("抓取后tray位姿（对齐到Stage0的180度等价分支）", report["tray_after_grip_aligned"])
        print("抓取前后tray变化：" + json.dumps(report["tray_before_after_delta"], ensure_ascii=False))
        print("视觉采集期间TCP变化：" + json.dumps(report["post_vision_tcp_delta"], ensure_ascii=False))
        print("视觉采集期间TCP静止检查：" + str(report["post_vision_tcp_stationary"]))
    print("以上XY距离是起终点净位移；姿态和位移均基于base。")
    print("Stage1已完成：" + str(report.get("grip_completed", False)))
    print("同步抬升{}已完成：{}".format(LIFT_LABEL, report.get("lift_completed", False)))
    print("数据文件：" + report["result_path"], flush=True)


def execute(args, report):
    import rospy
    from scipy.spatial.transform import Rotation
    from naviai_controller import NaviController, RobotModel
    from tray_control import SingleLayerTray, stage0, stage1
    from tray_control.common import vision
    from tray_control.common.runtime import operation
    from tray_control.observability.timing import phase
    from terminal import confirm, before_vision

    rospy.init_node("stage0_stage1_pose_record", anonymous=True)
    tray = SingleLayerTray(config_path=args.config, state_path=report["state_path"],
                           confirm=confirm, before_vision=before_vision)
    if not 0 < LIFT_HEIGHT_M <= 0.15:
        raise ValueError("测试抬升高度必须在(0, 0.15]m内")
    report["lift_height_m"] = LIFT_HEIGHT_M
    report["calibration_file"] = tray.config.calibration_file()
    report["calibration"] = stage0.load_calibration(report["calibration_file"])
    report["vision_backend"] = tray.config.common["vision"]["vision_backend"]
    robot = NaviController(auto_spin=False, model=RobotModel.WA2)

    # 等待只读控制器连接就绪；后续快照不长时间重试过期反馈。
    deadline = time.monotonic() + 8.0
    while not rospy.is_shutdown():
        try:
            stage1.read_tcp_pair(robot)
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            rospy.sleep(0.05)
    if rospy.is_shutdown():
        raise RuntimeError("ROS已关闭")

    prep = tray.prep(speed=0.010, acceleration=0.020)
    report["stage0"] = dict(prep.measurements)
    s0 = report["stage0"]
    raw_pose = list(s0["tray_pose"])
    if s0["symmetry_flipped"]:
        raw_pose[3:] = (Rotation.from_quat(raw_pose[3:])
                        * Rotation.from_euler("z", -math.pi)).as_quat().tolist()
    s0["vision_pose_before_symmetry_reconstructed"] = raw_pose
    calibration = report["calibration"]
    report["calibration_to_stage0_tray_delta"] = pose_delta(
        calibration["reference_tray_xyzquat"], s0["tray_pose"])
    report["calibration_to_stage0_tcp_delta"] = {
        side: pose_delta(calibration["reference_tcp_xyzquats"][side], s0["targets"][side])
        for side in ("left", "right")}
    report["tcp_before_grip"] = stage1.read_tcp_pair(robot)
    report["pregrasp_tracking_delta"] = {
        side: pose_delta(s0["targets"][side], report["tcp_before_grip"][side])
        for side in ("left", "right")}

    tray.grip(free_speed=0.006, contact_speed=0.001, acceleration=0.010)
    report["grip_completed"] = True
    report["tcp_after_grip"] = stage1.read_tcp_pair(robot)
    report["stage1_tcp_delta"] = {
        side: pose_delta(report["tcp_before_grip"][side], report["tcp_after_grip"][side])
        for side in ("left", "right")}
    tray.lift(speed=0.010, acceleration=0.020, height_m=LIFT_HEIGHT_M, robot=robot)
    report["lift_completed"] = True
    report["tcp_after_lift"] = stage1.read_tcp_pair(robot)
    # 只调用视觉封装；operation提供视觉准备回调，不发运动。
    with phase("Stage1", "抬升后视觉复测"), operation(confirm, before_vision):
        report["tray_after_grip"] = vision.estimate_tray_pose(
            argparse.Namespace(**tray.config.base()), "抓取并抬升{}后 T_base_tray".format(LIFT_LABEL))
    report["tcp_after_post_vision"] = stage1.read_tcp_pair(robot)
    aligned, raw_yaw, aligned_yaw, flipped = stage0.align_tray_half_turn_symmetry(
        report["tray_after_grip"], s0["tray_pose"])
    report["tray_after_grip_aligned"] = aligned
    report["tray_before_after_delta"] = dict(
        **pose_delta(s0["tray_pose"], aligned),
        raw_yaw_delta_deg=math.degrees(raw_yaw),
        aligned_yaw_delta_deg=math.degrees(aligned_yaw), symmetry_flipped=flipped)
    report["post_vision_tcp_delta"] = {
        side: dict(**pose_delta(report["tcp_after_lift"][side], report["tcp_after_post_vision"][side]),
                   rotation_difference_deg=math.degrees((
                       Rotation.from_quat(report["tcp_after_post_vision"][side][3:])
                       * Rotation.from_quat(report["tcp_after_lift"][side][3:]).inv()).magnitude()))
        for side in ("left", "right")}
    report["post_vision_tcp_stationary"] = all(
        math.sqrt(sum(v*v for v in d["delta_xyz_mm"])) <= 3.0
        and d["rotation_difference_deg"] <= 1.0
        for d in report["post_vision_tcp_delta"].values())
    if rospy.is_shutdown():
        raise RuntimeError("ROS已关闭，采集不得标记完成")
    report["completed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="执行真实预抓取、抓取及同步抬升{}".format(LIFT_LABEL))
    parser.add_argument("--config", default=str(ROOT / "tray_control/config/transport_tray.yaml"))
    parser.add_argument("--output-dir", help="本轮JSON与独立状态文件保存目录")
    parser.add_argument("--state-file", help="可选指定阶段状态文件；prep会重置其中本轮阶段记录")
    args = parser.parse_args()
    if not args.execute:
        print("计划：prep → 读取靠拢前TCP → grip → 读取抓取后TCP → lift同步抬升{} → 视觉复测 → 集中打印。".format(LIFT_LABEL))
        print("真实执行请加 --execute；完成后保持夹持，不转腰、不张爪。")
        return 0
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT / "pose_record_runs" / stamp
    output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, grip_completed=False, timestamp_beijing=stamp,
                  result_path=str(output / "result.json"),
                  state_path=str(Path(args.state_file).expanduser().resolve()) if args.state_file
                  else str(output / "tray_state.json"), arguments=vars(args))
    exit_code = 0
    try:
        execute(args, report)
    except (KeyboardInterrupt, Exception) as exc:
        report["error"] = "{}: {}".format(type(exc).__name__, exc)
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        Path(report["result_path"]).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print_report(report)
    return exit_code


if __name__ == "__main__":
    import sys
    if "--execute" in sys.argv:
        from tray_control.observability.timing import run_with_summary
        sys.exit(run_with_summary(main, "Stage0/1观测"))
    else:
        sys.exit(main())
