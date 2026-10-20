#!/usr/bin/env python3
"""V2 Stage0标定：采集参考tray和双TCP；也可只预览预抓取目标。

由上级test_stage0.py迁移，复用V2视觉和Stage0规划，不发送运动。
默认只打印计划；--calibrate采集并保存，--preview读取视觉并计算目标。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tray_control/config/transport_tray.yaml"
DEFAULT_PREVIEW_STATE = ROOT / "tray_control/runtime/stage0_preview.json"

# 直接运行文件时，将V2根目录加入搜索路径，供tray_control和terminal导入。
if __package__ in (None, "") and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--calibrate", action="store_true", help="视觉采集后人工对准双爪，读取TCP并覆盖标定；不运动"
    )
    mode.add_argument(
        "--preview", action="store_true", help="读取视觉并计算预抓取目标，写入预览状态；不运动"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="V2配置YAML路径")
    parser.add_argument(
        "--calibration-file", help="标定JSON；默认读取YAML，显式相对路径基于当前目录"
    )
    parser.add_argument("--label", help="托盘类别；默认读取YAML")
    parser.add_argument(
        "--vision-backend", choices=("tray-pose-yolo",), help="默认读取YAML，V2仅支持tray-pose-yolo"
    )
    parser.add_argument("--tray-pose-camera-to-head", help="相机外参npy；默认读取YAML")
    parser.add_argument("--tray-pose-rgbd-timeout", type=float, help="采图超时(s)；默认读取YAML")
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_PREVIEW_STATE),
        help="仅预览时写入的状态JSON，默认不占用主流程状态",
    )
    return parser.parse_args(argv)


def read_current_tcp_xyzquats(timeout_s: float = 8.0) -> Dict[str, List[float]]:
    """只读取当前左右电机TCP，不发送任何运动指令。"""
    import rospy
    from naviai_controller import ArmGroup, NaviController, RobotModel
    from tray_control.stage0 import validate_xyzquat

    robot = NaviController(auto_spin=False, model=RobotModel.WA2)
    deadline = time.monotonic() + timeout_s
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        poses: Dict[str, List[float]] = {}
        ready = True
        for side, arm in (("left", ArmGroup.LEFT), ("right", ArmGroup.RIGHT)):
            value = robot.get_tcp_rt(arm)
            age = robot.get_tcp_age(arm)
            if value is None or age is None or age > 0.5:
                ready = False
                break
            poses[side] = validate_xyzquat(value)
        if ready:
            return poses
        rospy.sleep(0.02)
    raise RuntimeError("左右TCP位姿在{:.1f}秒内未就绪".format(timeout_s))


def execute(options):
    """在本地配置下执行标定或预览；标定不修改搬运运行状态。"""
    import rospy
    from tray_control import stage0
    from tray_control.config import TrayConfig
    from tray_control.common import state, vision
    from tray_control.common.runtime import operation
    from terminal import confirm, before_vision

    config = TrayConfig(options.config)
    values = config.vision_args()
    for key in ("label", "vision_backend", "tray_pose_camera_to_head", "tray_pose_rgbd_timeout"):
        value = getattr(options, key)
        if value is not None:
            values[key] = value
    args = argparse.Namespace(**values)
    vision.validate_vision_arguments(args)
    calibration_file = (
        str(Path(options.calibration_file).expanduser().resolve())
        if options.calibration_file
        else config.calibration_file()
    )
    rospy.init_node("transport_tray_v2_stage0_calibrate", anonymous=True)
    print("标定文件：", calibration_file)
    print("相机外参：", args.tray_pose_camera_to_head)
    with operation(confirm, before_vision):
        if options.calibrate:
            pose = stage0.validate_xyzquat(vision.estimate_tray_pose(args, "T_base_tray"))
            print("参考tray BASE位姿：", pose)
            if not confirm(
                "calibration_capture",
                {
                    "prompt": "保持tray不动，将左右实际夹爪末端中心分别对准tray两侧中心预抓取点；"
                    "姿态调整正确后按Enter读取TCP并覆盖标定，输入其他内容或Ctrl+C取消："
                },
            ):
                print("已取消标定，未写入文件")
                return
            tcp = read_current_tcp_xyzquats()
            if rospy.is_shutdown():
                raise RuntimeError("ROS已关闭，不写入标定")
            stage0.save_calibration(
                calibration_file, pose, tcp, vision_backend=args.vision_backend, label=args.label
            )
            print("左右TCP BASE位姿：", tcp)
            print("Stage0标定已保存：", calibration_file)
            return
        plan = stage0.plan_pregrasp(args, calibration_file)
        if rospy.is_shutdown():
            raise RuntimeError("ROS已关闭，不保存预览")
        preview_file = str(Path(options.state_file).expanduser().resolve())
        state.save_state(
            dict(
                pickup_tray_pose=plan["tray_pose"],
                stage0_pregrasp_targets=plan["targets"],
                stage0_complete=False,
            ),
            preview_file,
        )
        print("预抓取左TCP：", plan["targets"]["left"])
        print("预抓取右TCP：", plan["targets"]["right"])
        print("仅计算，未发送运动；预览状态：", preview_file)


def main(argv=None):
    options = parse_args(argv)
    if not (options.calibrate or options.preview):
        print("计划：--calibrate识别tray→人工对准双爪→读取双TCP→保存标定；不发送运动。")
        print("核对：--preview识别tray并打印预抓取目标；默认写独立stage0_preview.json。")
        return
    execute(options)


if __name__ == "__main__":
    main()
