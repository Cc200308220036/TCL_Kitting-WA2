"""tray-pose-yolo视觉适配器；不依赖旧搬运包。"""
from __future__ import annotations
import argparse
import math
import os
import sys
import threading
from pathlib import Path
from typing import List
import numpy as np
from scipy.spatial.transform import Rotation
from .state import validate_pose7
SCRIPT_DIR = Path(__file__).resolve().parents[1]
CATKIN_SRC_DIR = Path(os.environ.get("TRAY_CATKIN_SRC", str(SCRIPT_DIR.parent.parent.parent))).expanduser().resolve()
NEW_FDPOSE_DIR = CATKIN_SRC_DIR / "Fdpose"
TRAY_POSE_SCRIPTS_DIR = CATKIN_SRC_DIR / "tray_pose" / "scripts"
_TRAY_POSE_ESTIMATOR = None
_RECORD_FRAME = threading.local()

def validate_vision_arguments(args):
    if args.vision_backend != "tray-pose-yolo":
        raise ValueError("v2仅提取当前tray-pose-yolo技术路线")
    if not str(args.label).strip():
        raise ValueError("label不能为空")
    if not math.isfinite(args.tray_pose_rgbd_timeout) or args.tray_pose_rgbd_timeout <= 0:
        raise ValueError("RGB-D超时必须为有限正数")
    if args.yolo_label not in (None,args.label):
        raise ValueError("YOLO标签必须与tray标签一致")

def _capture_tray_frame(args):
    """在机器人返回前固定同步RGB-D和该时刻的相机到BASE变换。"""
    global _TRAY_POSE_ESTIMATOR
    if not (TRAY_POSE_SCRIPTS_DIR / "estimate_rgbd_pose.py").is_file():
        raise RuntimeError("找不到新tray_pose接口: {}".format(TRAY_POSE_SCRIPTS_DIR))
    module_dir = str(TRAY_POSE_SCRIPTS_DIR)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    loaded_fdpose = sys.modules.get("fdpose")
    if loaded_fdpose is not None:
        loaded_file = Path(getattr(loaded_fdpose, "__file__", "") or "").resolve()
        if NEW_FDPOSE_DIR.resolve() not in loaded_file.parents:
            raise RuntimeError(
                "当前进程已加载其他fdpose包({})，不能安全调用新tray_pose接口；"
                "请在新Python进程运行".format(loaded_file)
            )
    # 此脚本已由各阶段初始化ROS；先采集同步帧，之后直到TF查询结束不能动头/腰。
    from estimate_rgbd_pose import RgbdPoseEstimator, _capture_rgbd

    color_bgr, depth = _capture_rgbd(timeout=args.tray_pose_rgbd_timeout)
    if _TRAY_POSE_ESTIMATOR is None:
        _TRAY_POSE_ESTIMATOR = RgbdPoseEstimator(
            init_ros=False,
            labels=[args.label],
            to_base=True,
            camera_to_head_path=args.tray_pose_camera_to_head,
            base_frame="BASE",
            head_frame="HEAD",
        )
    return (color_bgr.copy(), depth.copy(),
            _TRAY_POSE_ESTIMATOR.camera_to_base.T_base_camera().copy())


def capture_record_frame(args):
    from .runtime import vision_ready
    validate_vision_arguments(args)
    vision_ready("T_base_top")
    return _capture_tray_frame(args)


def record_from_frame(record, frame):
    """后台记录复用同一帧及变换，重试也不重新采集运动中的画面。"""
    _RECORD_FRAME.value = frame
    try:
        return record()
    finally:
        del _RECORD_FRAME.value


def has_record_frame():
    return getattr(_RECORD_FRAME, "value", None) is not None


def _estimate_tray_pose_yolo(args: argparse.Namespace, pose_name: str) -> List[float]:
    frame = getattr(_RECORD_FRAME, "value", None)
    color_bgr, depth, camera_to_base = frame if frame is not None else _capture_tray_frame(args)
    poses = _TRAY_POSE_ESTIMATOR.estimate(
        color_bgr, depth, labels=[args.label], color_order="bgr",
        T_target_camera=camera_to_base,
    )
    if not poses:
        raise RuntimeError("tray_pose YOLO+FDPose未返回{}的位姿".format(args.label))
    matching = [result for result in poses if result.label == args.label]
    if len(matching) != 1:
        raise RuntimeError(
            "tray_pose返回{}个{}目标；本阶段要求视野中恰好一个目标，避免选错tray"
            .format(len(matching), args.label)
        )
    T_base_tray = np.asarray(matching[0].T_object, dtype=np.float64)
    if (T_base_tray.shape != (4, 4)
            or not np.all(np.isfinite(T_base_tray))
            or not np.allclose(T_base_tray[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)):
        raise RuntimeError("tray_pose返回的T_base_tray不是有效4x4齐次矩阵")
    rotation = T_base_tray[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-2)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-2)):
        raise RuntimeError("tray_pose返回的T_base_tray旋转矩阵不正交，拒绝发送位姿")
    quaternion = Rotation.from_matrix(rotation).as_quat()
    # 四元数由Rotation生成；统一在重试入口校验最终七维位姿。
    return [*T_base_tray[:3, 3].tolist(), *quaternion.tolist()]


def _retry_tray_pose(estimator, args: argparse.Namespace, pose_name: str) -> List[float]:
    """识别或位姿校验失败后重新识别，总共最多三次。"""
    import rospy

    for attempt in range(1, 4):
        if rospy.is_shutdown():
            raise RuntimeError("ROS已关闭，停止{}视觉识别".format(pose_name))
        try:
            return validate_pose7(estimator(args, pose_name), pose_name)
        except Exception as exc:
            # Ctrl+C和SystemExit不会被捕获；ROS关闭也不继续重试。
            if rospy.is_shutdown():
                raise
            if attempt == 3:
                rospy.logerr("VISION_FAILED：%s识别第3/3次失败：%s", pose_name, exc)
                raise RuntimeError("{}连续3次识别失败，未获得唯一有效tray位姿：{}".format(
                    pose_name, exc)) from exc
            rospy.logwarn("VISION_RETRY：%s识别第%d/3次失败：%s；1秒后重新识别",
                          pose_name, attempt, exc)
            rospy.sleep(1.0)

def estimate_tray_pose(args, pose_name):
    from .runtime import vision_ready
    validate_vision_arguments(args)
    if not has_record_frame():
        vision_ready(pose_name)
    return _retry_tray_pose(_estimate_tray_pose_yolo,args,pose_name)


def record_first_layer(config):
    """视觉成功后直接覆盖首层状态，不备份，也不保留旧多层任务字段。"""
    import rospy
    from .state import save_state
    args = config.stage3_args(True)
    if not has_record_frame():
        rospy.sleep(args.vision_settle)
    pose = estimate_tray_pose(args, "T_base_top")
    if rospy.is_shutdown():
        raise RuntimeError('ROS已关闭，禁止更新首层状态')
    fresh = dict(top_tray_pose=pose, layer_index=1, top_pose_source='vision_first_layer_reset', top_pose_backend=args.vision_backend, top_pose_label=args.label, stage3_complete=True, package_steps={'record_top': 'SUCCEEDED'})
    save_state(fresh, config.state_path)
    return pose
