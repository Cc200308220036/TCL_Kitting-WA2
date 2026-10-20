#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO 分割 + FDPose 估姿，默认转到 BASE。

    from estimate_rgbd_pose import RgbdPoseEstimator

    estimator = RgbdPoseEstimator(init_ros=False)
    poses = estimator.estimate(color_bgr, depth, labels=["white_tray"])
    # poses[i].T_object 为 T_base_object
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_FDPOSE_SRC = Path("/root/catkin_ws/src/Fdpose/src")
_YOLO_SRC = Path("/root/catkin_ws/src/naviai_yolo_0909/src")
_DEPTH_SRC = Path("/root/catkin_ws/src/lingbot-depth-rqh")
for _path in (_SCRIPT_DIR, _FDPOSE_SRC, _YOLO_SRC, _DEPTH_SRC):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import rospy
from fdpose.pipeline import FdposePipeline

from camera_to_base import CameraToBase


def _enhance_overexposed_bgr(bgr):
    """压掉过曝高光并拉回局部对比，方便 YOLO 分出白盘边缘。"""
    import cv2
    import numpy as np

    img = np.ascontiguousarray(bgr)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32)
    hi = 175.0
    over = l_f > hi
    if np.any(over):
        l_f[over] = hi + (l_f[over] - hi) * 0.35
    l_ch = np.clip(l_f, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


class _DepthRefineAdapter:
    """把 lingbot-depth-rqh 的 DepthRefineClient 适配为 pipeline 期望的 refine 接口。

    pipeline 要求 depth client 提供 refine(color_bgr, depth) -> depth；
    DepthRefineClient.call_from_arrays(bgr, depth) 返回 (refined_u16, elapsed_ms)，
    这里丢弃耗时，只返回补全后的 uint16 深度（毫米）。
    """

    def __init__(self, service_name="/lingbot/depth_refine", timeout=10.0):
        from inference_depth_refine_client import DepthRefineClient, DepthRefineClientConfig
        self._client = DepthRefineClient(
            DepthRefineClientConfig(service_name=service_name, timeout=timeout)
        )

    def refine(self, color_bgr, depth):
        refined_u16, _elapsed_ms = self._client.call_from_arrays(color_bgr, depth)
        return refined_u16


def _depth_refine_factory(service_name, timeout_sec, **_kwargs):
    """from_ros_services 的 depth_refine_client_factory 回调。"""
    return _DepthRefineAdapter(service_name=service_name, timeout=timeout_sec)


class RgbdPoseEstimator:
    """复用一条 FdposePipeline：YOLO → FDPose，再经 CameraToBase 转到 BASE。"""

    def __init__(
        self,
        labels=None,
        init_ros=True,
        node_name="rgbd_pose_estimator",
        config_path=None,
        timeout=None,
        to_base=True,
        camera_to_head_path=None,
        base_frame="BASE",
        head_frame="HEAD",
        enable_depth_refine=True,
        depth_refine_service="/lingbot/depth_refine",
    ):
        if init_ros and not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)

        kwargs = {}
        if config_path is not None:
            kwargs["config_path"] = config_path
        if timeout is not None:
            kwargs["timeout"] = timeout
        if enable_depth_refine:
            kwargs["depth_refine_service"] = depth_refine_service
            kwargs["depth_refine_client_factory"] = _depth_refine_factory

        self.pipeline = FdposePipeline.from_ros_services(**kwargs)
        self.default_labels = list(labels) if labels is not None else ["white_tray"]
        self.to_base = bool(to_base)
        self.camera_to_base = None
        if self.to_base:
            extra = {}
            if camera_to_head_path is not None:
                extra["camera_to_head_path"] = camera_to_head_path
            self.camera_to_base = CameraToBase(
                base_frame=base_frame,
                head_frame=head_frame,
                **extra,
            )

    def estimate(
        self,
        color,
        depth,
        labels=None,
        color_order="bgr",
        load_mesh=True,
        T_target_camera=None,
        to_base=None,
    ):
        """对一帧 RGB-D 做分割和估姿。

        Returns:
            list[PoseResult]。默认 T_object 为 T_base_object。
            to_base=False 或不建 CameraToBase 时为 T_camera_object。
            未检出时返回空列表。
        """
        if labels is None:
            labels = self.default_labels

        use_base = self.to_base if to_base is None else bool(to_base)
        if T_target_camera is None and use_base and self.camera_to_base is not None:
            T_target_camera = self.camera_to_base.T_base_camera()

        if not any("white" in str(name).lower() for name in labels):
            return self.pipeline.process_rgbd(
                color=color,
                depth=depth,
                labels=labels,
                color_order=color_order,
                load_mesh=load_mesh,
                T_target_camera=T_target_camera,
            )

        pipe = self.pipeline
        labels = pipe._normalize_labels(labels)
        color_bgr = pipe._to_bgr(color, color_order)
        depth = pipe._check_depth(depth, color_bgr.shape[:2])
        color_seg = _enhance_overexposed_bgr(color_bgr)
        if load_mesh:
            pipe.fdpose.load_meshes(labels)
        response = pipe.segmentation.segment(
            color=color_seg, depth=depth, labels=labels,
        )
        if not response.have_objs or not response.items:
            return []
        detections = pipe._to_detection_inputs(response.items)
        depth_used = depth
        if pipe.depth_refine is not None:
            depth_used = pipe.depth_refine.refine(color_bgr, depth)
        depth_used = pipe._check_depth(depth_used, color_bgr.shape[:2])
        return pipe.fdpose.estimate(
            color_bgr=color_bgr,
            depth=depth_used,
            detections=detections,
            T_target_camera=T_target_camera,
        )


def _imgmsg_to_numpy(msg):
    import cv2
    import numpy as np

    enc = str(msg.encoding)
    if enc in ("rgb8", "bgr8", "8UC3"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        arr = np.ascontiguousarray(arr)
        if enc == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    if enc in ("16UC1", "mono16"):
        return np.ascontiguousarray(
            np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        )
    if enc == "32FC1":
        return np.ascontiguousarray(
            np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        )
    raise ValueError("unsupported encoding: %s" % enc)


def _capture_rgbd(
    color_topic="/zj_humanoid/sensor/realsense_head/color/image_raw",
    depth_topic="/zj_humanoid/sensor/realsense_head/aligned_depth_to_color/image_raw",
    timeout=15.0,
):
    import threading

    import message_filters
    from sensor_msgs.msg import Image

    event = threading.Event()
    packet = {}
    error = []

    def callback(color_msg, depth_msg):
        if event.is_set():
            return
        try:
            packet["color"] = _imgmsg_to_numpy(color_msg)
            packet["depth"] = _imgmsg_to_numpy(depth_msg)
        except Exception as exc:
            error.append(exc)
        event.set()

    color_sub = message_filters.Subscriber(color_topic, Image)
    depth_sub = message_filters.Subscriber(depth_topic, Image)
    sync = message_filters.ApproximateTimeSynchronizer(
        [color_sub, depth_sub], queue_size=10, slop=0.08
    )
    sync.registerCallback(callback)
    rospy.loginfo("Waiting synchronized RGBD: %s + %s", color_topic, depth_topic)
    rospy.sleep(1.0)
    if not event.wait(timeout):
        raise TimeoutError("Timeout waiting RGBD after %.1fs" % timeout)
    if error:
        raise error[0]
    return packet["color"], packet["depth"]


def _matrix_to_xyzquat(T):
    import numpy as np
    from scipy.spatial.transform import Rotation

    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [
        float(T[0, 3]),
        float(T[1, 3]),
        float(T[2, 3]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]


def main():
    import argparse

    import cv2

    parser = argparse.ArgumentParser(description="YOLO + FDPose, pose in BASE.")
    parser.add_argument("--rgb", default="", help="本地彩色图；省略则抓相机")
    parser.add_argument("--depth", default="", help="本地深度图；省略则抓相机")
    parser.add_argument("--labels", nargs="+", default=["white_tray"])
    parser.add_argument("--no-base", action="store_true", help="只返回相机系")
    parser.add_argument(
        "--depth-refine",
        action="store_true",
        help="启用深度补全（需先启动 lingbot-depth-rqh 服务）",
    )
    args = parser.parse_args()

    if not rospy.core.is_initialized():
        rospy.init_node("rgbd_pose_estimator", anonymous=True)

    if args.rgb and args.depth:
        color = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
        depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            raise RuntimeError("无法读取 --rgb / --depth")
    else:
        color, depth = _capture_rgbd()

    estimator = RgbdPoseEstimator(
        labels=args.labels,
        init_ros=False,
        to_base=not args.no_base,
        enable_depth_refine=args.depth_refine,
    )
    poses = estimator.estimate(color, depth, labels=args.labels)
    if not poses:
        print("no pose")
        return 1
    for pose in poses:
        xyzquat = _matrix_to_xyzquat(pose.T_object)
        print("label=", pose.label)
        print("T_object=")
        print(pose.T_object)
        print("[x, y, z, qx, qy, qz, qw] =", xyzquat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
