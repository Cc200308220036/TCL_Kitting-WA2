#!/usr/bin/env python3
# 运行：python tools/capture_one_pair.py [--rgb-topic 话题 --depth-topic 话题]。
# 实际执行：本文件订阅 ROS RGB-D 并保存 PNG；不调用 FDPose/YOLO Service。
# 配置：仅命令行参数，无 YAML；默认输出 data/samples/captures/。
"""从 ROS 相机话题抓拍一对同步 RGB-D，仅保存彩色图和毫米深度图，不调用模型。"""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import tempfile
from threading import Event, Lock
import time
from uuid import uuid4

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdpose.paths import SAMPLES_DIR


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-topic", default="/zj_humanoid/sensor/realsense_head/color/image_raw",
                        help="彩色图像话题")
    parser.add_argument("--depth-topic", default="/zj_humanoid/sensor/realsense_head/aligned_depth_to_color/image_raw",
                        help="已对齐到彩色图的深度话题；工具本身不执行空间对齐")
    parser.add_argument("--out-dir", type=Path, default=SAMPLES_DIR / "captures",
                        help="输出根目录，每次新建一个独立样例目录；相对路径基于当前终端目录")
    parser.add_argument("--wait", type=float, default=10.0, help="等待同步帧的最长秒数，必须大于 0")
    parser.add_argument("--slop", type=float, default=0.0,
                        help="允许的时间戳差（秒）；默认 0 为严格同步，例如 0.03 为近似同步")
    return parser


def capture_pair(rgb_topic, depth_topic, timeout, slop):
    # ROS 只在抓拍时导入，--help 和图像保存测试不需要 ROS/GPU 环境。
    import rospy
    from message_filters import ApproximateTimeSynchronizer, Subscriber, TimeSynchronizer
    from sensor_msgs.msg import Image

    ready, lock = Event(), Lock()
    pairs, subscribers = [], []

    def receive(color, depth):
        # 只使用收到的第一对同步帧，避免保存过程中被新帧替换。
        with lock:
            if not pairs:
                pairs.append((color, depth))
                ready.set()

    try:
        subscribers.append(Subscriber(rgb_topic, Image))
        subscribers.append(Subscriber(depth_topic, Image))
        if slop == 0:
            synchronizer = TimeSynchronizer(subscribers, 10)
        else:
            synchronizer = ApproximateTimeSynchronizer(subscribers, 10, slop, allow_headerless=False)
        synchronizer.registerCallback(receive)
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("未收到同步 RGB-D；检查话题数据、header.stamp，必要时使用 --slop 0.03")
            if ready.wait(min(remaining, .1)):
                return pairs[0]
        raise RuntimeError("ROS 已关闭，取消抓拍")
    finally:
        for subscriber in subscribers:
            subscriber.unregister()


def decode_pair(bridge, color_message, depth_message):
    color = np.asarray(bridge.imgmsg_to_cv2(color_message, desired_encoding="bgr8"))
    depth = np.asarray(bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough"))
    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3 or color.size == 0:
        raise ValueError("彩色图必须能转换成非空 uint8 H×W×3 BGR 图像")
    if depth.ndim != 2 or depth.shape != color.shape[:2]:
        raise ValueError(f"RGB/Depth 尺寸不一致：{color.shape} / {depth.shape}；请使用对齐后的深度话题")

    encoding = depth_message.encoding.upper()
    if encoding in ("16UC1", "MONO16") and depth.dtype == np.uint16:
        # 本工具与 FDPose Client 约定一致：uint16 深度单位为毫米，逐值保留。
        depth_mm = depth
    elif encoding == "32FC1" and depth.dtype == np.float32:
        # 米 → 毫米 PNG，最多约 0.5 mm 舍入误差；无效值统一保存为 0。
        valid = np.isfinite(depth) & (depth > 0)
        scaled = depth[valid].astype(np.float64) * 1000
        if np.any(scaled > np.iinfo(np.uint16).max):
            raise ValueError("深度超过 uint16 毫米 PNG 的 65.535 m 范围，拒绝截断保存")
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_mm[valid] = np.rint(scaled).astype(np.uint16)
    else:
        raise ValueError(f"不支持的深度格式：encoding={depth_message.encoding}, dtype={depth.dtype}")
    return color, depth_mm


def save_pair(out_dir, color_bgr, depth_mm):
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = root / f"{stamp}_{uuid4().hex[:8]}"
    # 两张图均写成功才发布样例目录；失败时清理本次临时文件，不碰已有样例。
    with tempfile.TemporaryDirectory(prefix=".capture_", dir=root) as temporary:
        staging = Path(temporary)
        staging.chmod(0o755)
        for relative, array in (("rgb/0.png", color_bgr), ("depth/0.png", depth_mm)):
            path = staging / relative
            path.parent.mkdir()
            path.parent.chmod(0o755)
            if not cv2.imwrite(str(path), array, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"图片保存失败：{relative}")
            path.chmod(0o644)
        staging.rename(destination)
    return destination


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.wait) or args.wait <= 0:
        parser.error("--wait 必须为大于 0 的有限秒数")
    if not math.isfinite(args.slop) or args.slop < 0:
        parser.error("--slop 必须为大于等于 0 的有限秒数")
    if not args.rgb_topic.strip() or not args.depth_topic.strip():
        parser.error("相机话题不能为空")
    try:
        import rospy
        from cv_bridge import CvBridge

        rospy.init_node("fdpose_capture_one_pair", anonymous=True)
        rospy.loginfo("Capture RGB-D: rgb=%s depth=%s slop=%.3fs", args.rgb_topic, args.depth_topic, args.slop)
        messages = capture_pair(args.rgb_topic, args.depth_topic, args.wait, args.slop)
        color, depth = decode_pair(CvBridge(), *messages)
        if messages[1].encoding.upper() == "32FC1":
            rospy.loginfo("32FC1 米制深度将舍入为 uint16 毫米 PNG；非有限值和非正值保存为 0")
        destination = save_pair(args.out_dir, color, depth)
        print(f"Saved RGB-D sample:\n  rgb:   {destination / 'rgb/0.png'}\n  depth: {destination / 'depth/0.png'} (uint16, mm)")
        return 0
    except Exception as exc:
        print(f"RGB-D capture failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
