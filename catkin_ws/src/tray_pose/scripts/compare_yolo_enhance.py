#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只跑 YOLO 分割，对比过曝增强前后的识别结果。

需要已启动 instance_segmentation_service。

    python compare_yolo_enhance.py
    python compare_yolo_enhance.py --rgb /path/to.jpg --labels white_tray
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_YOLO_SRC = Path("/root/catkin_ws/src/naviai_yolo_0909/src")
for _path in (_SCRIPT_DIR, _YOLO_SRC):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from estimate_rgbd_pose import _enhance_overexposed_bgr
from naviai_yolo.ros.seg_service_client import SegServiceClient


DEFAULT_COLOR_TOPIC = "/zj_humanoid/sensor/realsense_head/color/image_raw"
MASK_COLORS = (
    (0, 180, 255),
    (0, 220, 80),
    (255, 80, 80),
    (255, 0, 220),
)


def _imgmsg_to_bgr(msg):
    enc = str(msg.encoding)
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    arr = np.ascontiguousarray(arr)
    if enc == "rgb8":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def _capture_color(topic, timeout):
    rospy.loginfo("等待彩色图: %s", topic)
    msg = rospy.wait_for_message(topic, Image, timeout=timeout)
    return _imgmsg_to_bgr(msg)


def _draw_items(bgr, response, title):
    vis = bgr.copy()
    bridge = CvBridge()
    items = list(getattr(response, "items", []) or [])
    for i, item in enumerate(items):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        mask = bridge.imgmsg_to_cv2(item.mask, "mono16") > 0
        if mask.shape[:2] != vis.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (vis.shape[1], vis.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        overlay = vis.copy()
        overlay[mask] = (0.45 * np.asarray(color) + 0.55 * overlay[mask]).astype(np.uint8)
        vis = overlay
        x1, y1, x2, y2 = [int(v) for v in item.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{item.label} {float(item.confidence):.2f}"
        cv2.putText(
            vis, label, (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    cv2.putText(
        vis, f"{title}  n={len(items)}", (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return vis


def _summarize(name, response):
    items = list(getattr(response, "items", []) or [])
    print(f"{name}: have_objs={getattr(response, 'have_objs', False)}  n={len(items)}")
    for item in items:
        print(f"  {item.label}  conf={float(item.confidence):.3f}  bbox={list(item.bbox)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="对比 YOLO 增强前后分割")
    parser.add_argument("--rgb", default="", help="本地彩色图；省略则抓相机")
    parser.add_argument("--labels", nargs="+", default=["white_tray"])
    parser.add_argument(
        "--service", default="instance_segmentation_service",
    )
    parser.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--out",
        default=str(_SCRIPT_DIR.parent / "output" / "yolo_enhance_compare"),
    )
    args = parser.parse_args()

    if not rospy.core.is_initialized():
        rospy.init_node("compare_yolo_enhance", anonymous=True)

    if args.rgb:
        color = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
        if color is None:
            raise RuntimeError("无法读取 --rgb: %s" % args.rgb)
    else:
        color = _capture_color(args.color_topic, args.timeout)

    enhanced = _enhance_overexposed_bgr(color)
    client = SegServiceClient(service_name=args.service, wait_timeout_sec=args.timeout)
    before = client.segment(color=color, labels=args.labels)
    after = client.segment(color=enhanced, labels=args.labels)

    vis_before = _draw_items(color, before, "before")
    vis_after = _draw_items(enhanced, after, "after")
    compare = np.hstack([vis_before, vis_after])

    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "00_input.jpg"), color)
    cv2.imwrite(str(out_dir / "01_enhanced.jpg"), enhanced)
    cv2.imwrite(str(out_dir / "02_yolo_before.jpg"), vis_before)
    cv2.imwrite(str(out_dir / "03_yolo_after.jpg"), vis_after)
    cv2.imwrite(str(out_dir / "04_compare.jpg"), compare)

    _summarize("before", before)
    _summarize("after", after)
    print(f"saved: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
