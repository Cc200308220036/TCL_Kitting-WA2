"""抓拍一对同步的 color / depth 图并保存到 data/rgbd_captures 目录。

用法:
    conda activate seg
    source /opt/ros/noetic/setup.bash
    source ~/catkin_ws/devel/setup.bash
    python tools/capture_one_pair.py [--out-dir rgbd_captures] [--prefix test]

保存结果:
    <out-dir>/<prefix>_<timestamp>_color.png     # BGR 8UC3
    <out-dir>/<prefix>_<timestamp>_depth.png     # 16UC1, 单位 mm
    <out-dir>/<prefix>_<timestamp>_depth.npy     # 32FC1 时使用 NPY, 单位 m
"""

import os
import time
import argparse
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, TimeSynchronizer
from sensor_msgs.msg import Image
from naviai_yolo.configuration import load_config, resolve_output_path, validate_timeout
from naviai_yolo.ros.image_io import decode_color, decode_depth


class PairCapture:
    def __init__(self, out_dir, prefix):
        self.out_dir = out_dir
        self.prefix = prefix
        self.bridge = CvBridge()
        self.latest = None  # (rgb_msg, depth_msg)
        camera_config = load_config('camera.yaml', rospy.get_param('~camera_config', ''))
        rgb_topic = camera_config['camera_rgb_topic']
        depth_topic = camera_config['camera_depth_topic']
        rospy.loginfo(f"Subscribe: rgb={rgb_topic}, depth={depth_topic}")

        self.rgb_sub = Subscriber(rgb_topic, Image)
        self.depth_sub = Subscriber(depth_topic, Image)
        self.ts = TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.cb)

    def cb(self, rgb_msg, depth_msg):
        self.latest = (rgb_msg, depth_msg)

    def save_latest(self):
        if self.latest is None:
            rospy.logwarn("还没有收到同步帧, 请确认相机话题有数据 (rostopic hz <rgb_topic>).")
            return False

        rgb_msg, depth_msg = self.latest
        ts = int(time.time() * 1000)
        try:
            rgb_cv = decode_color(self.bridge, rgb_msg)
            depth_cv = decode_depth(self.bridge, depth_msg, rgb_cv.shape)
            os.makedirs(self.out_dir, exist_ok=True)
            color_path = os.path.join(self.out_dir, f"{self.prefix}_{ts}_color.png")
            suffix = 'png' if depth_cv.dtype == np.uint16 else 'npy'
            depth_path = os.path.join(self.out_dir, f"{self.prefix}_{ts}_depth.{suffix}")
            if not cv2.imwrite(color_path, rgb_cv):
                raise OSError(f'Failed to write color: {color_path}')
            if suffix == 'npy':
                np.save(depth_path, depth_cv, allow_pickle=False)
            elif not cv2.imwrite(depth_path, depth_cv):
                raise OSError(f'Failed to write depth: {depth_path}')
        except (CvBridgeError, ValueError, OSError, cv2.error) as exc:
            rospy.logerr(f'Capture failed (partial files may remain): {exc}')
            return False

        rospy.loginfo(f"Saved:\n  color: {color_path}\n  depth: {depth_path}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=str(resolve_output_path('data/rgbd_captures')))
    parser.add_argument('--prefix', default='cap')
    parser.add_argument('--wait', type=float, default=5.0,
                        help='最多等待多少秒收到同步帧')
    args = parser.parse_args(rospy.myargv()[1:])
    try:
        validate_timeout(args.wait, '--wait')
    except ValueError as exc:
        parser.error(str(exc))

    rospy.init_node('capture_one_pair', anonymous=True)
    cap = PairCapture(args.out_dir, args.prefix)

    deadline = time.monotonic() + args.wait
    rospy.loginfo(f"等待同步帧 (最多 {args.wait}s)...")
    while not rospy.is_shutdown() and cap.latest is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.1, remaining))

    return 0 if not rospy.is_shutdown() and cap.save_latest() else 1


if __name__ == '__main__':
    raise SystemExit(main())
