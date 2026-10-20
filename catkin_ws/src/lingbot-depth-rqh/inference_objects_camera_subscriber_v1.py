import argparse
import os
import time

import cv2
import message_filters
import numpy as np
import rospy
import torch
from threading import Lock
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image

from mdm.model.v2 import MDMModel


class LingbotDepthNode:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MDMModel.from_pretrained(args.model_path).to(self.device)
        self.model.eval()

        self.save_dir = args.save_dir
        self.frame_idx = 0
        self.data_lock = Lock()
        self.latest_bgr = None
        self.latest_rgb = None
        self.latest_depth_raw = None
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        info_msg = rospy.wait_for_message(
            args.camera_info_topic,
            CameraInfo,
            timeout=args.camera_info_timeout,
        )
        self.k_raw = np.array(info_msg.K, dtype=np.float32).reshape(3, 3)

        rgb_sub = message_filters.Subscriber(args.rgb_topic, Image)
        depth_sub = message_filters.Subscriber(args.depth_topic, Image)

        sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=args.queue_size,
            slop=args.slop,
        )
        sync.registerCallback(self.sync_callback)

        rospy.loginfo("LingbotDepthNode started.")
        rospy.loginfo("RGB topic: %s", args.rgb_topic)
        rospy.loginfo("Depth topic: %s", args.depth_topic)
        rospy.loginfo("CameraInfo topic: %s", args.camera_info_topic)

    def _to_meters(self, depth_img):
        if depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) / 1000.0
        else:
            depth_m = depth_img.astype(np.float32)
        return depth_m

    def _depth_to_color(self, depth_m):
        depth_vis = depth_m.copy()
        depth_vis = np.nan_to_num(depth_vis, nan=0.0, posinf=0.0, neginf=0.0)
        depth_vis = np.clip(depth_vis, 0.0, self.args.max_vis_depth_m)
        depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def _resize_to_width(self, image, target_w):
        h, w = image.shape[:2]
        if w == target_w:
            return image
        target_h = int(round(h * float(target_w) / float(w)))
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _resize_by_scale(self, image, scale):
        if scale >= 0.999:
            return image
        h, w = image.shape[:2]
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)

    def _add_panel_label(self, image, label):
        out = image.copy()
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
        x = max(8, out.shape[1] - text_w - 16)
        y = max(text_h + 8, 36)
        cv2.putText(
            out,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        return out

    def sync_callback(self, rgb_msg, depth_msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            rospy.logwarn("CvBridge convert failed: %s", exc)
            return

        with self.data_lock:
            self.latest_bgr = bgr
            self.latest_rgb = rgb
            self.latest_depth_raw = depth_raw

    def process_and_visualize_once(self):
        with self.data_lock:
            if self.latest_bgr is None or self.latest_rgb is None or self.latest_depth_raw is None:
                return
            bgr = self.latest_bgr.copy()
            rgb = self.latest_rgb.copy()
            depth_raw = self.latest_depth_raw.copy()

        print("bgr shape:", bgr.shape)
        h, w = rgb.shape[:2]
        depth_m = self._to_meters(depth_raw)

        k = self.k_raw.copy()
        k[0] /= float(w)
        k[1] /= float(h)

        image_t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)[None]
        depth_t = torch.tensor(depth_m, dtype=torch.float32, device=self.device)[None]
        intrinsics_t = torch.tensor(k, dtype=torch.float32, device=self.device)[None]

        start = time.time()
        with torch.no_grad():
            output = self.model.infer(
                image_t,
                depth_in=depth_t,
                intrinsics=intrinsics_t,
            )
        infer_ms = (time.time() - start) * 1000.0

        depth_pred = output["depth"].squeeze().detach().cpu().numpy()
        depth_pred = np.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)

        pred_color = self._depth_to_color(depth_pred)
        in_color = self._depth_to_color(depth_m)
        bgr_panel = self._add_panel_label(bgr, "RGB")
        in_panel = self._add_panel_label(in_color, "Input Depth")
        pred_panel = self._add_panel_label(pred_color, "Pred Depth")

        if self.args.layout == "horizontal":
            target_h = bgr_panel.shape[0]
            in_panel = cv2.resize(
                in_panel,
                (int(round(in_panel.shape[1] * float(target_h) / float(in_panel.shape[0]))), target_h),
                interpolation=cv2.INTER_LINEAR,
            )
            pred_panel = cv2.resize(
                pred_panel,
                (int(round(pred_panel.shape[1] * float(target_h) / float(pred_panel.shape[0]))), target_h),
                interpolation=cv2.INTER_LINEAR,
            )
            stacked = np.hstack([bgr_panel, in_panel, pred_panel])
        else:
            target_w = bgr_panel.shape[1]
            in_panel = self._resize_to_width(in_panel, target_w)
            pred_panel = self._resize_to_width(pred_panel, target_w)
            stacked = np.vstack([bgr_panel, in_panel, pred_panel])
        cv2.putText(
            stacked,
            f"infer: {infer_ms:.1f} ms",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        stacked_show = self._resize_by_scale(stacked, self.args.display_scale)
        cv2.imshow("LingbotDepth [RGB | InputDepth | PredDepth]", stacked_show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            rospy.signal_shutdown("User pressed q")
            return

        if self.save_dir and key == ord("s"):
            name = f"frame_{self.frame_idx:06d}"
            pred_u16 = (depth_pred * 1000.0).astype(np.uint16)
            in_u16 = (depth_m * 1000.0).astype(np.uint16)
            cv2.imwrite(os.path.join(self.save_dir, f"{name}_rgb.png"), bgr)
            cv2.imwrite(os.path.join(self.save_dir, f"{name}_depth_in.png"), in_u16)
            cv2.imwrite(os.path.join(self.save_dir, f"{name}_depth_pred.png"), pred_u16)
            cv2.imwrite(os.path.join(self.save_dir, f"{name}_depth_pred_vis.png"), pred_color)
            rospy.loginfo("Saved frame to %s with prefix %s", self.save_dir, name)

        self.frame_idx += 1

    def run(self):
        rospy.loginfo("Waiting for first synchronized RGB/Depth frame...")
        while not rospy.is_shutdown():
            with self.data_lock:
                has_frame = (
                    self.latest_bgr is not None
                    and self.latest_rgb is not None
                    and self.latest_depth_raw is not None
                )
            if has_frame:
                rospy.loginfo("First frame received, start processing loop.")
                break
            time.sleep(0.5)

        rate = rospy.Rate(self.args.process_hz)
        while not rospy.is_shutdown():
            self.process_and_visualize_once()
            rate.sleep()


def parse_args():
    parser = argparse.ArgumentParser(description="Subscribe camera topics and run Lingbot depth inference.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5/model.pt",
        help="MDMModel pretrained path",
    )
    # parser.add_argument("--rgb-topic", type=str, default="/zj_humanoid/sensor/realsense_up/color/image_raw")
    parser.add_argument("--rgb-topic", type=str, default="zj_humanoid/sensor/realsense_up/color/image_raw")
    parser.add_argument("--depth-topic", type=str, default="zj_humanoid/sensor/realsense_up/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", type=str, default="zj_humanoid/sensor/realsense_up/color/camera_info")
    parser.add_argument(
        "--camera-info-timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for reading CameraInfo once at startup",
    )
    parser.add_argument("--queue-size", type=int, default=10)
    parser.add_argument("--slop", type=float, default=0.08, help="Approximate sync tolerance in seconds")
    parser.add_argument("--max-vis-depth-m", type=float, default=2.0, help="Depth colormap clipping threshold in meters")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./results",
        help="Optional directory to save frames when pressing s",
    )
    parser.add_argument(
        "--process-hz",
        type=float,
        default=15.0,
        help="Processing and visualization loop frequency",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.6,
        help="Display resize scale in (0, 1], only affects imshow",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="horizontal",
        choices=["vertical", "horizontal"],
        help="Visualization layout style",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.display_scale = float(np.clip(args.display_scale, 0.1, 1.0))
    rospy.init_node("lingbot_depth_inference_subscriber", anonymous=True)
    node = LingbotDepthNode(args)
    node.run()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
