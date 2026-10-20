#!/usr/bin/env python3
import argparse
from dataclasses import asdict, dataclass
import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image
from navi_types.srv import DepthRefine
from threading import Lock
from cv_bridge import CvBridge, CvBridgeError

@dataclass
class DepthRefineClientConfig:
    service_name: str = "/lingbot/depth_refine"
    rgb_topic: str = None    # 如果为 None，则不订阅 RGB 话题
    depth_topic: str = None  # 如果为 None，则不订阅深度话题
    rgb_path: str = ""
    depth_path: str = ""
    save_dir: str = "./results"
    save_vis: bool = False
    max_vis_depth_m: float = 2.0
    display_scale: float = 0.6
    layout: str = "horizontal"
    queue_size: int = 10
    slop: float = 0.08
    timeout: float = 10.0


class DepthRefineClient:
    def __init__(self, config: DepthRefineClientConfig):
        self.config = config
        self.bridge = CvBridge()
        self.client = None
        # 复用同一个服务代理，减少多次调用时的重复开销。
        self._init_service_proxy()

        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.data_lock = Lock()
        self.latest_rgb = None
        self.latest_depth_raw = None
        if self.config.rgb_topic is not None and self.config.depth_topic is not None:
            rgb_sub = message_filters.Subscriber(self.config.rgb_topic, Image)
            depth_sub = message_filters.Subscriber(self.config.depth_topic, Image)

            sync = message_filters.ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub],
                queue_size=self.config.queue_size,
                slop=self.config.slop,
            )
            sync.registerCallback(self.sync_callback)
            rospy.loginfo("RGB topic: %s", self.config.rgb_topic)
            rospy.loginfo("Depth topic: %s", self.config.depth_topic)

    def sync_callback(self, rgb_msg, depth_msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            rospy.logwarn("CvBridge convert failed: %s", exc)
            return

        with self.data_lock:
            self.latest_rgb = rgb
            self.latest_depth_raw = depth_raw

    def _init_service_proxy(self):
        # 阻塞等待服务可用，然后创建（或重建）服务代理。
        while self.client is None:
            try:
                self.client = rospy.ServiceProxy(self.config.service_name, DepthRefine)
                break
            except rospy.ServiceException as exc:
                rospy.logwarn("Service not available, retrying: %s", exc)
                rospy.sleep(1.0)
            except rospy.ROSException as exc:
                rospy.logwarn("ROS exception: %s", exc)
                rospy.sleep(1.0)
            except Exception as exc:
                rospy.logwarn("Unexpected error: %s", exc)

    def get_latest_images(self):
        if self.config.rgb_topic is not None and self.config.depth_topic is not None:
            with self.data_lock:
                if self.latest_rgb is None or self.latest_depth_raw is None:
                    return None, None
                rgb = self.latest_rgb.copy()
                depth_raw = self.latest_depth_raw.copy()
                return rgb, depth_raw
        rospy.logwarn("RGB topic or depth topic is not set, returning None")
        return None, None

    @staticmethod
    def depth_to_color(depth_u16, max_depth_m):
        depth_m = depth_u16.astype(np.float32) / 1000.0
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, max_depth_m)
        depth_norm = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    @staticmethod
    def _resize_to_width(image, target_w):
        h, w = image.shape[:2]
        if w == target_w:
            return image
        target_h = int(round(h * float(target_w) / float(w)))
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _resize_by_scale(image, scale):
        if scale >= 0.999:
            return image
        h, w = image.shape[:2]
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _add_panel_label(image, label):
        out = image.copy()
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
        x = max(8, out.shape[1] - text_w - 16)
        y = max(text_h + 8, 36)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def build_visualization(self, rgb, depth_in, depth_pred, infer_ms=None):
        # 统一复用 subscriber 的三联图可视化风格。
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        in_color = self.depth_to_color(depth_in, self.config.max_vis_depth_m)
        pred_color = self.depth_to_color(depth_pred, self.config.max_vis_depth_m)

        bgr_panel = self._add_panel_label(bgr, "RGB")
        in_panel = self._add_panel_label(in_color, "Input Depth")
        pred_panel = self._add_panel_label(pred_color, "Pred Depth")

        if self.config.layout == "horizontal":
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

        if infer_ms is not None:
            cv2.putText(
                stacked,
                "infer: {:.1f} ms".format(infer_ms),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return self._resize_by_scale(stacked, self.config.display_scale)

    def load_inputs(self, rgb_path=None, depth_path=None):
        rgb_path = rgb_path or self.config.rgb_path
        depth_path = depth_path or self.config.depth_path

        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to read rgb image: {}".format(rgb_path))

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError("Failed to read depth image: {}".format(depth_path))
        self.validate_inputs(bgr, depth)
        return bgr, depth

    @staticmethod
    def validate_inputs(bgr, depth):
        # 同时支持本地加载和外部传入时的统一校验。
        if depth.ndim != 2:
            raise RuntimeError("Depth image must be single-channel, got shape: {}".format(depth.shape))

        if bgr.shape[:2] != depth.shape[:2]:
            raise RuntimeError(
                "RGB and depth size mismatch: rgb={} depth={}".format(bgr.shape[:2], depth.shape[:2])
            )

    def build_request_messages(self, bgr, depth):
        color_msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        color_msg.header.stamp = rospy.Time.now()

        # 深度编码跟随输入 dtype，避免类型误解码。
        if depth.dtype == np.uint16:
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        else:
            depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
        depth_msg.header = color_msg.header
        return color_msg, depth_msg

    def call_service(self, color_msg, depth_msg):
        rospy.loginfo("Calling depth refine service...")
        start = rospy.get_time()
        try:
            resp = self.client(color_msg, depth_msg)
        except rospy.ServiceException as exc:
            # 服务重启等瞬时异常时，重连一次再重试。
            rospy.logwarn("Service call failed once, reconnect and retry: %s", exc)
            self._init_service_proxy()
            resp = self.client(color_msg, depth_msg)
        elapsed_ms = (rospy.get_time() - start) * 1000.0
        rospy.loginfo("Service done in %.1f ms", elapsed_ms)
        return resp, elapsed_ms

    def save_refined_depth(self, refined_u16, output_path=None):
        output_path = output_path or self.config.output_path
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, refined_u16)
        rospy.loginfo("Saved refined depth: %s", output_path)

    def save_refined_visualization(self, refined_u16, output_path=None):
        output_path = output_path or self.config.output_path
        vis = self.depth_to_color(refined_u16, self.config.max_vis_depth_m)
        vis_path = os.path.splitext(output_path)[0] + "_vis.png"
        cv2.imwrite(vis_path, vis)
        rospy.loginfo("Saved refined depth visualization: %s", vis_path)

    def call_from_local_files(self, rgb_path=None, depth_path=None, output_path=None):
        # 本地文件模式：可选传入路径；未传时回退到配置路径。
        bgr, depth = self.load_inputs(rgb_path=rgb_path, depth_path=depth_path)
        refined_u16 = self.call_from_arrays(bgr, depth)
        self.save_refined_depth(refined_u16, output_path=output_path)
        if self.config.save_vis:
            self.save_refined_visualization(refined_u16, output_path=output_path)

    def call_from_arrays(self, bgr, depth):
        # 传参数组模式：外部直接传入 numpy 图像和深度图。
        self.validate_inputs(bgr, depth)
        color_msg, depth_msg = self.build_request_messages(bgr, depth)
        resp, elapsed_ms = self.call_service(color_msg, depth_msg)
        refined = self.bridge.imgmsg_to_cv2(resp.refined_depth_image, desired_encoding="passthrough")
        refined_u16 = refined.astype(np.uint16)
        return refined_u16, elapsed_ms


def build_config():
    parser = argparse.ArgumentParser(description="DepthRefine service test client.")
    parser.add_argument(
        "--service-name",
        type=str,
        default=None,
        help="Depth refine service name",
    )
    parser.add_argument(
        "--rgb-topic",
        type=str,
        default=None,
        help="RGB topic",
    )
    parser.add_argument(
        "--depth-topic",
        type=str,
        default=None,
        help="Depth topic",
    )
    parser.add_argument(
        "--save-vis",
        dest="save_vis",
        action="store_true",
        default=None,
        help="Save colorized depth visualization next to output file",
    )
    parser.add_argument(
        "--no-save-vis",
        dest="save_vis",
        action="store_false",
        default=None,
        help="Disable saving visualization",
    )
    parser.add_argument(
        "--max-vis-depth-m",
        type=float,
        default=None,
        help="Depth colormap clipping threshold in meters",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=None,
        help="Display resize scale in (0, 1], only affects imshow",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        choices=["vertical", "horizontal"],
        help="Visualization layout style",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout for waiting service in seconds",
    )
    args = parser.parse_args()

    # 先用 dataclass 默认值，再用命令行参数覆盖。
    cfg_dict = asdict(DepthRefineClientConfig())
    cli_overrides = {
        "service_name": args.service_name,
        "rgb_topic": args.rgb_topic,
        "depth_topic": args.depth_topic,
        "save_vis": args.save_vis,
        "max_vis_depth_m": args.max_vis_depth_m,
        "display_scale": args.display_scale,
        "layout": args.layout,
        "timeout": args.timeout,
    }
    cfg_dict.update({k: v for k, v in cli_overrides.items() if v is not None})
    cfg_dict["display_scale"] = float(np.clip(cfg_dict["display_scale"], 0.1, 1.0))
    return DepthRefineClientConfig(**cfg_dict)


def main():
    config = build_config()
    rospy.init_node("lingbot_depth_refine_client", anonymous=True)
    client = DepthRefineClient(config)
    while not rospy.is_shutdown():
        rgb, depth_raw = client.get_latest_images()
        if rgb is not None and depth_raw is not None:
            refined_u16, elapsed_ms = client.call_from_arrays(rgb, depth_raw)
            vis = client.build_visualization(rgb=rgb, depth_in=depth_raw, depth_pred=refined_u16, infer_ms=elapsed_ms)
            cv2.imshow("LingbotDepth [RGB | InputDepth | PredDepth]", vis)
            cv2.waitKey(1)
        rospy.sleep(0.1)
    cv2.destroyAllWindows()
    rospy.shutdown()

if __name__ == "__main__":
    main()
