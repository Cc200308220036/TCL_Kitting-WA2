#!/usr/bin/env python3
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image

from mdm.model.v2 import MDMModel
from navi_types.srv import DepthRefine, DepthRefineResponse


@dataclass
class DepthRefineServiceConfig:
    model_path: str = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
    service_name: str = "/lingbot/depth_refine"
    camera_info_topic: str = "/zj_humanoid/sensor/realsense_up/color/camera_info"
    camera_info_timeout: float = 5.0
    warmup: bool = True
    warmup_n: int = 3
    resolution_level: int = 3
    skip_pointcloud: bool = True

def resolve_model_path(model_ref: str) -> str:
    """Prefer local file path; fallback to Hugging Face repo id."""
    candidate = Path(model_ref)
    if candidate.exists():
        return str(candidate)

    # Make relative paths stable even when running script from other cwd.
    script_dir_candidate = Path(__file__).resolve().parent / model_ref
    if script_dir_candidate.exists():
        return str(script_dir_candidate)

    return model_ref


class LingbotDepthRefineService:
    def __init__(self, config: DepthRefineServiceConfig):
        self.config = config
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rospy.loginfo("Loading model from %s", config.model_path)
        self.model = MDMModel.from_pretrained(resolve_model_path(config.model_path)).to(self.device)
        self.model.eval()

        info_msg = rospy.wait_for_message(
            config.camera_info_topic,
            CameraInfo,
            timeout=config.camera_info_timeout,
        )
        self.k_raw = np.array(info_msg.K, dtype=np.float32).reshape(3, 3)
        self.image_width = int(info_msg.width)
        self.image_height = int(info_msg.height)

        self.warmup_n = max(1, int(config.warmup_n))
        if config.warmup:
            self._warmup_with_empty_frame()

        self.service = rospy.Service(config.service_name, DepthRefine, self.handle_request)
        rospy.loginfo("Depth refine service ready: %s", config.service_name)

    @staticmethod
    def _to_meters(depth_img):
        if depth_img.dtype == np.uint16:
            return depth_img.astype(np.float32) / 1000.0
        return depth_img.astype(np.float32)

    def _infer_depth(self, bgr, depth_raw):
        rgb = bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        depth_m = self._to_meters(depth_raw)

        image_t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)[None]
        depth_t = torch.tensor(depth_m, dtype=torch.float32, device=self.device)[None]
        intrinsics_t = None
        if not self.config.skip_pointcloud:
            k = self.k_raw.copy()
            k[0] /= float(w)
            k[1] /= float(h)
            intrinsics_t = torch.tensor(k, dtype=torch.float32, device=self.device)[None]

        with torch.no_grad():
            output = self.model.infer(
                image_t,
                depth_in=depth_t,
                intrinsics=intrinsics_t,
                resolution_level=self.config.resolution_level,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        # 这一步在 GPU 上完成清理和量化，只把最终 uint16 拷回 CPU，减少拷贝与 CPU 开销。
        depth_pred_t = output["depth"].squeeze().detach()
        depth_pred_t = torch.nan_to_num(depth_pred_t, nan=0.0, posinf=0.0, neginf=0.0)
        depth_pred_t = torch.clamp(depth_pred_t * 1000.0, 0.0, float(np.iinfo(np.uint16).max))
        depth_pred_np = depth_pred_t.contiguous().to(torch.float32).cpu().numpy()
        depth_pred = depth_pred_np.astype(np.uint16)
        return depth_pred

    def _warmup_with_empty_frame(self):
        if self.image_width <= 0 or self.image_height <= 0:
            rospy.logwarn("Skip warmup: invalid camera size (%d, %d).", self.image_width, self.image_height)
            return

        rospy.loginfo("Running warmup inference with empty frame (%d, %d)...", self.image_width, self.image_height)
        empty_bgr = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        empty_depth = np.zeros((self.image_height, self.image_width), dtype=np.uint16)
        for i in range(self.warmup_n):  
            start = rospy.get_time()
            _ = self._infer_depth(empty_bgr, empty_depth)
            warmup_ms = (rospy.get_time() - start) * 1000.0
            rospy.loginfo("time %d: Warmup inference done in %.1f ms.", i, warmup_ms)

    def handle_request(self, req):
        try:
            bgr = self.bridge.imgmsg_to_cv2(req.color_image, desired_encoding="bgr8")
            depth_raw = self.bridge.imgmsg_to_cv2(req.depth_image, desired_encoding="passthrough")
        except CvBridgeError as exc:
            rospy.logerr("CvBridge convert failed: %s", exc)
            raise rospy.ServiceException(str(exc))

        start_time = rospy.get_time()
        refined_u16 = self._infer_depth(bgr, depth_raw)
        infer_ms = (rospy.get_time() - start_time) * 1000.0

        out_msg = self.bridge.cv2_to_imgmsg(refined_u16, encoding="16UC1")
        out_msg.header = req.depth_image.header

        rospy.loginfo_throttle(
            1.0,
            "Depth refine request processed in %.1f ms, shape=(%d, %d)",
            infer_ms,
            refined_u16.shape[0],
            refined_u16.shape[1],
        )
        return DepthRefineResponse(refined_depth_image=out_msg)


def build_config():
    parser = argparse.ArgumentParser(description="Lingbot depth refine ROS service.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="MDMModel pretrained path",
    )
    parser.add_argument(
        "--service-name",
        type=str,
        default=None,
        help="Service name for depth refine",
    )
    parser.add_argument(
        "--camera-info-topic",
        type=str,
        default=None,
        help="CameraInfo topic used to initialize intrinsics",
    )
    parser.add_argument(
        "--camera-info-timeout",
        type=float,
        default=None,
        help="Timeout in seconds for reading CameraInfo once at startup",
    )
    parser.add_argument(
        "--warmup",
        dest="warmup",
        action="store_true",
        default=None,
        help="Run warmup inference with empty frame at startup",
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        default=None,
        help="Disable warmup inference",
    )
    parser.add_argument(
        "--warmup-n",
        type=int,
        default=None,
        help="Number of warmup inferences when warmup enabled",
    )
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=None,
        help="Model resolution level, lower is faster (default 9)",
    )
    parser.add_argument(
        "--skip-pointcloud",
        dest="skip_pointcloud",
        action="store_true",
        default=None,
        help="Skip point cloud generation when only refined depth is needed",
    )
    parser.add_argument(
        "--with-pointcloud",
        dest="skip_pointcloud",
        action="store_false",
        default=None,
        help="Keep point cloud generation inside model.infer",
    )

    args = parser.parse_args()

    cfg_dict = asdict(DepthRefineServiceConfig())

    cli_overrides = {
        "model_path": args.model_path,
        "service_name": args.service_name,
        "camera_info_topic": args.camera_info_topic,
        "camera_info_timeout": args.camera_info_timeout,
        "warmup": args.warmup,
        "warmup_n": args.warmup_n,
        "resolution_level": args.resolution_level,
        "skip_pointcloud": args.skip_pointcloud,
    }
    cfg_dict.update({k: v for k, v in cli_overrides.items() if v is not None})

    cfg = DepthRefineServiceConfig(**cfg_dict)
    cfg.warmup_n = max(1, int(cfg.warmup_n))
    cfg.resolution_level = int(np.clip(cfg.resolution_level, 0, 9))
    return cfg


def main():
    config = build_config()
    rospy.init_node("lingbot_depth_refine_service", anonymous=True)
    LingbotDepthRefineService(config)
    rospy.spin()


if __name__ == "__main__":
    main()
