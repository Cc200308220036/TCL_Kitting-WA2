#!/usr/bin/env python3
import argparse

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo

from mdm.model.v2 import MDMModel
from navi_types.srv import DepthRefine, DepthRefineResponse


class DepthRefineServiceServer:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MDMModel.from_pretrained(args.model_path).to(self.device)
        self.model.eval()

        info_msg = rospy.wait_for_message(
            args.camera_info_topic,
            CameraInfo,
            timeout=args.camera_info_timeout,
        )
        self.k_raw = np.array(info_msg.K, dtype=np.float32).reshape(3, 3)

        self.service = rospy.Service(args.service_name, DepthRefine, self.handle_refine)
        rospy.loginfo("DepthRefine service ready: %s", args.service_name)
        rospy.loginfo("CameraInfo topic: %s", args.camera_info_topic)

    @staticmethod
    def _to_meters(depth_img):
        if depth_img.dtype == np.uint16:
            return depth_img.astype(np.float32) / 1000.0
        return depth_img.astype(np.float32)

    def handle_refine(self, req):
        resp = DepthRefineResponse()

        try:
            try:
                rgb = self.bridge.imgmsg_to_cv2(req.color_image, desired_encoding="rgb8")
            except CvBridgeError:
                bgr = self.bridge.imgmsg_to_cv2(req.color_image, desired_encoding="bgr8")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            depth_raw = self.bridge.imgmsg_to_cv2(req.depth_image, desired_encoding="passthrough")
        except Exception as exc:
            rospy.logerr("DepthRefine request decode failed: %s", exc)
            resp.refined_depth_image = self.bridge.cv2_to_imgmsg(
                np.zeros((1, 1), dtype=np.uint16), encoding="mono16"
            )
            return resp

        h, w = rgb.shape[:2]
        depth_m = self._to_meters(depth_raw)

        k = self.k_raw.copy()
        k[0] /= float(w)
        k[1] /= float(h)

        image_t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1)[None]
        depth_t = torch.tensor(depth_m, dtype=torch.float32, device=self.device)[None]
        intrinsics_t = torch.tensor(k, dtype=torch.float32, device=self.device)[None]

        with torch.no_grad():
            output = self.model.infer(
                image_t,
                depth_in=depth_t,
                intrinsics=intrinsics_t,
            )

        depth_pred = output["depth"].squeeze().detach().cpu().numpy()
        depth_pred = np.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)
        depth_pred_u16 = np.clip(depth_pred * 1000.0, 0, 65535).astype(np.uint16)

        out_msg = self.bridge.cv2_to_imgmsg(depth_pred_u16, encoding="mono16")
        out_msg.header = req.depth_image.header
        resp.refined_depth_image = out_msg
        return resp


def parse_args():
    parser = argparse.ArgumentParser(description="Lingbot depth refine service server.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5/model.pt",
        help="MDMModel pretrained path",
    )
    parser.add_argument("--service-name", type=str, default="/lingbot_depth_refine")
    parser.add_argument("--camera-info-topic", type=str, default="/camera/color/camera_info")
    parser.add_argument("--camera-info-timeout", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("lingbot_depth_refine_service_server", anonymous=True)
    DepthRefineServiceServer(args)
    rospy.spin()


if __name__ == "__main__":
    main()
