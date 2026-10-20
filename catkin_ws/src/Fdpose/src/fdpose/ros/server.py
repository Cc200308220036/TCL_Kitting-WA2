#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current FDPose ROS server implementation (recommended).

Entry point:
    scripts/pose_estimator_track_node_roi.py

ROS services:
    /fdpose_load_mesh         Load and cache meshes from the mesh registry.
    /pose_estimation_service Estimate poses from the current RGB-D request.

Version difference:
    This is the V1 one-shot ROI version. The historical demo-data/history
    implementation has been removed; current deployments use ``server_v2.py``.

Important guarantees:
- Mesh loading keeps the same structure as the original server.
- Pose estimation uses ONLY the current service request:
    req.color_image, req.depth_image, item.mask
- Does NOT read demo_data/current.
- Does NOT copy demo_data/current.
- Does NOT use track_one/history; every request performs register().
- Crops current RGB/depth/mask by current mask ROI and shifts K cx/cy.
- Default: no image/debug file saving.
"""

import time
import os
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Keep the import order close to the original file: FoundationPose imports first.
from fdpose.paths import (
    CAMERA_CONFIG_DIR,
    DEBUG_DIR,
    MESH_REGISTRY_PATH,
    MESHES_DIR,
)
from fdpose.core.estimator import *  # noqa: F401,F403
from fdpose.core.datareader import *  # noqa: F401,F403

import rospy
import cv2
import numpy as np
import trimesh
import imageio
from cv_bridge import CvBridge

from naviai_manip_srvs.srv import PoseEst, PoseEstResponse
from naviai_manip_srvs.msg import ObjPose
from naviai_manip_srvs.srv import FdposeLoadMesh, FdposeLoadMeshResponse

from fdpose.ros.server_utils import (
    crop_by_mask,
    legacy_depth_to_meters,
    draw_pose_visualization,
    matrix_to_ros_pose,
    read_mesh_config,
)


class PoseEstimatorNode:
    def __init__(self):
        rospy.init_node("pose_estimator_track_node", anonymous=True)

        self.debug_dir = str(DEBUG_DIR)
        self.mesh_registry_path = str(MESH_REGISTRY_PATH)

        # Fast but still correct for one-shot: each request calls register().
        self.est_refine_iter = int(rospy.get_param("~est_refine_iter", 3))

        # ROI register parameters.
        self.use_roi = bool(rospy.get_param("~use_roi", True))
        self.roi_margin = int(rospy.get_param("~roi_margin", 40))
        self.min_mask_pixels = int(rospy.get_param("~min_mask_pixels", 50))

        # save_debug_files saves current request raw inputs and pose estimation results.
        self.save_debug_files = bool(rospy.get_param("~save_debug_files", True))

        # Save pose axis visualization on the current full RGB request image.
        self.save_axis_vis = bool(rospy.get_param("~save_axis_vis", True))
        self.axis_vis_dir = os.path.join(self.debug_dir, "vis_img")
        os.makedirs(self.axis_vis_dir, exist_ok=True)
        self.axis_vis_scale = float(rospy.get_param("~axis_vis_scale", 0.1))
        self.axis_vis_thickness = int(rospy.get_param("~axis_vis_thickness", 3))

        self.objects = {}
        self.mesh_config = self._read_mesh_config()
        self.glctx = None
        self.bridge = CvBridge()

        # Camera info. Same paths as your original server.
        self.camera_info_dir = rospy.get_param(
            "~camera_info_dir",
            str(CAMERA_CONFIG_DIR),
        )
        intrinsic_path = os.path.join(self.camera_info_dir, "intrinsic_data.npz")

        intrinsic_data = np.load(intrinsic_path, allow_pickle=True)
        self.K = intrinsic_data["K"].astype(np.float32)

        # Services: same service names as original.
        self.pose_estimation_service = rospy.Service(
            "pose_estimation_service",
            PoseEst,
            self.handle_pose_estimation,
        )
        self.mesh_service = rospy.Service(
            "fdpose_load_mesh",
            FdposeLoadMesh,
            self.load_meshes_from_yaml,
        )

        rospy.loginfo("Pose Estimation Service is ready")
        rospy.loginfo("Pose Estimator Node initialized")

    def matrix_to_ros_pose(self, matrix):
        return matrix_to_ros_pose(matrix)

    def _read_mesh_config(self):
        return read_mesh_config(self.mesh_registry_path, rospy.logerr)

    def _load_meshes_from_yaml(self, labels):
        """Original mesh loading structure, with only stronger exception logging."""
        if not labels:
            rospy.logwarn("No mesh labels provided for loading")
            return []
        if not self.mesh_config:
            rospy.logerr("Mesh configuration is empty; cannot load meshes")
            return []
        if self.glctx is None:
            self.glctx = dr.RasterizeCudaContext()

        loaded_labels = []
        for raw_label in labels:
            obj_label = str(raw_label).strip()

            if obj_label in self.objects:
                rospy.loginfo(f"Mesh already loaded: {obj_label}")
                loaded_labels.append(obj_label)
                continue

            obj_info = self.mesh_config.get(obj_label)
            if not obj_info:
                rospy.logwarn(f"No mesh configuration found for label: {obj_label}")
                continue

            mesh_path = str(MESHES_DIR / obj_info["path"])
            rospy.loginfo(f"Loading mesh label={obj_label}, path={mesh_path}")
            if not os.path.exists(mesh_path):
                rospy.logwarn(f"Mesh file not found: {mesh_path}")
                continue

            try:
                # Same loading logic as original.
                mesh = trimesh.load(mesh_path, force="mesh")

                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

                # Critical: float32 avoids torch Float/Double conflicts.
                mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)

                bbox = np.asarray(mesh.bounds, dtype=np.float32)

                model_pts = np.asarray(mesh.vertices, dtype=np.float32)
                model_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

                scorer = ScorePredictor()
                refiner = PoseRefinePredictor()

                self.objects[obj_label] = {
                    "mesh": mesh,
                    "bbox": bbox,
                    "pose_estimator": FoundationPose(
                        model_pts=model_pts,
                        model_normals=model_normals,
                        mesh=mesh,
                        scorer=scorer,
                        refiner=refiner,
                        debug_dir=self.debug_dir,
                        debug=0,
                        glctx=self.glctx,
                    ),
                }
                rospy.loginfo(f"Loaded mesh: {obj_label} from {mesh_path}")
                loaded_labels.append(obj_label)
            except Exception as e:
                rospy.logerr(f"Failed to load mesh {mesh_path}: {repr(e)}")
                rospy.logerr(traceback.format_exc())

        return loaded_labels

    def load_meshes_from_yaml(self, req):
        response = FdposeLoadMeshResponse(success=True)

        if req.cancel:
            if self.objects:
                rospy.loginfo("Unloading all loaded meshes")
            self.objects.clear()

        # 每次加载时重读 yaml，避免改 model_config 后必须重启节点
        self.mesh_config = self._read_mesh_config()

        requested_labels = [str(label).strip() for label in req.goods if label]
        if not requested_labels:
            if not req.cancel:
                rospy.logwarn("Mesh load request received with empty goods list")
            return response

        loaded_labels = self._load_meshes_from_yaml(requested_labels)
        failed_labels = [label for label in requested_labels if label not in loaded_labels]
        if failed_labels:
            rospy.logwarn(f"Failed to load meshes: {failed_labels}")
            response.success = False
        return response

    @staticmethod
    def _depth_mono16_to_meters(depth_image):
        return legacy_depth_to_meters(depth_image)

    def _crop_by_mask(self, color, depth, mask, K):
        return crop_by_mask(
            color, depth, mask, K,
            use_roi=self.use_roi,
            roi_margin=self.roi_margin,
            min_mask_pixels=self.min_mask_pixels,
        )

    def _save_current_debug(self, label, rgb_bgr, depth_mono16, mask_bool, pose):
        if not self.save_debug_files:
            return
        out_dir = str(DEBUG_DIR / "current_request")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{label}_rgb.png"), rgb_bgr)
        cv2.imwrite(os.path.join(out_dir, f"{label}_depth.png"), depth_mono16)
        cv2.imwrite(os.path.join(out_dir, f"{label}_mask.png"), (mask_bool.astype(np.uint8)) * 255)
        np.save(os.path.join(out_dir, f"{label}_pose.npy"), pose.reshape(4, 4))

    def _save_axis_visualization(self, label, color_rgb_full, pose):
        """Draw the estimated object pose in the camera image."""
        if not self.save_axis_vis:
            return

        os.makedirs(self.axis_vis_dir, exist_ok=True)

        vis = draw_pose_visualization(
            color_rgb_full, pose, self.K, self.objects[label]["bbox"],
            axis_scale=self.axis_vis_scale,
            axis_thickness=self.axis_vis_thickness,
        )

        ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
        axis_path = os.path.join(self.axis_vis_dir, f"{label}_axis_vis_{ts}.png")
        imageio.imwrite(axis_path, vis)
        rospy.loginfo(f"Saved axis visualization: {axis_path}")

    def handle_pose_estimation(self, req):
        t_total = time.time()

        # Current request images only. No demo_data/current reads.
        rgb_image = self.bridge.imgmsg_to_cv2(req.color_image, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(req.depth_image, "mono16")

        color_rgb_full = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        depth_m_full = self._depth_mono16_to_meters(depth_image)
        K_full = np.asarray(self.K, dtype=np.float32)

        obj_poses = []

        for i, item in enumerate(req.items):
            raw_label = str(item.label)
            obj_label = str(raw_label).strip()

            if obj_label not in self.objects:
                rospy.logwarn(
                    f"Skip item {i}: label={obj_label} not loaded. "
                    f"Loaded labels={list(self.objects.keys())}"
                )
                continue

            try:
                mask_image = self.bridge.imgmsg_to_cv2(item.mask, "mono16")
                mask_full = np.asarray(mask_image > 0, dtype=bool)
                mask_pixels = int(mask_full.sum())

                if mask_pixels < self.min_mask_pixels:
                    rospy.logwarn(
                        f"Skip item {i}: label={obj_label}, mask too small: {mask_pixels} pixels"
                    )
                    continue

                color_use, depth_use, mask_use, K_use, roi = self._crop_by_mask(
                    np.asarray(color_rgb_full, dtype=np.uint8),
                    np.asarray(depth_m_full, dtype=np.float32),
                    mask_full,
                    K_full,
                )

                estimator = self.objects[obj_label]["pose_estimator"]

                t_register = time.time()
                pose = estimator.register(
                    K=K_use,
                    rgb=color_use,
                    depth=depth_use,
                    ob_mask=mask_use,
                    iteration=self.est_refine_iter,
                )
                register_time = time.time() - t_register

                rospy.loginfo(
                    f"FDPose REGISTER label={obj_label}, "
                    f"roi={roi}, roi_size={color_use.shape[1]}x{color_use.shape[0]}, "
                    f"mask_pixels={mask_pixels}, time={register_time:.4f}s"
                )

                # Return pose in camera frame, as FoundationPose register() estimates.
                pose_result = ObjPose()
                pose_result.label = obj_label
                pose_result.pose = self.matrix_to_ros_pose(pose)
                obj_poses.append(pose_result)

                self._save_current_debug(obj_label, rgb_image, depth_image, mask_full, pose)
                self._save_axis_visualization(obj_label, color_rgb_full, pose)

            except Exception as e:
                rospy.logerr(f"FDPose failed for label={obj_label}: {repr(e)}")
                rospy.logerr(traceback.format_exc())
                continue

        rospy.loginfo(
            f"Pose service done: returned={len(obj_poses)}, total_time={time.time() - t_total:.4f}s"
        )
        return PoseEstResponse(obj_poses=obj_poses)

    def spin(self):
        rospy.spin()


def main():
    try:
        node = PoseEstimatorNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
