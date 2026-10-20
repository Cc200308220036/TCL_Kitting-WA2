#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimental FDPose ROS server V2.3.

This node is independent of the V1 ROS server and uses the same FoundationPose
core. Configuration lives in server_config; stateless tools live in server_utils.
Its behavior differences from V1 are:

- reject masks that contain too little valid depth before calling register();
- keep debug/visualization save failures separate from pose-estimation errors;
- share one ScorePredictor and PoseRefinePredictor across all loaded labels;
- support loading label-aligned external Mesh paths when provided by the request;
- support unloading selected mesh labels without affecting other labels;
- load defaults from YAML with ROS private-parameter overrides;
- optionally run a fixed RGB-D sample through the GPU inference path at startup;
- validate request image dimensions and serialize Pose/Mesh callbacks;
- expose independent V2 service names and timestamped raw/vis debug sessions.

The stable V1 implementation remains in :mod:`fdpose.ros.server`.
"""

import os
from pathlib import Path
from threading import Lock
import time
import traceback

import cv2
import numpy as np
import rospy
import torch
import trimesh
import yaml
from cv_bridge import CvBridge
import nvdiffrast.torch as dr

from naviai_manip_srvs.msg import ObjPose
from naviai_manip_srvs.srv import (
    FdposeLoadMesh,
    FdposeLoadMeshResponse,
    PoseEst,
    PoseEstResponse,
)

from fdpose.core.estimator import FoundationPose
from fdpose.learning.training.predict_pose_refine import PoseRefinePredictor
from fdpose.learning.training.predict_score import ScorePredictor
from fdpose.paths import (
    DEBUG_DIR,
    MESH_REGISTRY_PATH,
    MESHES_DIR,
)
from fdpose.ros.server_config import load_server_config
from fdpose.ros.debug_writer import DebugWriter
from fdpose.ros.server_utils import (
    crop_by_mask,
    depth_mono16_to_meters,
    matrix_to_ros_pose,
    read_mesh_config,
)

# 原先导入 V1 模块会设置这些环境变量；解耦后显式保留该副作用。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


class PoseEstimatorNodeV2:
    """V2 server with validation, shared predictors, and isolated debug errors."""

    def __init__(self):
        rospy.init_node("pose_estimator_track_node_v2", anonymous=True)
        self.config = load_server_config(rospy.get_param, rospy.logwarn)
        rospy.loginfo("FDPose V2 config: %s", self.config.config_path)

        # 底层 estimator(debug=0) 仍会 mkdir；仅传 runtime 根目录，避免全关时
        # 也创建调试子目录。正式调试文件完全由 DebugWriter 管理。
        self.debug_dir = str(DEBUG_DIR.parent)
        self.mesh_registry_path = str(MESH_REGISTRY_PATH)

        # rospy 不同连接可能并发回调；Pose 和 Mesh 管理必须共用这把锁。
        # 覆盖缓存、网络、渲染上下文及调试读写，不只保护 register()。
        self._request_lock = Lock()
        self.objects = {}
        self.mesh_config = self._read_mesh_config()
        self.glctx = None
        self.shared_scorer = None
        self.shared_refiner = None
        self.bridge = CvBridge()

        # 内参数据
        intrinsic_path = os.path.join(self.config.camera_info_dir, "intrinsic_data.npz")
        intrinsic_data = np.load(intrinsic_path, allow_pickle=True)
        self.K = intrinsic_data["K"].astype(np.float32)
        self.debug_writer = DebugWriter(DEBUG_DIR, self.config, self.K, rospy.logwarn)
        if self.debug_writer.enabled:
            rospy.loginfo("FDPose debug output: %s", self.debug_writer.run_dir)

        if self.config.warmup_enabled:
            try:
                self._run_warmup()
            except Exception as exc:
                rospy.logerr("FDPose V2 warmup failed: %r", exc)
                rospy.logerr(traceback.format_exc())

        self.pose_estimation_service = rospy.Service(
            self.config.pose_service_name,
            PoseEst,
            self.handle_pose_estimation,
        )
        self.mesh_service = rospy.Service(
            self.config.mesh_service_name,
            FdposeLoadMesh,
            self.load_meshes_from_yaml,
        )
        rospy.loginfo(
            "FDPose V2.3 ready: mesh_service=%s, pose_service=%s",
            self.config.mesh_service_name,
            self.config.pose_service_name,
        )

    def matrix_to_ros_pose(self, matrix):
        return matrix_to_ros_pose(matrix)

    def _read_mesh_config(self):
        return read_mesh_config(self.mesh_registry_path, rospy.logerr)

    @staticmethod
    def _depth_mono16_to_meters(depth_image):
        return depth_mono16_to_meters(depth_image)

    def _crop_by_mask(self, color, depth, mask, K):
        return crop_by_mask(
            color, depth, mask, K,
            use_roi=self.config.use_roi,
            roi_margin=self.config.roi_margin,
            min_mask_pixels=self.config.min_mask_pixels,
        )

    def spin(self):
        rospy.spin()

    def _get_shared_predictors(self):
        if self.shared_scorer is None and self.shared_refiner is None:
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            self.shared_scorer = scorer
            self.shared_refiner = refiner
            rospy.loginfo(
                "Initialized shared FDPose predictors: scorer_id=%s, refiner_id=%s",
                id(self.shared_scorer),
                id(self.shared_refiner),
            )
        elif self.shared_scorer is None or self.shared_refiner is None:
            raise RuntimeError("shared predictor state is incomplete")

        return self.shared_scorer, self.shared_refiner

    def _run_warmup(self):
        """Run one fixed RGB-D sample through the expensive GPU path."""

        warmup_dir = Path(self.config.warmup_dir)
        manifest_path = warmup_dir / "warmup.yaml"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Warmup manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        if not isinstance(manifest, dict):
            raise ValueError(f"Warmup manifest must be a mapping: {manifest_path}")

        label = str(manifest.get("label", "")).strip()
        if not label:
            raise ValueError(f"Warmup manifest has no label: {manifest_path}")

        def sample_path(key):
            relative = str(manifest.get(key, "")).strip()
            if not relative:
                raise ValueError(f"Warmup manifest has no {key}: {manifest_path}")
            path = (warmup_dir / relative).resolve()
            try:
                path.relative_to(warmup_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"Warmup {key} escapes sample directory") from exc
            if not path.is_file():
                raise FileNotFoundError(f"Warmup {key} not found: {path}")
            return path

        rgb_path = sample_path("rgb")
        depth_path = sample_path("depth")
        mask_path = sample_path("mask")

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb_bgr is None or depth_raw is None or mask_raw is None:
            raise ValueError(f"Failed to read Warmup RGB-D sample: {warmup_dir}")
        if depth_raw.ndim != 2:
            raise ValueError(f"Warmup depth must be 2D, got {depth_raw.shape}")
        if rgb_bgr.shape[:2] != depth_raw.shape or depth_raw.shape != mask_raw.shape:
            raise ValueError(
                "Warmup RGB/depth/mask size mismatch: "
                f"rgb={rgb_bgr.shape[:2]}, depth={depth_raw.shape}, "
                f"mask={mask_raw.shape}"
            )

        depth_unit = str(manifest.get("depth_unit", "mm")).strip().lower()
        depth_m = np.asarray(depth_raw, dtype=np.float32)
        if depth_unit == "mm":
            depth_m *= 0.001
        elif depth_unit != "m":
            raise ValueError(f"Unsupported Warmup depth_unit={depth_unit!r}")
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

        mask_bool = np.asarray(mask_raw > 0, dtype=bool)
        mask_pixels = int(mask_bool.sum())
        if mask_pixels < self.config.min_mask_pixels:
            raise ValueError(
                f"Warmup mask too small: {mask_pixels} < {self.config.min_mask_pixels}"
            )
        valid_pixels, required_pixels = self._valid_depth_stats(depth_m, mask_bool)
        if valid_pixels < required_pixels:
            raise ValueError(
                "Warmup valid depth is insufficient: "
                f"valid={valid_pixels}, required={required_pixels}"
            )

        intrinsic_name = str(manifest.get("camera_intrinsics", "")).strip()
        if intrinsic_name:
            sample_K = np.loadtxt(str(sample_path("camera_intrinsics"))).reshape(3, 3)
            if not np.allclose(sample_K, self.K, rtol=1e-4, atol=1e-3):
                rospy.logwarn(
                    "Warmup sample K differs from Service K; using Service K"
                )

        color_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        color_use, depth_use, mask_use, K_use, roi = self._crop_by_mask(
            np.asarray(color_rgb, dtype=np.uint8),
            depth_m,
            mask_bool,
            np.asarray(self.K, dtype=np.float32),
        )

        refine_iteration = int(manifest.get("refine_iteration", 1))
        if refine_iteration < 1:
            raise ValueError("Warmup refine_iteration must be at least 1")

        loaded_before = label in self.objects
        rospy.loginfo(
            "FDPose V2 Warmup start: label=%s, sample=%s, roi=%s",
            label,
            warmup_dir,
            roi,
        )
        start_time = time.time()
        try:
            loaded = self._load_meshes_from_yaml([label])
            if label not in loaded:
                raise RuntimeError(f"Failed to load Warmup Mesh label={label}")

            estimator = self.objects[label]["pose_estimator"]
            torch.cuda.synchronize()
            pose = estimator.register(
                K=K_use,
                rgb=color_use,
                depth=depth_use,
                ob_mask=mask_use,
                iteration=refine_iteration,
            )
            torch.cuda.synchronize()

            pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
            if not np.all(np.isfinite(pose)):
                raise ValueError("Warmup pose contains NaN or Inf")
        finally:
            if not loaded_before:
                self.objects.pop(label, None)

        rospy.loginfo(
            "FDPose V2 Warmup done: label=%s, time=%.4fs",
            label,
            time.time() - start_time,
        )

    def _load_meshes_from_yaml(self, labels):
        """Load per-label mesh state while sharing the inference networks."""
        if not labels:
            rospy.logwarn("No mesh labels provided for loading")
            return []
        if not self.mesh_config:
            rospy.logerr("Mesh configuration is empty; cannot load meshes")
            return []
        loaded_labels = []
        for raw_label in labels:
            obj_label = str(raw_label).strip()

            obj_info = self.mesh_config.get(obj_label)
            if not obj_info:
                rospy.logwarn("No mesh configuration found for label: %s", obj_label)
                continue

            mesh_path = str(MESHES_DIR / obj_info["path"])
            if self._load_mesh_from_path(obj_label, mesh_path):
                loaded_labels.append(obj_label)

        return loaded_labels

    def _load_mesh_from_path(self, obj_label, mesh_path):
        """Load one Mesh path into the per-label estimator cache."""

        obj_label = str(obj_label).strip()
        # 统一绝对路径和符号链接，注册表与显式路径入口都必须核对缓存来源。
        mesh_path = os.path.realpath(os.path.expanduser(str(mesh_path)))
        if obj_label in self.objects:
            loaded_path = self.objects[obj_label].get("mesh_path")
            if loaded_path != mesh_path:
                rospy.logwarn(
                    "Mesh source conflict: label=%s, loaded=%s, requested=%s; "
                    "unload this label first or use client.load_mesh(..., reload=True)",
                    obj_label, loaded_path, mesh_path,
                )
                return False
            rospy.loginfo("Mesh already loaded: %s", obj_label)
            return True

        rospy.loginfo("Loading V2 mesh label=%s, path=%s", obj_label, mesh_path)
        if not os.path.isfile(mesh_path):
            rospy.logwarn("Mesh file not found: %s", mesh_path)
            return False

        try:
            mesh = trimesh.load(mesh_path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

            mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
            bbox = np.asarray(mesh.bounds, dtype=np.float32)
            model_pts = np.asarray(mesh.vertices, dtype=np.float32)
            model_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            scorer, refiner = self._get_shared_predictors()
            # 注册表加载和外部路径加载均在这里复用同一个渲染上下文。
            if self.glctx is None:
                self.glctx = dr.RasterizeCudaContext()

            estimator = FoundationPose(
                model_pts=model_pts,
                model_normals=model_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=self.debug_dir,
                debug=0,
                glctx=self.glctx,
            )
            self.objects[obj_label] = {
                "mesh_path": mesh_path,
                "mesh": mesh,
                "bbox": bbox,
                "pose_estimator": estimator,
            }
            rospy.loginfo(
                "Loaded V2 mesh label=%s with shared scorer_id=%s, refiner_id=%s",
                obj_label,
                id(estimator.scorer),
                id(estimator.refiner),
            )
            return True
        except Exception as exc:
            rospy.logerr("Failed to load V2 mesh %s: %r", mesh_path, exc)
            rospy.logerr(traceback.format_exc())
            return False

    def load_meshes_from_yaml(self, req):
        """Load meshes, or unload selected meshes when ``cancel`` is true.

        A cancel request with labels unloads only those labels.  A cancel
        request without labels unloads all currently loaded meshes.
        """
        with self._request_lock:
            return self._handle_mesh_request(req)

    def _handle_mesh_request(self, req):
        """Mesh 回调内部实现；调用方持有 _request_lock。"""

        response = FdposeLoadMeshResponse(success=True)
        requested_labels = [
            str(label).strip() for label in req.goods
        ]
        # 非空 goods 中的空标签是坏请求，不可过滤后变成“卸载全部”。
        # 先整批校验，再执行任何加载/卸载，避免部分修改。
        if any(not label for label in requested_labels):
            rospy.logwarn("Reject Mesh request: goods contains an empty label")
            response.success = False
            return response
        mesh_paths = list(getattr(req, "mesh_paths", []))

        if req.cancel:
            if mesh_paths:
                rospy.logwarn("Ignoring mesh_paths in a Mesh unload request")
            if requested_labels:
                for label in requested_labels:
                    if self.objects.pop(label, None) is None:
                        rospy.logwarn("Mesh is not loaded, skip unload: %s", label)
                    else:
                        rospy.loginfo("Unloaded mesh: %s", label)
            else:
                if self.objects:
                    rospy.loginfo(
                        "Unloading all meshes: %s",
                        list(self.objects.keys()),
                    )
                self.objects.clear()
            return response

        self.mesh_config = self._read_mesh_config()
        if not requested_labels:
            rospy.logwarn("Mesh load request received with empty goods list")
            return response

        if mesh_paths and len(mesh_paths) != len(requested_labels):
            rospy.logwarn(
                "Mesh request size mismatch: goods=%d, mesh_paths=%d",
                len(requested_labels),
                len(mesh_paths),
            )
            response.success = False
            return response

        if not mesh_paths:
            loaded_labels = self._load_meshes_from_yaml(requested_labels)
        else:
            loaded_labels = []
            for label, mesh_path in zip(requested_labels, mesh_paths):
                if str(mesh_path).strip():
                    if self._load_mesh_from_path(label, mesh_path):
                        loaded_labels.append(label)
                else:
                    loaded_labels.extend(self._load_meshes_from_yaml([label]))

        failed_labels = [
            label for label in requested_labels if label not in loaded_labels
        ]
        if failed_labels:
            rospy.logwarn("Failed to load meshes: %s", failed_labels)
            response.success = False
        return response

    def _valid_depth_stats(self, depth_m, mask):
        valid_depth = (
            mask
            & np.isfinite(depth_m)
            & (depth_m >= self.config.min_depth_m)
            & (depth_m <= self.config.max_depth_m)
        )
        mask_pixels = int(mask.sum())
        valid_pixels = int(valid_depth.sum())
        required_pixels = max(
            self.config.min_valid_depth_pixels,
            int(np.ceil(mask_pixels * self.config.min_valid_depth_ratio)),
        )
        return valid_pixels, required_pixels

    def handle_pose_estimation(self, req):
        """串行保护完整请求；耗时区分等待锁和处理时间。"""
        t_total = time.perf_counter()
        with self._request_lock:
            lock_wait = time.perf_counter() - t_total
            response = self._handle_pose_request(req)

        rospy.loginfo(
            "FDPose V2 service done: returned=%d, lock_wait=%.4fs, total_time=%.4fs",
            len(response.obj_poses),
            lock_wait,
            time.perf_counter() - t_total,
        )
        return response

    def _handle_pose_request(self, req):
        """Pose 回调内部实现；调用方持有 _request_lock，包括调试保存阶段。"""
        t_processing = time.perf_counter()
        try:
            rgb_image = np.asarray(self.bridge.imgmsg_to_cv2(req.color_image, "bgr8"))
            depth_image = np.asarray(self.bridge.imgmsg_to_cv2(req.depth_image, "mono16"))
            if rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or rgb_image.size == 0:
                raise ValueError(f"RGB must be a non-empty HxWx3 image, got {rgb_image.shape}")
            if depth_image.ndim != 2 or depth_image.size == 0:
                raise ValueError(f"Depth must be a non-empty HxW image, got {depth_image.shape}")
            if rgb_image.shape[:2] != depth_image.shape:
                raise ValueError(
                    f"RGB/depth size mismatch: rgb={rgb_image.shape[:2]}, depth={depth_image.shape}"
                )

            color_rgb_full = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            depth_m_full = self._depth_mono16_to_meters(depth_image)
        except Exception as exc:
            # 整帧输入错误不能伪装成“没有目标”；保留服务失败语义并说明原因。
            rospy.logerr("FDPose V2 invalid RGB-D request: %s", exc)
            raise rospy.ServiceException(f"Invalid RGB-D request: {exc}") from exc

        K_full = np.asarray(self.K, dtype=np.float32)

        obj_poses = []
        debug_records = [] if self.debug_writer.enabled else None
        for i, item in enumerate(req.items):
            obj_label = str(item.label).strip()
            record = {"instance_id": i, "label": obj_label, "status": "skipped"}
            if debug_records is not None:
                debug_records.append(record)
            if obj_label not in self.objects:
                record["reason"] = "mesh_not_loaded"
                rospy.logwarn(
                    "Skip item %d: label=%s not loaded. Loaded labels=%s",
                    i,
                    obj_label,
                    list(self.objects.keys()),
                )
                continue

            try:
                mask_image = np.asarray(self.bridge.imgmsg_to_cv2(item.mask, "mono16"))
                if mask_image.ndim != 2 or mask_image.shape != depth_image.shape:
                    raise ValueError(
                        f"Mask must match RGB-D size: mask={mask_image.shape}, expected={depth_image.shape}"
                    )
                mask_full = np.asarray(mask_image > 0, dtype=bool)
                mask_pixels = int(mask_full.sum())
                if debug_records is not None and self.config.save_debug_files:
                    record["mask"] = mask_full

                if mask_pixels < self.config.min_mask_pixels:
                    record["reason"] = "mask_too_small"
                    rospy.logwarn(
                        "Skip item %d: label=%s, mask too small: %d pixels",
                        i,
                        obj_label,
                        mask_pixels,
                    )
                    continue

                valid_depth_pixels, required_depth_pixels = self._valid_depth_stats(
                    depth_m_full,
                    mask_full,
                )
                if valid_depth_pixels < required_depth_pixels:
                    record["reason"] = "insufficient_valid_depth"
                    rospy.logwarn(
                        "Skip item %d: label=%s, valid_depth=%d/%d, required=%d",
                        i,
                        obj_label,
                        valid_depth_pixels,
                        mask_pixels,
                        required_depth_pixels,
                    )
                    continue

                color_use, depth_use, mask_use, K_use, roi = self._crop_by_mask(
                    np.asarray(color_rgb_full, dtype=np.uint8),
                    np.asarray(depth_m_full, dtype=np.float32),
                    mask_full,
                    K_full,
                )

            except Exception as exc:
                record["reason"] = f"input_preparation_failed: {exc}"
                rospy.logwarn(
                    "Skip item %d: label=%s, input preparation failed: %r",
                    i,
                    obj_label,
                    exc,
                )
                continue

            try:
                estimator = self.objects[obj_label]["pose_estimator"]
                # register 返回 CPU NumPy Pose；这里记录调用墙钟耗时，不是纯 GPU 时间。
                t_register = time.perf_counter()
                pose = estimator.register(
                    K=K_use,
                    rgb=color_use,
                    depth=depth_use,
                    ob_mask=mask_use,
                    iteration=self.config.est_refine_iter,
                )
                register_time = time.perf_counter() - t_register

                pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
                if not np.all(np.isfinite(pose)):
                    raise ValueError("estimated pose contains NaN or Inf")

                pose_result = ObjPose()
                pose_result.label = obj_label
                pose_result.pose = self.matrix_to_ros_pose(pose)

            except Exception as exc:
                record.update(status="failed", reason=str(exc))
                rospy.logerr(
                    "FDPose V2 estimation failed for label=%s: %r",
                    obj_label,
                    exc,
                )
                rospy.logerr(traceback.format_exc())
                continue

            obj_poses.append(pose_result)
            record.update(status="success", pose=pose, register_ms=register_time * 1000)
            rospy.loginfo(
                "FDPose V2 REGISTER label=%s, roi=%s, roi_size=%dx%d, "
                "mask_pixels=%d, valid_depth_pixels=%d, time=%.4fs",
                obj_label,
                roi,
                color_use.shape[1],
                color_use.shape[0],
                mask_pixels,
                valid_depth_pixels,
                register_time,
            )

        if debug_records is not None:
            try:
                timing = self.debug_writer.save_request(
                    rgb_image, color_rgb_full, depth_image, depth_m_full, K_full,
                    debug_records, self.objects, (time.perf_counter() - t_processing) * 1000,
                )
                if timing is not None:
                    rospy.loginfo(
                        "FDPose DEBUG request=%s raw=%.2fms render=%.2fms "
                        "vis_write=%.2fms total=%.2fms",
                        timing["sample_id"], timing["raw_ms"], timing["vis_render_ms"],
                        timing["vis_write_ms"], timing["total_ms"],
                    )
            except Exception as exc:
                rospy.logwarn("FDPose V2 debug save failed (poses preserved): %r", exc)

        return PoseEstResponse(obj_poses=obj_poses)


def main():
    try:
        node = PoseEstimatorNodeV2()
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
