"""请求级调试记录与可视化；不依赖 ROS/GPU，不改变传入的图像、Mesh 或 Pose。"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import cv2
import numpy as np

from fdpose.ros.server_utils import draw_pose_visualization


def _mkdir(path):
    """只调整本次新建目录的权限，不修改已有父目录或全局 umask。"""
    if path.exists():
        return
    _mkdir(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        # 另一个进程可能同时创建共享输出根目录；不更改其权限。
        return
    path.chmod(0o755)


def _write_json(path, value):
    """先写完整的 0644 临时文件，再原子发布；不做昂贵的逐文件 fsync。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            os.fchmod(handle.fileno(), 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_image(path, image):
    # 较低 PNG 压缩级别，用磁盘空间换取更少编码时间。
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError(f"Failed to write image: {path}")
    path.chmod(0o644)


def _filename_label(label):
    """仅清理文件名后缀；JSON/服务中的原始 label 不变，编号保证实例区分。"""
    return "".join(char if char.isalnum() or char in "_-" else "_"
                   for char in str(label))[:48] or "object"


def sample_model_points(mesh, count=2000):
    """在原始 Mesh 表面采样；局部 RNG 不影响位姿估计使用的随机状态。"""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if faces.size:
        triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
        valid = np.isfinite(areas) & (areas > 0)
        if valid.any():
            triangles, areas = triangles[valid], areas[valid]
            rng = np.random.default_rng(0)
            chosen = triangles[rng.choice(len(triangles), count, p=areas / areas.sum())]
            root, v = np.sqrt(rng.random((count, 1))), rng.random((count, 1))
            return ((1 - root) * chosen[:, 0] + root * (1 - v) * chosen[:, 1]
                    + root * v * chosen[:, 2]).astype(np.float32)
    if not len(vertices):
        return np.empty((0, 3), dtype=np.float32)
    return vertices[np.linspace(0, len(vertices) - 1, min(count, len(vertices)), dtype=int)].copy()


def draw_model_projection(color_bgr, points, pose, K):
    """模型表面点的相机投影（黄色），不是实测点云；不进行遮挡推断。"""
    image = color_bgr.copy()
    camera = np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]
    camera = camera[np.all(np.isfinite(camera), axis=1) & (camera[:, 2] > 1e-6)]
    projected = camera @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    h, w = image.shape[:2]
    valid = (np.all(np.isfinite(uv), axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
             & (uv[:, 1] >= 0) & (uv[:, 1] < h))
    uv = uv[valid].astype(np.int32)
    dots = np.zeros((h, w), dtype=np.uint8)
    dots[uv[:, 1], uv[:, 0]] = 255
    image[cv2.dilate(dots, np.ones((3, 3), dtype=np.uint8)) > 0] = (0, 255, 255)
    return image


def colorize_depth(depth_m, low, high):
    """固定米制范围：近蓝远红；零、负数和非有限值显示黑色。"""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    values = np.zeros(depth_m.shape, dtype=np.uint8)
    values[valid] = np.round(np.clip((depth_m[valid] - low) / (high - low), 0, 1) * 255)
    image = cv2.applyColorMap(values, cv2.COLORMAP_JET)
    image[~valid] = 0
    return image


def draw_montage(pose_bgr, color_bgr, depth_m, points, pose, K, low, high):
    """每栏最大宽度 640，保持比例；三栏均使用同一完整图坐标，不使用 ROI 内参。"""
    h, w = color_bgr.shape[:2]
    width = min(w, 640)
    height = max(1, round(h * width / w))
    size = (width, height)
    color = cv2.resize(color_bgr, size, interpolation=cv2.INTER_AREA)
    scaled_K = np.asarray(K, dtype=np.float64).copy()
    scaled_K[0] *= width / w
    scaled_K[1] *= height / h
    panels = [cv2.resize(pose_bgr, size, interpolation=cv2.INTER_AREA),
              draw_model_projection(color, points, pose, scaled_K),
              colorize_depth(cv2.resize(depth_m, size, interpolation=cv2.INTER_NEAREST), low, high)]
    titles = ("Pose (camera)", "Model projection", "Input depth (m)")
    output = []
    for index, (panel, title) in enumerate(zip(panels, titles)):
        canvas = np.full((height + 72, width, 3), 28, dtype=np.uint8)
        canvas[32:32 + height] = panel
        cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        if index == 2:
            gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (10, 1))
            canvas[height + 36:height + 46] = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
            caption = f"{low:g} m (near)  ->  {high:g} m (far); invalid=black"
        else:
            caption = "Full-image camera frame" if index == 0 else "Surface samples; no occlusion test"
        cv2.putText(canvas, caption, (8, height + 64), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220, 220, 220), 1)
        output.append(canvas)
    return np.concatenate(output, axis=1)


class DebugWriter:
    """由 Service 的请求锁串行调用；写入同步执行，失败只告警。"""

    def __init__(self, root, config, K, warn):
        self.config, self.warn = config, warn
        self.run_dir = None
        self.sequence = 0
        if not (config.save_debug_files or config.save_axis_vis or config.save_debug_montage):
            return
        try:
            root = Path(root)
            _mkdir(root)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            self.run_dir = root / f"{stamp}_{uuid4().hex[:8]}"
            _mkdir(self.run_dir)
            _write_json(self.run_dir / "session.json", {
                "schema_version": 1, "server_version": "v2", "run_id": self.run_dir.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "config": vars(config), "K": np.asarray(K).tolist(),
                "pose_frame": "camera", "model_points": "original mesh surface",
            })
        except Exception as exc:
            self.run_dir = None
            self.warn(f"Debug recording disabled: {exc}")

    @property
    def enabled(self):
        return self.run_dir is not None

    def save_request(self, color_bgr, color_rgb, depth_raw, depth_m, K, records, objects, processing_ms):
        if not self.enabled:
            return None
        started = time.perf_counter()
        self.sequence += 1
        sample_id = f"{self.sequence:06d}"
        timing = {"raw_ms": 0.0, "vis_render_ms": 0.0, "vis_write_ms": 0.0}
        errors = []
        entries = [{key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in record.items() if key != "mask"} for record in records]
        meta = {"schema_version": 1, "sample_id": sample_id, "record_complete": False,
                "saved_at": datetime.now(timezone.utc).isoformat(), "pose_frame": "camera",
                "K": np.asarray(K).tolist(), "processing_ms": processing_ms,
                "depth": {"file": "depth.npy", "unit": "mm", "dtype": str(depth_raw.dtype),
                          "shape": list(depth_raw.shape)},
                "color": {"file": "color.png", "encoding": "bgr8"},
                "items": entries, "errors": errors, "timing_before_final_json_ms": timing}
        raw_dir = self.run_dir / "raw" / sample_id
        raw_ok = False
        if self.config.save_debug_files:
            start = time.perf_counter()
            try:
                _mkdir(raw_dir)
                _write_json(raw_dir / "results.json", meta)
                _write_image(raw_dir / "color.png", color_bgr)
                with (raw_dir / "depth.npy").open("wb") as handle:
                    np.save(handle, depth_raw, allow_pickle=False)
                (raw_dir / "depth.npy").chmod(0o644)
                for entry, record in zip(entries, records):
                    if "mask" in record:
                        _mkdir(raw_dir / "masks")
                        name = f"masks/{record['instance_id']:03d}.png"
                        _write_image(raw_dir / name, record["mask"].astype(np.uint8) * 255)
                        entry["mask"] = name
                raw_ok = True
            except Exception as exc:
                errors.append(f"raw: {exc}")
                self.warn(f"Debug raw {sample_id} failed: {exc}")
            timing["raw_ms"] += (time.perf_counter() - start) * 1000

        if self.config.save_axis_vis or self.config.save_debug_montage:
            for entry, record in zip(entries, records):
                if record["status"] != "success":
                    continue
                try:
                    start = time.perf_counter()
                    mesh_record = objects[record["label"]]
                    pose = record["pose"]
                    pose_bgr = draw_pose_visualization(
                        color_rgb, pose, K, mesh_record["bbox"],
                        axis_scale=self.config.axis_vis_scale,
                        axis_thickness=self.config.axis_vis_thickness,
                    )[:, :, ::-1].copy()
                    cv2.putText(pose_bgr, f"{sample_id}/{record['instance_id']:03d} {record['label']}",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                except Exception as exc:
                    errors.append(f"pose render {record['instance_id']}: {exc}")
                    self.warn(f"Debug pose render {sample_id} failed: {exc}")
                    continue
                finally:
                    timing["vis_render_ms"] += (time.perf_counter() - start) * 1000
                for kind, enabled in (("pose", self.config.save_axis_vis),
                                      ("montage", self.config.save_debug_montage)):
                    if not enabled:
                        continue
                    try:
                        image = pose_bgr
                        if kind == "montage":
                            start = time.perf_counter()
                            try:
                                # 缓存属于当前 Mesh 记录，卸载/重载 Mesh 时自然失效。
                                if "debug_points" not in mesh_record:
                                    mesh_record["debug_points"] = sample_model_points(mesh_record["mesh"])
                                image = draw_montage(pose_bgr, color_bgr, depth_m,
                                                     mesh_record["debug_points"], pose, K,
                                                     self.config.min_depth_m, self.config.max_depth_m)
                            finally:
                                timing["vis_render_ms"] += (time.perf_counter() - start) * 1000
                        start = time.perf_counter()
                        try:
                            _mkdir(self.run_dir / "vis")
                            label_suffix = _filename_label(record["label"])
                            name = f"{sample_id}_{record['instance_id']:03d}_{kind}_{label_suffix}.png"
                            _write_image(self.run_dir / "vis" / name, image)
                        finally:
                            timing["vis_write_ms"] += (time.perf_counter() - start) * 1000
                        entry[kind + "_image"] = f"vis/{name}"
                    except Exception as exc:
                        errors.append(f"{kind} {record['instance_id']}: {exc}")
                        self.warn(f"Debug {kind} {sample_id} failed: {exc}")

        if self.config.save_debug_files:
            start = time.perf_counter()
            try:
                meta["record_complete"] = raw_ok
                _write_json(raw_dir / "results.json", meta)
            except Exception as exc:
                self.warn(f"Debug metadata {sample_id} failed: {exc}")
            timing["raw_ms"] += (time.perf_counter() - start) * 1000
        return dict(timing, sample_id=sample_id, total_ms=(time.perf_counter() - started) * 1000)
