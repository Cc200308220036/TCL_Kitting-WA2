"""真实 OpenCV/文件系统回归；仅隔离 ROS/GPU，临时图片自动清理。"""

import ast
import itertools
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

from test_mesh_interfaces import load_subject, ROOT


def load_cpu_pose_drawer():
    # 原样编译 core/utils.py 的纯绘图函数，跳过该模块的 GPU 顶层导入。
    names = {"to_homo", "project_3d_to_2d", "_draw_dashed_line", "draw_xyz_axis", "draw_posed_3d_box"}
    tree = ast.parse((ROOT / "src/fdpose/core/utils.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "cv2": cv2}
    exec(compile(tree, "core/utils.py CPU visualization functions", "exec"), namespace)
    shared = load_subject("debug_shared_tools", "src/fdpose/ros/server_utils.py")
    def draw(*args, **kwargs):
        with patch.dict(sys.modules, {"fdpose.core.utils": SimpleNamespace(**namespace)}):
            return shared.draw_pose_visualization(*args, **kwargs)
    return draw


@unittest.skipIf(cv2 is None, "OpenCV is required for image-output tests")
class DebugWriterTests(unittest.TestCase):
    def setUp(self):
        self.module = load_subject("actual_debug_writer", "src/fdpose/ros/debug_writer.py", {"cv2": cv2})
        self.module.draw_pose_visualization = Mock(side_effect=load_cpu_pose_drawer())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "debug"
        self.warn = Mock()
        self.color = np.full((120, 160, 3), [40, 90, 130], dtype=np.uint8)
        self.depth = np.full((120, 160), 1000, dtype=np.uint16)
        self.K = np.array([[150, 0, 80], [0, 150, 60], [0, 0, 1]], dtype=np.float32)
        vertices = np.array([[-.1, -.1, 0], [.1, -.1, 0], [0, .1, 0]], dtype=np.float32)
        self.mesh = SimpleNamespace(vertices=vertices, faces=np.array([[0, 1, 2]]))
        self.objects = {"part": {"mesh": self.mesh, "bbox": np.array([[-.1, -.1, 0], [.1, .1, .05]])}}
        self.pose = np.eye(4)
        self.pose[2, 3] = 1
        self.records = [{"instance_id": 0, "label": "part", "status": "success", "pose": self.pose,
                         "mask": np.ones((120, 160), dtype=bool)}]

    def writer(self, raw=True, pose=True, montage=False):
        config = SimpleNamespace(save_debug_files=raw, save_axis_vis=pose, save_debug_montage=montage,
                                 axis_vis_scale=.1, axis_vis_thickness=2, min_depth_m=.05, max_depth_m=3.)
        return self.module.DebugWriter(self.root, config, self.K, self.warn)

    def save(self, writer):
        return writer.save_request(self.color, self.color[:, :, ::-1].copy(), self.depth,
                                   self.depth.astype(np.float32) * .001, self.K,
                                   self.records, self.objects, 12.5)

    def test_all_eight_switch_combinations_and_pose_render_reuse(self):
        for raw, pose, montage in itertools.product((False, True), repeat=3):
            with self.subTest(raw=raw, pose=pose, montage=montage):
                self.module.draw_pose_visualization.reset_mock()
                writer = self.writer(raw, pose, montage)
                result = self.save(writer)
                if not (raw or pose or montage):
                    self.assertFalse(self.root.exists())
                    self.assertIsNone(result)
                    continue
                self.assertEqual((writer.run_dir / "raw").exists(), raw)
                self.assertEqual((writer.run_dir / "vis/000001_000_pose_part.png").exists(), pose)
                self.assertEqual((writer.run_dir / "vis/000001_000_montage_part.png").exists(), montage)
                self.assertEqual(self.module.draw_pose_visualization.call_count, int(pose or montage))
                self.assertGreaterEqual(result["total_ms"], 0)
        self.warn.assert_not_called()

    def test_montage_disabled_skips_sampling_projection_and_depth_color(self):
        with patch.object(self.module, "sample_model_points") as sample, \
             patch.object(self.module, "draw_montage") as montage:
            self.save(self.writer())
        sample.assert_not_called()
        montage.assert_not_called()

    def test_same_label_instances_and_repeated_requests_never_overwrite(self):
        self.records.append(dict(self.records[0], instance_id=1))
        writer = self.writer(montage=True)
        with patch.object(self.module, "sample_model_points", wraps=self.module.sample_model_points) as sample:
            self.save(writer)
            self.save(writer)
        sample.assert_called_once_with(self.mesh)
        self.assertTrue((writer.run_dir / "raw/000001/masks/001.png").exists())
        self.assertTrue((writer.run_dir / "raw/000002/results.json").exists())
        self.assertEqual(len(list((writer.run_dir / "vis").glob("*.png"))), 8)
        meta = json.loads((writer.run_dir / "raw/000001/results.json").read_text())
        self.assertEqual([r["instance_id"] for r in meta["items"]], [0, 1])
        self.assertEqual(meta["pose_frame"], "camera")
        np.testing.assert_allclose(meta["items"][0]["pose"], self.pose)

    def test_permissions_are_readable_even_with_restrictive_umask(self):
        previous = os.umask(0o077)
        try:
            writer = self.writer(montage=True)
            self.save(writer)
        finally:
            os.umask(previous)
        for path in [self.root, *self.root.rglob("*")]:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755 if path.is_dir() else 0o644, path)

    def test_json_replace_is_atomic_readable_and_cleans_failed_temporary(self):
        writer = self.writer()
        path = writer.run_dir / "atomic.json"
        self.module._write_json(path, {"old": 1})
        def fail_replace(source, target):
            self.assertEqual(json.loads(path.read_text()), {"old": 1})
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o644)
            self.assertEqual(json.loads(source.read_text()), {"new": "中文"})
            raise OSError("replace failed")
        with patch.object(self.module.os, "replace", side_effect=fail_replace), self.assertRaises(OSError):
            self.module._write_json(path, {"new": "中文"})
        self.assertEqual(json.loads(path.read_text()), {"old": 1})
        self.assertFalse(list(writer.run_dir.glob("*.tmp")))

    def test_raw_data_and_inputs_are_preserved(self):
        self.depth[0, 0] = 0
        snapshots = [value.copy() for value in (self.color, self.depth, self.K, self.pose, self.mesh.vertices)]
        writer = self.writer(montage=True)
        self.save(writer)
        folder = writer.run_dir / "raw/000001"
        np.testing.assert_array_equal(np.load(folder / "depth.npy"), self.depth)
        np.testing.assert_array_equal(cv2.imread(str(folder / "color.png")), self.color)
        meta = json.loads((folder / "results.json").read_text())
        self.assertTrue(meta["record_complete"])
        self.assertEqual(meta["depth"]["unit"], "mm")
        for before, after in zip(snapshots, (self.color, self.depth, self.K, self.pose, self.mesh.vertices)):
            np.testing.assert_array_equal(before, after)

    def test_model_sampling_preserves_global_rng_and_original_coordinates(self):
        state = np.random.get_state()
        points = self.module.sample_model_points(self.mesh)
        current = np.random.get_state()
        np.testing.assert_array_equal(state[1], current[1])
        self.assertEqual(state[2:], current[2:])
        self.assertEqual(points.shape, (2000, 3))
        np.testing.assert_array_equal(points[:, 2], 0)
        self.assertTrue(np.all(np.abs(points[:, :2]) <= .100001))

    def test_projection_uses_camera_K_and_rejects_behind_or_offscreen_points(self):
        color = np.zeros_like(self.color)
        points = np.array([[0, 0, 0], [0, 0, -2], [1e20, 1, 0], [np.nan, 0, 0]])
        projected = self.module.draw_model_projection(color, points, self.pose, self.K)
        np.testing.assert_array_equal(projected[60, 80], [0, 255, 255])
        self.assertEqual(np.count_nonzero(projected[:, :, 1]), 9)
        np.testing.assert_array_equal(color, 0)

    def test_depth_invalid_black_and_fixed_metric_color_range(self):
        depth = np.array([[0, np.nan, np.inf, -1, .05, 3]], dtype=np.float32)
        image = self.module.colorize_depth(depth, .05, 3)
        np.testing.assert_array_equal(image[0, :4], 0)
        expected = cv2.applyColorMap(np.array([[0, 255]], dtype=np.uint8), cv2.COLORMAP_JET)
        np.testing.assert_array_equal(image[0, 4:], expected[0])

    def test_montage_is_bounded_and_pose_image_remains_full_resolution(self):
        self.color = np.full((720, 1280, 3), 80, dtype=np.uint8)
        self.depth = np.full((720, 1280), 1000, dtype=np.uint16)
        self.records[0]["mask"] = np.ones((720, 1280), dtype=bool)
        writer = self.writer(montage=True)
        self.save(writer)
        pose = cv2.imread(str(writer.run_dir / "vis/000001_000_pose_part.png"))
        montage = cv2.imread(str(writer.run_dir / "vis/000001_000_montage_part.png"))
        self.assertEqual(pose.shape, (720, 1280, 3))
        self.assertEqual(montage.shape, (432, 1920, 3))

    def test_raw_failure_does_not_prevent_visualization_and_is_marked_incomplete(self):
        writer = self.writer()
        real_write = self.module._write_image
        def write(path, image):
            if path.name == "color.png":
                raise OSError("raw disk error")
            real_write(path, image)
        with patch.object(self.module, "_write_image", side_effect=write):
            self.save(writer)
        self.assertTrue((writer.run_dir / "vis/000001_000_pose_part.png").exists())
        meta = json.loads((writer.run_dir / "raw/000001/results.json").read_text())
        self.assertFalse(meta["record_complete"])
        self.assertTrue(meta["errors"])

    def test_single_image_failure_does_not_prevent_montage(self):
        writer = self.writer(montage=True)
        real_write = self.module._write_image
        def write(path, image):
            if path.name.endswith("_pose_part.png"):
                raise OSError("pose write failed")
            real_write(path, image)
        with patch.object(self.module, "_write_image", side_effect=write):
            self.save(writer)
        self.assertTrue((writer.run_dir / "vis/000001_000_montage_part.png").exists())

    def test_no_successful_pose_keeps_raw_status_without_rendering(self):
        self.records = [{"instance_id": 0, "label": "part", "status": "skipped", "reason": "mask_too_small"}]
        writer = self.writer(montage=True)
        self.save(writer)
        self.module.draw_pose_visualization.assert_not_called()
        meta = json.loads((writer.run_dir / "raw/000001/results.json").read_text())
        self.assertEqual(meta["items"][0]["reason"], "mask_too_small")

    def test_init_failure_disables_recording_only(self):
        with patch.object(self.module, "_write_json", side_effect=OSError("read-only disk")):
            writer = self.writer()
        self.assertFalse(writer.enabled)
        self.assertIsNone(self.save(writer))
        self.warn.assert_called_once()

    def test_label_suffix_and_json_image_links_match(self):
        label = "wp_2"
        self.objects[label] = self.objects.pop("part")
        self.records[0]["label"] = label
        self.records.append(dict(self.records[0], instance_id=1))
        writer = self.writer(montage=True)
        self.save(writer)
        meta = json.loads((writer.run_dir / "raw/000001/results.json").read_text())
        for index, entry in enumerate(meta["items"]):
            for kind in ("pose", "montage"):
                name = f"vis/000001_{index:03d}_{kind}_wp_2.png"
                self.assertEqual(entry[kind + "_image"], name)
                self.assertTrue((writer.run_dir / name).is_file())
            self.assertEqual(entry["label"], label)

    def test_filename_label_is_safe_and_bounded_without_changing_metadata(self):
        label = "part/../unsafe\\name"
        self.objects[label] = self.objects.pop("part")
        self.records[0]["label"] = label
        writer = self.writer()
        self.save(writer)
        meta = json.loads((writer.run_dir / "raw/000001/results.json").read_text())
        self.assertEqual(meta["items"][0]["label"], label)
        self.assertTrue((writer.run_dir / "vis/000001_000_pose_part____unsafe_name.png").exists())
        self.assertEqual(self.module._filename_label("工件_2"), "工件_2")
        self.assertEqual(self.module._filename_label(""), "object")
        self.assertEqual(len(self.module._filename_label("a" * 1000)), 48)


if __name__ == "__main__":
    unittest.main()
