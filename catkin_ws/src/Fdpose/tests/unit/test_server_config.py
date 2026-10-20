"""V2 配置加载回归：只模拟 YAML 解析边界，无 ROS/GPU 依赖。"""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from test_mesh_interfaces import load_subject


class ServerConfigTests(unittest.TestCase):
    def setUp(self):
        self.module = load_subject("server_config_subject", "src/fdpose/ros/server_config.py")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.module.PROJECT_ROOT = self.root
        self.path = self.root / "settings.yaml"
        self.path.write_text("# parser boundary mocked in this unit test\n", encoding="utf-8")
        self.warn = Mock()

    def load(self, values=None, overrides=None):
        self.module.yaml.safe_load.return_value = values
        params = {"~config_path": "settings.yaml", **(overrides or {})}
        return self.module.load_server_config(lambda key, default: params.get(key, default), self.warn)

    def test_missing_file_keeps_defaults_and_warns(self):
        cfg = self.load(overrides={"~config_path": "missing.yaml"})
        self.warn.assert_called_once()
        defaults = self.module.ServerConfig()
        for field in fields(defaults):
            if field.name != "config_path":
                self.assertEqual(getattr(cfg, field.name), getattr(defaults, field.name))
        self.assertEqual(cfg.config_path, str(self.root / "missing.yaml"))
        self.module.yaml.safe_load.assert_not_called()

    def test_every_setting_can_be_overridden_by_ros(self):
        yaml_values = {
            "est_refine_iter": 2, "use_roi": True, "roi_margin": 10,
            "min_mask_pixels": 30, "min_depth_m": 0.1, "max_depth_m": 2.0,
            "min_valid_depth_pixels": 20, "min_valid_depth_ratio": 0.1,
            "save_debug_files": True, "save_axis_vis": True, "save_debug_montage": False,
            "axis_vis_scale": 0.2, "axis_vis_thickness": 2,
            "camera_info_dir": "camera_a", "warmup_enabled": False,
            "warmup_dir": "sample_a", "pose_service_name": "/pose_a",
            "mesh_service_name": "/mesh_a",
        }
        expected = {
            "est_refine_iter": 5, "use_roi": False, "roi_margin": 12,
            "min_mask_pixels": 40, "min_depth_m": 0.2, "max_depth_m": 4.0,
            "min_valid_depth_pixels": 10, "min_valid_depth_ratio": 0.3,
            "save_debug_files": False, "save_axis_vis": False, "save_debug_montage": True,
            "axis_vis_scale": 0.3, "axis_vis_thickness": 5,
            "camera_info_dir": str(self.root / "camera_b"), "warmup_enabled": True,
            "warmup_dir": str(self.root / "sample_b"), "pose_service_name": "/pose_b",
            "mesh_service_name": "/mesh_b",
        }
        yaml_cfg = self.load(yaml_values)
        for key, value in yaml_values.items():
            if key in ("camera_info_dir", "warmup_dir"):
                value = str(self.root / value)
            self.assertEqual(getattr(yaml_cfg, key), value)
        cfg = self.load(yaml_values, {"~" + key: value for key, value in expected.items()})
        for key, value in expected.items():
            self.assertEqual(getattr(cfg, key), value)
        self.assertEqual(len(expected), len(fields(cfg)) - 1)

    def test_relative_absolute_and_home_paths(self):
        cfg = self.load({"camera_info_dir": "calibration", "warmup_dir": "~/fdpose_sample"})
        self.assertEqual(cfg.camera_info_dir, str(self.root / "calibration"))
        self.assertEqual(cfg.warmup_dir, str(Path("~/fdpose_sample").expanduser().resolve()))
        cfg = self.load(overrides={"~config_path": str(self.path)})
        self.assertEqual(cfg.config_path, str(self.path))

    def test_unknown_keys_warn(self):
        cfg = self.load({"unused_key": 123})
        self.warn.assert_called_once_with("Unknown FDPose V2 config key: %s", "unused_key")
        self.assertFalse(hasattr(cfg, "unused_key"))

    def test_invalid_types_are_rejected_in_yaml_and_ros(self):
        for name, value in (("use_roi", "false"), ("warmup_enabled", "false"),
                            ("save_debug_files", 0), ("est_refine_iter", "4"),
                            ("est_refine_iter", 1.5), ("roi_margin", True),
                            ("min_depth_m", "0.2"), ("pose_service_name", 123),
                            ("mesh_service_name", " "), ("camera_info_dir", "")):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    self.load({name: value})
                with self.assertRaises(ValueError):
                    self.load({}, {"~" + name: value})

    def test_ranges_and_nonfinite_values_are_rejected(self):
        for name in ("min_depth_m", "max_depth_m", "min_valid_depth_ratio", "axis_vis_scale"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.load({name: value})
        for name, value in (("est_refine_iter", 0), ("est_refine_iter", -1),
                            ("roi_margin", -1), ("min_mask_pixels", 0),
                            ("axis_vis_thickness", 0), ("axis_vis_scale", 0)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load({name: value})

    def test_original_depth_validation_is_preserved(self):
        for values in ({"min_depth_m": -0.1}, {"min_depth_m": 3.0},
                       {"min_valid_depth_pixels": 0}, {"min_valid_depth_ratio": 1.1},
                       {"min_valid_depth_ratio": -0.1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.load(values)
        for ratio in (0.0, 1.0):
            self.assertEqual(self.load({"min_valid_depth_ratio": ratio}).min_valid_depth_ratio, ratio)

    def test_non_mapping_and_parser_errors_propagate(self):
        for value in (["not", "a mapping"], [], False, 0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a mapping"):
                self.load(value)
        self.module.yaml.safe_load.side_effect = ValueError("invalid yaml")
        with self.assertRaisesRegex(ValueError, "invalid yaml"):
            self.load()

    def test_loaded_config_is_immutable(self):
        cfg = self.load({})
        with self.assertRaises(FrozenInstanceError):
            cfg.roi_margin = 100


if __name__ == "__main__":
    unittest.main()
