"""共享工具及 V1/V2 接线回归；不调用真实 ROS、绘图库或 GPU。"""

from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch, mock_open

import numpy as np

from test_mesh_interfaces import load_subject


class ServerToolsTests(unittest.TestCase):
    def setUp(self):
        self.tools = load_subject("server_tools_subject", "src/fdpose/ros/server_utils.py")
        self.config_module = load_subject("tools_config_subject", "src/fdpose/ros/server_config.py")

    def load_server(self, version):
        overrides = {"fdpose.ros.server_utils": self.tools,
                     "fdpose.ros.server_config": self.config_module}
        if version == 2:
            overrides["fdpose.ros.server"] = None  # 任何对 V1 的导入都会失败。
        return load_subject("tools_server_" + str(version),
                            "src/fdpose/ros/server" + ("_v2" if version == 2 else "") + ".py",
                            overrides)

    def test_v2_has_no_v1_base_or_import(self):
        server = self.load_server(2)
        self.assertEqual(server.PoseEstimatorNodeV2.__bases__, (object,))

    def test_depth_conversion_preserves_legacy_threshold(self):
        for values, expected in (([0, 1000, 2000], [0, 1, 2]),
                                 ([0, 1, 2], [0, 1, 2]),
                                 ([20], [20]), ([21], [0.021]),
                                 ([np.nan, np.inf, -np.inf], [0, 0, 0])):
            with self.subTest(values=values):
                original = np.array(values, dtype=np.float32)
                snapshot = original.copy()
                converted = self.tools.legacy_depth_to_meters(original)
                np.testing.assert_allclose(converted, expected)
                np.testing.assert_array_equal(original, snapshot)
                self.assertEqual(converted.dtype, np.float32)

    def test_v2_depth_is_always_millimeters(self):
        depth = np.ones((3, 3), dtype=np.uint16)
        depth[1, 1] = 1000
        original = depth.copy()
        converted = self.tools.depth_mono16_to_meters(depth)
        np.testing.assert_allclose(converted, depth.astype(np.float32) / 1000)
        np.testing.assert_array_equal(depth, original)
        np.testing.assert_allclose(self.tools.depth_mono16_to_meters(
            np.array([0, 1, 20, 21, 65535], dtype=np.uint16)), [0, .001, .02, .021, 65.535])
        self.assertEqual(converted.dtype, np.float32)

    def test_depth_conversion_is_wired_to_the_correct_version(self):
        depth = np.array([[1]], dtype=np.uint16)
        v1, v2 = self.load_server(1), self.load_server(2)
        np.testing.assert_allclose(v1.PoseEstimatorNode._depth_mono16_to_meters(depth), [[1]])
        np.testing.assert_allclose(v2.PoseEstimatorNodeV2._depth_mono16_to_meters(depth), [[.001]])

    def test_roi_crops_all_inputs_and_shifts_only_principal_point(self):
        color = np.arange(6 * 8 * 3).reshape(6, 8, 3)
        depth = np.ones((6, 8), dtype=np.float32)
        mask = np.zeros((6, 8), dtype=bool)
        mask[2:4, 3:5] = True
        K = np.array([[100, 0, 4], [0, 100, 3], [0, 0, 1]], dtype=np.float32)
        args = (color, depth, mask, K)
        crop = self.tools.crop_by_mask(*args, use_roi=True, roi_margin=1, min_mask_pixels=4)
        self.assertEqual(crop[-1], (2, 1, 6, 5))
        for actual, original in zip(crop[:3], args[:3]):
            np.testing.assert_array_equal(actual, original[1:5, 2:6])
            self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(crop[3], [[100, 0, 2], [0, 100, 2], [0, 0, 1]])
        self.assertEqual(K[0, 2], 4)
        self.assertEqual(K[1, 2], 3)
        full = self.tools.crop_by_mask(*args, use_roi=False, roi_margin=1, min_mask_pixels=100)
        for actual, original in zip(full[:4], args):
            self.assertIs(actual, original)
        with self.assertRaisesRegex(ValueError, "mask too small"):
            self.tools.crop_by_mask(*args, use_roi=True, roi_margin=1, min_mask_pixels=5)
        clipped = self.tools.crop_by_mask(*args, use_roi=True, roi_margin=100, min_mask_pixels=1)
        self.assertEqual(clipped[-1], (0, 0, 8, 6))

    def test_servers_match_complete_mesh_labels(self):
        for version in (1, 2):
            with self.subTest(version=version):
                server = self.load_server(version)
                cls = server.PoseEstimatorNode if version == 1 else server.PoseEstimatorNodeV2
                node = cls.__new__(cls)
                node.glctx = object()
                node.objects = {"wp_2": object()}
                node.mesh_config = {"wp_2": {"path": "unused.ply"}}
                if version == 2:
                    node.objects["wp_2"] = {
                        "mesh_path": str((server.MESHES_DIR / "unused.ply").resolve())}
                self.assertEqual(node._load_meshes_from_yaml([" wp_2 ", "wp_2_unstable"]), ["wp_2"])
                node.objects["wp_2_unstable"] = object()
                if version == 2:
                    node.mesh_config["wp_2_unstable"] = {"path": "unstable.ply"}
                    node.objects["wp_2_unstable"] = {
                        "mesh_path": str((server.MESHES_DIR / "unstable.ply").resolve())}
                self.assertEqual(node._load_meshes_from_yaml(["wp_2_unstable"]), ["wp_2_unstable"])

    def test_v2_unload_does_not_rewrite_label(self):
        server = self.load_server(2)
        node = server.PoseEstimatorNodeV2.__new__(server.PoseEstimatorNodeV2)
        node._request_lock = server.Lock()
        original = object()
        node.objects = {"wp_2": original, "wp_2_unstable": object()}
        server.FdposeLoadMeshResponse = lambda **kwargs: SimpleNamespace(**kwargs)
        response = node.load_meshes_from_yaml(SimpleNamespace(cancel=True, goods=[" wp_2_unstable "], mesh_paths=[]))
        self.assertTrue(response.success)
        self.assertEqual(node.objects, {"wp_2": original})

    def test_registry_returns_meshes_and_keeps_error_behavior(self):
        self.tools.yaml.YAMLError = ValueError
        self.tools.yaml.safe_load.return_value = {"meshes": {"a": {"path": "a.ply"}}}
        logerr = Mock()
        with patch.object(self.tools.os.path, "exists", return_value=True), \
             patch("builtins.open", mock_open()):
            self.assertEqual(self.tools.read_mesh_config("registry.yaml", logerr),
                             {"a": {"path": "a.ply"}})
            self.tools.yaml.safe_load.side_effect = ValueError("bad YAML")
            self.assertEqual(self.tools.read_mesh_config("registry.yaml", logerr), {})
        logerr.assert_called_once()
        with patch.object(self.tools.os.path, "exists", return_value=False):
            self.assertEqual(self.tools.read_mesh_config("missing", logerr), {})

    def test_registry_rejects_invalid_structure(self):
        self.tools.yaml.YAMLError = ValueError
        for value in ([], ["bad"], False, "bad", {"meshes": []}, {"meshes": "bad"}, {}):
            with self.subTest(value=value), patch.object(self.tools.os.path, "exists", return_value=True), \
                 patch("builtins.open", mock_open()):
                self.tools.yaml.safe_load.return_value = value
                logerr = Mock()
                self.assertEqual(self.tools.read_mesh_config("registry.yaml", logerr), {})
                logerr.assert_called_once()

    def test_registry_skips_bad_entries_and_loads_valid_mesh(self):
        self.tools.yaml.safe_load.return_value = {"meshes": {
            "missing_path": {"scale": 1}, "scalar": "mesh.ply",
            "empty_path": {"path": ""}, "numeric_path": {"path": 123},
            " ": {"path": "space.ply"}, 3: {"path": "numeric.ply"},
            "good": {"path": "good.ply"}}}
        logerr = Mock()
        with patch.object(self.tools.os.path, "exists", return_value=True), \
             patch("builtins.open", mock_open()):
            registry = self.tools.read_mesh_config("registry.yaml", logerr)
        self.assertEqual(registry, {"good": {"path": "good.ply"}})
        self.assertEqual(logerr.call_count, 6)
        server = self.load_server(2)
        node = server.PoseEstimatorNodeV2.__new__(server.PoseEstimatorNodeV2)
        node.mesh_config = registry
        node._load_mesh_from_path = Mock(return_value=True)
        self.assertEqual(node._load_meshes_from_yaml(["missing_path", "good"]), ["good"])
        node._load_mesh_from_path.assert_called_once_with("good", str(server.MESHES_DIR / "good.ply"))

    def test_registry_read_permission_error_returns_empty_mapping(self):
        self.tools.yaml.YAMLError = ValueError
        logerr = Mock()
        with patch.object(self.tools.os.path, "exists", return_value=True), \
             patch("builtins.open", side_effect=PermissionError("unreadable")):
            self.assertEqual(self.tools.read_mesh_config("registry.yaml", logerr), {})
        logerr.assert_called_once()

    def test_pose_message_keeps_xyz_and_xyzw_order(self):
        pose_message = SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace())
        rotation = Mock()
        rotation.from_matrix.return_value.as_quat.return_value = [0.1, 0.2, 0.3, 0.4]
        matrix = np.eye(4)
        matrix[:3, 3] = [1, 2, 3]
        with patch.dict(sys.modules, {
            "geometry_msgs": SimpleNamespace(),
            "geometry_msgs.msg": SimpleNamespace(Pose=lambda: pose_message),
            "scipy": SimpleNamespace(), "scipy.spatial": SimpleNamespace(),
            "scipy.spatial.transform": SimpleNamespace(Rotation=rotation),
        }):
            result = self.tools.matrix_to_ros_pose(matrix)
        self.assertEqual(vars(result.position), {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertEqual(vars(result.orientation), {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.4})
        np.testing.assert_array_equal(rotation.from_matrix.call_args.args[0], matrix[:3, :3])

    def test_drawing_uses_camera_pose_full_image_K_and_mesh_bbox(self):
        box = Mock(return_value="box_image")
        axis = Mock(return_value="axis_image")
        color = np.zeros((6, 8, 3), dtype=np.uint8)
        pose = np.eye(4)
        pose[:3, 3] = [0.1, 0.2, 1.0]
        K, bbox = np.eye(3), np.ones((2, 3))
        with patch.dict(sys.modules, {"fdpose.core.utils": SimpleNamespace(
                draw_posed_3d_box=box, draw_xyz_axis=axis)}):
            result = self.tools.draw_pose_visualization(color, pose, K, bbox,
                                                       axis_scale=0.1, axis_thickness=3)
        self.assertEqual(result, "axis_image")
        self.assertIs(box.call_args.args[0], K)
        self.assertIs(box.call_args.kwargs["bbox"], bbox)
        self.assertIsNot(box.call_args.kwargs["img"], color)
        np.testing.assert_array_equal(box.call_args.kwargs["img"], color)
        np.testing.assert_allclose(axis.call_args.kwargs["ob_in_cam"], pose)
        self.assertIs(axis.call_args.kwargs["K"], K)
        self.assertTrue(axis.call_args.kwargs["is_input_rgb"])
        self.assertEqual(axis.call_args.kwargs["transparency"], 0)

    def test_v1_keeps_legacy_visualization_output(self):
        for version in (1,):
            with self.subTest(version=version):
                server = self.load_server(version)
                cls = server.PoseEstimatorNode if version == 1 else server.PoseEstimatorNodeV2
                node = cls.__new__(cls)
                cfg = self.config_module.ServerConfig(min_mask_pixels=1, roi_margin=0)
                if version == 2:
                    node.config = cfg
                else:
                    node.use_roi, node.roi_margin, node.min_mask_pixels = True, 0, 1
                    node.save_axis_vis, node.axis_vis_scale, node.axis_vis_thickness = True, 0.1, 3
                color = np.zeros((3, 4, 3), dtype=np.uint8)
                depth, mask, node.K = np.ones((3, 4)), np.ones((3, 4), dtype=bool), np.eye(3)
                node.objects = {"object": {"bbox": np.ones((2, 3))}}
                node.axis_vis_dir = "unused_output"
                self.assertEqual(node._crop_by_mask(color, depth, mask, node.K)[-1], (0, 0, 4, 3))
                pose = np.eye(4)
                with patch.object(server.os, "makedirs"), \
                     patch.object(server, "draw_pose_visualization", return_value="image") as draw:
                    node._save_axis_visualization("object", color, pose)
                self.assertIs(draw.call_args.args[0], color)
                self.assertIs(draw.call_args.args[1], pose)
                self.assertIs(draw.call_args.args[2], node.K)
                write = server.imageio.imwrite.call_args.args
                self.assertTrue(write[0].startswith("unused_output/object_axis_vis_"))
                self.assertEqual(write[1], "image")

    def test_v2_initialization_warms_up_before_registering_services(self):
        for fail_warmup in (False, True):
            with self.subTest(fail_warmup=fail_warmup):
                server = self.load_server(2)
                cfg = self.config_module.ServerConfig(warmup_enabled=True, save_axis_vis=False,
                                                       pose_service_name="/pose_test", mesh_service_name="/mesh_test")
                steps = []

                def warmup():
                    steps.append("warmup")
                    if fail_warmup:
                        raise RuntimeError("warmup unavailable")

                server.rospy.Service.side_effect = lambda name, *args: steps.append(name)
                with patch.object(server, "load_server_config", return_value=cfg), \
                     patch.object(server.np, "load", return_value={"K": np.eye(3)}), \
                     patch.object(server.PoseEstimatorNodeV2, "_read_mesh_config", return_value={}), \
                     patch.object(server.PoseEstimatorNodeV2, "_run_warmup", side_effect=warmup):
                    node = server.PoseEstimatorNodeV2()
                self.assertIs(node.config, cfg)
                self.assertEqual(steps, ["warmup", "/pose_test", "/mesh_test"])
                self.assertIsNone(node.shared_scorer)
                node.spin()
                server.rospy.spin.assert_called_once()

    def test_v2_request_keeps_roi_pose_and_isolates_debug_failures(self):
        server = self.load_server(2)
        node = server.PoseEstimatorNodeV2.__new__(server.PoseEstimatorNodeV2)
        node._request_lock = server.Lock()
        node.config = self.config_module.ServerConfig(min_mask_pixels=1, min_valid_depth_pixels=1,
                                                     min_valid_depth_ratio=1.0, roi_margin=0)
        node.K = np.array([[100, 0, 3], [0, 100, 2], [0, 0, 1]], dtype=np.float32)
        color = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        depth = np.full((4, 6), 1000, dtype=np.uint16)
        depth[0, 0] = 0
        mask = np.zeros((4, 6), dtype=np.uint16)
        mask[1:3, 2:4] = 65535
        bad_mask = np.zeros_like(mask)
        bad_mask[0, 0] = 65535
        images = {"color": color, "depth": depth, "good": mask, "bad": bad_mask}
        node.bridge = Mock()
        node.bridge.imgmsg_to_cv2.side_effect = lambda image, encoding: images[image]
        server.cv2.cvtColor.side_effect = lambda image, code: image[..., ::-1].copy()
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [0.1, 0.2, 1.0]
        estimator = Mock()
        estimator.register.return_value = pose
        node.objects = {"object": {"pose_estimator": estimator}}
        server.ObjPose = SimpleNamespace
        server.PoseEstResponse = lambda **kw: SimpleNamespace(**kw)
        node.matrix_to_ros_pose = Mock(return_value="camera_pose")
        node.debug_writer = Mock(enabled=True)
        node.debug_writer.save_request.side_effect = OSError("disk unavailable")
        request = SimpleNamespace(color_image="color", depth_image="depth", items=[
            SimpleNamespace(label="object", mask="good"),
            SimpleNamespace(label="object", mask="bad"),
            SimpleNamespace(label="object_unstable", mask="good"),
        ])
        result = node.handle_pose_estimation(request)
        self.assertEqual(len(result.obj_poses), 1)
        self.assertEqual(result.obj_poses[0].pose, "camera_pose")
        estimator.register.assert_called_once()
        inputs = estimator.register.call_args.kwargs
        np.testing.assert_array_equal(inputs["rgb"], color[1:3, 2:4, ::-1])
        np.testing.assert_allclose(inputs["depth"], np.ones((2, 2)))
        np.testing.assert_array_equal(inputs["K"], [[100, 0, 1], [0, 100, 1], [0, 0, 1]])
        self.assertEqual(inputs["iteration"], 3)
        np.testing.assert_array_equal(node.matrix_to_ros_pose.call_args.args[0], pose)
        self.assertEqual(node.debug_writer.save_request.call_args.args[1].shape, color.shape)
        self.assertEqual(node.debug_writer.save_request.call_args.args[5][1]["reason"],
                         "insufficient_valid_depth")


if __name__ == "__main__":
    unittest.main()
