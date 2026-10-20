"""Mesh 加载及本地入口回归；只模拟 ROS/GPU 边界，不连接服务或 GPU。"""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_subject(name, relative_path, overrides=None):
    # 导入真正的被测模块；在局部作用域替代本机没有的 ROS/GPU 依赖。
    dependencies = {key: MagicMock() for key in (
        "rospy", "cv_bridge", "naviai_manip_srvs", "naviai_manip_srvs.msg",
        "naviai_manip_srvs.srv", "yaml", "scipy", "scipy.spatial",
        "scipy.spatial.transform", "cv2", "imageio", "torch", "trimesh", "nvdiffrast",
        "nvdiffrast.torch", "fdpose.core.estimator", "fdpose.core.datareader",
        "fdpose.ros.debug_writer",
        "fdpose.learning.training.predict_score", "fdpose.learning.training.predict_pose_refine",
    )}
    dependencies.update(overrides or {})
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**dependencies, name: module}):
        spec.loader.exec_module(module)
    return module


class MeshInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.client_module = load_subject("mesh_client_subject", "src/fdpose/ros/client.py")
        self.client = self.client_module.FdposeClient()
        self.client._ensure_ready = Mock()
        self.client.clear_meshes = Mock()
        self.client._load_mesh_proxy = Mock(return_value=SimpleNamespace(success=True))
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.mesh_path = Path(self.temporary.name) / "model.ply"
        self.mesh_path.write_bytes(b"mesh fixture; geometry loader is mocked")

    def test_old_message_rejected_before_clearing(self):
        self.client_module.FdposeLoadMeshRequest = SimpleNamespace
        with self.assertRaises(self.client_module.FdposeClientError):
            self.client.load_meshes(["wp_2"], clear_existing=True, mesh_paths=[self.mesh_path])
        self.client.clear_meshes.assert_not_called()
        self.client._load_mesh_proxy.assert_not_called()

    def test_invalid_path_rejected_before_clearing(self):
        with self.assertRaises(FileNotFoundError):
            self.client.load_meshes(["wp_2"], clear_existing=True, mesh_paths=[self.mesh_path / "missing"])
        self.client.clear_meshes.assert_not_called()

    def test_valid_message_clears_before_loading(self):
        self.client_module.FdposeLoadMeshRequest = lambda: SimpleNamespace(mesh_paths=[])
        order = Mock()
        order.attach_mock(self.client.clear_meshes, "clear")
        order.attach_mock(self.client._load_mesh_proxy, "load")
        self.client.load_meshes(["wp_2"], clear_existing=True, mesh_paths=[self.mesh_path])
        self.assertEqual([call[0] for call in order.mock_calls], ["clear", "load"])
        request = self.client._load_mesh_proxy.call_args.args[0]
        self.assertEqual(request.goods, ["wp_2"])
        self.assertEqual(request.mesh_paths, [str(self.mesh_path.resolve())])

    def test_registered_label_still_supports_old_message(self):
        self.client_module.FdposeLoadMeshRequest = SimpleNamespace
        self.client.load_meshes(["wp_2"])
        self.client._load_mesh_proxy.assert_called_once()

    def test_external_and_registered_meshes_share_one_context(self):
        server = load_subject("mesh_server_subject", "src/fdpose/ros/server_v2.py")
        node = server.PoseEstimatorNodeV2.__new__(server.PoseEstimatorNodeV2)
        node.objects, node.glctx, node.debug_dir = {}, None, self.temporary.name
        node._get_shared_predictors = lambda: ("scorer", "refiner")
        node.mesh_config = {"registered": {"path": str(self.mesh_path)}}
        server.trimesh.Scene = type("Scene", (), {})
        server.trimesh.load.return_value = SimpleNamespace(
            vertices=np.zeros((3, 3)), vertex_normals=np.ones((3, 3)), bounds=np.zeros((2, 3)))
        server.FoundationPose = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        server.dr.RasterizeCudaContext = Mock(return_value=object())
        self.assertTrue(node._load_mesh_from_path("external", self.mesh_path))
        self.assertIsNotNone(node.glctx)
        self.assertEqual(node._load_meshes_from_yaml(["registered"]), ["registered"])
        server.dr.RasterizeCudaContext.assert_called_once()
        for record in node.objects.values():
            self.assertIs(record["pose_estimator"].glctx, node.glctx)

    def test_depth_convention_is_integer_mm_float_m(self):
        for depth in (np.array([[1000]], dtype=np.uint16), np.array([[1.0]], dtype=np.float32)):
            np.testing.assert_array_equal(self.client._depth_to_mono16(depth), [[1000]])

    def test_single_mesh_label_is_not_rewritten_for_registered_or_external_load(self):
        self.client.load_meshes = Mock()
        self.client.load_mesh(" wp_2_unstable ")
        self.client.load_meshes.assert_called_once_with(
            ["wp_2_unstable"], mesh_paths=None, reload=False)
        self.client.load_meshes.reset_mock()
        self.client.load_mesh("wp_2_unstable", mesh_path=str(self.mesh_path))
        self.client.load_meshes.assert_called_once_with(
            ["wp_2_unstable"], mesh_paths=[str(self.mesh_path)], reload=False)

    def test_label_path_safety_validation_is_preserved(self):
        for label in ("", " ", "../mesh", "a/b", "a\\b"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.client.load_mesh(label)
        self.client._load_mesh_proxy.assert_not_called()

    def test_estimate_one_matches_complete_label(self):
        result = self.client_module.PoseResult("wp_2_unstable", np.eye(4))
        self.client.estimate = Mock(return_value=[result])
        args = (" wp_2_unstable ", np.zeros((2, 2, 3), dtype=np.uint8),
                np.ones((2, 2), dtype=np.uint16), np.ones((2, 2), dtype=np.uint8))
        self.assertIs(self.client.estimate_one(*args), result)
        self.assertEqual(self.client.estimate.call_args.args[2][0].label, "wp_2_unstable")
        self.client.estimate.return_value = [self.client_module.PoseResult("wp_2", np.eye(4))]
        with self.assertRaises(self.client_module.FdposeClientError):
            self.client.estimate_one(*args)

    def test_local_entry_delegates_to_public_client(self):
        cli = load_subject("local_client_subject", "tools/clients/test_fdpose_client.py",
                           {"fdpose.ros.client": self.client_module})
        args = cli.build_parser().parse_args([])
        self.assertEqual(args.load_mesh_service, "/fdpose_v2/load_mesh")
        self.assertEqual(args.pose_service, "/fdpose_v2/pose_estimation")
        images = (np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2), dtype=np.uint16),
                  np.ones((2, 2), dtype=np.uint8))
        cli.load_local_images = Mock(return_value=images)
        result = self.client_module.PoseResult("wp_2", np.eye(4))
        fake_client = Mock()
        fake_client.estimate_one.return_value = result
        with patch.object(cli, "FdposeClient", return_value=fake_client) as factory, \
             patch.object(cli, "Rotation") as rotation, contextlib.redirect_stdout(io.StringIO()):
            rotation.from_matrix.return_value.as_quat.return_value = [0, 0, 0, 1]
            self.assertEqual(cli.run_local_test(args), 0)
        factory.assert_called_once_with(mesh_service=args.load_mesh_service,
                                        pose_service=args.pose_service, timeout=args.timeout)
        fake_client.load_mesh.assert_called_once_with(args.label)
        call = fake_client.estimate_one.call_args.kwargs
        self.assertIs(call["color_bgr"], images[0])
        self.assertIs(call["depth"], images[1])
        self.assertIs(call["mask"], images[2])

    def test_local_test_service_names_can_be_overridden_for_v1(self):
        cli = load_subject("local_test_override", "tools/clients/test_fdpose_client.py",
                           {"fdpose.ros.client": self.client_module})
        args = cli.build_parser().parse_args([
            "--load-mesh-service", "/fdpose_load_mesh",
            "--pose-service", "/pose_estimation_service",
        ])
        self.assertEqual(args.load_mesh_service, "/fdpose_load_mesh")
        self.assertEqual(args.pose_service, "/pose_estimation_service")

    def test_local_test_main_initializes_node_and_calls_test_runner(self):
        cli = load_subject("local_test_main", "tools/clients/test_fdpose_client.py",
                           {"fdpose.ros.client": self.client_module})
        with patch.object(sys, "argv", ["test_fdpose_client.py"]), \
             patch.object(cli, "run_local_test", return_value=0) as runner:
            self.assertEqual(cli.main(), 0)
        cli.rospy.init_node.assert_called_once_with("fdpose_local_test_client", anonymous=True)
        args = runner.call_args.args[0]
        self.assertEqual(args.load_mesh_service, self.client_module.DEFAULT_MESH_SERVICE)
        self.assertEqual(args.pose_service, self.client_module.DEFAULT_POSE_SERVICE)


if __name__ == "__main__":
    unittest.main()
