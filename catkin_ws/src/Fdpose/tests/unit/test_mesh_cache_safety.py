"""串联真实 Client 与 V2 Mesh 回调，模拟 ROS 消息、几何解析和 GPU。"""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from test_mesh_interfaces import load_subject


class MeshCacheSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "original.ply"
        self.source.write_bytes(b"old mesh; geometry parsing is mocked")
        self.new_source = self.root / "replacement.ply"
        self.new_source.write_bytes(b"new mesh; geometry parsing is mocked")
        self.registry_path = self.root / "registry.yaml"
        self.registry_path.write_text("meshes: {}\n", encoding="utf-8")
        self.registry = {}

        self.server = load_subject("cache_safety_server", "src/fdpose/ros/server_v2.py")
        self.server.MESHES_DIR = self.root
        self.server.FdposeLoadMeshResponse = lambda **kw: SimpleNamespace(**kw)
        self.server.trimesh.Scene = type("Scene", (), {})
        self.server.trimesh.load.side_effect = lambda *args, **kw: SimpleNamespace(
            vertices=np.zeros((3, 3)), vertex_normals=np.ones((3, 3)), bounds=np.zeros((2, 3)))
        self.server.FoundationPose = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
        self.node = self.server.PoseEstimatorNodeV2.__new__(self.server.PoseEstimatorNodeV2)
        self.node._request_lock = self.server.Lock()
        self.node.objects = {}
        self.node.mesh_config = self.registry
        self.node._read_mesh_config = Mock(side_effect=lambda: dict(self.registry))
        self.node.glctx = None
        self.node.debug_dir = str(self.root)
        self.node._get_shared_predictors = lambda: ("scorer", "refiner")

        self.module = load_subject("cache_safety_client", "src/fdpose/ros/client.py")
        self.module.MESH_REGISTRY_PATH = self.registry_path
        self.module.FdposeLoadMeshRequest = lambda: SimpleNamespace(mesh_paths=[])
        self.module.rospy.ServiceException = type("ServiceError", (Exception,), {})
        self.client = self.make_client()

    def make_client(self):
        client = self.module.FdposeClient()
        client._ensure_ready = Mock()
        client._read_mesh_registry = Mock(side_effect=lambda: {"meshes": dict(self.registry)})
        client._load_mesh_proxy = Mock(wraps=self.node.load_meshes_from_yaml)
        return client

    def load_two(self):
        self.client.load_mesh("part", self.source)
        self.client.load_mesh("other", self.source)
        self.client._load_mesh_proxy.reset_mock()
        self.client._ensure_ready.reset_mock()

    def test_invalid_client_clear_rejected_before_connection_or_mutation(self):
        self.load_two()
        before = dict(self.node.objects)
        for labels in ("", " ", [""], [" "], ["part", ""], ["", "part"], [None], [1]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                self.client.clear_meshes(labels)
            self.assertEqual(self.node.objects, before)
        self.client._ensure_ready.assert_not_called()
        self.client._load_mesh_proxy.assert_not_called()

    def test_only_explicit_empty_clear_unloads_all(self):
        for labels in (None, []):
            with self.subTest(labels=labels):
                self.load_two()
                self.client.clear_meshes(labels)
                self.assertEqual(self.node.objects, {})
                self.assertTrue(self.source.is_file())

    def test_selected_clear_trims_and_deduplicates_without_touching_other(self):
        self.load_two()
        other = self.node.objects["other"]
        self.client.clear_meshes([" part ", "part"])
        self.assertEqual(self.node.objects, {"other": other})
        self.assertEqual(self.client._load_mesh_proxy.call_args.args[0].goods, ["part"])

    def test_direct_server_invalid_labels_reject_entire_load_or_unload(self):
        self.load_two()
        before = dict(self.node.objects)
        for cancel in (True, False):
            for goods in ([""], [" "], ["part", ""], ["", "part"]):
                with self.subTest(cancel=cancel, goods=goods):
                    result = self.node.load_meshes_from_yaml(SimpleNamespace(
                        cancel=cancel, goods=goods, mesh_paths=[]))
                    self.assertFalse(result.success)
                    self.assertEqual(self.node.objects, before)

    def test_loaded_source_is_recorded_and_same_path_reuses_estimator(self):
        self.client.load_mesh("part", self.source)
        record = self.node.objects["part"]
        self.assertEqual(record["mesh_path"], str(self.source))
        self.client.load_mesh("part", self.source)
        self.assertIs(self.node.objects["part"], record)
        self.server.FoundationPose.assert_called_once()

    def test_relative_path_and_symlink_resolve_to_same_cached_source(self):
        self.client.load_mesh("part", self.source)
        alias = self.root / "alias.ply"
        alias.symlink_to(self.source)
        for path in (str(alias), os.path.relpath(self.source)):
            with self.subTest(path=path):
                result = self.node.load_meshes_from_yaml(SimpleNamespace(
                    cancel=False, goods=["part"], mesh_paths=[path]))
                self.assertTrue(result.success)
        self.server.FoundationPose.assert_called_once()

    def test_changed_external_source_fails_without_discarding_old_cache(self):
        self.load_two()
        before = dict(self.node.objects)
        with self.assertRaisesRegex(self.module.FdposeClientError, "reload=True"):
            self.client.load_mesh("part", self.new_source)
        self.assertEqual(self.node.objects, before)
        warning = [call for call in self.server.rospy.logwarn.call_args_list
                   if "Mesh source conflict" in str(call.args[0])][-1]
        self.assertEqual(warning.args[1:], ("part", str(self.source), str(self.new_source)))
        self.assertEqual(self.server.FoundationPose.call_count, 2)

    def test_different_client_cannot_silently_reuse_another_source(self):
        self.client.load_mesh("part", self.source)
        other_client = self.make_client()
        with self.assertRaises(self.module.FdposeClientError):
            other_client.load_mesh("part", self.new_source)
        self.assertEqual(self.node.objects["part"]["mesh_path"], str(self.source))
        self.server.FoundationPose.assert_called_once()

    def test_changed_registry_source_is_checked_even_if_label_is_loaded(self):
        self.registry["part"] = {"path": self.source.name}
        self.client.load_mesh("part")
        old = self.node.objects["part"]
        self.registry["part"] = {"path": self.new_source.name}
        with self.assertRaises(self.module.FdposeClientError):
            self.client.load_mesh("part")
        self.assertIs(self.node.objects["part"], old)
        self.server.FoundationPose.assert_called_once()

    def test_missing_registry_label_does_not_falsely_confirm_cached_source(self):
        self.registry["part"] = {"path": self.source.name}
        self.client.load_mesh("part")
        old = self.node.objects["part"]
        self.registry.clear()
        with self.assertRaises(self.module.FdposeClientError):
            self.client.load_mesh("part")
        self.assertIs(self.node.objects["part"], old)

    def test_explicit_reload_changes_only_requested_cache(self):
        self.load_two()
        other = self.node.objects["other"]
        self.client.load_mesh("part", self.new_source, reload=True)
        self.assertEqual(self.node.objects["part"]["mesh_path"], str(self.new_source))
        self.assertIs(self.node.objects["other"], other)
        self.assertEqual([call.args[0].cancel for call in self.client._load_mesh_proxy.call_args_list],
                         [True, False])

    def test_same_path_content_updates_still_require_explicit_reload(self):
        self.client.load_mesh("part", self.source)
        old = self.node.objects["part"]
        self.source.write_bytes(b"updated content at the same path")
        self.client.load_mesh("part", self.source)
        self.assertIs(self.node.objects["part"], old)
        self.client.load_mesh("part", self.source, reload=True)
        self.assertIsNot(self.node.objects["part"], old)

    def test_unloaded_external_mesh_needs_path_again(self):
        self.load_two()
        self.client.clear_meshes(["part"])
        with self.assertRaises(self.module.FdposeClientError):
            self.client.load_mesh("part")
        self.assertNotIn("part", self.node.objects)
        self.client.load_mesh("part", self.source)
        self.assertEqual(self.node.objects["part"]["mesh_path"], str(self.source))
        self.assertIn("other", self.node.objects)

    def test_explicit_reload_restores_registered_asset_without_local_state(self):
        self.registry["part"] = {"path": self.source.name}
        self.client.load_mesh("part", self.new_source)
        with self.assertRaises(self.module.FdposeClientError):
            self.client.load_mesh("part")
        self.assertEqual(self.node.objects["part"]["mesh_path"], str(self.new_source))
        self.client.load_mesh("part", reload=True)
        self.assertEqual(self.node.objects["part"]["mesh_path"], str(self.source))
        self.assertEqual(self.registry, {"part": {"path": self.source.name}})

    def test_external_load_does_not_edit_assets_or_client_registry(self):
        files = {path: path.read_bytes() for path in self.root.iterdir()}
        with unittest.mock.patch.object(self.client, "register_mesh") as register:
            self.client.load_mesh("part", self.source)
            register.assert_not_called()
        self.assertEqual({path: path.read_bytes() for path in self.root.iterdir()}, files)
        self.client._read_mesh_registry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
