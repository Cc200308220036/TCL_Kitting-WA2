"""独立复制注册及直接路径加载回归：真实文件/YAML，模拟 ROS/GPU。"""

from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import yaml
except ImportError:
    yaml = None

from test_mesh_interfaces import load_subject


@unittest.skipIf(yaml is None, "Mesh registration tests require PyYAML")
class ClientRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_subject("registration_subject", "src/fdpose/ros/client.py", {"yaml": yaml})
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.registry = self.assets / "registry.yaml"
        self.registry.write_text("# empty registry\nmeshes: {}\n", encoding="utf-8")
        self.module.MESHES_DIR = self.assets
        self.module.MESH_REGISTRY_PATH = self.registry
        self.source = self.root / "model.ply"
        self.source.write_bytes(b"mesh fixture; geometry is not parsed by Client")
        self.new_source = self.root / "new.ply"
        self.new_source.write_bytes(b"another mesh")
        self.module.FdposeLoadMeshRequest = lambda: SimpleNamespace(mesh_paths=[])
        self.client = self.module.FdposeClient()
        self.client._ensure_ready = Mock()
        self.client._load_mesh_proxy = Mock(return_value=SimpleNamespace(success=True))

    def read_registry(self):
        return yaml.safe_load(self.registry.read_text(encoding="utf-8"))["meshes"]

    def test_register_copies_file_writes_flow_yaml_and_does_not_contact_ros(self):
        self.module.rospy.core.is_initialized.return_value = False
        target = Path(self.client.register_mesh(" part ", self.source))
        self.assertEqual(target, self.assets / "objects" / "part" / "model.ply")
        self.assertEqual(target.read_bytes(), self.source.read_bytes())
        self.assertEqual(self.read_registry(), {"part": {"path": "objects/part/model.ply"}})
        self.client._ensure_ready.assert_not_called()
        self.client._load_mesh_proxy.assert_not_called()

    def test_duplicate_label_always_errors_even_for_same_source_or_saved_path(self):
        target = Path(self.client.register_mesh("part", self.source))
        original = self.registry.read_bytes()
        for source in (self.source, self.new_source, target):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "already registered"):
                self.client.register_mesh("part", source)
            self.assertEqual(self.registry.read_bytes(), original)
            self.assertEqual(list(target.parent.iterdir()), [target])
            self.assertEqual(target.read_bytes(), self.source.read_bytes())

    def test_existing_target_directory_is_never_overwritten_or_removed(self):
        directory = self.assets / "objects" / "part"
        directory.mkdir(parents=True)
        existing = directory / self.source.name
        existing.write_bytes(b"user asset without registry entry")
        original = self.registry.read_bytes()
        with self.assertRaises(FileExistsError):
            self.client.register_mesh("part", self.source)
        self.assertEqual(existing.read_bytes(), b"user asset without registry entry")
        self.assertEqual(self.registry.read_bytes(), original)

    def test_register_preserves_other_labels_and_source_file(self):
        self.registry.write_text(yaml.safe_dump({
            "meshes": {"old": {"path": str(self.source), "note": "keep"}},
        }), encoding="utf-8")
        original_source = self.source.read_bytes()
        self.client.register_mesh("part", self.source)
        self.assertEqual(self.read_registry()["old"], {"path": str(self.source), "note": "keep"})
        self.assertEqual(self.source.read_bytes(), original_source)

    def test_invalid_label_or_source_does_not_write_assets(self):
        for label in ("", " ", "../part"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.client.register_mesh(label, self.source)
        for source in (None, "", self.root / "missing.ply"):
            with self.subTest(source=source), self.assertRaises((ValueError, FileNotFoundError)):
                self.client.register_mesh("part", source)
        self.assertEqual(list(self.assets.iterdir()), [self.registry])
        self.assertEqual(self.read_registry(), {})

    def test_missing_registry_errors_without_copying(self):
        self.registry.unlink()
        with self.assertRaises(FileNotFoundError):
            self.client.register_mesh("part", self.source)
        self.assertEqual(list(self.assets.iterdir()), [])

    def test_registry_write_failure_removes_only_new_uncommitted_copy(self):
        old = Path(self.client.register_mesh("old", self.source))
        original = self.registry.read_bytes()
        with patch.object(self.module.os, "replace", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                self.client.register_mesh("part", self.new_source)
        self.assertEqual(self.registry.read_bytes(), original)
        self.assertFalse((self.assets / "objects" / "part").exists())
        self.assertEqual(old.read_bytes(), self.source.read_bytes())
        self.assertEqual(list(self.assets.glob(".registry.yaml.*")),
                         [self.assets / ".registry.yaml.lock"])

    def test_partial_copy_failure_leaves_no_registration_or_partial_file(self):
        original = self.registry.read_bytes()

        def fail_copy(source, target):
            Path(target).write_bytes(b"partial")
            raise OSError("copy error")

        with patch.object(self.module.shutil, "copy2", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "copy error"):
                self.client.register_mesh("part", self.source)
        self.assertFalse((self.assets / "objects" / "part").exists())
        self.assertEqual(self.registry.read_bytes(), original)
        self.assertTrue(self.source.is_file())

    def test_concurrent_different_labels_do_not_lose_entries(self):
        def register(index):
            return self.module.FdposeClient().register_mesh(f"part_{index}", self.source)

        with ThreadPoolExecutor(max_workers=4) as executor:
            paths = list(executor.map(register, range(8)))
        self.assertEqual(set(self.read_registry()), {f"part_{index}" for index in range(8)})
        self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_concurrent_same_label_has_exactly_one_success(self):
        def register(index):
            try:
                return self.module.FdposeClient().register_mesh("part", self.source)
            except ValueError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(register, range(4)))
        self.assertEqual(sum(isinstance(result, str) for result in results), 1)
        self.assertEqual(sum(isinstance(result, ValueError) for result in results), 3)
        self.assertEqual(set(self.read_registry()), {"part"})
        self.assertEqual(len(list((self.assets / "objects" / "part").iterdir())), 1)

    def test_direct_load_never_registers_or_reads_client_registry(self):
        self.registry.write_text("invalid: [", encoding="utf-8")
        original = self.registry.read_bytes()
        with patch.object(self.client, "register_mesh") as register, \
             patch.object(self.client, "_read_mesh_registry", side_effect=AssertionError("unexpected read")):
            self.client.load_mesh("part", self.source)
            register.assert_not_called()
        request = self.client._load_mesh_proxy.call_args.args[0]
        self.assertEqual(request.goods, ["part"])
        self.assertEqual(request.mesh_paths, [str(self.source)])
        self.assertEqual(self.registry.read_bytes(), original)
        self.assertEqual(list(self.assets.iterdir()), [self.registry])

    def test_register_and_load_are_explicit_separate_operations(self):
        self.client.register_mesh("part", self.source)
        self.client._load_mesh_proxy.assert_not_called()
        self.client.load_mesh("part")
        self.assertEqual(self.client._load_mesh_proxy.call_args.args[0].mesh_paths, [])

    def test_client_does_not_remember_external_path(self):
        self.client.load_mesh("part", self.source)
        self.client.clear_meshes(["part"])
        self.client.load_mesh("part")
        requests = [call.args[0] for call in self.client._load_mesh_proxy.call_args_list]
        self.assertEqual([request.cancel for request in requests], [False, True, False])
        self.assertEqual(requests[-1].mesh_paths, [])  # 交给服务端查表，不复用旧路径。
        self.assertEqual(self.read_registry(), {})

    def test_batch_mixes_explicit_path_and_registry_lookup(self):
        self.client.load_meshes(["part", "wp_2"], mesh_paths=[self.source, None])
        request = self.client._load_mesh_proxy.call_args.args[0]
        self.assertEqual(request.goods, ["part", "wp_2"])
        self.assertEqual(request.mesh_paths, [str(self.source), ""])
        self.assertEqual(self.read_registry(), {})

    def test_reload_unloads_only_requested_label_then_loads_explicit_path(self):
        self.client.load_mesh("part", self.new_source, reload=True)
        requests = [call.args[0] for call in self.client._load_mesh_proxy.call_args_list]
        self.assertEqual([request.cancel for request in requests], [True, False])
        self.assertEqual([request.goods for request in requests], [["part"], ["part"]])
        self.assertEqual(requests[-1].mesh_paths, [str(self.new_source)])

    def test_reload_validates_path_and_message_before_unloading(self):
        for source in ("", self.root / "missing.ply"):
            with self.subTest(source=source), self.assertRaises((ValueError, FileNotFoundError)):
                self.client.load_mesh("part", source, reload=True)
        self.module.FdposeLoadMeshRequest = SimpleNamespace
        with self.assertRaises(self.module.FdposeClientError):
            self.client.load_mesh("part", self.source, reload=True)
        self.client._load_mesh_proxy.assert_not_called()

    def test_failed_direct_load_does_not_change_registry(self):
        original = self.registry.read_bytes()
        self.client._load_mesh_proxy.return_value.success = False
        with self.assertRaises(self.module.FdposeClientError):
            self.client.load_mesh("part", self.source)
        self.assertEqual(self.registry.read_bytes(), original)
        self.assertEqual(list(self.assets.iterdir()), [self.registry])

    def test_removed_state_and_options_are_not_silently_accepted(self):
        self.assertFalse(hasattr(self.client, "_temporary_mesh_paths"))
        self.assertFalse(hasattr(self.client, "unregister_mesh"))
        self.assertEqual(list(inspect.signature(self.client.register_mesh).parameters), ["label", "mesh_path"])
        for method in (self.client.load_mesh, self.client.register_mesh):
            for kwargs in ({"copy_to_registry": False}, {"overwrite": True}):
                with self.subTest(method=method.__name__, kwargs=kwargs), self.assertRaises(TypeError):
                    method("part", self.source, **kwargs)
        self.client._load_mesh_proxy.assert_not_called()
        self.assertEqual(list(self.assets.iterdir()), [self.registry])


if __name__ == "__main__":
    unittest.main()
