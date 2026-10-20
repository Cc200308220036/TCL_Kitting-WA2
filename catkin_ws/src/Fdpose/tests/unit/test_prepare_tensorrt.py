import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "build"
    / "prepare_tensorrt.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_tensorrt", SCRIPT_PATH)
prepare_tensorrt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_tensorrt)


class PrepareTensorRTTest(unittest.TestCase):
    def test_install_failure_restores_old_files_and_removes_new_files(self):
        for old_files_exist in (True, False):
            with self.subTest(old_files_exist=old_files_exist), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                engine_dir = root / "engines"
                build_dir = root / "build"
                staging = root / "staging"
                for path in (engine_dir, build_dir, staging):
                    path.mkdir()
                extension = staging / "pyTRTEng.so"
                extension.write_bytes(b"new extension")
                engines = {}
                targets = [build_dir / extension.name, engine_dir / "manifest.json"]
                for key, _, name, _, _ in prepare_tensorrt.ENGINE_SPECS:
                    engines[key] = staging / name
                    engines[key].write_bytes(b"new engine")
                    targets.append(engine_dir / name)
                for target in targets:
                    if old_files_exist:
                        target.write_bytes(b"old content")
                original_install = prepare_tensorrt._install_file_atomic
                calls = []

                def fail_on_second_install(source, destination):
                    calls.append(destination)
                    if len(calls) == 2:
                        raise OSError("simulated disk failure")
                    original_install(source, destination)

                with patch.multiple(prepare_tensorrt, PROJECT_ROOT=root, ENGINE_DIR=engine_dir,
                                    NATIVE_BUILD_DIR=build_dir, MANIFEST_PATH=engine_dir / "manifest.json"), \
                     patch.object(prepare_tensorrt, "_install_file_atomic", side_effect=fail_on_second_install):
                    with self.assertRaisesRegex(OSError, "simulated"):
                        prepare_tensorrt._install_artifacts(extension, engines, {}, {})
                for target in targets:
                    if old_files_exist:
                        self.assertEqual(target.read_bytes(), b"old content")
                    else:
                        self.assertFalse(target.exists())
                self.assertEqual(list(root.glob(".fdpose-trt-backup-*")), [])

    def test_build_lock_rejects_second_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(prepare_tensorrt, "PROJECT_ROOT", Path(temp_dir)):
            with prepare_tensorrt._build_lock():
                with self.assertRaisesRegex(prepare_tensorrt.PrepareError, "another"):
                    with prepare_tensorrt._build_lock():
                        self.fail("second writer acquired lock")
            with prepare_tensorrt._build_lock():
                pass

    def test_failed_validation_never_installs_staging(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(prepare_tensorrt, "PROJECT_ROOT", Path(temp_dir)), \
             patch.object(prepare_tensorrt, "_build_extension", return_value=Path("extension")), \
             patch.object(prepare_tensorrt, "_build_engines", return_value={}), \
             patch.object(prepare_tensorrt, "_validate_artifacts", side_effect=prepare_tensorrt.PrepareError("bad output")), \
             patch.object(prepare_tensorrt, "_install_artifacts") as install:
            with self.assertRaisesRegex(prepare_tensorrt.PrepareError, "bad output"):
                prepare_tensorrt.prepare(Path("trtexec"), {}, {}, 1)
            install.assert_not_called()
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_environment_ignores_log_timestamps_and_detects_library_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            binary = Path(temp_dir) / "trtexec"
            library = Path(temp_dir) / "libnvinfer.so"
            binary.write_bytes(b"binary")
            library.write_bytes(b"library version one")
            help_text = ["[01:02:03] TensorRT v8601 --help"]

            def capture(command):
                if command[0] == "ldd":
                    return f"libnvinfer.so => {library} (0x1234)"
                if command[-1] == "--help":
                    return help_text[0]
                return "CUDA compiler version"

            with patch.object(prepare_tensorrt, "_probe_torch", return_value={}), \
                 patch.object(prepare_tensorrt, "_read_l4t_release", return_value="test"), \
                 patch.object(prepare_tensorrt, "_run_capture", side_effect=capture):
                first = prepare_tensorrt.detect_environment(binary)
                help_text[0] = "[05:06:07] TensorRT v8601 --help"
                self.assertEqual(first, prepare_tensorrt.detect_environment(binary))
                library.write_bytes(b"library version two")
                self.assertNotEqual(first, prepare_tensorrt.detect_environment(binary))

    def test_manifest_matches_outputs_and_detects_corruption(self):
        old_root = prepare_tensorrt.PROJECT_ROOT
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                prepare_tensorrt.PROJECT_ROOT = root
                extension = root / "native" / "tensorrt" / "build" / "pyTRTEng.so"
                extension.parent.mkdir(parents=True)
                extension.write_bytes(b"extension")

                engines = {}
                for name in ("score", "refine_b168", "refine_b1"):
                    path = root / "assets" / "engines" / f"{name}.eng"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(name.encode("utf-8"))
                    engines[name] = path

                environment = {"tensorrt": "test"}
                inputs = {"onnx_sha256": {"test": "value"}}
                outputs = {
                    "extension": prepare_tensorrt._output_record(extension),
                }
                outputs.update(
                    {
                        name: prepare_tensorrt._output_record(path)
                        for name, path in engines.items()
                    }
                )
                manifest = {
                    "schema_version": prepare_tensorrt.MANIFEST_SCHEMA_VERSION,
                    "environment": environment,
                    "inputs": inputs,
                    "outputs": outputs,
                }

                matches, _ = prepare_tensorrt.manifest_matches(
                    manifest, environment, inputs
                )
                self.assertTrue(matches)

                engines["score"].write_bytes(b"changed")
                matches, reason = prepare_tensorrt.manifest_matches(
                    manifest, environment, inputs
                )
                self.assertFalse(matches)
                self.assertIn("checksum", reason)
        finally:
            prepare_tensorrt.PROJECT_ROOT = old_root

    def test_environment_change_requires_rebuild(self):
        manifest = {
            "schema_version": prepare_tensorrt.MANIFEST_SCHEMA_VERSION,
            "environment": {"tensorrt": "8.5"},
            "inputs": {},
            "outputs": {},
        }

        matches, reason = prepare_tensorrt.manifest_matches(
            manifest,
            {"tensorrt": "8.6"},
            {},
        )

        self.assertFalse(matches)
        self.assertIn("environment changed", reason)


if __name__ == "__main__":
    unittest.main()
