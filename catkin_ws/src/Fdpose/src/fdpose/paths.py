"""Central paths for the FDPose source-tree layout."""

from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

CONFIG_DIR = PROJECT_ROOT / "config"
CAMERA_CONFIG_DIR = CONFIG_DIR / "camera"

ASSETS_DIR = PROJECT_ROOT / "assets"
WARMUP_DIR = ASSETS_DIR / "warmup"
MESHES_DIR = ASSETS_DIR / "meshes"
MESH_REGISTRY_PATH = MESHES_DIR / "registry.yaml"
# Compatibility alias for external code that still imports the old constant.
MODEL_CONFIG_PATH = MESH_REGISTRY_PATH
WEIGHTS_DIR = ASSETS_DIR / "checkpoints"
TRT_DIR = ASSETS_DIR / "engines"
ONNX_DIR = ASSETS_DIR / "onnx"

NATIVE_DIR = PROJECT_ROOT / "native"
MYCPP_BUILD_DIR = NATIVE_DIR / "mycpp" / "build"
CUSTOM_OPS_BUILD_DIR = NATIVE_DIR / "custom_ops" / "build"
TRT_BUILD_DIR = NATIVE_DIR / "tensorrt" / "build"

RUNTIME_DIR = PROJECT_ROOT / "runtime"
DEBUG_DIR = RUNTIME_DIR / "debug"
LOG_DIR = RUNTIME_DIR / "logs"
PID_DIR = RUNTIME_DIR / "pids"

DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"


def ensure_native_modules_on_path() -> None:
    """Expose in-place native extension builds to Python."""
    for build_dir in (MYCPP_BUILD_DIR, CUSTOM_OPS_BUILD_DIR, TRT_BUILD_DIR):
        path = str(build_dir)
        if path not in sys.path:
            sys.path.insert(0, path)
