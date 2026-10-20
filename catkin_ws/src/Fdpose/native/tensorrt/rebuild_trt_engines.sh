#!/usr/bin/env bash
# 运行：bash native/tensorrt/rebuild_trt_engines.sh（历史兼容入口）。
# 实际执行：tools/build/prepare_tensorrt.py --force，强制重新准备扩展和 Engine。
# 配置：PYTHON 环境变量及准备工具参数；不读取 ROS/服务部署 YAML。
# Compatibility wrapper. The Python tool now builds the native extension,
# generates all three engines, validates them, and writes the environment
# manifest in one operation.
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

exec "$PYTHON" "$PROJECT_ROOT/tools/build/prepare_tensorrt.py" --force "$@"
