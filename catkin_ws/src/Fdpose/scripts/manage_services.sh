#!/usr/bin/env bash
# 运行：bash scripts/manage_services.sh start|stop|status。
# 实际执行：tools/manage_services.py；这是 Python 管理器的 Shell 薄入口。
# 配置：config/fdpose_services.yaml（start --config 可覆盖）；不读取 pipeline.conf。
# 统一入口：start / stop / status。控制端只需 Python 3 + PyYAML。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SERVICE_MANAGER_PYTHON:-python3}" "$SCRIPT_DIR/../tools/manage_services.py" "$@"
