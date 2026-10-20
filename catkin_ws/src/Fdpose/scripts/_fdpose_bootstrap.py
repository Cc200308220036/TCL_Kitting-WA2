# 用途：由 scripts/ 下的 ROS 入口导入，为源码目录添加 src/ 搜索路径。
# 不直接运行，不启动服务，也不读取配置文件。
"""Make the source-tree Python package importable by compatibility scripts."""

from pathlib import Path
import sys


def add_src_to_path() -> None:
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
