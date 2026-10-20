#!/usr/bin/env python3
"""Entry point for naviai_yolo.ros.seg_action_server."""

import sys
from pathlib import Path

source_path = Path(__file__).resolve().parents[2] / 'src'
if (source_path / 'naviai_yolo').is_dir():
    sys.path.insert(0, str(source_path))

from naviai_yolo.ros.seg_action_server import SegActionServer, main


if __name__ == '__main__':
    main()
