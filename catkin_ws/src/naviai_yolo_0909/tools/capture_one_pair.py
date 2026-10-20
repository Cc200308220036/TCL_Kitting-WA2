#!/usr/bin/env python3
"""Entry point for naviai_yolo.ros.capture_one_pair."""

import sys
from pathlib import Path

source_path = Path(__file__).resolve().parents[1] / 'src'
if (source_path / 'naviai_yolo').is_dir():
    sys.path.insert(0, str(source_path))

from naviai_yolo.ros.capture_one_pair import PairCapture, main


if __name__ == '__main__':
    raise SystemExit(main())
