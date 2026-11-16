#!/usr/bin/env python3
"""Entry script for Qwen Station (also the PyInstaller entry point)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_station.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
