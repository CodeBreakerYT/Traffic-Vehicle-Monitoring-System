"""
Resolves bundled-asset paths correctly both in normal `python main.py`
execution and inside a PyInstaller --onefile executable, where data files
are extracted to a temporary directory (sys._MEIPASS) rather than the
current working directory.
"""

import sys
from pathlib import Path


def resource_path(*parts) -> str:
    base = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else Path(__file__).resolve().parent
    return str(base.joinpath(*parts))
