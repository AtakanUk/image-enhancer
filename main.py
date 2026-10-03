"""Image Enhancer — entry point.

Run with:  python main.py     (or run.bat on Windows)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app.main_window import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
