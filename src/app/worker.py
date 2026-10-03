"""Background worker thread: processes the file list in order and signals the GUI."""
from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from core import ffmpeg as ff
from core.jobs import Cancelled, Runner, Settings


class Worker(QThread):
    log = Signal(str)
    file_progress = Signal(float, str)      # 0..1, status text
    overall_progress = Signal(int, int)     # completed, total
    file_done = Signal(str, str)            # source, output
    file_failed = Signal(str, str)          # source, error
    all_done = Signal(int, int, bool)       # succeeded, failed, cancelled

    def __init__(self, files: list[Path], settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.files = list(files)
        self.settings = settings
        self.runner = Runner()

    def cancel(self) -> None:
        self.runner.cancel_event.set()

    def run(self) -> None:
        ok = fail = 0
        cancelled = False
        total = len(self.files)

        for i, src in enumerate(self.files):
            if self.runner.cancel_event.is_set():
                cancelled = True
                break
            self.overall_progress.emit(i, total)
            try:
                is_video = src.suffix.lower() in ff.VIDEO_EXT
                fn = self.runner.run_video if is_video else self.runner.run_image
                dst = fn(src, self.settings, self.log.emit,
                         lambda f, s: self.file_progress.emit(f, s))
                ok += 1
                self.file_done.emit(str(src), str(dst))
            except Cancelled:
                cancelled = True
                self.log.emit("Processing cancelled.")
                break
            except Exception as exc:                      # noqa: BLE001
                fail += 1
                self.log.emit(f"ERROR — {src.name}: {exc}")
                self.log.emit(traceback.format_exc(limit=3).strip())
                self.file_failed.emit(str(src), str(exc))

        self.overall_progress.emit(total if not cancelled else ok + fail, total)
        self.runner.release()
        self.all_done.emit(ok, fail, cancelled)
