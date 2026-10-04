"""Image Enhancer — main window."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from PySide6.QtCore import QSettings, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSlider, QSpinBox,
    QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from core import ffmpeg as ff
from core.jobs import (AUTO_MODEL, PRESETS, Settings, build_plan,
                       probe_dims, resolve_model)
from core.models import REGISTRY
from .worker import Worker

MEDIA_EXT = ff.IMAGE_EXT | ff.VIDEO_EXT


# ------------------------------------------------------------------- file list

class DropList(QListWidget):
    """File list that accepts drag and drop."""

    def __init__(self, on_change) -> None:
        super().__init__()
        self.on_change = on_change
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths: list[Path] = []
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                paths += [c for c in sorted(p.rglob("*"))
                          if c.suffix.lower() in MEDIA_EXT]
            elif p.suffix.lower() in MEDIA_EXT:
                paths.append(p)
        self.add_paths(paths)
        e.acceptProposedAction()

    def add_paths(self, paths: list[Path]) -> int:
        existing = {self.item(i).data(Qt.UserRole) for i in range(self.count())}
        added = 0
        for p in paths:
            key = str(p.resolve())
            if key in existing:
                continue
            kind = "VIDEO" if p.suffix.lower() in ff.VIDEO_EXT else "PHOTO"
            item = QListWidgetItem(f"[{kind}]  {p.name}")
            item.setData(Qt.UserRole, key)
            item.setToolTip(key)
            self.addItem(item)
            existing.add(key)
            added += 1
        if added:
            self.on_change()
        return added

    def paths(self) -> list[Path]:
        return [Path(self.item(i).data(Qt.UserRole)) for i in range(self.count())]

    def remove_selected(self) -> None:
        for item in self.selectedItems():
            self.takeItem(self.row(item))
        self.on_change()


# ----------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Enhancer — AI Upscaling")
        self.resize(1180, 780)
        self.worker: Worker | None = None
        self.qs = QSettings("ImageEnhancer", "ImageEnhancer")
        self.last_output: Path | None = None

        self._build_ui()
        self._load_prefs()
        self._refresh_estimate()

    # -- UI setup ---------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([680, 500])

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.addWidget(splitter, 1)
        lay.addWidget(self._build_bottom())
        self.setCentralWidget(root)
        self.statusBar().showMessage(self._device_line())

    def _build_left(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 8, 0)

        head = QLabel("Files")
        head.setObjectName("h1")
        lay.addWidget(head)

        hint = QLabel("Drag and drop photos or videos here "
                      "(folders work too).")
        hint.setWordWrap(True)
        hint.setObjectName("dim")
        lay.addWidget(hint)

        self.list = DropList(self._refresh_estimate)
        self.list.currentRowChanged.connect(lambda _: self._refresh_estimate())
        lay.addWidget(self.list, 1)

        row = QHBoxLayout()
        for text, slot in (("Add files...", self._add_files),
                           ("Add folder...", self._add_folder),
                           ("Remove selected", self.list.remove_selected),
                           ("Clear list", self._clear_list)):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        lay.addLayout(row)

        self.estimate = QLabel("—")
        self.estimate.setWordWrap(True)
        self.estimate.setObjectName("estimate")
        self.estimate.setMinimumHeight(72)
        lay.addWidget(self.estimate)
        return box

    def _build_right(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._grp_target())
        lay.addWidget(self._grp_model())
        lay.addWidget(self._grp_photo())
        lay.addWidget(self._grp_video())
        lay.addWidget(self._grp_output())
        lay.addWidget(self._grp_advanced())
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    # -- settings groups --------------------------------------------------

    def _grp_target(self) -> QGroupBox:
        g = QGroupBox("1 · Target resolution")
        lay = QVBoxLayout(g)

        self.rb_preset = QRadioButton("Preset")
        self.rb_scale = QRadioButton("Multiplier")
        self.rb_custom = QRadioButton("Custom size")
        self.rb_preset.setChecked(True)

        self.cb_preset = QComboBox()
        for label, h in PRESETS:
            self.cb_preset.addItem(label, h)
        self.cb_preset.setCurrentIndex(3)          # 4K

        self.cb_scale = QComboBox()
        for m in (1.5, 2, 3, 4, 6, 8):
            self.cb_scale.addItem(f"x{m:g}  ({m:g} times the source)", float(m))
        self.cb_scale.setCurrentIndex(3)

        self.sp_w = QSpinBox(); self.sp_w.setRange(16, 32000); self.sp_w.setValue(3840)
        self.sp_h = QSpinBox(); self.sp_h.setRange(16, 32000); self.sp_h.setValue(2160)
        self.ck_ratio = QCheckBox("Keep aspect ratio (width is automatic)")
        self.ck_ratio.setChecked(True)

        for w in (self.rb_preset, self.cb_preset, self.rb_scale, self.cb_scale,
                  self.rb_custom):
            lay.addWidget(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Width")); row.addWidget(self.sp_w)
        row.addWidget(QLabel("x")); row.addWidget(QLabel("Height")); row.addWidget(self.sp_h)
        lay.addLayout(row)
        lay.addWidget(self.ck_ratio)

        note = QLabel("Presets set the height; the width follows the source "
                      "aspect ratio. If the target is not an exact model "
                      "multiple, the AI upscales one step further and scales "
                      "down to the target, which looks cleaner than upscaling "
                      "to it directly.")
        note.setWordWrap(True); note.setObjectName("dim")
        lay.addWidget(note)

        for w in (self.rb_preset, self.rb_scale, self.rb_custom):
            w.toggled.connect(self._sync_target_enabled)
        self.cb_preset.currentIndexChanged.connect(self._refresh_estimate)
        self.cb_scale.currentIndexChanged.connect(self._refresh_estimate)
        self.sp_w.valueChanged.connect(self._refresh_estimate)
        self.sp_h.valueChanged.connect(self._refresh_estimate)
        self.ck_ratio.toggled.connect(self._sync_target_enabled)
        return g

    def _grp_model(self) -> QGroupBox:
        g = QGroupBox("2 · AI model")
        lay = QVBoxLayout(g)

        self.cb_model = QComboBox()
        self.cb_model.addItem("Auto — choose by file type (recommended)",
                              AUTO_MODEL)
        for key, spec in REGISTRY.items():
            self.cb_model.addItem(spec.label, key)
        lay.addWidget(self.cb_model)

        self.lbl_model_hint = QLabel("")
        self.lbl_model_hint.setWordWrap(True)
        self.lbl_model_hint.setObjectName("dim")
        lay.addWidget(self.lbl_model_hint)

        self.row_denoise = QWidget()
        drow = QHBoxLayout(self.row_denoise)
        drow.setContentsMargins(0, 6, 0, 0)
        drow.addWidget(QLabel("Denoise"))
        self.sl_denoise = QSlider(Qt.Horizontal)
        self.sl_denoise.setRange(0, 100); self.sl_denoise.setValue(50)
        self.lbl_denoise = QLabel("0.50")
        drow.addWidget(self.sl_denoise, 1); drow.addWidget(self.lbl_denoise)
        lay.addWidget(self.row_denoise)

        dn_note = QLabel("0 = keeps fine texture and film grain · "
                         "1 = removes noise aggressively")
        dn_note.setWordWrap(True); dn_note.setObjectName("dim")
        lay.addWidget(dn_note)
        self.dn_note = dn_note

        self.cb_model.currentIndexChanged.connect(self._sync_model)
        self.sl_denoise.valueChanged.connect(
            lambda v: (self.lbl_denoise.setText(f"{v/100:.2f}")))
        return g

    def _grp_photo(self) -> QGroupBox:
        g = QGroupBox("3 · Photo enhancement")
        lay = QVBoxLayout(g)

        self.ck_face = QCheckBox("Face restoration (GFPGAN v1.4)")
        self.ck_face.setChecked(True)
        lay.addWidget(self.ck_face)

        self.row_face = QWidget()
        frow = QHBoxLayout(self.row_face)
        frow.setContentsMargins(20, 0, 0, 0)
        frow.addWidget(QLabel("Strength"))
        self.sl_face = QSlider(Qt.Horizontal)
        self.sl_face.setRange(0, 100)
        self.sl_face.setValue(80)
        self.lbl_face = QLabel("0.80")
        frow.addWidget(self.sl_face, 1)
        frow.addWidget(self.lbl_face)
        lay.addWidget(self.row_face)

        face_note = QLabel(
            "Faces are detected on the source, rebuilt by a separate network "
            "and blended softly into the upscaled image. Lower strength stays "
            "closer to the person's real features; higher strength gives a "
            "crisper but more 'generated' face.")
        face_note.setWordWrap(True)
        face_note.setObjectName("dim")
        lay.addWidget(face_note)

        self.ck_ensemble = QCheckBox("x8 self-ensemble — cleaner, noticeably slower")
        lay.addWidget(self.ck_ensemble)

        ens_note = QLabel(
            "Processes the image in 8 orientations separately and averages "
            "the results. Measured gain: +0.9 dB PSNR, at ~2.5x the time with "
            "the General model and ~7.5x with Photo x4.")
        ens_note.setWordWrap(True)
        ens_note.setObjectName("dim")
        lay.addWidget(ens_note)

        scope = QLabel("Both settings in this section apply to photos only.")
        scope.setWordWrap(True)
        scope.setObjectName("dim")
        lay.addWidget(scope)

        self.ck_face.toggled.connect(self.row_face.setVisible)
        self.sl_face.valueChanged.connect(
            lambda v: self.lbl_face.setText(f"{v/100:.2f}"))
        return g

    def _grp_video(self) -> QGroupBox:
        g = QGroupBox("4 · Video settings")
        form = QFormLayout(g)

        self.cb_enc = QComboBox()
        for key, label in ff.ENCODERS.items():
            self.cb_enc.addItem(label, key)
        form.addRow("Codec", self.cb_enc)

        qrow = QWidget(); ql = QHBoxLayout(qrow); ql.setContentsMargins(0, 0, 0, 0)
        self.sl_quality = QSlider(Qt.Horizontal)
        self.sl_quality.setRange(1, 40); self.sl_quality.setValue(20)
        self.sl_quality.setInvertedAppearance(True)     # right = higher quality
        self.lbl_quality = QLabel("20")
        ql.addWidget(QLabel("smaller file")); ql.addWidget(self.sl_quality, 1)
        ql.addWidget(QLabel("higher quality")); ql.addWidget(self.lbl_quality)
        self.sl_quality.valueChanged.connect(lambda v: self.lbl_quality.setText(str(v)))
        form.addRow("Quality (CQ/CRF)", qrow)

        self.cb_audio = QComboBox()
        self.cb_audio.addItem("Copy audio as is", "copy")
        self.cb_audio.addItem("Re-encode as AAC 256k", "aac")
        self.cb_audio.addItem("Remove audio", "none")
        form.addRow("Audio", self.cb_audio)

        self.cb_container = QComboBox()
        for ext in (".mp4", ".mkv", ".mov"):
            self.cb_container.addItem(ext, ext)
        form.addRow("Container", self.cb_container)

        self.sp_preview = QDoubleSpinBox()
        self.sp_preview.setRange(0, 3600); self.sp_preview.setValue(0)
        self.sp_preview.setSuffix(" s"); self.sp_preview.setSpecialValueText("Whole video")
        form.addRow("Only the first", self.sp_preview)

        tip = QLabel("Use this to check your settings on a 10-second clip "
                     "before starting a long video.")
        tip.setWordWrap(True); tip.setObjectName("dim")
        form.addRow("", tip)
        return g

    def _grp_output(self) -> QGroupBox:
        g = QGroupBox("5 · Output")
        lay = QVBoxLayout(g)

        row = QHBoxLayout()
        self.ed_out = QLabel()
        self.ed_out.setObjectName("path")
        self.ed_out.setWordWrap(True)
        btn = QPushButton("Change...")
        btn.clicked.connect(self._pick_out_dir)
        row.addWidget(self.ed_out, 1); row.addWidget(btn)
        lay.addLayout(row)

        form = QFormLayout()
        self.cb_imgfmt = QComboBox()
        self.cb_imgfmt.addItem("PNG (lossless, recommended)", "png")
        self.cb_imgfmt.addItem("JPEG (smaller file)", "jpg")
        self.cb_imgfmt.addItem("Same as source", "same")
        form.addRow("Photo format", self.cb_imgfmt)

        self.sp_jpegq = QSpinBox(); self.sp_jpegq.setRange(60, 100); self.sp_jpegq.setValue(95)
        form.addRow("JPEG quality", self.sp_jpegq)
        lay.addLayout(form)

        self.ck_overwrite = QCheckBox("Overwrite files with the same name "
                                      "(otherwise a number is appended)")
        lay.addWidget(self.ck_overwrite)

        self.btn_open = QPushButton("Open output folder")
        self.btn_open.clicked.connect(self._open_out_dir)
        lay.addWidget(self.btn_open)
        return g

    def _grp_advanced(self) -> QGroupBox:
        g = QGroupBox("6 · Advanced")
        form = QFormLayout(g)

        self.cb_tile = QComboBox()
        self.cb_tile.addItem("Auto (based on VRAM)", -1)
        self.cb_tile.addItem("No tiling — fastest, uses the most VRAM", 0)
        for t in (1024, 768, 512, 384, 256, 192):
            self.cb_tile.addItem(f"{t} px tiles", t)
        form.addRow("Tile size", self.cb_tile)

        self.ck_half = QCheckBox("FP16 (half precision) — 2x faster on GPU")
        self.ck_half.setChecked(True)
        form.addRow("", self.ck_half)

        self.sp_batch = QSpinBox(); self.sp_batch.setRange(0, 16); self.sp_batch.setValue(0)
        self.sp_batch.setSpecialValueText("Auto")
        form.addRow("Video frame batch", self.sp_batch)

        note = QLabel("If you run out of memory (CUDA out of memory), lower "
                      "the tile size. Tile size does not change the result; "
                      "it only splits the work into pieces.")
        note.setWordWrap(True); note.setObjectName("dim")
        form.addRow("", note)
        return g

    def _build_bottom(self) -> QWidget:
        box = QFrame()
        box.setObjectName("bottom")
        lay = QVBoxLayout(box)

        self.pb_file = QProgressBar(); self.pb_file.setTextVisible(False)
        self.pb_all = QProgressBar(); self.pb_all.setFormat("%v / %m files")
        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setObjectName("status")

        lay.addWidget(self.lbl_status)
        lay.addWidget(self.pb_file)
        lay.addWidget(self.pb_all)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        self.log.setFont(QFont("Consolas", 9))
        lay.addWidget(self.log)

        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_start = QPushButton("START")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumSize(160, 40)
        self.btn_start.clicked.connect(self._start)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumSize(110, 40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        row.addWidget(self.btn_cancel); row.addWidget(self.btn_start)
        lay.addLayout(row)
        return box

    # -- state sync -------------------------------------------------------

    def _sync_target_enabled(self) -> None:
        self.cb_preset.setEnabled(self.rb_preset.isChecked())
        self.cb_scale.setEnabled(self.rb_scale.isChecked())
        custom = self.rb_custom.isChecked()
        self.sp_h.setEnabled(custom)
        self.ck_ratio.setEnabled(custom)
        self.sp_w.setEnabled(custom and not self.ck_ratio.isChecked())
        self._refresh_estimate()

    def _sync_model(self) -> None:
        key = self.cb_model.currentData()
        if key == AUTO_MODEL:
            self.lbl_model_hint.setText(
                "Photos use the most detailed model; videos use a model that "
                "is much faster per frame. If unsure, leave this selected.")
            has_dn = True
        else:
            spec = REGISTRY[key]
            self.lbl_model_hint.setText(spec.hint)
            has_dn = spec.wdn_url is not None
        self.row_denoise.setVisible(has_dn)
        self.dn_note.setVisible(has_dn)
        self._refresh_estimate()

    def _current_settings(self) -> Settings:
        st = Settings()
        st.model_key = self.cb_model.currentData()
        st.denoise = self.sl_denoise.value() / 100.0

        if self.rb_preset.isChecked():
            st.target_mode, st.target_h = "preset", self.cb_preset.currentData()
        elif self.rb_scale.isChecked():
            st.target_mode, st.scale_mult = "scale", self.cb_scale.currentData()
        else:
            st.target_mode = "custom"
            st.target_h = self.sp_h.value()
            st.target_w = 0 if self.ck_ratio.isChecked() else self.sp_w.value()

        st.face_enhance = self.ck_face.isChecked()
        st.face_strength = self.sl_face.value() / 100.0
        st.ensemble = self.ck_ensemble.isChecked()

        st.tile = self.cb_tile.currentData()
        st.half = self.ck_half.isChecked()
        st.batch = self.sp_batch.value()

        st.encoder = self.cb_enc.currentData()
        st.quality = self.sl_quality.value()
        st.audio_mode = self.cb_audio.currentData()
        st.container = self.cb_container.currentData()
        st.preview_seconds = self.sp_preview.value()

        st.out_dir = Path(self.ed_out.text())
        st.image_format = self.cb_imgfmt.currentData()
        st.jpeg_quality = self.sp_jpegq.value()
        st.overwrite = self.ck_overwrite.isChecked()
        return st

    def _refresh_estimate(self) -> None:
        paths = self.list.paths()
        if not paths:
            self.estimate.setText("The list is empty. Add files to see what "
                                  "will be done with them.")
            return

        item = self.list.currentItem()
        src = Path(item.data(Qt.UserRole)) if item else paths[0]
        st = self._current_settings()
        try:
            is_video = src.suffix.lower() in ff.VIDEO_EXT
            w, h, extra = probe_dims(src)
            plan = build_plan(w, h, st, is_video=is_video)
            model = REGISTRY[resolve_model(st, is_video)].label
            self.estimate.setText(
                f"{src.name}{extra}\n{plan.describe()}\nModel: {model}")
        except Exception as exc:                          # noqa: BLE001
            self.estimate.setText(f"{src.name}: could not be read — {exc}")

    def _device_line(self) -> str:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            gb = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
            return f"GPU: {name} · {gb:.0f} GB VRAM · CUDA {torch.version.cuda}"
        return "No GPU found — processing will run on the CPU and be very slow."

    # -- actions ----------------------------------------------------------

    def _add_files(self) -> None:
        pats = " ".join(f"*{e}" for e in sorted(MEDIA_EXT))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select photos or videos", self.qs.value("last_dir", ""),
            f"Media files ({pats});;All files (*)")
        if files:
            self.qs.setValue("last_dir", str(Path(files[0]).parent))
            self.list.add_paths([Path(f) for f in files])

    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select folder", self.qs.value("last_dir", ""))
        if d:
            self.qs.setValue("last_dir", d)
            found = [p for p in sorted(Path(d).rglob("*"))
                     if p.suffix.lower() in MEDIA_EXT]
            n = self.list.add_paths(found)
            self._append_log(f"Added {n} file(s) from {d}.")

    def _clear_list(self) -> None:
        self.list.clear()
        self._refresh_estimate()

    def _pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    def _open_out_dir(self) -> None:
        target = self.last_output.parent if self.last_output else Path(self.ed_out.text())
        target.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32" and self.last_output and self.last_output.exists():
            # Open Explorer with the last output file selected.
            subprocess.Popen(["explorer", "/select,", str(self.last_output)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _start(self) -> None:
        files = self.list.paths()
        if not files:
            QMessageBox.information(self, "Empty list",
                                    "Add at least one photo or video first.")
            return
        missing = [p for p in files if not p.exists()]
        if missing:
            QMessageBox.warning(self, "File not found",
                                "These files no longer exist:\n" +
                                "\n".join(p.name for p in missing[:10]))
            return

        st = self._current_settings()
        st.out_dir.mkdir(parents=True, exist_ok=True)
        self._save_prefs()

        self.log.clear()
        self._append_log(self._device_line())
        self._set_running(True)
        self.pb_all.setMaximum(len(files))
        self.pb_all.setValue(0)

        self.worker = Worker(files, st)
        self.worker.log.connect(self._append_log)
        self.worker.file_progress.connect(self._on_file_progress)
        self.worker.overall_progress.connect(
            lambda done, total: self.pb_all.setValue(done))
        self.worker.file_done.connect(self._on_file_done)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker:
            self.lbl_status.setText("Cancelling...")
            self.btn_cancel.setEnabled(False)
            self.worker.cancel()

    # -- signal handlers --------------------------------------------------

    def _append_log(self, text: str) -> None:
        self.log.append(text)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    def _on_file_progress(self, frac: float, status: str) -> None:
        self.pb_file.setMaximum(1000)
        self.pb_file.setValue(int(max(0.0, min(frac, 1.0)) * 1000))
        self.lbl_status.setText(status)

    def _on_file_done(self, src: str, dst: str) -> None:
        self.last_output = Path(dst)

    def _on_all_done(self, ok: int, fail: int, cancelled: bool) -> None:
        self._set_running(False)
        self.pb_file.setValue(0)
        if cancelled:
            msg = f"Cancelled. {ok} file(s) had already been completed."
        elif fail:
            msg = f"Done — {ok} succeeded, {fail} failed. See the log for details."
        else:
            msg = f"Done — {ok} file(s) processed."
        self.lbl_status.setText(msg)
        self._append_log(msg)
        self.worker = None

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.list.setEnabled(not running)

    # -- preferences ------------------------------------------------------

    def _load_prefs(self) -> None:
        q = self.qs
        default_out = str((Path(__file__).resolve().parents[2] / "output"))
        self.ed_out.setText(q.value("out_dir", default_out))

        mode = q.value("target_mode", "preset")
        {"preset": self.rb_preset, "scale": self.rb_scale,
         "custom": self.rb_custom}.get(mode, self.rb_preset).setChecked(True)
        self.cb_preset.setCurrentIndex(int(q.value("preset_idx", 3)))
        self.cb_scale.setCurrentIndex(int(q.value("scale_idx", 3)))
        self.sp_w.setValue(int(q.value("target_w", 3840)))
        self.sp_h.setValue(int(q.value("target_h", 2160)))
        self.ck_ratio.setChecked(q.value("keep_ratio", "true") == "true")

        idx = self.cb_model.findData(q.value("model", AUTO_MODEL))
        self.cb_model.setCurrentIndex(max(idx, 0))
        self.sl_denoise.setValue(int(q.value("denoise", 50)))

        for cb, key, default in ((self.cb_enc, "encoder", "hevc_nvenc"),
                                 (self.cb_audio, "audio", "copy"),
                                 (self.cb_container, "container", ".mp4"),
                                 (self.cb_imgfmt, "imgfmt", "png")):
            i = cb.findData(q.value(key, default))
            cb.setCurrentIndex(max(i, 0))
        self.sl_quality.setValue(int(q.value("quality", 20)))
        self.sp_jpegq.setValue(int(q.value("jpegq", 95)))
        i = self.cb_tile.findData(int(q.value("tile", -1)))
        self.cb_tile.setCurrentIndex(max(i, 0))
        self.ck_half.setChecked(q.value("half", "true") == "true")
        self.ck_face.setChecked(q.value("face", "true") == "true")
        self.sl_face.setValue(int(q.value("face_strength", 80)))
        self.ck_ensemble.setChecked(q.value("ensemble", "false") == "true")
        self.row_face.setVisible(self.ck_face.isChecked())

        self._sync_target_enabled()
        self._sync_model()

    def _save_prefs(self) -> None:
        q = self.qs
        q.setValue("out_dir", self.ed_out.text())
        q.setValue("target_mode", "preset" if self.rb_preset.isChecked()
                   else "scale" if self.rb_scale.isChecked() else "custom")
        q.setValue("preset_idx", self.cb_preset.currentIndex())
        q.setValue("scale_idx", self.cb_scale.currentIndex())
        q.setValue("target_w", self.sp_w.value())
        q.setValue("target_h", self.sp_h.value())
        q.setValue("keep_ratio", "true" if self.ck_ratio.isChecked() else "false")
        q.setValue("model", self.cb_model.currentData())
        q.setValue("denoise", self.sl_denoise.value())
        q.setValue("encoder", self.cb_enc.currentData())
        q.setValue("audio", self.cb_audio.currentData())
        q.setValue("container", self.cb_container.currentData())
        q.setValue("imgfmt", self.cb_imgfmt.currentData())
        q.setValue("quality", self.sl_quality.value())
        q.setValue("jpegq", self.sp_jpegq.value())
        q.setValue("tile", self.cb_tile.currentData())
        q.setValue("half", "true" if self.ck_half.isChecked() else "false")
        q.setValue("face", "true" if self.ck_face.isChecked() else "false")
        q.setValue("face_strength", self.sl_face.value())
        q.setValue("ensemble", "true" if self.ck_ensemble.isChecked() else "false")

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            ans = QMessageBox.question(
                self, "Job in progress",
                "A job is still running. Cancel it and quit?")
            if ans != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.cancel()
            self.worker.wait(8000)
        self._save_prefs()
        event.accept()


STYLE = """
QWidget { font-size: 13px; }
QLabel#h1 { font-size: 17px; font-weight: 600; padding-bottom: 2px; }
QLabel#dim { font-size: 11px; }
QLabel#status { font-weight: 600; }
QLabel#estimate {
    border: 1px solid palette(mid); border-radius: 6px;
    padding: 8px; background: palette(alternate-base);
}
QLabel#path {
    border: 1px solid palette(mid); border-radius: 4px;
    padding: 5px; background: palette(base);
}
QGroupBox {
    font-weight: 600; border: 1px solid palette(mid);
    border-radius: 8px; margin-top: 10px; padding: 12px 10px 10px 10px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QFrame#bottom { border-top: 1px solid palette(mid); padding-top: 8px; }
QPushButton { padding: 6px 12px; border-radius: 5px; }
QPushButton#primary {
    background: #2d7d46; color: white; font-weight: 700; font-size: 15px;
    border: none; border-radius: 6px;
}
QPushButton#primary:hover { background: #359152; }
QPushButton#primary:disabled { background: palette(mid); color: palette(dark); }
QProgressBar { border: 1px solid palette(mid); border-radius: 5px; height: 16px; }
QProgressBar::chunk { background: #3a8ee6; border-radius: 4px; }
"""


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    """Blend b into a by t (0..1)."""
    return QColor(round(a.red() + (b.red() - a.red()) * t),
                  round(a.green() + (b.green() - a.green()) * t),
                  round(a.blue() + (b.blue() - a.blue()) * t))


def apply_style(app: QApplication) -> None:
    """Fusion style with a palette that stays readable in light and dark themes.

    In Windows dark mode Qt's palette has a navy AlternateBase and a Mid color
    almost identical to the background, which hides borders and secondary
    text. Those roles are derived from the window and text colors instead.
    """
    app.setStyle("Fusion")
    pal = app.palette()
    window, text = pal.color(QPalette.Window), pal.color(QPalette.WindowText)
    pal.setColor(QPalette.AlternateBase, _mix(window, text, 0.06))
    pal.setColor(QPalette.Mid, _mix(window, text, 0.3))
    app.setPalette(pal)
    dim = _mix(window, text, 0.6).name()
    app.setStyleSheet(STYLE + f"QLabel#dim {{ color: {dim}; }}\n")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Image Enhancer")
    apply_style(app)
    win = MainWindow()
    win.show()
    return app.exec()
