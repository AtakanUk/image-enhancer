"""Job pipelines: single-image and video upscaling with progress reporting."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import ffmpeg as ff
from .models import DEFAULT_IMAGE_MODEL, DEFAULT_VIDEO_MODEL, REGISTRY
from .upscaler import Plan, Upscaler, auto_tile, plan_resize

# Target resolution presets: label -> height (0 = custom/multiplier)
PRESETS: list[tuple[str, int]] = [
    ("HD  — 1280 x 720", 720),
    ("Full HD  — 1920 x 1080", 1080),
    ("2K QHD  — 2560 x 1440", 1440),
    ("4K UHD  — 3840 x 2160", 2160),
    ("5K  — 5120 x 2880", 2880),
    ("8K UHD  — 7680 x 4320", 4320),
]


@dataclass
class Settings:
    model_key: str = "photo_x4"
    denoise: float = 0.5
    # target: "preset" (target_h), "custom" (target_w x target_h), "scale" (multiplier)
    target_mode: str = "preset"
    target_h: int = 2160
    target_w: int = 0
    scale_mult: float = 4.0
    tile: int = -1               # -1 = auto, 0 = no tiling
    half: bool = True            # FP16
    ensemble: bool = False       # x8 self-ensemble (slow, cleaner)
    # face restoration (photos only)
    face_enhance: bool = True
    face_strength: float = 0.8
    # video
    encoder: str = "hevc_nvenc"
    quality: int = 20
    audio_mode: str = "copy"     # copy | aac | none
    container: str = ".mp4"
    preview_seconds: float = 0.0  # 0 = whole video
    batch: int = 0               # 0 = auto
    # output
    out_dir: Path = field(default_factory=lambda: Path("output"))
    image_format: str = "png"    # png | jpg | same
    jpeg_quality: int = 95
    overwrite: bool = False


Log = Callable[[str], None]
Prog = Callable[[float, str], None]   # (0..1, status text)


class Cancelled(RuntimeError):
    pass


_EXIF_ORIENTATION_TAG = 274


def _exif_orientation(path: Path) -> int:
    """EXIF orientation tag (1..8); 1 if it cannot be read."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
    except Exception:
        return 1


def _apply_orientation(img: np.ndarray, orient: int) -> np.ndarray:
    if orient == 2:
        return cv2.flip(img, 1)
    if orient == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orient == 4:
        return cv2.flip(img, 0)
    if orient == 5:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)
    if orient == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orient == 7:
        return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orient == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _read_image(path: Path) -> np.ndarray:
    """Image reader that also works with Unicode paths (cv2.imread does not).

    IMREAD_UNCHANGED keeps alpha and 16-bit depth but ignores EXIF rotation,
    so it is applied by hand to keep phone photos upright.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {path.name}")
    return _apply_orientation(img, _exif_orientation(path))


_dims_cache: dict[tuple[str, float, int], tuple[int, int, str]] = {}


def probe_dims(path: Path) -> tuple[int, int, str]:
    """(width, height, extra_info) without decoding the whole file.

    The result is cached by path + modification time; the UI calls this on
    every settings change, so it has to be cheap.
    """
    stat = path.stat()
    key = (str(path), stat.st_mtime, stat.st_size)
    if key in _dims_cache:
        return _dims_cache[key]

    if path.suffix.lower() in ff.VIDEO_EXT:
        info = ff.probe(path)
        result = (info.width, info.height, f" · {float(info.fps):.2f} fps")
    else:
        from PIL import Image
        with Image.open(path) as im:          # lazy: does not decode pixels
            w, h = im.size
            if _exif_orientation(path) >= 5:  # 5..8 -> rotated by 90 degrees
                w, h = h, w
        result = (w, h, "")

    _dims_cache[key] = result
    return result


def _to_bgr3(img: np.ndarray) -> np.ndarray:
    """The face detector expects 8-bit, 3-channel BGR."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    return np.ascontiguousarray(img)


def _write_image(path: Path, img: np.ndarray, jpeg_quality: int) -> None:
    ext = path.suffix.lower()
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.dtype == np.uint16:
            img = (img / 257).astype(np.uint8)
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, max(jpeg_quality, 90)]

    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


AUTO_MODEL = "auto"


def resolve_model(st: Settings, is_video: bool) -> str:
    """Resolve the 'auto' choice to a model based on the file type.

    Photos get the heavy model with the most detail; videos get the light
    model, which is many times faster per frame. Otherwise a long video
    would take hours.
    """
    if st.model_key != AUTO_MODEL:
        return st.model_key
    return DEFAULT_VIDEO_MODEL if is_video else DEFAULT_IMAGE_MODEL


def compute_target(src_w: int, src_h: int, st: Settings) -> tuple[int, int]:
    if st.target_mode == "scale":
        return round(src_w * st.scale_mult), round(src_h * st.scale_mult)
    if st.target_mode == "custom":
        return st.target_w, st.target_h
    return 0, st.target_h          # 0 -> derived from the aspect ratio


def build_plan(src_w: int, src_h: int, st: Settings, is_video: bool) -> Plan:
    tw, th = compute_target(src_w, src_h, st)
    scale = REGISTRY[resolve_model(st, is_video)].scale
    # yuv420p in video codecs does not accept odd dimensions.
    return plan_resize(src_w, src_h, tw, th, model_scale=scale,
                       even_dims=is_video)


def unique_path(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    stem, suffix, n = path.stem, path.suffix, 2
    while True:
        cand = path.with_name(f"{stem} ({n}){suffix}")
        if not cand.exists():
            return cand
        n += 1


def _label_for(plan: Plan) -> str:
    return f"{plan.out_w}x{plan.out_h}"


class Runner:
    """Caches the model instance so consecutive files do not reload it."""

    def __init__(self) -> None:
        self._up: Upscaler | None = None
        self._key: tuple | None = None
        self._faces = None
        self.cancel_event = threading.Event()

    # -- model cache ------------------------------------------------------

    def _get_upscaler(self, st: Settings, model_key: str, src_w: int, src_h: int,
                      log: Log, prog: Prog, ensemble: bool = False) -> Upscaler:
        tile = st.tile
        if tile < 0:
            tile = auto_tile(src_w, src_h, REGISTRY[model_key].kind)
        key = (model_key, round(st.denoise, 3), st.half, ensemble)

        if self._up is None or self._key != key:
            if self._up is not None:
                self._up.release()
            log(f"Loading model: {REGISTRY[model_key].label}")
            self._up = Upscaler(
                model_key, denoise=st.denoise, half=st.half, tile=tile,
                ensemble=ensemble, progress=lambda msg, f: prog(f, msg),
            )
            self._key = key
            log(f"Device: {self._up.device.type.upper()}"
                f"{' (FP16)' if self._up.half else ' (FP32)'}"
                f"{' · x8 self-ensemble' if ensemble else ''}")
        self._up.tile = tile
        return self._up

    def _check(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled("Cancelled by user.")

    def _get_face_enhancer(self, log: Log, prog: Prog):
        if self._faces is None:
            from .faces import FaceEnhancer
            log("Loading face restoration model (GFPGAN v1.4)...")
            device = self._up.device if self._up else None
            if device is None:
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._faces = FaceEnhancer(device, progress=lambda m, f: prog(f, m),
                                       log=log)
        return self._faces

    def release(self) -> None:
        if self._up is not None:
            self._up.release()
            self._up = None
            self._key = None
        if self._faces is not None:
            self._faces.release()
            self._faces = None

    # -- image ------------------------------------------------------------

    def run_image(self, src: Path, st: Settings, log: Log, prog: Prog) -> Path:
        img = _read_image(src)
        h, w = img.shape[:2]

        model_key = resolve_model(st, is_video=False)
        plan = build_plan(w, h, st, is_video=False)
        log(f"{src.name}: {plan.describe()}")
        if plan.passes == 0:
            log("  Warning: target is not larger than the source, so AI upscaling is skipped.")

        up = self._get_upscaler(st, model_key, w, h, log, prog,
                                ensemble=st.ensemble)
        self._check()

        state = {"t0": time.monotonic()}
        # The face stage owns the last slice of the progress bar; keep the
        # tile progress from overwriting it.
        tile_span = 0.85 if st.face_enhance else 1.0

        def on_tile(done: int, total: int) -> None:
            self._check()
            prog(done / total * tile_span, f"{src.name} — tile {done}/{total}")

        post_ai = None
        if st.face_enhance:
            def post_ai(canvas: np.ndarray, factor: int) -> np.ndarray:
                self._check()
                prog(0.9, f"{src.name} — detecting faces...")
                # Faces are detected on the source image (the alignment
                # landmarks are defined there) and pasted onto the upscaled
                # canvas.
                fe = self._get_face_enhancer(log, prog)
                self._check()
                result, n = fe.enhance(_to_bgr3(img), canvas, factor,
                                       st.face_strength)
                log(f"  Restored {n} face(s) (strength {st.face_strength:.2f})."
                    if n else "  No faces found; image left as is.")
                return result

            if img.dtype == np.uint16:
                log("  Note: face restoration works in 8-bit, so the output will be 8-bit.")

        prog(0.0, f"Processing {src.name}...")
        out_img = up.process(img, plan, on_tile, post_ai=post_ai)

        ext = src.suffix.lower() if st.image_format == "same" else f".{st.image_format}"
        if ext not in ff.IMAGE_EXT:
            ext = ".png"
        dst = unique_path(st.out_dir / f"{src.stem}_{_label_for(plan)}{ext}",
                          st.overwrite)
        _write_image(dst, out_img, st.jpeg_quality)

        log(f"  Done ({time.monotonic() - state['t0']:.1f} s) -> {dst.name}")
        return dst

    # -- video ------------------------------------------------------------

    def run_video(self, src: Path, st: Settings, log: Log, prog: Prog) -> Path:
        info = ff.probe(src)
        model_key = resolve_model(st, is_video=True)
        plan = build_plan(info.width, info.height, st, is_video=True)

        dur = st.preview_seconds if st.preview_seconds > 0 else 0.0
        total = info.n_frames
        if dur > 0:
            total = min(total or 10 ** 9, int(round(dur * float(info.fps))))

        log(f"{src.name}: {plan.describe()}")
        log(f"  {float(info.fps):.3f} fps, {total or '?'} frames, "
            f"audio: {info.audio_codec or 'none'}")
        if dur > 0:
            log(f"  Preview mode: first {dur:g} seconds only.")
        if plan.passes == 0:
            log("  Warning: target is not larger than the source, so AI upscaling is skipped.")

        # Self-ensemble and face restoration are not applied to video: the
        # former multiplies the time by 8, the latter causes flicker because
        # every frame is reconstructed independently.
        up = self._get_upscaler(st, model_key, info.width, info.height,
                                log, prog, ensemble=False)
        self._check()

        batch = st.batch or self._auto_batch(info.width, info.height)
        frame_bytes = info.width * info.height * 3

        dst = unique_path(
            st.out_dir / f"{src.stem}_{_label_for(plan)}{st.container}", st.overwrite)
        dst.parent.mkdir(parents=True, exist_ok=True)

        dec = ff.spawn(ff.decode_cmd(src, duration=dur), stdout=subprocess.PIPE)
        enc = ff.spawn(
            ff.encode_cmd(dst, plan.out_w, plan.out_h, info.fps_str,
                          encoder=st.encoder, quality=st.quality,
                          src_for_audio=src, audio_mode=st.audio_mode,
                          audio_codec=info.audio_codec, duration=dur),
            stdin=subprocess.PIPE)

        dec_err, enc_err = _drain(dec), _drain(enc)
        done, t0 = 0, time.monotonic()

        try:
            while True:
                self._check()
                frames = []
                for _ in range(batch):
                    raw = _read_exact(dec.stdout, frame_bytes)
                    if raw is None:
                        break
                    frames.append(np.frombuffer(raw, np.uint8).reshape(
                        info.height, info.width, 3))
                if not frames:
                    break

                out = up.process_batch(np.stack(frames), plan)
                enc.stdin.write(out.tobytes())

                done += len(frames)
                elapsed = time.monotonic() - t0
                fps_now = done / max(elapsed, 1e-6)
                if total:
                    eta = (total - done) / max(fps_now, 1e-6)
                    prog(min(done / total, 1.0),
                         f"{src.name} — frame {done}/{total} · "
                         f"{fps_now:.1f} fps · {_hms(eta)} left")
                else:
                    prog(0.0, f"{src.name} — frame {done} · {fps_now:.1f} fps")

            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError(f"Encoder error: {enc_err.text()[:500]}")
            if dec.wait() not in (0, None) and done == 0:
                raise RuntimeError(f"Decoder error: {dec_err.text()[:500]}")

        except Cancelled:
            _kill(enc); _kill(dec)
            dst.unlink(missing_ok=True)
            raise
        except BrokenPipeError as exc:
            _kill(dec); enc.wait(timeout=5)
            raise RuntimeError(f"Encoder exited unexpectedly: "
                               f"{enc_err.text()[:500]}") from exc
        finally:
            _kill(dec, grace=0.5)
            _kill(enc)

        secs = time.monotonic() - t0
        size_mb = dst.stat().st_size / (1 << 20) if dst.exists() else 0
        log(f"  Done: {done} frames, {_hms(secs)}, {size_mb:.1f} MB -> {dst.name}")
        return dst

    @staticmethod
    def _auto_batch(w: int, h: int) -> int:
        px = w * h
        if px > 900_000:      # 1080p and above
            return 1
        if px > 350_000:      # around 720p
            return 2
        return 4


# ------------------------------------------------------------------ helpers

class _Drain:
    """Reads ffmpeg's stderr in the background so the pipe never fills up and blocks."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._buf: list[bytes] = []
        self._proc = proc
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for line in iter(self._proc.stderr.readline, b""):
                self._buf.append(line)
        except Exception:
            pass

    def text(self) -> str:
        self._thread.join(timeout=2)
        return b"".join(self._buf).decode("utf-8", "replace").strip()


def _drain(proc: subprocess.Popen) -> _Drain:
    return _Drain(proc)


def _read_exact(stream, n: int) -> bytes | None:
    """Read exactly n bytes; return None once the stream has ended."""
    chunks, remaining = [], n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _kill(proc: subprocess.Popen, grace: float = 0.0) -> None:
    if proc.poll() is not None:
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=max(grace, 2.0))
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
