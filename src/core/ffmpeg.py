"""ffmpeg/ffprobe wrappers.

Frames never touch the disk: they are read from the decoder as raw BGR,
processed, and handed to the encoder as raw BGR. Disk usage therefore stays
constant even for long videos.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

# Do not pop up a console window (Windows, when running from the GUI).
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0

# Audio codecs the mp4 container carries without issues; others become AAC.
_MP4_SAFE_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac", "opus"}

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".m2ts"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}


class FFmpegMissing(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegMissing("ffmpeg was not found on PATH.")
    return exe


def ffprobe_bin() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        # Try the ffprobe next to ffmpeg.
        ffmpeg = Path(ffmpeg_bin())
        sibling = ffmpeg.with_name("ffprobe" + ffmpeg.suffix)
        if sibling.exists():
            return str(sibling)
        raise FFmpegMissing("ffprobe was not found on PATH.")
    return exe


@dataclass
class VideoInfo:
    width: int              # display width (rotation applied)
    height: int
    fps: Fraction
    n_frames: int
    duration: float
    has_audio: bool
    audio_codec: str
    v_codec: str

    @property
    def fps_str(self) -> str:
        return f"{self.fps.numerator}/{self.fps.denominator}"


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, creationflags=_NO_WINDOW,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[:400]}")
    data = json.loads(out.stdout)

    vs = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if vs is None:
        raise RuntimeError("The file has no video stream.")
    aud = next((s for s in data["streams"] if s.get("codec_type") == "audio"), None)

    w, h = int(vs["width"]), int(vs["height"])

    # Rotation: ffmpeg already rotates frames while decoding, so at +-90
    # degrees width and height swap.
    rot = 0
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(float(vs.get("tags", {}).get("rotate", 0) or 0))
    if abs(rot) % 180 == 90:
        w, h = h, w

    fps = Fraction(vs.get("avg_frame_rate") or "0/1")
    if fps <= 0:
        fps = Fraction(vs.get("r_frame_rate") or "30/1")
    if fps <= 0:
        fps = Fraction(30, 1)

    duration = float(data.get("format", {}).get("duration") or vs.get("duration") or 0.0)
    n_frames = int(vs.get("nb_frames") or 0)
    if n_frames <= 0:
        n_frames = int(round(duration * float(fps))) if duration else 0

    return VideoInfo(
        width=w, height=h, fps=fps, n_frames=n_frames, duration=duration,
        has_audio=aud is not None,
        audio_codec=(aud or {}).get("codec_name", ""),
        v_codec=vs.get("codec_name", ""),
    )


def decode_cmd(src: str | Path, start: float = 0.0, duration: float = 0.0) -> list[str]:
    """Dump the source to stdout as raw bgr24 (cv2 channel order)."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-map", "0:v:0", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return cmd


ENCODERS = {
    "hevc_nvenc": "HEVC / H.265 (NVENC - GPU, recommended)",
    "h264_nvenc": "H.264 (NVENC - GPU, most compatible)",
    "av1_nvenc":  "AV1 (NVENC - GPU, smallest file)",
    "libx265":    "HEVC / H.265 (CPU - best compression)",
    "libx264":    "H.264 (CPU)",
}


def encode_cmd(dst: str | Path, width: int, height: int, fps_str: str,
               encoder: str = "hevc_nvenc", quality: int = 20,
               src_for_audio: str | Path | None = None,
               audio_mode: str = "copy", audio_codec: str = "",
               start: float = 0.0, duration: float = 0.0) -> list[str]:
    """Build the command that reads raw bgr24 from stdin and writes the file.

    Audio is taken from the source file as a second input and copied.
    """
    dst = Path(dst)
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", fps_str, "-i", "-"]

    want_audio = audio_mode != "none" and src_for_audio is not None
    if want_audio:
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(src_for_audio)]
        if duration > 0:
            cmd += ["-t", f"{duration:.3f}"]

    cmd += ["-map", "0:v:0"]
    if want_audio:
        cmd += ["-map", "1:a:0?"]

    cmd += ["-c:v", encoder, "-pix_fmt", "yuv420p"]
    if encoder.endswith("nvenc"):
        # p6 = slow/high quality; lower cq = higher quality.
        cmd += ["-preset", "p6", "-tune", "hq", "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0", "-bf", "3"]
        # The H.264 High profile allows the 8x8 transform -> better compression.
        if encoder == "h264_nvenc":
            cmd += ["-profile:v", "high"]
        elif encoder == "hevc_nvenc":
            cmd += ["-profile:v", "main"]
    else:
        cmd += ["-preset", "slow", "-crf", str(quality)]

    if want_audio:
        if audio_mode == "copy" and (
            dst.suffix.lower() != ".mp4" or audio_codec in _MP4_SAFE_AUDIO
        ):
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "256k"]
        # Deliberately no -shortest: in most files the audio is a few
        # milliseconds shorter than the video, and -shortest then cuts the
        # video at a GOP boundary and drops the last frames (measured: 3 of
        # 60 frames lost in a test). Audio running a few ms long is better
        # than losing frames.

    if dst.suffix.lower() in (".mp4", ".m4v", ".mov"):
        cmd += ["-movflags", "+faststart"]

    cmd += [str(dst)]
    return cmd


def spawn(cmd: list[str], stdin=None, stdout=None) -> subprocess.Popen:
    return subprocess.Popen(
        cmd, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE,
        creationflags=_NO_WINDOW, bufsize=0,
    )
