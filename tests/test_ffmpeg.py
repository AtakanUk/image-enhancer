from pathlib import Path

import pytest

from core import ffmpeg as ff


@pytest.fixture(autouse=True)
def fake_ffmpeg(monkeypatch):
    # Command building must not depend on ffmpeg being installed.
    monkeypatch.setattr(ff, "ffmpeg_bin", lambda: "ffmpeg")


def _value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_decode_streams_raw_bgr_to_stdout():
    cmd = ff.decode_cmd("in.mkv", duration=10)
    assert cmd[-1] == "-"
    assert _value(cmd, "-pix_fmt") == "bgr24"
    assert _value(cmd, "-t") == "10.000"
    assert _value(cmd, "-map") == "0:v:0"


def test_encode_reads_raw_frames_from_stdin():
    cmd = ff.encode_cmd("out.mp4", 3840, 2160, "30000/1001")
    assert _value(cmd, "-s") == "3840x2160"
    assert _value(cmd, "-r") == "30000/1001"
    assert cmd[cmd.index("-i") + 1] == "-"
    assert cmd[-1] == "out.mp4"


@pytest.mark.parametrize("dst, codec, expected", [
    ("out.mp4", "aac", "copy"),
    ("out.mp4", "pcm_s16le", "aac"),   # mp4 cannot carry PCM -> re-encode
    ("out.mkv", "pcm_s16le", "copy"),
])
def test_audio_copy_or_reencode(dst, codec, expected):
    cmd = ff.encode_cmd(dst, 640, 360, "25/1", src_for_audio="in.mov",
                        audio_mode="copy", audio_codec=codec)
    assert _value(cmd, "-c:a") == expected
    assert "1:a:0?" in cmd
    assert "-shortest" not in cmd


def test_audio_can_be_dropped():
    cmd = ff.encode_cmd("out.mp4", 640, 360, "25/1", src_for_audio="in.mov",
                        audio_mode="none")
    assert "-c:a" not in cmd and "1:a:0?" not in cmd


def test_nvenc_uses_cq_and_cpu_encoders_use_crf():
    nvenc = ff.encode_cmd("out.mp4", 640, 360, "25/1", encoder="hevc_nvenc", quality=18)
    x264 = ff.encode_cmd("out.mp4", 640, 360, "25/1", encoder="libx264", quality=18)
    assert _value(nvenc, "-cq") == "18" and "-crf" not in nvenc
    assert _value(x264, "-crf") == "18" and "-cq" not in x264


@pytest.mark.parametrize("dst, faststart", [("a.mp4", True), ("a.mov", True), ("a.mkv", False)])
def test_faststart_only_for_mp4_family(dst, faststart):
    assert ("+faststart" in ff.encode_cmd(dst, 64, 64, "25/1")) is faststart


def test_ffprobe_falls_back_to_binary_next_to_ffmpeg(monkeypatch, tmp_path):
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.touch()
    ffprobe.touch()
    monkeypatch.setattr(ff, "ffmpeg_bin", lambda: str(ffmpeg))
    monkeypatch.setattr(ff.shutil, "which", lambda name: None)
    assert Path(ff.ffprobe_bin()) == ffprobe


def test_missing_ffprobe_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ff, "ffmpeg_bin", lambda: str(tmp_path / "ffmpeg"))
    monkeypatch.setattr(ff.shutil, "which", lambda name: None)
    with pytest.raises(ff.FFmpegMissing):
        ff.ffprobe_bin()
