import io

import numpy as np
import pytest
from PIL import Image

from core import jobs
from core.jobs import (AUTO_MODEL, Settings, build_plan, compute_target,
                       probe_dims, resolve_model, unique_path)


def test_compute_target_modes():
    st = Settings(target_mode="scale", scale_mult=2.5)
    assert compute_target(640, 480, st) == (1600, 1200)
    st = Settings(target_mode="custom", target_w=1000, target_h=500)
    assert compute_target(640, 480, st) == (1000, 500)
    st = Settings(target_mode="preset", target_h=2160)
    assert compute_target(640, 480, st) == (0, 2160)


def test_auto_model_depends_on_file_type():
    st = Settings(model_key=AUTO_MODEL)
    assert resolve_model(st, is_video=False) == "photo_x4"
    assert resolve_model(st, is_video=True) == "general"
    assert resolve_model(Settings(model_key="anime_x4"), is_video=True) == "anime_x4"


def test_video_plans_have_even_dimensions():
    st = Settings(model_key="general", target_mode="custom", target_w=1001, target_h=563)
    plan = build_plan(640, 360, st, is_video=True)
    assert plan.out_w % 2 == 0 and plan.out_h % 2 == 0
    assert build_plan(640, 360, st, is_video=False).out_w == 1001


def test_unique_path_appends_a_number(tmp_path):
    p = tmp_path / "photo_3840x2160.png"
    assert unique_path(p, overwrite=False) == p
    p.touch()
    assert unique_path(p, overwrite=True) == p
    assert unique_path(p, overwrite=False).name == "photo_3840x2160 (2).png"
    (tmp_path / "photo_3840x2160 (2).png").touch()
    assert unique_path(p, overwrite=False).name == "photo_3840x2160 (3).png"


def _jpeg_with_orientation(path, w, h, orientation):
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (w, h), (200, 30, 30)).save(path, exif=exif)


def test_exif_rotation_and_unicode_path(tmp_path):
    path = tmp_path / "café ünïcode.jpg"
    _jpeg_with_orientation(path, 40, 20, orientation=6)   # rotated 90 degrees

    img = jobs._read_image(path)
    assert img.shape[:2] == (40, 20)                      # displayed upright
    assert probe_dims(path) == (20, 40, "")


def test_image_roundtrip_formats(tmp_path):
    img = np.zeros((8, 10, 4), np.uint16)
    jobs._write_image(tmp_path / "a.png", img, 95)
    jobs._write_image(tmp_path / "a.jpg", img, 95)          # alpha and 16-bit dropped
    assert jobs._read_image(tmp_path / "a.png").shape == (8, 10, 4)
    assert jobs._read_image(tmp_path / "a.jpg").shape == (8, 10, 3)


def test_read_exact_handles_short_reads():
    class Trickle(io.BytesIO):
        def read(self, n=-1):
            return super().read(min(n, 3))

    stream = Trickle(b"abcdefgh")
    assert jobs._read_exact(stream, 5) == b"abcde"
    assert jobs._read_exact(stream, 5) is None             # only 3 bytes left


@pytest.mark.parametrize("secs, text", [(5, "0:05"), (65, "1:05"), (3725, "1:02:05"), (-3, "0:00")])
def test_hms(secs, text):
    assert jobs._hms(secs) == text
