import numpy as np
import pytest
import torch

import core.upscaler as upscaler
from core.arch.rrdbnet import RRDBNet
from core.arch.srvgg import SRVGGNetCompact
from core.upscaler import Plan, Upscaler, _augment, _unaugment, plan_resize


@pytest.fixture
def make_upscaler(monkeypatch):
    """Upscaler backed by a tiny random network instead of downloaded weights."""
    def fake_load_model(spec, denoise=0.5, progress=None):
        torch.manual_seed(0)
        if spec.kind == "rrdb":
            return RRDBNet(3, 3, scale=spec.params["scale"], num_feat=8,
                           num_block=1, num_grow_ch=4).eval()
        return SRVGGNetCompact(3, 3, num_feat=8, num_conv=2,
                               upscale=spec.scale).eval()

    monkeypatch.setattr(upscaler, "load_model", fake_load_model)

    def factory(model_key="general", **kwargs):
        kwargs.setdefault("tile", 0)
        return Upscaler(model_key, device="cpu", half=False, **kwargs)
    return factory


def _image(h, w, channels=3, dtype=np.uint8, seed=0):
    rng = np.random.default_rng(seed)
    hi = 65535 if dtype == np.uint16 else 255
    return rng.integers(0, hi + 1, size=(h, w, channels), dtype=dtype)


def test_dihedral_transforms_round_trip():
    t = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    for i in range(8):
        assert torch.equal(_unaugment(_augment(t, i), i), t)
    # On a square input all 8 transforms are distinct.
    sq = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    assert len({tuple(_augment(sq, i).flatten().tolist()) for i in range(8)}) == 8


@pytest.mark.parametrize("model_key", ["general", "photo_x4", "photo_x2"])
def test_tiled_output_matches_single_pass(make_upscaler, model_key):
    up = make_upscaler(model_key)
    img = _image(70, 90)
    plan = plan_resize(90, 70, 90 * up.scale, 70 * up.scale, up.scale)

    whole = up.process(img, plan)
    up.tile = 32
    tiled = up.process(img, plan)

    assert whole.shape == (70 * up.scale, 90 * up.scale, 3)
    assert np.abs(whole.astype(int) - tiled.astype(int)).max() <= 1


def test_x2_model_accepts_odd_sizes(make_upscaler):
    up = make_upscaler("photo_x2")
    out = up.process(_image(37, 29), Plan(29, 37, 58, 74, 1, 2, False))
    assert out.shape == (74, 58, 3)


def test_alpha_and_16bit_are_preserved(make_upscaler):
    up = make_upscaler("general")
    img = _image(12, 16, channels=4, dtype=np.uint16)
    plan = plan_resize(16, 12, 48, 36, up.scale)

    out = up.process(img, plan)
    assert out.shape == (36, 48, 4)
    assert out.dtype == np.uint16


def test_grayscale_input_becomes_bgr(make_upscaler):
    up = make_upscaler("general")
    out = up.process(_image(10, 10)[:, :, 0], plan_resize(10, 10, 40, 40, 4))
    assert out.shape == (40, 40, 3)


def test_post_ai_hook_gets_integer_factor_and_8bit(make_upscaler):
    up = make_upscaler("general")
    seen = {}

    def hook(canvas, factor):
        seen["shape"], seen["factor"], seen["dtype"] = canvas.shape, factor, canvas.dtype
        return canvas

    out = up.process(_image(10, 12, dtype=np.uint16), plan_resize(12, 10, 0, 30, 4),
                     post_ai=hook)
    assert seen == {"shape": (40, 48, 3), "factor": 4, "dtype": np.uint8}
    assert out.dtype == np.uint8 and out.shape == (30, 36, 3)


def test_batch_processing_shape(make_upscaler):
    up = make_upscaler("general")
    frames = np.stack([_image(18, 32, seed=i) for i in range(3)])
    plan = plan_resize(32, 18, 0, 54, up.scale, even_dims=True)
    out = up.process_batch(frames, plan)
    assert out.shape == (3, 54, 96, 3) and out.dtype == np.uint8


def test_ensemble_reports_progress_over_all_eight_runs(make_upscaler):
    up = make_upscaler("general", ensemble=True, tile=8)
    calls = []
    out = up.process(_image(16, 16), plan_resize(16, 16, 64, 64, 4),
                     on_tile=lambda done, total: calls.append((done, total)))
    assert out.shape == (64, 64, 3)
    assert calls[-1] == (32, 32)          # 4 tiles x 8 transforms
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)


def test_out_of_memory_halves_tile_and_retries(make_upscaler, monkeypatch):
    up = make_upscaler("general", tile=512)
    real = up._forward_whole
    state = {"failed": False}

    def flaky(t):
        if not state["failed"]:
            state["failed"] = True
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(t)

    monkeypatch.setattr(up, "_forward_whole", flaky)
    out = up.process(_image(20, 20), plan_resize(20, 20, 80, 80, 4))
    assert up.tile == 256
    assert out.shape == (80, 80, 3)
