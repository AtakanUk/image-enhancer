import pytest

from core.upscaler import PASS_TOLERANCE, plan_resize


@pytest.mark.parametrize("src, passes, ai_size", [
    ((1280, 720), 1, (5120, 2880)),   # 720p -> 4K: one pass, then downscale
    ((854, 480), 1, (3416, 1920)),    # 480p -> 4K: remaining stretch 1.125x, still one pass
    ((426, 240), 2, (6816, 3840)),    # 240p -> 4K: needs a second pass
])
def test_passes_to_4k(src, passes, ai_size):
    plan = plan_resize(*src, 0, 2160, model_scale=4)
    assert plan.passes == passes
    assert (plan.ai_w, plan.ai_h) == ai_size
    assert plan.out_h == 2160


def test_width_follows_aspect_ratio():
    plan = plan_resize(1280, 720, 0, 2160, model_scale=4)
    assert (plan.out_w, plan.out_h) == (3840, 2160)
    assert plan.downscale


def test_height_follows_aspect_ratio():
    plan = plan_resize(1000, 500, 3000, 0, model_scale=4)
    assert (plan.out_w, plan.out_h) == (3000, 1500)


def test_exact_model_factor_has_no_resize():
    plan = plan_resize(1920, 1080, 3840, 2160, model_scale=2)
    assert plan.passes == 1
    assert (plan.ai_w, plan.ai_h) == (plan.out_w, plan.out_h)
    assert "downscale" not in plan.describe() and "stretch" not in plan.describe()


def test_target_not_larger_than_source_skips_ai():
    plan = plan_resize(3840, 2160, 0, 1080, model_scale=4)
    assert plan.passes == 0
    assert plan.downscale
    assert "no AI" in plan.describe()


def test_small_upscale_still_runs_one_pass():
    # Any upscale gets one pass: the model also cleans compression artifacts.
    plan = plan_resize(1000, 1000, 1100, 1100, model_scale=4)
    assert plan.passes == 1


def test_second_pass_only_above_tolerance():
    # 4 * PASS_TOLERANCE is the largest factor a single x4 pass covers.
    edge = int(100 * 4 * PASS_TOLERANCE)
    assert plan_resize(100, 100, edge, edge, model_scale=4).passes == 1
    assert plan_resize(100, 100, edge + 4, edge + 4, model_scale=4).passes == 2


def test_max_passes_caps_the_plan():
    plan = plan_resize(10, 10, 100_000, 100_000, model_scale=2, max_passes=3)
    assert plan.passes == 3


def test_even_dims_for_video():
    plan = plan_resize(641, 361, 0, 1081, model_scale=4, even_dims=True)
    assert plan.out_w % 2 == 0 and plan.out_h % 2 == 0
