from pathlib import Path

import pytest
import torch

import core.models as models
from core.arch.gfpgan_clean import GFPGANv1Clean
from core.arch.rrdbnet import RRDBNet
from core.arch.srvgg import SRVGGNetCompact


def test_architectures_do_not_import_basicsr():
    for path in (Path(models.__file__).parent / "arch").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from basicsr" not in source and "import basicsr" not in source, path.name


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_rrdbnet_output_size(scale):
    net = RRDBNet(3, 3, scale=scale, num_feat=8, num_block=1, num_grow_ch=4).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, 16, 24))
    assert out.shape == (1, 3, 16 * scale, 24 * scale)


def test_srvgg_output_size():
    net = SRVGGNetCompact(3, 3, num_feat=8, num_conv=2, upscale=4).eval()
    with torch.no_grad():
        assert net(torch.rand(2, 3, 10, 12)).shape == (2, 3, 40, 48)


def test_gfpgan_forward_runs_standalone():
    net = GFPGANv1Clean(out_size=64, num_style_feat=32, channel_multiplier=1,
                        narrow=0.25, different_w=True, input_is_latent=True,
                        sft_half=True).eval()
    with torch.no_grad():
        image, _ = net(torch.rand(1, 3, 64, 64), return_rgb=False)
    assert image.shape == (1, 3, 64, 64)


def test_registry_specs_build_matching_networks():
    # Every registry entry must describe a network the loader can build.
    for spec in models.REGISTRY.values():
        cls = RRDBNet if spec.kind == "rrdb" else SRVGGNetCompact
        cls(num_in_ch=3, num_out_ch=3, **spec.params)
        assert spec.url.endswith(spec.filename)


def test_dni_blends_the_two_weight_sets(monkeypatch, tmp_path):
    spec = models.REGISTRY["general"]
    torch.manual_seed(0)
    strong = SRVGGNetCompact(3, 3, **spec.params).state_dict()
    weak = SRVGGNetCompact(3, 3, **spec.params).state_dict()
    torch.save({"params": strong}, tmp_path / spec.filename)
    torch.save({"params": weak}, tmp_path / spec.wdn_filename)
    monkeypatch.setattr(models, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(models, "_download",
                        lambda *a, **k: pytest.fail("must not download"))

    net = models.load_model(spec, denoise=0.25)
    for name, value in net.state_dict().items():
        expected = 0.25 * strong[name] + 0.75 * weak[name]
        assert torch.allclose(value, expected, atol=1e-6), name
    assert not any(p.requires_grad for p in net.parameters())

    pure = models.load_model(spec, denoise=1.0)      # no blending, wdn file unused
    for name, value in pure.state_dict().items():
        assert torch.equal(value, strong[name])
