"""Model registry and downloader.

Weights are downloaded from the official Real-ESRGAN releases into models/ on
first use and loaded from disk on later runs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests
import torch

from .arch.rrdbnet import RRDBNet
from .arch.srvgg import SRVGGNetCompact

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
_BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    filename: str
    url: str
    scale: int
    kind: str                     # "rrdb" | "srvgg"
    params: dict = field(default_factory=dict)
    hint: str = ""
    min_bytes: int = 1_000_000
    # Weak-denoise counterpart of realesr-general-x4v3, used for DNI blending
    wdn_filename: str | None = None
    wdn_url: str | None = None


REGISTRY: dict[str, ModelSpec] = {
    "general": ModelSpec(
        key="general",
        label="General — Photo / Video (balanced, fast)",
        filename="realesr-general-x4v3.pth",
        url=f"{_BASE}/v0.2.5.0/realesr-general-x4v3.pth",
        scale=4, kind="srvgg",
        params=dict(num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
        hint="Works on any content. Default for video; supports the denoise slider.",
        min_bytes=200_000,
        wdn_filename="realesr-general-wdn-x4v3.pth",
        wdn_url=f"{_BASE}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    ),
    "photo_x4": ModelSpec(
        key="photo_x4",
        label="Photo x4 — Highest detail (slow)",
        filename="RealESRGAN_x4plus.pth",
        url=f"{_BASE}/v0.1.0/RealESRGAN_x4plus.pth",
        scale=4, kind="rrdb",
        params=dict(num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        hint="Sharpest result on real photos. Too slow for video.",
    ),
    "photo_x2": ModelSpec(
        key="photo_x2",
        label="Photo x2 — Small upscale, high fidelity",
        filename="RealESRGAN_x2plus.pth",
        url=f"{_BASE}/v0.2.1/RealESRGAN_x2plus.pth",
        scale=2, kind="rrdb",
        params=dict(num_feat=64, num_block=23, num_grow_ch=32, scale=2),
        hint="Most natural result when upscaling by about 2x, e.g. 1080p -> 4K.",
    ),
    "anime_x4": ModelSpec(
        key="anime_x4",
        label="Anime / Illustration x4",
        filename="RealESRGAN_x4plus_anime_6B.pth",
        url=f"{_BASE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        scale=4, kind="rrdb",
        params=dict(num_feat=64, num_block=6, num_grow_ch=32, scale=4),
        hint="Cartoons, anime, illustrations. Looks plastic on real photos.",
    ),
    "anime_video": ModelSpec(
        key="anime_video",
        label="Anime video (very fast)",
        filename="realesr-animevideov3.pth",
        url=f"{_BASE}/v0.2.5.0/realesr-animevideov3.pth",
        scale=4, kind="srvgg",
        params=dict(num_feat=64, num_conv=16, upscale=4, act_type="prelu"),
        hint="Fastest option for anime videos.",
        min_bytes=100_000,
    ),
}

# Face restoration network. Kept apart from the upscaling models because it is
# not a choice in the model list but an optional stage that can be toggled.
FACE_SPEC = ModelSpec(
    key="gfpgan14",
    label="GFPGAN v1.4 — face restoration",
    filename="GFPGANv1.4.pth",
    url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    scale=1, kind="gfpgan",
    hint="Reconstructs facial detail in portraits.",
    min_bytes=100_000_000,
)

DEFAULT_IMAGE_MODEL = "photo_x4"
DEFAULT_VIDEO_MODEL = "general"

ProgressFn = Callable[[str, float], None]


def _download(url: str, dest: Path, min_bytes: int, progress: ProgressFn | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(f"Downloading {dest.name}", done / total)
    if tmp.stat().st_size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete download: {dest.name} ({url}).")
    os.replace(tmp, dest)


def ensure_weights(spec: ModelSpec, progress: ProgressFn | None = None) -> Path:
    """Return the path to the weight file, downloading it first if needed."""
    path = MODEL_DIR / spec.filename
    if not path.exists() or path.stat().st_size < spec.min_bytes:
        _download(spec.url, path, spec.min_bytes, progress)
    return path


def _extract_state(obj):
    if isinstance(obj, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in obj:
                return obj[key]
    return obj


def load_model(spec: ModelSpec, denoise: float = 0.5,
               progress: ProgressFn | None = None) -> torch.nn.Module:
    """Build the network, load its weights and put it in eval mode.

    denoise: only used by the "general" model. 1.0 = strong denoising,
    0.0 = weak (keeps noise and fine texture). Values in between are a linear
    blend of the two weight sets (the DNI method from Real-ESRGAN).
    """
    weights_path = ensure_weights(spec, progress)
    state = _extract_state(torch.load(weights_path, map_location="cpu", weights_only=True))

    if spec.wdn_url and denoise < 0.999:
        wdn_spec = ModelSpec(
            key=spec.key + "_wdn", label="", filename=spec.wdn_filename,
            url=spec.wdn_url, scale=spec.scale, kind=spec.kind,
            min_bytes=spec.min_bytes,
        )
        wdn_path = ensure_weights(wdn_spec, progress)
        wdn_state = _extract_state(torch.load(wdn_path, map_location="cpu", weights_only=True))
        # Real-ESRGAN DNI: dni_weight = [denoise, 1 - denoise] for [x4v3, wdn].
        # denoise=1 -> pure x4v3 (strong denoising),
        # denoise=0 -> pure wdn (keeps noise and fine texture).
        state = {k: v * denoise + wdn_state[k] * (1.0 - denoise) for k, v in state.items()}

    if spec.kind == "rrdb":
        net = RRDBNet(num_in_ch=3, num_out_ch=3, **spec.params)
    else:
        net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, **spec.params)

    net.load_state_dict(state, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net
