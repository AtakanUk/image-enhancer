"""Inference engine: tiled upscaling + target-resolution planning."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch

from .models import REGISTRY, ModelSpec, load_model

TileFn = Callable[[int, int], None]      # (tiles_done, tiles_total)
PostAiFn = Callable[[np.ndarray, int], np.ndarray]  # (image, factor) -> image


# ---------------------------------------------------------------- planning

@dataclass(frozen=True)
class Plan:
    """Describes how to get from the source size to the target size."""
    src_w: int
    src_h: int
    out_w: int
    out_h: int
    passes: int          # how many times the model runs
    model_scale: int
    downscale: bool      # whether the model output is reduced to the target

    @property
    def ai_w(self) -> int:
        return self.src_w * self.model_scale ** self.passes

    @property
    def ai_h(self) -> int:
        return self.src_h * self.model_scale ** self.passes

    def describe(self) -> str:
        if self.passes == 0:
            return (f"{self.src_w}x{self.src_h} -> {self.out_w}x{self.out_h} "
                    "(no AI needed, resize only)")
        chain = f"AI x{self.model_scale}" + (f", {self.passes} passes" if self.passes > 1 else "")
        tail = ""
        if (self.ai_w, self.ai_h) != (self.out_w, self.out_h):
            direction = "downscale" if self.downscale else "stretch"
            tail = f" -> {self.out_w}x{self.out_h} {direction}"
        return (f"{self.src_w}x{self.src_h} -> {chain} -> "
                f"{self.ai_w}x{self.ai_h}{tail}")


# Running the model one more time only pays off if the stretch left after the
# last pass exceeds this factor. The reason was measured (see README, "How many
# passes"): as soon as the model processes its own output it leaves the
# distribution it was trained on and hallucinates on top of hallucinated
# detail, so PSNR and SSIM dropped in every test. With this threshold, jobs
# like 480p -> 4K finish in a single pass instead of producing a 16x
# intermediate image: more accurate and ~40x faster.
PASS_TOLERANCE = 1.5


def plan_resize(src_w: int, src_h: int, target_w: int, target_h: int,
                model_scale: int, max_passes: int = 3,
                even_dims: bool = False,
                tolerance: float = PASS_TOLERANCE) -> Plan:
    """Work out how many model passes are needed to reach the target.

    If target_w <= 0 it is derived from target_h, keeping the aspect ratio
    (and vice versa).
    """
    if target_w <= 0:
        target_w = round(src_w * target_h / src_h)
    if target_h <= 0:
        target_h = round(src_h * target_w / src_w)
    if even_dims:
        target_w += target_w % 2
        target_h += target_h % 2

    needed = max(target_w / src_w, target_h / src_h)
    passes, reached = 0, 1.0
    while passes < max_passes:
        remaining = needed / reached
        # The first pass only requires *some* upscaling: even for small
        # factors it is worth it because the model removes compression
        # artifacts.
        limit = 1.0 if passes == 0 else tolerance
        if remaining <= limit + 1e-6:
            break
        reached *= model_scale
        passes += 1

    return Plan(src_w, src_h, target_w, target_h, passes, model_scale,
                downscale=(reached > needed + 1e-6) or passes == 0)


# ------------------------------------------------------------------ engine

class Upscaler:
    # The tile overlap must cover the network's receptive field; too little
    # leaves visible seams at tile borders. The values were picked by
    # measurement (see README, "Tile overlap"): with them the tiled output is
    # indistinguishable from a single-pass run (max diff <= 5/255).
    _PAD_BY_KIND = {"srvgg": 32, "rrdb4": 56, "rrdb2": 128}

    def __init__(self, model_key: str, denoise: float = 0.5, device: str = "cuda",
                 half: bool = True, tile: int = 512, tile_pad: int = -1,
                 ensemble: bool = False, progress=None):
        self.spec: ModelSpec = REGISTRY[model_key]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.type == "cuda"
        self.tile = tile
        self.tile_pad = tile_pad if tile_pad >= 0 else self._default_pad()
        self.ensemble = ensemble
        self.scale = self.spec.scale

        net = load_model(self.spec, denoise=denoise, progress=progress)
        self.net = net.to(self.device)
        if self.half:
            self.net = self.net.half()

    # -- internal helpers -------------------------------------------------

    def _default_pad(self) -> int:
        """Tile overlap based on the model's receptive field.

        RRDBNet x2 halves the input with pixel_unshuffle first, so its
        receptive field in source pixels is twice as large and it needs the
        widest overlap.
        """
        if self.spec.kind != "rrdb":
            return self._PAD_BY_KIND["srvgg"]
        return self._PAD_BY_KIND["rrdb2" if self.spec.params.get("scale", 4) == 2
                                 else "rrdb4"]

    @property
    def _mod(self) -> int:
        """Number the input size must be divisible by.

        RRDBNet x2/x1 shrink the input with pixel_unshuffle, which requires
        divisible dimensions. The other models accept any size.
        """
        if self.spec.kind == "rrdb":
            return {1: 4, 2: 2}.get(self.spec.params.get("scale", 4), 1)
        return 1

    def _forward_whole(self, t: torch.Tensor) -> torch.Tensor:
        mod = self._mod
        ph = pw = 0
        if mod > 1:
            _, _, h, w = t.shape
            ph, pw = (mod - h % mod) % mod, (mod - w % mod) % mod
            if ph or pw:
                t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")

        out = self.net(t)
        if ph or pw:
            h, w = out.shape[-2:]
            out = out[:, :, : h - ph * self.scale, : w - pw * self.scale]
        return out

    def _forward_tiled(self, t: torch.Tensor, on_tile: TileFn | None) -> torch.Tensor:
        b, c, h, w = t.shape
        s, tile, pad = self.scale, self.tile, self.tile_pad
        out = torch.empty((b, c, h * s, w * s), dtype=t.dtype, device=t.device)

        nx, ny = math.ceil(w / tile), math.ceil(h / tile)
        total, done = nx * ny, 0
        for yi in range(ny):
            for xi in range(nx):
                ix0, ix1 = xi * tile, min((xi + 1) * tile, w)
                iy0, iy1 = yi * tile, min((yi + 1) * tile, h)
                # Take extra context around the tile; this overlap prevents seams.
                px0, px1 = max(ix0 - pad, 0), min(ix1 + pad, w)
                py0, py1 = max(iy0 - pad, 0), min(iy1 + pad, h)

                patch = self._forward_whole(t[:, :, py0:py1, px0:px1])

                ox0, oy0 = (ix0 - px0) * s, (iy0 - py0) * s
                out[:, :, iy0 * s:iy1 * s, ix0 * s:ix1 * s] = patch[
                    :, :, oy0:oy0 + (iy1 - iy0) * s, ox0:ox0 + (ix1 - ix0) * s
                ]
                done += 1
                if on_tile:
                    on_tile(done, total)
        return out

    def _forward(self, t: torch.Tensor, on_tile: TileFn | None) -> torch.Tensor:
        try:
            return (self._forward_whole(t) if self.tile <= 0
                    else self._forward_tiled(t, on_tile))
        except torch.cuda.OutOfMemoryError:
            # Out of VRAM: shrink the tile and try once more.
            torch.cuda.empty_cache()
            self.tile = max(128, (self.tile if self.tile > 0 else 1024) // 2)
            return self._forward_tiled(t, on_tile)

    # -- public API -------------------------------------------------------

    def _run_passes(self, t: torch.Tensor, plan: Plan,
                    on_tile: TileFn | None) -> torch.Tensor:
        """Run the model as many times as the plan says.

        With self-ensemble enabled, each pass runs separately on the 8
        dihedral transforms of the image (4 rotations x mirror); the results
        are transformed back and averaged. This cancels the model's
        orientation bias: wrongly hallucinated details do not line up across
        the variants and fade out, while real structure is reinforced. The
        cost is 8x the time.
        """
        for _ in range(plan.passes):
            if not self.ensemble:
                t = self._forward(t, on_tile).clamp_(0, 1)
                continue

            acc = None
            for i in range(8):
                cb = None
                if on_tile:
                    def cb(done, total, _i=i):
                        on_tile(_i * total + done, total * 8)
                y = _unaugment(self._forward(_augment(t, i), cb), i)
                y = y.float().clamp_(0, 1)
                acc = y if acc is None else acc.add_(y)
            t = acc.div_(8.0)
            if self.half:
                t = t.half()
        return t

    @torch.inference_mode()
    def process(self, img: np.ndarray, plan: Plan,
                on_tile: TileFn | None = None,
                post_ai: PostAiFn | None = None) -> np.ndarray:
        """Upscale a BGR image (uint8/uint16, optional alpha) according to the plan.

        If post_ai is given, it is called after the model passes but before
        the final resize to the target: it receives (image, total_factor) and
        returns the modified image. Face restoration uses this hook because
        the faces must be pasted onto a canvas with an integer scale factor.
        """
        alpha = None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            alpha, img = img[:, :, 3], img[:, :, :3]

        src_dtype = img.dtype
        max_val = 65535.0 if src_dtype == np.uint16 else 255.0

        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        t = t.unsqueeze(0).to(self.device)
        t = (t.half() if self.half else t.float()).div_(max_val)

        t = self._run_passes(t, plan, on_tile)

        if post_ai is not None:
            arr = self._to_bgr8(t)
            arr = post_ai(arr, self.scale ** plan.passes)
            t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
            t = t.unsqueeze(0).to(self.device)
            t = (t.half() if self.half else t.float()).div_(255.0)
            max_val = 255.0
            src_dtype = np.uint8      # GFPGAN produces 8-bit output

        t = self._fit(t, plan)
        arr = t.float().mul_(max_val).add_(0.5).clamp_(0, max_val)
        out = arr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(src_dtype)

        if alpha is not None:
            a = cv2.resize(alpha, (plan.out_w, plan.out_h),
                           interpolation=cv2.INTER_LANCZOS4)
            if a.dtype != out.dtype:
                a = (a / 257).astype(out.dtype) if a.dtype == np.uint16 \
                    else (a.astype(np.uint16) * 257)
            out = np.dstack([out, a])
        return out

    @staticmethod
    def _to_bgr8(t: torch.Tensor) -> np.ndarray:
        arr = t.float().clamp(0, 1).mul_(255.0).round_()
        return arr.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()

    @torch.inference_mode()
    def process_batch(self, frames: np.ndarray, plan: Plan) -> np.ndarray:
        """For video: (N,H,W,3) uint8 stack -> (N,out_h,out_w,3) uint8."""
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device)
        t = (t.half() if self.half else t.float()).div_(255.0)

        t = self._run_passes(t, plan, None)

        t = self._fit(t, plan)
        t = t.float().mul_(255.0).round_().clamp_(0, 255).to(torch.uint8)
        return t.permute(0, 2, 3, 1).cpu().numpy()

    def _fit(self, t: torch.Tensor, plan: Plan) -> torch.Tensor:
        """Bring the model output to the exact target size."""
        if t.shape[-1] == plan.out_w and t.shape[-2] == plan.out_h:
            return t
        if plan.out_w < t.shape[-1]:
            # Downscaling: area averaging gives the cleanest result (no moire).
            out = torch.nn.functional.interpolate(
                t.float(), size=(plan.out_h, plan.out_w), mode="area")
        else:
            out = torch.nn.functional.interpolate(
                t.float(), size=(plan.out_h, plan.out_w),
                mode="bicubic", align_corners=False)
        return out.clamp_(0, 1)

    def release(self) -> None:
        self.net = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _augment(t: torch.Tensor, i: int) -> torch.Tensor:
    """The i-th of the 8 dihedral transforms of the image (D4 group)."""
    if i & 4:
        t = torch.flip(t, [-1])
    k = i & 3
    return torch.rot90(t, k, [-2, -1]) if k else t


def _unaugment(t: torch.Tensor, i: int) -> torch.Tensor:
    """Inverse of _augment: undoes the operations in reverse order."""
    k = i & 3
    if k:
        t = torch.rot90(t, -k, [-2, -1])
    return torch.flip(t, [-1]) if i & 4 else t


def auto_tile(src_w: int, src_h: int, model_kind: str) -> int:
    """Pick a sensible tile size from free VRAM and the image size.

    Returns 0 to process the image in one piece (fastest, no seams).
    """
    if not torch.cuda.is_available():
        return 192

    free_bytes, _ = torch.cuda.mem_get_info()
    gb = free_bytes / (1 << 30)
    ladder = {
        "rrdb": [(18, 1024), (10, 768), (6, 512), (3, 384)],
        "srvgg": [(12, 2048), (8, 1536), (5, 1024), (3, 640)],
    }[model_kind]

    tile = 256
    for need, candidate in ladder:
        if gb >= need:
            tile = candidate
            break
    # No need to tile at all if the image is already smaller than one tile.
    return 0 if max(src_w, src_h) <= tile else tile
