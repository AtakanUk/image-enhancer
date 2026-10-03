"""Face restoration stage (GFPGAN v1.4).

Upscaling models treat the whole frame the same way, but faces are small,
dense in detail, and the first thing the eye goes to. This stage handles them
separately:

  1. Faces are detected on the *source* image and aligned to 512x512 from 5
     landmarks (facexlib / RetinaFace).
  2. Each crop goes through GFPGAN, a generative network trained
     specifically on faces, which reconstructs it.
  3. The result is pasted onto the *upscaled* background through an inverse
     affine transform, blended softly with a face parsing mask.

Detection runs on the source image because the alignment landmarks are
defined there, while pasting happens on the upscaled canvas. That is why the
upscale factor has to be an integer (the model's own factor); the final
resize to the target happens after this stage.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch

from .arch.gfpgan_clean import GFPGANv1Clean
from .models import FACE_SPEC, MODEL_DIR, ensure_weights

ProgressFn = Callable[[str, float], None]

# The configuration GFPGAN v1.4 was trained with.
_ARCH_KWARGS = dict(
    out_size=512, num_style_feat=512, channel_multiplier=2,
    decoder_load_path=None, fix_decoder=False, num_mlp=8,
    input_is_latent=True, different_w=True, narrow=1, sft_half=True,
)


class FaceEnhancer:
    def __init__(self, device: torch.device, progress: ProgressFn | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self.device = device
        self.log = log or (lambda _: None)

        weights = ensure_weights(FACE_SPEC, progress)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = state.get("params_ema", state)

        net = GFPGANv1Clean(**_ARCH_KWARGS)
        net.load_state_dict(state, strict=True)
        # The StyleGAN2 decoder can be unstable in FP16, so the face stage
        # always runs in FP32. Only 512x512 crops are processed, so the cost
        # is low.
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

        self._helper = None
        self._helper_scale = None

    # -- internal helpers -------------------------------------------------

    def _get_helper(self, upscale: int):
        """FaceRestoreHelper is expensive to build; reuse it while the factor is unchanged."""
        if self._helper is None or self._helper_scale != upscale:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            self._helper = FaceRestoreHelper(
                upscale, face_size=512, crop_ratio=(1, 1),
                det_model="retinaface_resnet50", save_ext="png",
                use_parse=True, device=self.device,
                model_rootpath=str(MODEL_DIR),
            )
            self._helper_scale = upscale
        return self._helper

    @torch.inference_mode()
    def _restore_crop(self, face_bgr: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(
            cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)))
        t = t.permute(2, 0, 1).float().div_(255.0).sub_(0.5).div_(0.5)
        out = self.net(t.unsqueeze(0).to(self.device), return_rgb=False)[0]
        out = out.squeeze(0).clamp_(-1, 1).add_(1).mul_(127.5)
        arr = out.round_().clamp_(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # -- public API -------------------------------------------------------

    def enhance(self, source_bgr: np.ndarray, background_bgr: np.ndarray,
                upscale: int, strength: float = 1.0) -> tuple[np.ndarray, int]:
        """Restore the faces and paste them onto the upscaled background.

        source_bgr:     source image the faces are detected on (original size)
        background_bgr: upscaled image the faces are pasted onto
        upscale:        integer factor between the two
        strength:       1.0 full restoration, 0.0 untouched (blend in between)

        Returns (result, number_of_faces). Without faces the background is
        returned unchanged.
        """
        helper = self._get_helper(upscale)
        helper.clean_all()
        helper.read_image(source_bgr)

        n_faces = helper.get_face_landmarks_5(
            only_center_face=False, eye_dist_threshold=5)
        if not n_faces:
            return background_bgr, 0

        helper.align_warp_face()
        for cropped in helper.cropped_faces:
            restored = self._restore_crop(cropped)
            if strength < 0.999:
                # Blend the restored face with the original crop: lower
                # values stay closer to the person's actual features.
                restored = cv2.addWeighted(restored, strength,
                                           cropped, 1.0 - strength, 0.0)
            helper.add_restored_face(restored)

        helper.get_inverse_affine(None)
        result = helper.paste_faces_to_input_image(upsample_img=background_bgr)
        return result, len(helper.cropped_faces)

    def release(self) -> None:
        self.net = None
        self._helper = None
        self._helper_scale = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
