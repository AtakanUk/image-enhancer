"""SRVGGNetCompact, the generator behind realesr-general-x4v3 / realesr-animevideov3.

Much lighter than RRDBNet; this is the default model for video.

Adapted from BasicSR (basicsr/archs/srvgg_arch.py), Apache-2.0,
https://github.com/XPixelGroup/BasicSR. Modified: condensed into a standalone,
inference-only module without the basicsr dependency (no registry).

Layer names must match the official weights.
"""
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16,
                 upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale

        def activation():
            if act_type == "relu":
                return nn.ReLU(inplace=True)
            if act_type == "prelu":
                return nn.PReLU(num_parameters=num_feat)
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(activation())
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(activation())
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # The learned residual is added on top of a nearest-neighbour upscaled base.
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")
