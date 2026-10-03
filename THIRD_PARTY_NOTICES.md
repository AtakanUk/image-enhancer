# Third-party notices

The network architectures in `src/core/arch/` are derived from the projects below, both licensed under the Apache License 2.0. Each file's docstring states where it comes from and what was changed.

| File | Source | Changes |
| --- | --- | --- |
| `rrdbnet.py` | [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR): `basicsr/archs/rrdbnet_arch.py` | Condensed into a standalone, inference-only module (no basicsr imports, registry or weight initialisation) |
| `srvgg.py` | [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR): `basicsr/archs/srvgg_arch.py` (Real-ESRGAN) | Condensed into a standalone, inference-only module (no basicsr import or registry) |
| `gfpgan_clean.py` | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN): `gfpgan/archs/gfpganv1_clean_arch.py` | basicsr import and registry decorator removed |
| `stylegan2_clean.py` | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN): `gfpgan/archs/stylegan2_clean_arch.py` | basicsr imports and registry decorator removed; `default_init_weights` inlined |

## Copyright and license texts

- **BasicSR**: Copyright 2018-2022 BasicSR Authors. Apache License 2.0, full text in [`LICENSES/BasicSR.txt`](LICENSES/BasicSR.txt).
- **GFPGAN**: Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved. Apache License 2.0, full text (including the licenses of its third-party components) in [`LICENSES/GFPGAN.txt`](LICENSES/GFPGAN.txt). GFPGAN's license lists its StyleGAN2 code as modified from [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) (MIT, Copyright (c) 2019 Kim Seonghyeon). The "clean" variant used here does not include the custom CUDA kernels that fall under the NVIDIA license.

## Model weights

Pretrained weights (Real-ESRGAN, GFPGAN, facexlib) are not included in this repository. They are downloaded at runtime from the authors' official releases and remain under their respective licenses.
