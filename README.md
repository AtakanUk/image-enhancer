# Image Enhancer

A desktop app that upscales photos and videos with AI to a resolution you choose: 720p to 4K, 8K, a multiplier, or a custom size. Built with PySide6 and PyTorch on top of Real-ESRGAN (upscaling) and GFPGAN (face restoration) weights.

**Stack:** Python 3.12 · PyTorch (CUDA, FP16) · PySide6 · OpenCV · facexlib · ffmpeg

---

## Highlights

- **No `basicsr` dependency.** `basicsr` breaks on torchvision ≥ 0.17, so the network architectures (RRDBNet, SRVGGNetCompact, GFPGANv1Clean, StyleGAN2) are kept as standalone modules whose layer names match the official weights.
- **Target-resolution planning.** Models scale by fixed factors (×2/×4); the planner picks the number of passes and super-samples down to the exact target. Measurements showed that a *second* model pass almost always hurts (4.5× upscale: one pass = 36.61 dB PSNR in 0.27 s vs. two passes = 34.16 dB in 11.2 s), so an extra pass is used only when the remaining stretch exceeds 1.5×.
- **Seam-free tiled inference.** The tile overlap for each architecture was chosen by comparing tiled and single-pass output (max diff ≤ 5/255). On CUDA OOM the tile size is halved and the job retried automatically.
- **Face restoration (photos).** Faces are detected on the *source* image, aligned to 512×512 (RetinaFace landmarks), restored with GFPGAN v1.4, and pasted back onto the upscaled image through an inverse affine transform and a soft parsing mask. A strength slider blends restored and original faces.
- **×8 self-ensemble.** Runs the 8 dihedral transforms and averages them: +0.9 dB PSNR at 2.5–7.5× the cost.
- **Streaming video pipeline.** Frames are piped from ffmpeg as raw BGR straight into the encoder. No frames are written to disk, so disk usage stays constant even for hours of video. NVENC or libx265; audio is stream-copied (or converted to AAC when mp4 can't carry it).
- **Noise control** via DNI: linear interpolation between two weight sets of the general model.
- Weights download on first use (only the models you actually pick).

## Measured results

RTX 4090, FP16. A 4K photo was downscaled to 720p, upscaled back to 4K and compared with the original:

| Method | PSNR | SSIM | Sharpness | Time |
| --- | --- | --- | --- | --- |
| Lanczos (no AI) | 43.90 | 0.982 | 1.7 | — |
| General model | 40.55 | 0.973 | 14.0 | 0.18 s |
| Photo ×4 model | 38.11 | 0.965 | 13.0 | 1.13 s |
| **Original 4K** | — | — | **17.6** | — |

PSNR rewards blur. Lanczos gets the highest score but has a tenth of the original's sharpness. The AI models get close to the original's texture density; since the detail they generate does not match the original pixel for pixel, their PSNR is lower, but the difference is obvious to the eye.

Video speed (720p → 4K, General model): **~8.6 fps**, so a 5-minute 30 fps clip takes about 17 minutes.

## Getting started

### Requirements

- Python 3.12 (developed with `torch 2.6.0+cu124`)
- PyTorch with CUDA. A GPU is not required, but processing on the CPU is very slow.
- ffmpeg and ffprobe on `PATH` (needed for video)

### Install and run

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python main.py          # or run.bat on Windows
```

Model weights are downloaded into `models/` the first time they are needed, and only the ones you use. Upscaling models are 2–64 MB each; face restoration additionally needs GFPGAN (333 MB), the face detector (104 MB) and the face parser (81 MB).

## Usage

1. Drag and drop photos or videos onto the file list (folders work too; all media files inside are picked up).
2. Choose a **target resolution**: a preset (720p … 8K), a multiplier (×2, ×4 …) or a custom size.
3. Choose a **model**. If unsure, keep *Auto*.
4. **Face restoration** is on by default for photos; adjust its strength if needed.
5. Press **START**. Results go to `output/` unless you pick another folder.

The panel under the file list shows exactly what will happen to the selected file before you start:

```text
holiday.jpg
1280x720 -> AI x4 -> 5120x2880 -> 3840x2160 downscale
Model: Photo x4 — Highest detail (slow)
```

## Models

| Model | Best for |
| --- | --- |
| **Auto** | `Photo x4` for photos, `General` for videos. Recommended. |
| **General** | Any content; balanced and fast. Has a denoise control. Default for video. |
| **Photo x4** | Highest detail on real photos. Too slow for video. |
| **Photo x2** | Most natural result for upscales close to 2×, such as 1080p → 4K. |
| **Anime / Illustration x4** | Cartoons, anime, illustrations. |
| **Anime video** | Fastest option for anime videos. |

**Denoise** (General model only): `0` keeps fine texture and film grain, `1` removes noise aggressively. It is a linear blend of two weight sets (Real-ESRGAN's DNI method).

## How it works

### Reaching the target

The models upscale by fixed factors (×2 or ×4). When the target does not line up with those factors, the app goes one step further and scales down to the target. For 720p → 4K it first produces 5120×2880 and then reduces it to 3840×2160. This super-sampling gives a cleaner result than upscaling to the target directly.

If the target is smaller than the source, the AI is skipped, the image is only resized, and the log shows a warning.

### How many passes

Running the model a **second time** gains almost nothing, contrary to what one might expect. Measured on 324×480 → 2160p (a 4.5× upscale, RTX 4090):

| Path | PSNR | SSIM | Time |
| --- | --- | --- | --- |
| 1 pass → 1296×1920 → Lanczos stretch | **36.61** | **0.9516** | **0.27 s** |
| 2 passes → 5184×7680 → downscale | 34.16 | 0.9476 | 11.20 s |

The same happened at 5×, 6×, 7× and 8×: a single pass won every time. In the second pass the model processes its own output, which is outside the "degraded image" distribution it was trained on, so it hallucinates detail on top of hallucinated detail.

This test used a Lanczos stretch, while the app does that final stretch on the GPU with bicubic interpolation. A re-check on four public-domain test images (4.5×, Photo x4 and General models) put bicubic within 0.07 dB of Lanczos, and one pass beat two passes in all eight cases, by 0.3–1.3 dB.

An extra pass is therefore only used when the stretch left after the last pass **exceeds 1.5×**. In practice: 720p → 4K is one pass, 480p → 4K is one pass (it used to build a 16× intermediate image and was ~40× slower), and 240p → 4K is two passes.

### Face restoration

The upscaling models treat the whole frame the same way, but faces are small, dense in detail and the first thing the eye goes to. This stage handles them separately:

1. Faces are detected on the **source** image and aligned to 512×512 from 5 landmarks (RetinaFace).
2. Each crop goes through a generative network trained specifically on faces (GFPGAN v1.4).
3. The result is pasted onto the **upscaled** image through an inverse affine transform, blended softly with a face parsing mask.

The **strength** slider blends the restored face with the original: lower values stay closer to the person's real features, higher values give a crisper but more "generated" face. The default is 0.80. It takes about 1 second per face on an RTX 4090. If no face is found, the image is left as is.

It is not applied to video: every frame is reconstructed independently, which causes flicker between frames.

### ×8 self-ensemble

Processes the image in its 8 dihedral orientations (4 rotations × mirror), transforms the results back and averages them. This cancels the model's orientation bias: wrongly hallucinated details do not line up across the variants and fade, while real structure is reinforced.

| Model | Standard | ×8 self-ensemble | Time cost |
| --- | --- | --- | --- |
| General | 40.55 dB | **41.43 dB** | ~2.5× |
| Photo x4 | 38.11 dB | **38.98 dB** | ~7.5× |

+0.9 dB is a significant gain in super-resolution. It is not applied to video because it multiplies the time by 8.

### Tile overlap

Tiles are processed with an overlap; if the overlap does not cover the network's receptive field, visible seams remain at the tile borders. The values were chosen by measurement, comparing against a single-pass (1×1) output, as the max difference out of 255:

| Model type | Overlap | Tiled vs. single pass |
| --- | --- | --- |
| SRVGG (General, Anime video) | 32 | ≤ 1 |
| RRDBNet ×4 (Photo x4, Anime x4) | 56 | ≤ 5 |
| RRDBNet ×2 (Photo x2) | 128 | ≤ 2 |

RRDBNet ×2 needs the widest overlap: it first halves the input with `pixel_unshuffle`, which doubles its receptive field in source pixels.

### Video

- **Codec**: the NVENC (GPU) options are fast; `libx265` gives smaller files but is very slow on the CPU.
- **Quality (CQ/CRF)**: lower number = higher quality and bigger file. 20 is a good starting point.
- **Audio**: copied as is by default (no quality loss). If the mp4 container can't carry the audio codec, it is converted to AAC automatically.
- **Only the first N seconds**: try your settings on a short clip before committing to a long video.

Frames are never written to disk: they are read from the decoder as raw BGR and passed straight to the encoder, so disk usage stays constant even for a video that takes hours.

## Running out of GPU memory

If the log shows `CUDA out of memory`, lower **Advanced → Tile size** (512 → 384 → 256). The tile size does not change the result; it only splits the work into pieces. On an out-of-memory error the app also halves the tile size by itself and retries once.

## Known limitations

- 10-bit and HDR video is processed as 8-bit SDR (the pipeline is `bgr24`).
- Variable frame rate (VFR) video is converted to a constant frame rate.
- Subtitles and secondary audio tracks are not carried over.
- Face restoration and ×8 self-ensemble are photo-only (because of flicker and the 8× cost, respectively).
- Face restoration works in 8-bit; with a 16-bit source the output becomes 8-bit (noted in the log).

## Project structure

```text
main.py                     entry point
run.bat                     Windows launcher
src/core/
  arch/rrdbnet.py           RRDBNet (x4plus, x2plus, anime_6B)
  arch/srvgg.py             SRVGGNetCompact (general, animevideo)
  arch/gfpgan_clean.py      GFPGANv1Clean, face restoration network
  arch/stylegan2_clean.py   StyleGAN2 decoder used by GFPGAN
  models.py                 model registry, downloader, DNI blending
  upscaler.py               tiled inference + target-resolution planning
  faces.py                  detect → restore → soft paste-back
  ffmpeg.py                 ffprobe/ffmpeg wrappers, codec presets
  jobs.py                   photo and video pipelines
src/app/
  main_window.py            PySide6 UI
  worker.py                 background worker thread
models/                     downloaded weights (not committed)
```

## License

MIT for this project's code. The files in `src/core/arch/` are adapted from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)/[BasicSR](https://github.com/XPixelGroup/BasicSR) (Apache-2.0) and [GFPGAN](https://github.com/TencentARC/GFPGAN) (Apache-2.0); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are downloaded from their official releases and are subject to their own licenses.
