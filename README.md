# cudacaption

Local command-line utility for transcribing video files into timestamped text using Whisper, optimized for NVIDIA CUDA GPUs.

## Requirements

- `pyenv` Python `3.11.10`
- `ffmpeg` installed and available on `PATH`
- NVIDIA driver and CUDA-compatible environment, such as an RTX 4050, for acceleration

The project installs CUDA user-space runtime libraries (cuBLAS/cuDNN) via pip so GPU
inference works without manual system library path setup.

## Tested Stack

This project was validated on the following stack:

- OS: Linux
- Python: 3.11.10
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- NVIDIA driver: 590.48.01
- CUDA (driver-reported): 13.1
- ffmpeg: available on PATH
- Core packages:
  - faster-whisper 1.2.1
  - ctranslate2 4.7.1
  - nvidia-cublas-cu12 12.9.2.10
  - nvidia-cudnn-cu12 9.21.1.3
  - nvidia-cuda-runtime-cu12 12.9.79
- Vision path tested with:
  - transformers 4.46.3
  - torch 2.11.0
  - timm 1.0.26
  - einops 0.8.2

## Setup

```bash
pyenv install 3.11.10
pyenv local 3.11.10
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Usage

```bash
cudacaption ./myvideo.mp4
```

Optional flags:

```bash
cudacaption ./myvideo.mp4 --model medium --language en --output-dir ./transcripts
cudacaption ./myvideo.mp4 --cpu
cudacaption ./myvideo.mp4 --word-timestamps
cudacaption ./myvideo.mp4 --extract-keyframes --keyframe-mode scene --max-keyframes 40
cudacaption ./myvideo.mp4 --extract-keyframes --keyframe-mode interval --keyframe-interval-sec 30 --max-keyframes 12
cudacaption ./myvideo.mp4 --extract-keyframes --vision-model microsoft/Florence-2-base --vision-prompt-profile slides
cudacaption ./myvideo.mp4 --extract-keyframes --vision-model Salesforce/blip-image-captioning-base
```

Keyframe/vision flags:

- `--extract-keyframes` enable sparse frame extraction
- `--keyframe-mode` `scene` or `interval` (scene falls back to interval when no scene cuts are found)
- `--keyframe-interval-sec` sampling interval used by interval mode
- `--scene-threshold` scene-cut sensitivity for scene mode
- `--max-keyframes` cap number of processed frames
- `--vision-model` optional image-to-text model name
- `--vision-prompt-profile` `slides`, `meeting`, or `code-demo`

## Output

For an input file `myvideo.mp4`, outputs are written as:

- `myvideo.srt`
- `myvideo.vtt`
- `myvideo.json`
- `myvideo.txt`

By default outputs go beside the input file, unless `--output-dir` is provided.

When `--extract-keyframes` is enabled, additional artifacts are written:

- `myvideo.visual.json` (keyframe metadata or visual captions)
- `myvideo.timeline.json` (merged audio + visual events)
- `myvideo.keyframes/` (persisted extracted frame images)

When `--vision-model` is set:

- `myvideo.visual.json` includes per-frame `caption` and model-specific `ocr_text` (Florence-2 path)
- `myvideo.timeline.json` merges audio segments and visual caption events

Notes:

- `microsoft/Florence-2-base` provides richer slide-focused output and OCR-like text extraction.
- `Salesforce/blip-image-captioning-base` is supported and runs successfully, but captions are generally less precise for dense slide text.

## Git Workflow

- Commit messages follow conventional style, for example:
  - `feat: add language override flag`
  - `fix: improve CUDA runtime error messaging`
  - `docs: clarify GPU setup steps`
- Pre-commit hooks are configured in `.pre-commit-config.yaml` and run automatically after installation.
- Transcript output formats (`*.json`, `*.srt`, `*.vtt`, `*.txt`) are git-ignored to prevent accidental commits.

## Branch Protection (GitHub)

Recommended `main` branch protection settings:

- Require a pull request before merging
- Require at least 1 approval
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Block force pushes and branch deletion
- Include administrators in enforcement
