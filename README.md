# cudacaption

Local command-line utility for transcribing video files into timestamped text using Whisper, optimized for NVIDIA CUDA GPUs.

## Requirements

- `pyenv` Python `3.11.10`
- `ffmpeg` installed and available on `PATH`
- NVIDIA driver and CUDA-compatible environment, such as an RTX 4050, for acceleration

The project installs CUDA user-space runtime libraries (cuBLAS/cuDNN) via pip so GPU
inference works without manual system library path setup.

## Setup

```bash
pyenv install 3.11.10
pyenv local 3.11.10
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For local git safeguards (lint/format/secret scan hooks):

```bash
pip install -e ".[dev]"
pre-commit install
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
```

## Output

For an input file `myvideo.mp4`, outputs are written as:

- `myvideo.srt`
- `myvideo.vtt`
- `myvideo.json`
- `myvideo.txt`

By default outputs go beside the input file, unless `--output-dir` is provided.

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
