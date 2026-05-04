# cudacaption

Local command-line utility for transcribing video files into timestamped text using Whisper, optimized for NVIDIA CUDA GPUs.

## Requirements

- `pyenv` Python `3.11.10`
- `ffmpeg` installed and available on `PATH`
- NVIDIA driver and CUDA-compatible environment for RTX 4050 acceleration

## Setup

```bash
pyenv install 3.11.10
pyenv local 3.11.10
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
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
