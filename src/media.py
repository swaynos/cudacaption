from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not found on PATH")


def extract_audio_wav(video_path: Path, wav_path: Path) -> None:
    ensure_ffmpeg_available()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(wav_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg audio extraction failed: {msg}")
