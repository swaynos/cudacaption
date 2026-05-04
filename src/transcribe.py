from __future__ import annotations

import glob
import os
import site
import ctypes
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscribeConfig:
    model_name: str = "medium"
    language: str | None = None
    device: str = "cuda"
    compute_type: str = "float16"
    word_timestamps: bool = False


def _ensure_cuda_library_path() -> None:
    site_packages = site.getsitepackages()
    if not site_packages:
        return
    nvidia_lib_glob = os.path.join(site_packages[0], "nvidia", "*", "lib")
    lib_dirs = [p for p in glob.glob(nvidia_lib_glob) if os.path.isdir(p)]
    if not lib_dirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    existing = current.split(":") if current else []
    merged = lib_dirs + [p for p in existing if p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(merged))

    preload_names = ("libcublas.so.12", "libcudnn.so.9", "libcudart.so.12")
    for lib_dir in lib_dirs:
        for name in preload_names:
            candidate = os.path.join(lib_dir, name)
            if os.path.exists(candidate):
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)


def transcribe_file(audio_path: Path, config: TranscribeConfig) -> list[dict]:
    if config.device == "cuda":
        _ensure_cuda_library_path()

    from faster_whisper import WhisperModel

    model = WhisperModel(
        config.model_name,
        device=config.device,
        compute_type=config.compute_type,
    )

    segments, _ = model.transcribe(
        str(audio_path),
        language=config.language,
        word_timestamps=config.word_timestamps,
    )

    out: list[dict] = []
    for seg in segments:
        out.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
        )
    return out
