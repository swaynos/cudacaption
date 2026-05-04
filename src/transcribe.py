from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass
class TranscribeConfig:
    model_name: str = "medium"
    language: str | None = None
    device: str = "cuda"
    compute_type: str = "float16"
    word_timestamps: bool = False


def transcribe_file(audio_path: Path, config: TranscribeConfig) -> list[dict]:
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
