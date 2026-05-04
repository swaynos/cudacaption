from __future__ import annotations

import json
from pathlib import Path


def format_timestamp(seconds: float, decimal_marker: str = ",") -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1_000
    millis = total_ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"


def write_json(segments: list[dict], output_path: Path) -> None:
    output_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")


def write_txt(segments: list[dict], output_path: Path) -> None:
    lines = [seg["text"].strip() for seg in segments if seg.get("text")]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(segments: list[dict], output_path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(float(seg["start"]), decimal_marker=",")
        end = format_timestamp(float(seg["end"]), decimal_marker=",")
        text = str(seg["text"]).strip()
        lines.extend([str(i), f"{start} --> {end}", text, ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[dict], output_path: Path) -> None:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = format_timestamp(float(seg["start"]), decimal_marker=".")
        end = format_timestamp(float(seg["end"]), decimal_marker=".")
        text = str(seg["text"]).strip()
        lines.extend([f"{start} --> {end}", text, ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
