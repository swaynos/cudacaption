from __future__ import annotations

import re
import subprocess
from pathlib import Path

from media import ensure_ffmpeg_available


SHOWINFO_RE = re.compile(r"pts_time:(?P<pts>[0-9]+(?:\.[0-9]+)?)")


def _extract_with_filter(video_path: Path, frames_dir: Path, vf: str) -> list[dict]:
    ensure_ffmpeg_available()
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frames_dir / "frame_%05d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"{vf},showinfo",
        "-vsync",
        "vfr",
        str(output_pattern),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg keyframe extraction failed: {msg}")

    timestamps: list[float] = []
    for line in proc.stderr.splitlines():
        match = SHOWINFO_RE.search(line)
        if match:
            timestamps.append(float(match.group("pts")))

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    count = min(len(frame_paths), len(timestamps))
    out: list[dict] = []
    for idx in range(count):
        out.append({"timestamp": timestamps[idx], "frame_path": str(frame_paths[idx])})
    return out


def extract_keyframes(
    video_path: Path,
    frames_dir: Path,
    mode: str = "scene",
    interval_sec: float = 15.0,
    scene_threshold: float = 0.35,
    max_keyframes: int = 40,
) -> list[dict]:
    if mode == "scene":
        vf = f"select='gt(scene,{scene_threshold})'"
        frames = _extract_with_filter(video_path, frames_dir, vf)
        if not frames:
            frames = _extract_with_filter(
                video_path, frames_dir, f"fps=1/{max(interval_sec, 0.1)}"
            )
    else:
        frames = _extract_with_filter(
            video_path, frames_dir, f"fps=1/{max(interval_sec, 0.1)}"
        )

    return frames[: max(1, max_keyframes)]
