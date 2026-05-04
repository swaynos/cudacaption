from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

from media import extract_audio_wav
from keyframes import extract_keyframes
from transcribe import TranscribeConfig, transcribe_file
from writers import write_json, write_srt, write_txt, write_unified_json, write_vtt
from vision import analyze_keyframes


def format_runtime_error(exc: Exception, device: str) -> str:
    message = str(exc)
    if device == "cuda" and (
        "libcublas" in message.lower()
        or "cuda" in message.lower()
        or "cudnn" in message.lower()
    ):
        return (
            "CUDA runtime is unavailable. Install compatible NVIDIA/CUDA libraries "
            "for ctranslate2, or re-run with --cpu.\n"
            f"Original error: {message}"
        )
    return message


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cudacaption",
        description="Transcribe a video file into timestamped outputs using Whisper.",
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument(
        "--model", default="medium", help="Whisper model (default: medium)"
    )
    parser.add_argument("--language", default=None, help="Language hint, e.g. en")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: input file directory)",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Enable word-level timestamps during transcription",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU instead of CUDA GPU",
    )
    parser.add_argument(
        "--extract-keyframes", action="store_true", help="Extract sparse keyframes"
    )
    parser.add_argument(
        "--keyframe-mode", choices=["scene", "interval"], default="scene"
    )
    parser.add_argument("--keyframe-interval-sec", type=float, default=15.0)
    parser.add_argument("--scene-threshold", type=float, default=0.35)
    parser.add_argument("--max-keyframes", type=int, default=40)
    parser.add_argument(
        "--vision-model",
        default=None,
        help="Image-to-text model name for keyframe analysis",
    )
    parser.add_argument(
        "--vision-prompt-profile",
        default="slides",
        choices=["slides", "meeting", "code-demo"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    input_path = Path(args.video_path).expanduser().resolve()

    if not input_path.exists() or not input_path.is_file():
        print(f"Error: file not found or unreadable: {input_path}", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    json_out = output_dir / f"{stem}.json"
    srt_out = output_dir / f"{stem}.srt"
    vtt_out = output_dir / f"{stem}.vtt"
    txt_out = output_dir / f"{stem}.txt"
    visual_out = output_dir / f"{stem}.visual.json"
    merged_out = output_dir / f"{stem}.timeline.json"
    keyframe_dir_out = output_dir / f"{stem}.keyframes"

    device = "cpu" if args.cpu else "cuda"
    compute_type = "int8" if args.cpu else "float16"

    print(f"Input: {input_path}")
    print(f"Device: {device} ({compute_type})")

    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="cudacaption_") as tmpdir:
            wav_path = Path(tmpdir) / f"{stem}.wav"
            print("Extracting audio with ffmpeg...")
            extract_audio_wav(input_path, wav_path)

            print(f"Transcribing with model '{args.model}'...")
            segments = transcribe_file(
                wav_path,
                TranscribeConfig(
                    model_name=args.model,
                    language=args.language,
                    device=device,
                    compute_type=compute_type,
                    word_timestamps=args.word_timestamps,
                ),
            )

            visual_events: list[dict] = []
            if args.extract_keyframes:
                frames_dir = Path(tmpdir) / "keyframes"
                print("Extracting keyframes...")
                keyframes = extract_keyframes(
                    input_path,
                    frames_dir,
                    mode=args.keyframe_mode,
                    interval_sec=args.keyframe_interval_sec,
                    scene_threshold=args.scene_threshold,
                    max_keyframes=args.max_keyframes,
                )
                keyframe_dir_out.mkdir(parents=True, exist_ok=True)
                persisted_keyframes: list[dict] = []
                for idx, frame in enumerate(keyframes, start=1):
                    src = Path(frame["frame_path"])
                    dst = keyframe_dir_out / f"frame_{idx:05d}.jpg"
                    shutil.copy2(src, dst)
                    persisted_keyframes.append(
                        {"timestamp": float(frame["timestamp"]), "frame_path": str(dst)}
                    )
                keyframes = persisted_keyframes
                if args.vision_model and keyframes:
                    print(f"Analyzing keyframes with model '{args.vision_model}'...")
                    visual_events = analyze_keyframes(
                        keyframes,
                        model_name=args.vision_model,
                        device=device,
                        prompt_profile=args.vision_prompt_profile,
                    )
                else:
                    visual_events = keyframes

        write_json(segments, json_out)
        write_srt(segments, srt_out)
        write_vtt(segments, vtt_out)
        write_txt(segments, txt_out)
        if args.extract_keyframes:
            write_json(visual_events, visual_out)
            merged_timeline = sorted(
                [
                    {"time": float(s["start"]), "type": "audio", "text": s["text"]}
                    for s in segments
                ]
                + [
                    {
                        "time": float(v.get("timestamp", 0.0)),
                        "type": "visual",
                        "text": v.get("caption", v.get("frame_path", "")),
                    }
                    for v in visual_events
                ],
                key=lambda e: e["time"],
            )
            write_unified_json(
                {
                    "video": str(input_path),
                    "audio_segments": segments,
                    "visual_keyframes": visual_events,
                    "merged_timeline": merged_timeline,
                },
                merged_out,
            )
    except Exception as exc:
        error_message = format_runtime_error(exc, device)
        print(f"Error: transcription failed: {error_message}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start
    print("Done.")
    print(f"- {json_out}")
    print(f"- {srt_out}")
    print(f"- {vtt_out}")
    print(f"- {txt_out}")
    if args.extract_keyframes:
        print(f"- {visual_out}")
        print(f"- {merged_out}")
        print(f"- {keyframe_dir_out}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
