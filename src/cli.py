from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from media import extract_audio_wav
from transcribe import TranscribeConfig, transcribe_file
from writers import write_json, write_srt, write_txt, write_vtt


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cudacaption",
        description="Transcribe a video file into timestamped outputs using Whisper.",
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--model", default="medium", help="Whisper model (default: medium)")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    input_path = Path(args.video_path).expanduser().resolve()

    if not input_path.exists() or not input_path.is_file():
        print(f"Error: file not found or unreadable: {input_path}", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    json_out = output_dir / f"{stem}.json"
    srt_out = output_dir / f"{stem}.srt"
    vtt_out = output_dir / f"{stem}.vtt"
    txt_out = output_dir / f"{stem}.txt"

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

        write_json(segments, json_out)
        write_srt(segments, srt_out)
        write_vtt(segments, vtt_out)
        write_txt(segments, txt_out)
    except Exception as exc:
        print(f"Error: transcription failed: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start
    print("Done.")
    print(f"- {json_out}")
    print(f"- {srt_out}")
    print(f"- {vtt_out}")
    print(f"- {txt_out}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
