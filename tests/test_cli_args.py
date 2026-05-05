from __future__ import annotations

from cli import parse_args, resolve_runtime_target


def test_parse_args_sets_english_override_flag() -> None:
    args = parse_args(["sample.mp4", "--english"])
    assert args.english is True
    assert args.video_path == "sample.mp4"


def test_parse_args_accepts_language_flag() -> None:
    args = parse_args(["sample.mp4", "--language", "en"])
    assert args.language == "en"


def test_parse_args_defaults_for_keyframe_controls() -> None:
    args = parse_args(["sample.mp4", "--extract-keyframes"])
    assert args.extract_keyframes is True
    assert args.keyframe_mode == "scene"
    assert args.max_keyframes == 40


def test_resolve_runtime_target_cpu_flag() -> None:
    device, compute_type, note = resolve_runtime_target(force_cpu=True)
    assert device == "cpu"
    assert compute_type == "int8"
    assert note is None
