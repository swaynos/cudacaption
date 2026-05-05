from __future__ import annotations

import pytest

from cli import _validate_early_captions, _validate_early_transcript


def test_validate_early_transcript_passes_for_normal_text() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "text": "This is a valid sentence in English."},
        {"start": 2.0, "end": 5.0, "text": "The transcript should pass this sanity check."},
    ]
    _validate_early_transcript(segments, sanity_segments=2, sanity_window_sec=30.0)


def test_validate_early_transcript_fails_for_garbage() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "text": "xXx++--001122"},
        {"start": 2.0, "end": 4.0, "text": "+++==//"},
    ]
    with pytest.raises(RuntimeError, match="garbled"):
        _validate_early_transcript(segments, sanity_segments=2, sanity_window_sec=30.0)


def test_validate_early_captions_fails_on_empty_caption() -> None:
    visual = [
        {"caption": "A dashboard screenshot"},
        {"caption": ""},
    ]
    with pytest.raises(RuntimeError, match="empty caption"):
        _validate_early_captions(visual, count=5)


def test_validate_early_transcript_fails_when_no_segments() -> None:
    with pytest.raises(RuntimeError, match="no segments generated"):
        _validate_early_transcript([], sanity_segments=2, sanity_window_sec=30.0)


def test_validate_early_transcript_fails_when_too_short() -> None:
    segments = [{"start": 0.0, "end": 0.5, "text": "ok"}]
    with pytest.raises(RuntimeError, match="too short"):
        _validate_early_transcript(segments, sanity_segments=1, sanity_window_sec=30.0)


def test_validate_early_captions_passes_with_non_empty_entries() -> None:
    visual = [
        {"caption": "Slide title and summary"},
        {"caption": "Diagram with labeled components"},
    ]
    _validate_early_captions(visual, count=2)
