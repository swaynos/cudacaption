# Whisper Video Transcription Utility Spec

## Objective

Build a local command-line utility named `cudacaption` that transcribes a provided video file into timestamped text using Whisper, with an implementation optimized to use the laptop's NVIDIA RTX 4050 GPU when available.

Expand scope to include **key-frame visual understanding** for slide-based or screen-shared videos (for example Zoom meetings), by extracting sparse representative frames and running an image-to-text vision model (Florence-2 or similar) that fits within ~6 GB VRAM.

Primary invocation goal:

```bash
cudacaption ./myvideo.mp4
```

## Runtime and Environment Requirements

- Python runtime: `3.11.10` managed via `pyenv`.
- Project-local virtual environment created from `pyenv` Python `3.11.10`.
- GPU expectation: transcription should use the NVIDIA RTX 4050 to accelerate inference.
- OS target: Linux laptop environment.

## Functional Requirements

1. Accept a single positional argument: input video path.
2. Validate that the path exists and is a readable file.
3. Extract audio from the input video into a Whisper-friendly format (mono, 16 kHz WAV).
4. Run Whisper transcription with segment timestamps.
5. Produce output artifacts:
   - `*.srt` (subtitle format with timestamps)
   - `*.vtt` (web subtitle format)
   - `*.json` (structured segments with `start`, `end`, `text`)
   - Optional plain text `*.txt`
6. Exit with clear error messages and non-zero code for invalid input or runtime failure.
7. Print output file paths after successful completion.
8. Optionally extract key frames at scene changes or fixed intervals suitable for mostly-static content.
9. Optionally run image-to-text analysis on extracted key frames and produce visual insights.
10. Merge audio transcription + visual key-frame outputs into a unified JSON timeline.

### New Scope: Visual Key-Frame Intelligence

For slide-heavy presentations and static shared-screen meetings, the tool should infer non-audio context from key frames, including:

- Slide titles and section headers visible on screen.
- Bullet points, chart labels, and table headings (OCR-like extraction via VLM captioning/OCR prompts).
- Speaker-shared links, action items, and deadlines shown visually but not spoken.
- Scene/slide transitions with coarse timestamps.
- Screen modality tags (slides, code editor, browser, spreadsheet, whiteboard, camera-only).
- Optional confidence/quality signals for extracted visual text.

## Non-Functional Requirements

- Performance: use CUDA path on RTX 4050 for meaningful speedup over CPU.
- Privacy: local-first processing (no cloud dependency required).
- Usability: one-command workflow for common use.
- Reliability: deterministic outputs for same input and model settings.
- VRAM fit: default visual model path must run on laptops with ~6 GB VRAM.

## Technical Approach

### 1) Tooling and Dependency Strategy

- Use `faster-whisper` as the transcription backend for strong performance and easy CUDA use.
- Use `ffmpeg` for audio extraction/preprocessing.
- Use `ffmpeg` frame extraction filters (scene detect / fps sampling) for key frames.
- Use a lightweight local vision-language model (Florence-2 or equivalent) for frame-to-text.
- Use a lightweight CLI layer (standard `argparse` initially).
- Dependency installation through `pip` in a `pyenv`-backed virtualenv.

Suggested Python dependencies:

- `faster-whisper`
- `transformers` (for Florence-2 or similar VLM)
- `torch`
- `pillow`
- `ctranslate2` (typically pulled as dependency)
- `srt` (if custom SRT writing is needed)
- `rich` (optional, for cleaner CLI logs)

System dependencies:

- `ffmpeg`
- NVIDIA driver + CUDA runtime compatible with selected `faster-whisper`/`ctranslate2` build.

### 2) GPU Utilization Plan (RTX 4050)

- Default execution should request GPU:
  - `device="cuda"`
  - `compute_type="float16"` (default preferred on RTX 4050)
- On startup, log selected device and compute type.
- If CUDA initialization fails, either:
  - fail fast with a clear GPU troubleshooting message (preferred for this spec), or
  - allow optional `--cpu` fallback mode for explicit non-GPU runs.

### 3) CLI Contract

Initial command shape:

```bash
cudacaption <video_path> [--model medium] [--language en] [--output-dir ./transcripts]
```

Recommended options:

- `video_path` (required positional)
- `--model` (default `medium`; configurable)
- `--language` (optional language hint)
- `--output-dir` (default: same directory as input)
- `--word-timestamps` (optional; enables word-level timing when needed)
- `--cpu` (optional explicit CPU mode)
- `--extract-keyframes` (enable key-frame extraction)
- `--keyframe-mode` (`scene` or `interval`; default `scene`)
- `--keyframe-interval-sec` (used when `interval` mode is selected)
- `--vision-model` (default to a 6 GB VRAM-friendly model preset)
- `--vision-prompt-profile` (e.g. `slides`, `meeting`, `code-demo`)
- `--max-keyframes` (cap processed frames for long recordings)

### 4) Processing Pipeline

1. Resolve and validate input path.
2. Derive output basename from input filename.
3. Extract audio:
    - `ffmpeg -i <video> -ac 1 -ar 16000 -vn <temp.wav>`
4. Run transcription via `faster-whisper` model instance.
5. If visual mode enabled, extract key frames:
   - Scene mode: ffmpeg scene-change threshold based extraction.
   - Interval mode: fixed time sampling (e.g. every N seconds).
6. Run frame-to-text inference on key frames using vision model.
7. Align visual outputs to nearest timestamps and de-duplicate repeated slide content.
8. Serialize segments into:
    - JSON: array of `{start, end, text}` with seconds and/or timestamp strings.
    - SRT/VTT: segmented subtitle files.
    - Visual JSON: key-frame records with `timestamp`, `caption`, `ocr_text`, `tags`.
    - Unified JSON: merged audio+visual timeline.
9. Emit completion summary with paths and elapsed time.
10. Cleanup temporary artifacts.

### 4.1) Visual Output Schema (Draft)

```json
{
  "video": "meeting.mp4",
  "audio_segments": [{"start": 0.0, "end": 4.2, "text": "..."}],
  "visual_keyframes": [
    {
      "timestamp": 62.5,
      "frame_path": ".../kf_0062.5.jpg",
      "slide_id": "slide_03",
      "caption": "Slide titled Project Milestones",
      "ocr_text": "Milestone 1 ...",
      "tags": ["slides", "roadmap"],
      "confidence": 0.82
    }
  ],
  "merged_timeline": [
    {"time": 62.0, "type": "audio", "text": "..."},
    {"time": 62.5, "type": "visual", "text": "Slide titled Project Milestones"}
  ]
}
```

### 5) Project Structure

Suggested layout:

```text
whisper/
  spec.md
  pyproject.toml
  README.md
  src/
    cudacaption/
      __init__.py
      cli.py
      transcribe.py
      media.py
      writers.py
```

## Implementation Milestones

1. Bootstrap project with Python `3.11.10` via `pyenv` and virtualenv.
2. Add CLI skeleton and argument parsing.
3. Implement media extraction wrapper around `ffmpeg`.
4. Implement Whisper transcription service using `faster-whisper` with CUDA config.
5. Implement JSON/SRT/VTT writers.
6. Add logging, error handling, and exit codes.
7. Test end-to-end on a real Zoom recording using RTX 4050.
8. Add key-frame extraction module with `scene` and `interval` modes.
9. Integrate Florence-2 (or equivalent) vision inference with VRAM-aware defaults.
10. Add visual JSON + merged timeline serializers.
11. Evaluate on at least two slide-based meeting recordings and compare usefulness vs audio-only output.

## Verification and Acceptance Criteria

The utility is accepted when all conditions are met:

1. Running `cudacaption ./myvideo.mp4` completes without manual intervention.
2. Outputs `myvideo.srt`, `myvideo.vtt`, and `myvideo.json` in expected location.
3. Logs confirm CUDA device usage (RTX 4050 path) unless `--cpu` is specified.
4. Output timestamps align reasonably with spoken segments.
5. Invalid file path returns a clear error and non-zero exit code.
6. For slide-based inputs, key-frame mode extracts meaningful visual text and slide transitions.
7. Visual model path runs within available laptop VRAM budget (~6 GB target).
8. Unified JSON timeline includes both transcription and visual events with coherent timestamps.

## Testing Plan

- Unit tests:
  - argument validation
  - path and naming logic
  - output serialization format correctness
- Integration tests:
  - ffmpeg extraction invocation
  - full transcription run on a short sample clip
- Runtime validation:
  - confirm GPU utilization via logs and performance comparison vs CPU run.
  - confirm visual model memory footprint stays within expected VRAM on RTX 4050-class hardware.
  - compare extracted visual text against known slide content on sample meeting videos.

## Risks and Mitigations

- CUDA/driver mismatch: document known-good dependency versions and install steps.
- VRAM limits for larger models: default to `medium`, allow model override.
- Poor source audio quality: rely on ffmpeg normalization options in future iterations.
- Visual hallucination or OCR noise: use prompt constraints, confidence scoring, and de-duplication.
- Repeated near-identical frames: enforce perceptual-hash or text-similarity dedupe.
- Long videos with many transitions: add `--max-keyframes` and interval fallback.

## Deliverables

1. `cudacaption` CLI utility installable/runnable in the project virtualenv.
2. Documentation for setup with `pyenv` Python `3.11.10` and RTX 4050 GPU checks.
3. Timestamped transcription outputs (SRT, VTT, JSON) for provided video inputs.
4. Optional key-frame visual analysis outputs and merged audio+visual timeline JSON.
