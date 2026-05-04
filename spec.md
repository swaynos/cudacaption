# Whisper Video Transcription Utility Spec

## Objective

Build a local command-line utility named `cudacaption` that transcribes a provided video file into timestamped text using Whisper, with an implementation optimized to use the laptop's NVIDIA RTX 4050 GPU when available.

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

## Non-Functional Requirements

- Performance: use CUDA path on RTX 4050 for meaningful speedup over CPU.
- Privacy: local-first processing (no cloud dependency required).
- Usability: one-command workflow for common use.
- Reliability: deterministic outputs for same input and model settings.

## Technical Approach

### 1) Tooling and Dependency Strategy

- Use `faster-whisper` as the transcription backend for strong performance and easy CUDA use.
- Use `ffmpeg` for audio extraction/preprocessing.
- Use a lightweight CLI layer (standard `argparse` initially).
- Dependency installation through `pip` in a `pyenv`-backed virtualenv.

Suggested Python dependencies:

- `faster-whisper`
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

### 4) Processing Pipeline

1. Resolve and validate input path.
2. Derive output basename from input filename.
3. Extract audio:
   - `ffmpeg -i <video> -ac 1 -ar 16000 -vn <temp.wav>`
4. Run transcription via `faster-whisper` model instance.
5. Serialize segments into:
   - JSON: array of `{start, end, text}` with seconds and/or timestamp strings.
   - SRT/VTT: segmented subtitle files.
6. Emit completion summary with paths and elapsed time.
7. Cleanup temporary artifacts.

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

## Verification and Acceptance Criteria

The utility is accepted when all conditions are met:

1. Running `cudacaption ./myvideo.mp4` completes without manual intervention.
2. Outputs `myvideo.srt`, `myvideo.vtt`, and `myvideo.json` in expected location.
3. Logs confirm CUDA device usage (RTX 4050 path) unless `--cpu` is specified.
4. Output timestamps align reasonably with spoken segments.
5. Invalid file path returns a clear error and non-zero exit code.

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

## Risks and Mitigations

- CUDA/driver mismatch: document known-good dependency versions and install steps.
- VRAM limits for larger models: default to `medium`, allow model override.
- Poor source audio quality: rely on ffmpeg normalization options in future iterations.

## Deliverables

1. `cudacaption` CLI utility installable/runnable in the project virtualenv.
2. Documentation for setup with `pyenv` Python `3.11.10` and RTX 4050 GPU checks.
3. Timestamped transcription outputs (SRT, VTT, JSON) for provided video inputs.
