# Photo Selector

## Install (editable)

```powershell
pip install -e .
```

## Developer setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install in editable mode:

```powershell
pip install -e .
```

Run lint and tests:

```powershell
ruff check .
pytest -q
```

## Dependency management

Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

Install dev tools:

```powershell
pip install -r requirements-dev.txt
```

Regenerate the lock file (requires pip-tools):

```powershell
pip-compile --output-file=requirements.lock requirements.txt
```

## Quality checks

Run lint and tests locally:

```powershell
ruff check .
pytest -q
```

## CI checks

Required checks for branch protection:

- `lint`
- `test`

## Config file

Use `--config` to load settings from a YAML file. CLI flags override config values.

Supported keys (recommended nested `photo:` and `video:`):

- `input`
- `output`
- `model`
- `base_url`
- `target_count` (photo only)
- `max_source_seconds` (video only)
- `photo.dedupe_enabled`
- `photo.dedupe_hamming_threshold`
- `video.preset`
- `video.max_source_seconds`
- `video.use_hwaccel`
- `video.hwaccel` (legacy)
- `video.concat_in_digest_folder`
- `video.delete_split_files`
- `video.dedupe_enabled`
- `video.dedupe_hamming_threshold`
- `video.dedupe_scope`
- `video.max_selected_clips`
- `video.target_digest_seconds`

Top-level keys are still supported for backward compatibility.

Example (recommended nested):

```yaml
model: gemma3:4b
base_url: http://localhost:11434
input: C:\path\to\input
output: C:\path\to\output
target_count: 120
max_source_seconds: 30
photo:
  dedupe_enabled: true
  dedupe_hamming_threshold: 6
video:
  max_source_seconds: 30
  preset: youtube16x9
  use_hwaccel: true
  concat_in_digest_folder: true
  delete_split_files: false
  dedupe_enabled: true
  dedupe_hamming_threshold: 6
  dedupe_scope: per_source_video
  max_selected_clips: 20
  target_digest_seconds: 90

Example (top-level legacy keys):

```yaml
model: gemma3:4b
base_url: http://localhost:11434
input: C:\path\to\input
output: C:\path\to\output
target_count: 120
max_source_seconds: 30
concat_in_digest_folder: true
```
```

## Output structure

All outputs are organized under the `--output` directory:

- `selected/`: Selected images
- `scores/`: Score database and manifests
- `temp/`: Intermediate artifacts (clips, frames, audio, concat lists)
- `digest_clips/`: Selected clips per source

Final concatenated digests are written to the output root.
Concatenation operates on selected clips only, ordered by clip start time.

## Startup dependency checks

At startup, the CLI validates:

- `ffmpeg` is available in PATH
- NVENC encoder availability when `--use-hwaccel` is set
- Ollama server connectivity at `--ollama-base-url` (or `OLLAMA_BASE_URL`)

If a dependency is missing, the command exits with a clear error message and non-zero code.
Use `--debug` to show stack traces.

Example checks:

```powershell
photo-selector --input "C:\path\to\photos" --output "output" --target-count 10 --model gemma3:4b --dry-run
photo-video-digest --input "C:\path\to\videos" --output "output" --max-source-seconds 30 --preset youtube16x9 --use-hwaccel --dry-run
```

## Photo selection

Score and select photos using Ollama.

```powershell
photo-selector --input "C:\path\to\photos" --output "output" --target-count 120 --model gemma3:4b
```

### Options

- `--input`: Input directory with photos.
- `--output`: Output directory.
- `--target-count`: Number of photos to select.
- `--model`: Ollama model name (for example `gemma3:4b`).
- `--resume`: Skip already processed files based on stored hashes.
- `--force`: Recompute scores even if cached.
- `--photo-dedupe` / `--no-photo-dedupe`: Enable or disable near-duplicate filtering.
- `--photo-dedupe-hamming-threshold`: Hamming distance threshold (default 6).
- `--dry-run`: Print an execution plan without writing files.
- `--debug`: Show stack traces on errors.
- `--log-format`: `plain` or `json` (default `plain`).
- `--ollama-base-url`: Ollama base URL (default `http://localhost:11434`).

## Video digest (per-source)

Run the video digest pipeline. Each source video is split into clips, scored, and concatenated into a per-source digest.

```powershell
photo-video-digest --input "C:\path\to\videos" --output "output" --max-source-seconds 30 --min-clip 2 --max-clip 6 --preset clips_only --model gemma3:4b --concat-in-digest-folder
```

### Options

- `--max-source-seconds`: Max total duration per source video.
- `--min-clip`: Minimum clip length in seconds.
- `--max-clip`: Maximum clip length in seconds.
- `--model`: Ollama model name (for example `gemma3:4b`).
- `--ollama-base-url`: Ollama base URL (default `http://localhost:11434`).
- `--preset`: `youtube16x9`, `shorts9x16`, or `clips_only`.
- `--keep-temp`: Keep intermediate files under `output/temp`.
- `--delete-split-files`: When `--keep-temp` is not set, delete split clips and temp artifacts after a successful run.
- `--concat-in-digest-folder`: Also write `output/digest_clips/<source_stem>/digest.mp4`.
- `--use-hwaccel`: Use NVENC for splitting and concatenation (NVIDIA GPUs).
- `--video-dedupe` / `--no-video-dedupe`: Enable or disable near-duplicate filtering.
- `--video-dedupe-hamming-threshold`: Hamming distance threshold (default 6).
- `--video-dedupe-scope`: `global` or `per_source_video` (default `per_source_video`).
- `--video-max-selected-clips`: Max selected clips per source (default 20).
- `--video-target-digest-seconds`: Max digest seconds per source (default 90).
- `--dry-run`: Print an execution plan without writing files.
- `--debug`: Show stack traces on errors.
- `--log-format`: `plain` or `json` (default `plain`).

The video manifest also includes `job_state`, which records per-step status for split, score, select, and concat.

## Logging

Structured logging can be enabled with `--log-format json`. Each line is a JSON object.
The final summary always includes:

- `total_files`
- `processed`
- `skipped`
- `failed`
- `duration_seconds`

## Scoring schema

LLM output is normalized into a strict schema:

- `overall_score`: 0.0 to 1.0
- `sharpness`: 0.0 to 1.0
- `subject_visibility`: 0.0 to 1.0
- `composition`: 0.0 to 1.0
- `duplication_penalty`: 0.0 to 1.0
- `reasoning`: short text

Missing fields are safely defaulted, and legacy `score` is mapped to `overall_score`.
