# Photo Selector

## Install (editable)

```powershell
pip install -e .
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

## Output structure

All outputs are organized under the `--output` directory:

- `selected/`: Selected images
- `scores/`: Score database and manifests
- `temp/`: Intermediate artifacts (clips, frames, audio, concat lists)
- `digest_clips/`: Selected clips per source

Final concatenated digests are written to the output root.

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
- `--concat-in-digest-folder`: Also write `output/digest_clips/<source_stem>/digest.mp4`.
- `--use-hwaccel`: Use NVENC for splitting and concatenation (NVIDIA GPUs).
- `--dry-run`: Print an execution plan without writing files.
- `--debug`: Show stack traces on errors.
- `--log-format`: `plain` or `json` (default `plain`).

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
