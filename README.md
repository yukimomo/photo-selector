# Photo Selector

## Install (editable)

```powershell
pip install -e .
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
