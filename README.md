# vlm-annotations

Qwen3-VL video annotation harness for `DorianAtSchool/pick_place`.

This project:
- resolves episodes/videos from the Hugging Face dataset cache,
- samples frames with `ffmpeg`,
- calls an external vLLM OpenAI-compatible server,
- writes structured pick/place events + summaries,
- supports model sweep comparisons (default: Qwen3-VL 4B + 8B).

## Install

Python 3.10+ recommended.

```bash
pip install -e .
```

or full dependency set:

```bash
pip install -r requirements.txt
```

## Environment

Copy `.env.example` to `.env` and set values as needed:

- `HF_HOME` (default `.cache/hf`)
- `HUGGINGFACE_HUB_TOKEN` (optional)
- `VLLM_BASE_URL` (default `http://localhost:8000/v1`)
- `VLLM_API_KEY` (default `EMPTY`)

## External vLLM server

The CLI expects the vLLM server to already be running. Example:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --port 8000
```

For sweeps with multiple models, switch/restart the server between models (or pass `--no-pause-between-models` if you automate switching).

## CLI

### Inspect dataset metadata

```bash
pipeline inspect-hf \
  --dataset-id DorianAtSchool/pick_place \
  --camera-key observation.images.wrist
```

Machine-readable output:

```bash
pipeline inspect-hf --json
```

### Annotate one model

```bash
pipeline annotate-hf \
  --dataset-id DorianAtSchool/pick_place \
  --camera-key observation.images.wrist \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --max-episodes 5 \
  --fps 1.0 \
  --max-frames 16 \
  --out out
```

### Sweep models (smoke test default)

Uses default model list:
- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`

```bash
pipeline sweep-hf \
  --dataset-id DorianAtSchool/pick_place \
  --camera-key observation.images.wrist \
  --max-episodes 5 \
  --fps 1.0 \
  --max-frames 16 \
  --out out
```

Override model list:

```bash
pipeline sweep-hf \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --no-pause-between-models
```

### Qwen3.5 FP8 auto sweep (no manual switching)

The repository includes [`run_auto_sweep.sh`](/home/smahmud/Documents/vlm-annotations/run_auto_sweep.sh), which:
- starts/stops vLLM automatically per model,
- retries startup at lower context length if needed,
- runs `pipeline annotate-hf`,
- continues even if one model fails,
- writes a sweep comparison markdown report.

Default model set:
- `Qwen/Qwen3.5-35B-A3B-FP8`
- `Qwen/Qwen3.5-27B-FP8`

Run it:

```bash
./run_auto_sweep.sh
```

Optional overrides:

```bash
DATASET=DorianAtSchool/pick_place \
CAMERA=observation.images.wrist \
MAX_EPISODES=5 \
PRIMARY_MAX_MODEL_LEN=32768 \
FALLBACK_MAX_MODEL_LEN=16384 \
./run_auto_sweep.sh
```

## Output layout

Each run writes:

- `out/<run_id>/manifest.json`
- `out/<run_id>/<model_alias>/episodes/<episode_index>/annotation.json`
- `out/<run_id>/<model_alias>/episodes/<episode_index>/frame_index.jsonl`
- `out/<run_id>/<model_alias>/events.jsonl`
- `out/<run_id>/<model_alias>/summaries.jsonl`
- `out/<run_id>/<model_alias>/metrics.json`
- `out/<run_id>/aggregate_metrics.json`
- `out/<run_id>/comparison.md`

## Tests

```bash
pytest
```
