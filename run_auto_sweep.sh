#!/usr/bin/env bash
set -u -o pipefail

MODELS=(
  "Qwen/Qwen3.5-35B-A3B-FP8"
  "Qwen/Qwen3.5-27B-FP8"
)

DATASET="${DATASET:-DorianAtSchool/pick_place}"
CAMERA="${CAMERA:-observation.images.wrist}"
MAX_EPISODES="${MAX_EPISODES:-5}"
FPS="${FPS:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-16}"
OUT_DIR="${OUT_DIR:-out}"
BASE_RUN_ID="${BASE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STREAM_ANNOTATE_OUTPUT="${STREAM_ANNOTATE_OUTPUT:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
PRIMARY_MAX_MODEL_LEN="${PRIMARY_MAX_MODEL_LEN:-32768}"
FALLBACK_MAX_MODEL_LEN="${FALLBACK_MAX_MODEL_LEN:-16384}"
VLLM_START_TIMEOUT_SEC="${VLLM_START_TIMEOUT_SEC:-900}"

RESULTS_TSV="${OUT_DIR}/${BASE_RUN_ID}_sweep_results.tsv"
COMPARISON_MD="${OUT_DIR}/${BASE_RUN_ID}_comparison.md"

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2
}

usage() {
  cat <<EOF
Usage: ./run_auto_sweep.sh

Runs an automatic two-model Qwen3.5 FP8 sweep:
  - Qwen/Qwen3.5-35B-A3B-FP8
  - Qwen/Qwen3.5-27B-FP8

Optional environment overrides:
  DATASET, CAMERA, MAX_EPISODES, FPS, MAX_FRAMES, OUT_DIR, BASE_RUN_ID
  STREAM_ANNOTATE_OUTPUT (default: 1, stream annotate output live)
  VLLM_PORT, VLLM_BASE_URL, GPU_MEMORY_UTILIZATION
  PRIMARY_MAX_MODEL_LEN (default: 32768)
  FALLBACK_MAX_MODEL_LEN (default: 16384)
  VLLM_START_TIMEOUT_SEC (default: 900)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    log "ERROR: required command not found: ${cmd}"
    exit 1
  fi
}

stop_vllm() {
  pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
  sleep 2
}

wait_for_vllm() {
  local pid="$1"
  local timeout="$2"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if curl -sf "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      # vLLM may re-parent workers during startup; if any matching process is
      # still alive, keep waiting instead of declaring a crash.
      if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
        return 1
      fi
    fi
    if (( "$(date +%s)" - start_ts >= timeout )); then
      return 2
    fi
    sleep 2
  done
}

launch_vllm() {
  local model="$1"
  local max_len="$2"
  local log_path="$3"
  local pid=""

  stop_vllm
  log "Starting vLLM for ${model} (max_model_len=${max_len})"
  .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "${model}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${max_len}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    >"${log_path}" 2>&1 &
  pid=$!

  if wait_for_vllm "${pid}" "${VLLM_START_TIMEOUT_SEC}"; then
    printf '%s\n' "${pid}"
    return 0
  fi

  local wait_rc=$?
  if [[ "${wait_rc}" -eq 2 ]]; then
    log "vLLM start timed out for ${model} (max_model_len=${max_len})."
  else
    log "vLLM crashed during startup for ${model} (max_model_len=${max_len})."
  fi
  kill "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true
  return 1
}

append_result() {
  local model="$1"
  local run_id="$2"
  local status="$3"
  local used_len="$4"
  local vllm_log="$5"
  local annotate_log="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${model}" "${run_id}" "${status}" "${used_len}" "${vllm_log}" "${annotate_log}" \
    >>"${RESULTS_TSV}"
}

run_annotate() {
  local model="$1"
  local run_id="$2"
  local annotate_log="$3"
  local -a cmd=(
    .venv/bin/python -m vlm_pipeline.cli annotate-hf
    --dataset-id "${DATASET}"
    --camera-key "${CAMERA}"
    --model-id "${model}"
    --max-episodes "${MAX_EPISODES}"
    --fps "${FPS}"
    --max-frames "${MAX_FRAMES}"
    --out "${OUT_DIR}"
    --run-id "${run_id}"
  )

  if [[ "${STREAM_ANNOTATE_OUTPUT}" == "1" ]]; then
    # Preserve TTY rendering (progress bars) when possible and still log output.
    if [[ -t 1 ]] && command -v script >/dev/null 2>&1; then
      local cmd_str
      printf -v cmd_str '%q ' "${cmd[@]}"
      script -q -f -e -c "${cmd_str}" "${annotate_log}"
      return $?
    fi
    "${cmd[@]}" 2>&1 | tee "${annotate_log}"
    return $?
  fi

  "${cmd[@]}" >"${annotate_log}" 2>&1
  return $?
}

main() {
  require_cmd curl
  require_cmd pkill
  if [[ ! -x ".venv/bin/python" ]]; then
    log "ERROR: .venv/bin/python not found or not executable."
    exit 1
  fi

  mkdir -p "${OUT_DIR}"
  printf 'model\trun_id\tstatus\tmax_model_len\tvllm_log\tannotate_log\n' >"${RESULTS_TSV}"
  log "Sweep run_id prefix: ${BASE_RUN_ID}"

  for model in "${MODELS[@]}"; do
    local alias="${model//\//_}"
    local run_id="${BASE_RUN_ID}_${alias}"
    local vllm_log="${OUT_DIR}/${run_id}.vllm.log"
    local annotate_log="${OUT_DIR}/${run_id}.annotate.log"
    local used_len=""
    local status="ok"
    local server_pid=""

    server_pid="$(launch_vllm "${model}" "${PRIMARY_MAX_MODEL_LEN}" "${vllm_log}")" || true
    if [[ -n "${server_pid}" ]]; then
      used_len="${PRIMARY_MAX_MODEL_LEN}"
    else
      log "Retrying ${model} with max_model_len=${FALLBACK_MAX_MODEL_LEN}"
      server_pid="$(launch_vllm "${model}" "${FALLBACK_MAX_MODEL_LEN}" "${vllm_log}")" || true
      if [[ -n "${server_pid}" ]]; then
        used_len="${FALLBACK_MAX_MODEL_LEN}"
      else
        status="server_failed"
        append_result "${model}" "${run_id}" "${status}" "-" "${vllm_log}" "${annotate_log}"
        log "Skipping annotation for ${model}; see ${vllm_log}"
        continue
      fi
    fi

    log "Running annotate-hf for ${model} (run_id=${run_id})"
    if run_annotate "${model}" "${run_id}" "${annotate_log}"; then
      status="ok"
      log "Annotation completed for ${model}"
    else
      status="annotate_failed"
      log "Annotation failed for ${model}; see ${annotate_log}"
    fi

    append_result "${model}" "${run_id}" "${status}" "${used_len}" "${vllm_log}" "${annotate_log}"
    stop_vllm
  done

  log "Writing comparison summary: ${COMPARISON_MD}"
  .venv/bin/python - \
    "${RESULTS_TSV}" \
    "${COMPARISON_MD}" \
    "${OUT_DIR}" \
    "${BASE_RUN_ID}" \
    "${DATASET}" \
    "${CAMERA}" \
    "${MAX_EPISODES}" <<'PY'
import csv
import json
from pathlib import Path
import sys

results_tsv = Path(sys.argv[1])
comparison_md = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
base_run_id = sys.argv[4]
dataset = sys.argv[5]
camera = sys.argv[6]
max_episodes = sys.argv[7]

rows = []
with results_tsv.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows.extend(reader)

lines = [
    f"# Qwen3.5 FP8 Sweep Comparison ({base_run_id})",
    "",
    f"- Dataset: `{dataset}`",
    f"- Camera: `{camera}`",
    f"- Episodes: `{max_episodes}`",
    "",
    "| Model | Status | Max Len | Schema Valid | Task Consistency | Avg Latency (s) | Avg Tokens |",
    "|---|---|---:|---:|---:|---:|---:|",
]

for row in rows:
    model = row["model"]
    run_id = row["run_id"]
    status = row["status"]
    max_len = row["max_model_len"]
    alias = model.replace("/", "_")
    metrics_path = out_dir / run_id / alias / "metrics.json"

    schema_valid = "n/a"
    task_consistency = "n/a"
    avg_latency = "n/a"
    avg_tokens = "n/a"
    if metrics_path.exists():
        try:
            metrics_doc = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = metrics_doc.get("metrics", {})
            schema_valid = f"{float(metrics.get('schema_valid_rate', 0.0)):.3f}"
            task_consistency = f"{float(metrics.get('task_consistency_rate', 0.0)):.3f}"
            avg_latency = f"{float(metrics.get('avg_latency_s', 0.0)):.3f}"
            avg_tokens = f"{float(metrics.get('avg_total_tokens', 0.0)):.1f}"
        except Exception:
            pass

    lines.append(
        f"| {model} | {status} | {max_len} | {schema_valid} | "
        f"{task_consistency} | {avg_latency} | {avg_tokens} |"
    )

comparison_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

  stop_vllm
  log "Sweep finished."
  log "Per-model logs + status: ${RESULTS_TSV}"
  log "Comparison markdown: ${COMPARISON_MD}"
}

trap stop_vllm EXIT

main "$@"
