from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
from typing import Any

from .config import (
    DEFAULT_CAMERA_KEY,
    DEFAULT_DATASET_ID,
    DEFAULT_FPS,
    DEFAULT_MAX_EPISODES,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_IDS,
    default_hf_home,
    default_hf_token,
    default_vllm_api_key,
    default_vllm_base_url,
    schema_path,
)
from .episode_index import EpisodeRef, pick_episode_indices, sanitize_model_alias
from .evaluate import (
    EpisodeEvalRecord,
    build_episode_eval_record,
    compute_aggregate_metrics,
    compute_model_metrics,
)
from .frame_sampler import sample_episode_frames
from .hf_dataset import HFDatasetAdapter
from .parser import load_schema, parse_with_optional_repair
from .prompting import build_annotation_messages, build_repair_messages
from .vllm_client import VLLMClient
from .writers import write_comparison_markdown, write_json, write_jsonl


class _SimpleProgressBar:
    def __init__(self, total: int, label: str, enabled: bool = True) -> None:
        self.total = max(1, total)
        self.label = label
        self.enabled = enabled
        self._is_tty = enabled and sys.stderr.isatty()
        self._last_rendered = -1

    def update(self, current: int, status: str = "") -> None:
        clamped = max(0, min(current, self.total))
        if clamped == self._last_rendered and status == "":
            return
        self._last_rendered = clamped

        pct = clamped / self.total
        if self._is_tty:
            width = max(12, min(40, shutil.get_terminal_size((80, 20)).columns - 48))
            filled = int(width * pct)
            bar = "#" * filled + "-" * (width - filled)
            line = f"\r{self.label} [{bar}] {clamped}/{self.total} {pct * 100:5.1f}%"
            if status:
                line += f" | {status}"
            print(line, end="", file=sys.stderr, flush=True)
        elif self.enabled:
            line = f"{self.label}: {clamped}/{self.total}"
            if status:
                line += f" | {status}"
            print(line, file=sys.stderr, flush=True)

    def close(self) -> None:
        if self._is_tty:
            print(file=sys.stderr, flush=True)


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pipeline", description="Qwen3-VL annotation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-hf", help="Inspect HF episode metadata")
    _add_dataset_args(inspect_parser)
    inspect_parser.add_argument("--max-episodes-preview", type=int, default=5)
    inspect_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    inspect_parser.set_defaults(func=_run_inspect)

    annotate_parser = subparsers.add_parser("annotate-hf", help="Annotate episodes for a single served model")
    _add_annotation_args(annotate_parser, require_model=True)
    annotate_parser.set_defaults(func=_run_annotate)

    sweep_parser = subparsers.add_parser("sweep-hf", help="Run a model sweep over shared episode inputs")
    _add_annotation_args(sweep_parser, require_model=False)
    sweep_parser.add_argument(
        "--pause-between-models",
        dest="pause_between_models",
        action="store_true",
        default=True,
        help="Pause between models to let you switch the external vLLM server",
    )
    sweep_parser.add_argument(
        "--no-pause-between-models",
        dest="pause_between_models",
        action="store_false",
        help="Do not pause between models",
    )
    sweep_parser.set_defaults(func=_run_sweep)

    return parser.parse_args(argv)


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--camera-key", default=DEFAULT_CAMERA_KEY)
    parser.add_argument("--hf-home", type=Path, default=default_hf_home())
    parser.add_argument("--hf-token", default=default_hf_token())


def _add_annotation_args(parser: argparse.ArgumentParser, require_model: bool) -> None:
    _add_dataset_args(parser)
    parser.add_argument(
        "--episode-index",
        type=int,
        action="append",
        default=None,
        help="Episode index to annotate. Repeat for multiple episodes.",
    )
    parser.add_argument("--max-episodes", type=int, default=DEFAULT_MAX_EPISODES)
    parser.add_argument("--model-id", action="append", default=None, required=require_model)
    parser.add_argument("--vllm-base-url", default=default_vllm_base_url())
    parser.add_argument("--api-key", default=default_vllm_api_key())
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--include-task-hint", dest="include_task_hint", action="store_true", default=True)
    parser.add_argument("--no-task-hint", dest="include_task_hint", action="store_false")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--run-id", default=None)


def _select_episode_refs(args: argparse.Namespace, adapter: HFDatasetAdapter) -> list[EpisodeRef]:
    rows = adapter.load_episode_rows()
    available_indices = sorted(int(row["episode_index"]) for row in rows)
    selected = pick_episode_indices(available_indices, args.episode_index, args.max_episodes)
    return adapter.resolve_episode_refs(camera_key=args.camera_key, episode_indices=selected)


def _merge_usage(*usage_dicts: dict[str, Any]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for usage in usage_dicts:
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                merged[key] = int(merged.get(key, 0) + int(value))
    return merged


def _pause_for_model_switch(model_id: str) -> None:
    prompt = f"\nSwitch external vLLM server to model '{model_id}', then press Enter to continue... "
    try:
        input(prompt)
    except EOFError:
        pass


def _build_fallback_annotation(summary: str) -> dict[str, Any]:
    return {
        "completion_status": "failed",
        "summary": summary,
        "events": [],
    }


def _run_model(
    run_root: Path,
    dataset_id: str,
    model_id: str,
    episode_refs: list[EpisodeRef],
    client: VLLMClient,
    schema: dict[str, Any],
    include_task_hint: bool,
    fps: float,
    max_frames: int,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_alias = sanitize_model_alias(model_id)
    model_root = run_root / model_alias
    episodes_root = model_root / "episodes"

    eval_records: list[EpisodeEvalRecord] = []
    events_rows: list[dict[str, Any]] = []
    summaries_rows: list[dict[str, Any]] = []
    progress = _SimpleProgressBar(total=len(episode_refs), label=f"{model_alias} episodes")
    progress.update(0, "starting")

    try:
        for processed_count, episode in enumerate(episode_refs, start=1):
            progress.update(
                processed_count - 1,
                f"episode {episode.episode_index:03d} running",
            )

            episode_dir = episodes_root / f"{episode.episode_index:03d}"
            frames_dir = episode_dir / "frames"
            frame_index_path = episode_dir / "frame_index.jsonl"
            annotation_path = episode_dir / "annotation.json"

            parse_valid = False
            parse_repaired = False
            error_message: str | None = None
            annotation_payload: dict[str, Any] = _build_fallback_annotation("Annotation failed")
            usage: dict[str, int] = {}
            timings: dict[str, float] = {"inference_s": 0.0, "repair_s": 0.0, "total_s": 0.0}
            initial_response_text = ""
            repair_response_text: str | None = None

            try:
                episode_start = perf_counter()
                frame_records = sample_episode_frames(
                    episode=episode,
                    output_dir=frames_dir,
                    fps=fps,
                    max_frames=max_frames,
                )
                frame_rows = [frame.to_json(episode_dir) for frame in frame_records]
                write_jsonl(frame_index_path, frame_rows)

                if not frame_records:
                    raise RuntimeError("No frames were sampled for this episode")

                messages = build_annotation_messages(
                    frame_records=frame_records,
                    task_text=episode.task_text,
                    include_task_hint=include_task_hint,
                    schema=schema,
                )
                initial_response = client.chat_completion(
                    model_id=model_id,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                initial_response_text = initial_response.content
                usage = _merge_usage(usage, initial_response.usage)
                timings["inference_s"] = initial_response.latency_s
                timings["total_s"] = initial_response.latency_s

                def _repair_fn(parse_error: str) -> str:
                    nonlocal usage, timings, repair_response_text
                    repair_messages = build_repair_messages(
                        raw_response=initial_response_text,
                        parse_error=parse_error,
                        schema=schema,
                    )
                    repair_response = client.chat_completion(
                        model_id=model_id,
                        messages=repair_messages,
                        temperature=0.0,
                        max_tokens=max_tokens,
                    )
                    repair_response_text = repair_response.content
                    usage = _merge_usage(usage, repair_response.usage)
                    timings["repair_s"] = repair_response.latency_s
                    timings["total_s"] = timings["inference_s"] + repair_response.latency_s
                    return repair_response.content

                parse_result = parse_with_optional_repair(
                    raw_text=initial_response_text,
                    schema=schema,
                    repair_fn=_repair_fn,
                )
                parse_valid = parse_result.ok
                parse_repaired = parse_result.repaired and parse_result.ok
                error_message = parse_result.error
                if parse_result.ok and parse_result.annotation is not None:
                    annotation_payload = parse_result.annotation
                else:
                    annotation_payload = _build_fallback_annotation("Schema parse failed")

            except Exception as exc:
                error_message = str(exc)
                annotation_payload = _build_fallback_annotation("Runtime failure during annotation")
            finally:
                timings["total_s"] = max(timings.get("total_s", 0.0), perf_counter() - episode_start)

            annotation_doc = {
                "episode_index": episode.episode_index,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "camera_key": episode.camera_key,
                "task_text": episode.task_text,
                "events": annotation_payload.get("events", []),
                "summary": annotation_payload.get("summary", ""),
                "completion_status": annotation_payload.get("completion_status", "failed"),
                "raw_response": {
                    "initial": initial_response_text,
                    "repair": repair_response_text,
                },
                "usage": usage,
                "timings": timings,
                "parse": {
                    "valid": parse_valid,
                    "repaired": parse_repaired,
                    "error": error_message,
                },
                "episode_window": {
                    "start_s": episode.start_s,
                    "end_s": episode.end_s,
                    "source_video": str(episode.video_path),
                },
            }
            write_json(annotation_path, annotation_doc)

            for event in annotation_doc.get("events", []):
                if isinstance(event, dict):
                    row = {"episode_index": episode.episode_index, "model_id": model_id}
                    row.update(event)
                    events_rows.append(row)

            summaries_rows.append(
                {
                    "episode_index": episode.episode_index,
                    "model_id": model_id,
                    "completion_status": annotation_doc.get("completion_status"),
                    "summary": annotation_doc.get("summary"),
                    "parse_valid": parse_valid,
                    "parse_repaired": parse_repaired,
                }
            )

            eval_records.append(
                build_episode_eval_record(
                    episode_index=episode.episode_index,
                    task_text=episode.task_text,
                    annotation=annotation_payload if parse_valid else None,
                    schema_valid=parse_valid,
                    parse_repaired=parse_repaired,
                    latency_s=float(timings.get("total_s", 0.0)),
                    total_tokens=int(usage.get("total_tokens", 0)),
                )
            )
            completion_status = str(annotation_doc.get("completion_status", "failed"))
            progress.update(
                processed_count,
                f"episode {episode.episode_index:03d} {completion_status}",
            )
    finally:
        progress.close()

    write_jsonl(model_root / "events.jsonl", events_rows)
    write_jsonl(model_root / "summaries.jsonl", summaries_rows)

    metrics = compute_model_metrics(eval_records)
    metrics_payload = {
        "model_id": model_id,
        "model_alias": model_alias,
        "metrics": metrics,
        "episodes": [asdict(record) for record in eval_records],
    }
    write_json(model_root / "metrics.json", metrics_payload)

    return (
        {
            "model_id": model_id,
            "model_alias": model_alias,
            "model_root": str(model_root),
            "episode_count": len(eval_records),
        },
        metrics,
    )


def _run_inspect(args: argparse.Namespace) -> int:
    adapter = HFDatasetAdapter(
        dataset_id=args.dataset_id,
        hf_home=args.hf_home,
        token=args.hf_token,
    )
    info = adapter.load_info()
    rows = adapter.load_episode_rows()
    camera_keys = adapter.available_camera_keys()

    preview_rows = sorted(rows, key=lambda row: int(row["episode_index"]))[: args.max_episodes_preview]
    preview = []
    for row in preview_rows:
        task_text = (row.get("tasks") or [""])[0]
        episode_idx = int(row["episode_index"])
        record = {
            "episode_index": episode_idx,
            "task_text": task_text,
            "length": int(row.get("length", 0)),
        }
        for camera in camera_keys:
            start_key = f"videos/{camera}/from_timestamp"
            end_key = f"videos/{camera}/to_timestamp"
            if start_key in row and end_key in row:
                record[f"{camera}.start_s"] = float(row[start_key])
                record[f"{camera}.end_s"] = float(row[end_key])
        preview.append(record)

    payload = {
        "dataset_id": args.dataset_id,
        "hf_home": str(args.hf_home),
        "total_episodes": len(rows),
        "camera_keys": camera_keys,
        "fps": info.get("fps"),
        "preview": preview,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Dataset: {payload['dataset_id']}")
        print(f"HF cache: {payload['hf_home']}")
        print(f"Total episodes: {payload['total_episodes']}")
        print(f"Camera keys: {', '.join(payload['camera_keys'])}")
        print(f"Dataset FPS: {payload['fps']}")
        print("Preview:")
        for row in payload["preview"]:
            print(f"  - episode {row['episode_index']}: {row['task_text']} (length={row['length']})")

    return 0


def _run_annotate(args: argparse.Namespace) -> int:
    model_id = args.model_id[0]

    adapter = HFDatasetAdapter(
        dataset_id=args.dataset_id,
        hf_home=args.hf_home,
        token=args.hf_token,
    )
    episode_refs = _select_episode_refs(args, adapter)
    run_id = args.run_id or _now_run_id()
    run_root = args.out / run_id
    schema = load_schema(schema_path())

    manifest = {
        "run_id": run_id,
        "command": "annotate-hf",
        "dataset_id": args.dataset_id,
        "camera_key": args.camera_key,
        "model_ids": [model_id],
        "episode_indices": [ref.episode_index for ref in episode_refs],
        "settings": {
            "fps": args.fps,
            "max_frames": args.max_frames,
            "max_tokens": args.max_tokens,
            "include_task_hint": args.include_task_hint,
            "vllm_base_url": args.vllm_base_url,
        },
    }

    client = VLLMClient(base_url=args.vllm_base_url, api_key=args.api_key)
    model_manifest, model_metrics = _run_model(
        run_root=run_root,
        dataset_id=args.dataset_id,
        model_id=model_id,
        episode_refs=episode_refs,
        client=client,
        schema=schema,
        include_task_hint=args.include_task_hint,
        fps=args.fps,
        max_frames=args.max_frames,
        max_tokens=args.max_tokens,
    )

    manifest["models"] = [model_manifest]
    write_json(run_root / "manifest.json", manifest)

    aggregate_metrics = compute_aggregate_metrics({model_manifest["model_alias"]: model_metrics})
    write_json(run_root / "aggregate_metrics.json", aggregate_metrics)
    write_comparison_markdown(
        path=run_root / "comparison.md",
        run_id=run_id,
        dataset_id=args.dataset_id,
        camera_key=args.camera_key,
        model_metrics={model_manifest["model_alias"]: model_metrics},
    )

    print(f"Run complete: {run_root}")
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    model_ids = args.model_id if args.model_id else list(DEFAULT_MODEL_IDS)

    adapter = HFDatasetAdapter(
        dataset_id=args.dataset_id,
        hf_home=args.hf_home,
        token=args.hf_token,
    )
    episode_refs = _select_episode_refs(args, adapter)
    run_id = args.run_id or _now_run_id()
    run_root = args.out / run_id
    schema = load_schema(schema_path())

    manifest = {
        "run_id": run_id,
        "command": "sweep-hf",
        "dataset_id": args.dataset_id,
        "camera_key": args.camera_key,
        "model_ids": model_ids,
        "episode_indices": [ref.episode_index for ref in episode_refs],
        "settings": {
            "fps": args.fps,
            "max_frames": args.max_frames,
            "max_tokens": args.max_tokens,
            "include_task_hint": args.include_task_hint,
            "vllm_base_url": args.vllm_base_url,
            "pause_between_models": args.pause_between_models,
        },
        "models": [],
    }

    client = VLLMClient(base_url=args.vllm_base_url, api_key=args.api_key)
    metrics_by_alias: dict[str, dict[str, Any]] = {}
    model_progress = _SimpleProgressBar(total=len(model_ids), label="sweep models")
    model_progress.update(0, "starting")

    try:
        for idx, model_id in enumerate(model_ids):
            model_progress.update(idx, f"running {sanitize_model_alias(model_id)}")
            if idx > 0 and args.pause_between_models:
                _pause_for_model_switch(model_id)

            try:
                model_manifest, model_metrics = _run_model(
                    run_root=run_root,
                    dataset_id=args.dataset_id,
                    model_id=model_id,
                    episode_refs=episode_refs,
                    client=client,
                    schema=schema,
                    include_task_hint=args.include_task_hint,
                    fps=args.fps,
                    max_frames=args.max_frames,
                    max_tokens=args.max_tokens,
                )
                manifest["models"].append({**model_manifest, "status": "ok"})
                metrics_by_alias[model_manifest["model_alias"]] = model_metrics
                model_progress.update(idx + 1, f"done {model_manifest['model_alias']}")
            except Exception as exc:
                alias = sanitize_model_alias(model_id)
                manifest["models"].append(
                    {
                        "model_id": model_id,
                        "model_alias": alias,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                metrics_by_alias[alias] = {
                    "episode_count": 0,
                    "schema_valid_rate": 0.0,
                    "parse_repair_rate": 0.0,
                    "avg_latency_s": 0.0,
                    "avg_total_tokens": 0.0,
                    "completion_rate": 0.0,
                    "robot_match_rate": 0.0,
                    "object_match_rate": 0.0,
                    "task_consistency_rate": 0.0,
                }
                model_progress.update(idx + 1, f"failed {alias}")
    finally:
        model_progress.close()

    write_json(run_root / "manifest.json", manifest)
    aggregate_metrics = compute_aggregate_metrics(metrics_by_alias)
    write_json(run_root / "aggregate_metrics.json", aggregate_metrics)
    write_comparison_markdown(
        path=run_root / "comparison.md",
        run_id=run_id,
        dataset_id=args.dataset_id,
        camera_key=args.camera_key,
        model_metrics=metrics_by_alias,
    )

    print(f"Sweep complete: {run_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
