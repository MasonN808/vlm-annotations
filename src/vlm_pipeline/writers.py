from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_comparison_markdown(
    path: Path,
    run_id: str,
    dataset_id: str,
    camera_key: str,
    model_metrics: dict[str, dict[str, Any]],
) -> None:
    lines = [
        f"# Model Sweep Comparison ({run_id})",
        "",
        f"- Dataset: `{dataset_id}`",
        f"- Camera: `{camera_key}`",
        "",
        "| Model | Episodes | Schema Valid | Repair Rate | Avg Latency (s) | Avg Tokens | Task Consistency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for alias, metrics in model_metrics.items():
        lines.append(
            "| {alias} | {episode_count:.0f} | {schema_valid_rate:.3f} | {parse_repair_rate:.3f} | "
            "{avg_latency_s:.3f} | {avg_total_tokens:.1f} | {task_consistency_rate:.3f} |".format(
                alias=alias,
                episode_count=float(metrics.get("episode_count", 0)),
                schema_valid_rate=float(metrics.get("schema_valid_rate", 0.0)),
                parse_repair_rate=float(metrics.get("parse_repair_rate", 0.0)),
                avg_latency_s=float(metrics.get("avg_latency_s", 0.0)),
                avg_total_tokens=float(metrics.get("avg_total_tokens", 0.0)),
                task_consistency_rate=float(metrics.get("task_consistency_rate", 0.0)),
            )
        )

    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
