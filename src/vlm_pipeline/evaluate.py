from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from .episode_index import parse_task_entities


@dataclass(frozen=True)
class EpisodeEvalRecord:
    episode_index: int
    schema_valid: bool
    parse_repaired: bool
    latency_s: float
    total_tokens: int
    completion_status: str
    expected_robot: str | None
    expected_object: str | None
    predicted_robot: str | None
    predicted_object: str | None


def extract_predicted_entities(annotation: dict[str, Any]) -> tuple[str | None, str | None]:
    events = annotation.get("events") if isinstance(annotation, dict) else None
    if not isinstance(events, list):
        return None, None

    robot_votes: dict[str, int] = {}
    object_votes: dict[str, int] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        robot = str(ev.get("actor_robot", "")).strip().lower()
        target = str(ev.get("target_object", "")).strip().lower()
        if robot and robot != "unknown":
            robot_votes[robot] = robot_votes.get(robot, 0) + 1
        if target and target != "unknown":
            object_votes[target] = object_votes.get(target, 0) + 1

    robot_pred = max(robot_votes, key=robot_votes.get) if robot_votes else None
    object_pred = max(object_votes, key=object_votes.get) if object_votes else None
    return robot_pred, object_pred


def build_episode_eval_record(
    episode_index: int,
    task_text: str,
    annotation: dict[str, Any] | None,
    schema_valid: bool,
    parse_repaired: bool,
    latency_s: float,
    total_tokens: int,
) -> EpisodeEvalRecord:
    expected_robot, expected_object = parse_task_entities(task_text)
    predicted_robot, predicted_object = extract_predicted_entities(annotation or {})
    completion_status = "failed"
    if isinstance(annotation, dict):
        completion_status = str(annotation.get("completion_status", "failed"))

    return EpisodeEvalRecord(
        episode_index=episode_index,
        schema_valid=schema_valid,
        parse_repaired=parse_repaired,
        latency_s=latency_s,
        total_tokens=total_tokens,
        completion_status=completion_status,
        expected_robot=expected_robot,
        expected_object=expected_object,
        predicted_robot=predicted_robot,
        predicted_object=predicted_object,
    )


def _ratio(matches: list[bool]) -> float:
    if not matches:
        return 0.0
    return sum(1 for m in matches if m) / len(matches)


def compute_model_metrics(records: list[EpisodeEvalRecord]) -> dict[str, Any]:
    if not records:
        return {
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

    schema_valid_rate = _ratio([r.schema_valid for r in records])
    parse_repair_rate = _ratio([r.parse_repaired for r in records])
    completion_rate = _ratio([r.completion_status in {"completed", "partial"} for r in records])

    robot_matches = [
        r.expected_robot is not None and r.predicted_robot == r.expected_robot
        for r in records
        if r.expected_robot is not None
    ]
    object_matches = [
        r.expected_object is not None and r.predicted_object == r.expected_object
        for r in records
        if r.expected_object is not None
    ]
    pair_matches = [
        (r.expected_robot is not None and r.predicted_robot == r.expected_robot)
        and (r.expected_object is not None and r.predicted_object == r.expected_object)
        for r in records
        if r.expected_robot is not None and r.expected_object is not None
    ]

    return {
        "episode_count": len(records),
        "schema_valid_rate": schema_valid_rate,
        "parse_repair_rate": parse_repair_rate,
        "avg_latency_s": mean([r.latency_s for r in records]),
        "avg_total_tokens": mean([r.total_tokens for r in records]),
        "completion_rate": completion_rate,
        "robot_match_rate": _ratio(robot_matches),
        "object_match_rate": _ratio(object_matches),
        "task_consistency_rate": _ratio(pair_matches),
    }


def compute_aggregate_metrics(model_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for alias, metrics in model_metrics.items():
        row = {"model_alias": alias}
        row.update(metrics)
        models.append(row)

    sorted_by_consistency = sorted(
        models,
        key=lambda row: row.get("task_consistency_rate", 0.0),
        reverse=True,
    )
    best = sorted_by_consistency[0]["model_alias"] if sorted_by_consistency else None
    return {
        "models": models,
        "best_model_by_task_consistency": best,
    }
