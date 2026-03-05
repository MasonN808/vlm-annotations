from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from .frame_sampler import FrameRecord


def _to_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_annotation_messages(
    frame_records: list[FrameRecord],
    task_text: str,
    include_task_hint: bool,
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    schema_text = json.dumps(schema, indent=2)
    task_line = f"Task hint: {task_text}\n" if include_task_hint and task_text else ""
    frame_lines = "\n".join(
        f"- frame_id={f.frame_id}, timestamp_s={f.timestamp_s:.3f}" for f in frame_records
    )
    instruction = (
        "You are annotating a pick-and-place robot episode.\n"
        "IMPORTANT: Do not reveal reasoning. Do not output thinking traces.\n"
        "Start your response with '{' and end it with '}'.\n"
        f"{task_line}"
        "Return STRICT JSON only with no markdown fences.\n"
        "The JSON MUST validate against this schema:\n"
        f"{schema_text}\n"
        "Rules:\n"
        "- Use event labels from the enum only.\n"
        "- confidence must be between 0 and 1.\n"
        "- actor_robot should be blue/red/unknown.\n"
        "- target_object should be one of cube colors or unknown.\n"
        "- evidence_frame_ids should reference provided frame_id values.\n"
        "Provided frames:\n"
        f"{frame_lines}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    for frame in frame_records:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _to_data_url(frame.path)},
            }
        )
    return [
        {
            "role": "system",
            "content": (
                "You produce high-precision robotic event annotations in strict JSON. "
                "Never include chain-of-thought, 'thinking process', or explanation text."
            ),
        },
        {"role": "user", "content": content},
    ]


def build_repair_messages(
    raw_response: str,
    parse_error: str,
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    schema_text = json.dumps(schema, indent=2)
    prompt = (
        "Fix the annotation output to strict JSON only.\n"
        "Do not add markdown, explanations, or thinking text.\n"
        "Output ONLY one JSON object.\n"
        "Start with '{' and end with '}'.\n"
        f"Validation error: {parse_error}\n"
        "Target schema:\n"
        f"{schema_text}\n"
        "Original output to repair:\n"
        f"{raw_response}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You repair JSON to match a schema exactly. "
                "Return exactly one JSON object and nothing else."
            ),
        },
        {"role": "user", "content": prompt},
    ]
