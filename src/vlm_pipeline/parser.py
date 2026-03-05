from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Callable

import jsonschema


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    annotation: dict[str, Any] | None
    error: str | None
    repaired: bool
    repaired_raw_text: str | None


def load_schema(schema_path: Path) -> dict[str, Any]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

    start = stripped.find("{")
    if start == -1:
        return stripped

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]

    return stripped[start:]


def _parse_and_validate(raw_text: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    candidate = _extract_json_candidate(raw_text)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}"

    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        return None, f"Schema validation error: {exc.message}"

    return data, None


def parse_with_optional_repair(
    raw_text: str,
    schema: dict[str, Any],
    repair_fn: Callable[[str], str] | None = None,
) -> ParseResult:
    parsed, error = _parse_and_validate(raw_text, schema)
    if parsed is not None:
        return ParseResult(ok=True, annotation=parsed, error=None, repaired=False, repaired_raw_text=None)

    if repair_fn is None:
        return ParseResult(ok=False, annotation=None, error=error, repaired=False, repaired_raw_text=None)

    repaired_text = repair_fn(error or "Unknown parse error")
    repaired_data, repaired_error = _parse_and_validate(repaired_text, schema)
    if repaired_data is not None:
        return ParseResult(
            ok=True,
            annotation=repaired_data,
            error=None,
            repaired=True,
            repaired_raw_text=repaired_text,
        )

    return ParseResult(
        ok=False,
        annotation=None,
        error=repaired_error,
        repaired=True,
        repaired_raw_text=repaired_text,
    )
