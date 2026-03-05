from __future__ import annotations

from vlm_pipeline.config import schema_path
from vlm_pipeline.parser import load_schema, parse_with_optional_repair


def test_parser_accepts_valid_json():
    schema = load_schema(schema_path())
    raw = '{"completion_status":"completed","summary":"done","events":[]}'
    result = parse_with_optional_repair(raw_text=raw, schema=schema)
    assert result.ok
    assert not result.repaired


def test_parser_rejects_invalid_schema_without_repair():
    schema = load_schema(schema_path())
    raw = '{"completion_status":"completed","summary":"","events":[]}'
    result = parse_with_optional_repair(raw_text=raw, schema=schema)
    assert not result.ok
    assert "Schema validation error" in (result.error or "")


def test_parser_succeeds_with_one_repair_pass():
    schema = load_schema(schema_path())

    def repair_fn(_: str) -> str:
        return '{"completion_status":"partial","summary":"fixed","events":[]}'

    result = parse_with_optional_repair(raw_text="not-json", schema=schema, repair_fn=repair_fn)
    assert result.ok
    assert result.repaired
