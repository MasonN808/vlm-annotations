from __future__ import annotations

import json

from vlm_pipeline.cli import main


def test_inspect_hf_json_output(sample_hf_dataset, capsys):
    rc = main(
        [
            "inspect-hf",
            "--dataset-id",
            "dummy/pick_place",
            "--hf-home",
            str(sample_hf_dataset),
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["total_episodes"] == 3
    assert "observation.images.wrist" in payload["camera_keys"]
