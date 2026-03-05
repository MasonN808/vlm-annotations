from __future__ import annotations

import json

import vlm_pipeline.cli as cli
from vlm_pipeline.cli import main


def test_sweep_writes_outputs_when_one_model_fails(sample_hf_dataset, monkeypatch, tmp_path):
    original_run_model = cli._run_model

    def fake_run_model(*args, **kwargs):
        model_id = kwargs["model_id"]
        if model_id.endswith("8B-Instruct"):
            raise RuntimeError("intentional test failure")
        return original_run_model(*args, **kwargs)

    monkeypatch.setattr(cli, "_run_model", fake_run_model)

    # Mock actual inference for the successful model path.
    import vlm_pipeline.vllm_client as vllm_client

    def fake_chat_completion(self, model_id, messages, temperature=0.0, max_tokens=1024):
        payload = {
            "completion_status": "completed",
            "summary": "ok",
            "events": [],
        }
        return vllm_client.ChatCompletionResult(
            content=json.dumps(payload),
            raw={"choices": [{"message": {"content": json.dumps(payload)}}]},
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_s=0.1,
        )

    monkeypatch.setattr(vllm_client.VLLMClient, "chat_completion", fake_chat_completion)

    out_root = tmp_path / "out"
    rc = main(
        [
            "sweep-hf",
            "--dataset-id",
            "dummy/pick_place",
            "--hf-home",
            str(sample_hf_dataset),
            "--camera-key",
            "observation.images.wrist",
            "--episode-index",
            "0",
            "--model-id",
            "Qwen/Qwen3-VL-4B-Instruct",
            "--model-id",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--no-pause-between-models",
            "--out",
            str(out_root),
            "--run-id",
            "sweeptest",
        ]
    )

    assert rc == 0
    run_root = out_root / "sweeptest"
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))

    statuses = {m["model_alias"]: m["status"] for m in manifest["models"]}
    assert statuses["Qwen_Qwen3-VL-4B-Instruct"] == "ok"
    assert statuses["Qwen_Qwen3-VL-8B-Instruct"] == "failed"
    assert (run_root / "aggregate_metrics.json").exists()
    assert (run_root / "comparison.md").exists()
