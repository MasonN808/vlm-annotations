from __future__ import annotations

import json
from pathlib import Path

import vlm_pipeline.vllm_client as vllm_client
from vlm_pipeline.cli import main


def test_annotate_hf_end_to_end_with_mocked_vllm(sample_hf_dataset, monkeypatch, tmp_path):
    def fake_chat_completion(self, model_id, messages, temperature=0.0, max_tokens=1024):
        payload = {
            "completion_status": "completed",
            "summary": "Blue robot picks and places the target cube.",
            "events": [
                {
                    "label": "reach",
                    "start_s": 0.2,
                    "end_s": 0.8,
                    "confidence": 0.88,
                    "actor_robot": "blue",
                    "target_object": "red",
                    "evidence_frame_ids": [0],
                    "notes": "arm moves toward cube"
                }
            ]
        }
        return vllm_client.ChatCompletionResult(
            content=json.dumps(payload),
            raw={"choices": [{"message": {"content": json.dumps(payload)}}]},
            usage={"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
            latency_s=0.5,
        )

    monkeypatch.setattr(vllm_client.VLLMClient, "chat_completion", fake_chat_completion)

    out_root = tmp_path / "out"
    rc = main(
        [
            "annotate-hf",
            "--dataset-id",
            "dummy/pick_place",
            "--hf-home",
            str(sample_hf_dataset),
            "--camera-key",
            "observation.images.wrist",
            "--model-id",
            "Qwen/Qwen3-VL-4B-Instruct",
            "--episode-index",
            "0",
            "--fps",
            "1.0",
            "--max-frames",
            "4",
            "--out",
            str(out_root),
            "--run-id",
            "testrun",
        ]
    )

    assert rc == 0
    run_root = out_root / "testrun"
    assert (run_root / "manifest.json").exists()
    assert (run_root / "aggregate_metrics.json").exists()

    model_alias = "Qwen_Qwen3-VL-4B-Instruct"
    annotation_path = run_root / model_alias / "episodes" / "000" / "annotation.json"
    assert annotation_path.exists()

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert annotation["parse"]["valid"] is True
    assert annotation["completion_status"] == "completed"
