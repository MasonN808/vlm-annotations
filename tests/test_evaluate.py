from __future__ import annotations

from vlm_pipeline.evaluate import build_episode_eval_record, compute_model_metrics


def test_compute_model_metrics_with_task_consistency():
    annotation = {
        "completion_status": "completed",
        "summary": "Blue robot picked and placed red cube",
        "events": [
            {
                "label": "grasp",
                "start_s": 1.0,
                "end_s": 1.5,
                "confidence": 0.9,
                "actor_robot": "blue",
                "target_object": "red",
                "evidence_frame_ids": [1, 2],
                "notes": "gripper closes",
            }
        ],
    }
    record = build_episode_eval_record(
        episode_index=0,
        task_text="As the blue robot, pick up the red cube and place it in the bin.",
        annotation=annotation,
        schema_valid=True,
        parse_repaired=False,
        latency_s=1.2,
        total_tokens=128,
    )
    metrics = compute_model_metrics([record])

    assert metrics["episode_count"] == 1
    assert metrics["schema_valid_rate"] == 1.0
    assert metrics["task_consistency_rate"] == 1.0
    assert metrics["robot_match_rate"] == 1.0
    assert metrics["object_match_rate"] == 1.0
