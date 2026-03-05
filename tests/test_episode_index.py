from __future__ import annotations

from vlm_pipeline.hf_dataset import HFDatasetAdapter


def test_resolve_episode_refs_maps_video_path_and_timestamps(sample_hf_dataset):
    adapter = HFDatasetAdapter(dataset_id="dummy/pick_place", hf_home=sample_hf_dataset)
    refs = adapter.resolve_episode_refs(camera_key="observation.images.wrist", episode_indices=[1])

    assert len(refs) == 1
    ref = refs[0]
    assert ref.episode_index == 1
    assert ref.start_s == 2.0
    assert ref.end_s == 4.0
    assert ref.video_path == (
        sample_hf_dataset
        / "videos"
        / "observation.images.wrist"
        / "chunk-000"
        / "file-000.mp4"
    )
