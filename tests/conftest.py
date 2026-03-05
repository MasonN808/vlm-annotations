from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture()
def sample_hf_dataset(tmp_path: Path) -> Path:
    hf_home = tmp_path / "hf"
    meta_dir = hf_home / "meta" / "episodes" / "chunk-000"
    video_dir = hf_home / "videos" / "observation.images.wrist" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "fps": 15,
        "features": {
            "observation.images.wrist": {"dtype": "video"},
            "observation.images.side": {"dtype": "video"},
        },
    }
    (hf_home / "meta").mkdir(parents=True, exist_ok=True)
    (hf_home / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    rows = {
        "episode_index": [0, 1, 2],
        "tasks": [
            ["As the blue robot, pick up the red cube and place it in the bin."],
            ["As the red robot, pick up the green cube and place it in the bin."],
            ["As the blue robot, pick up the yellow cube and place it in the bin."],
        ],
        "length": [30, 30, 30],
        "videos/observation.images.wrist/chunk_index": [0, 0, 0],
        "videos/observation.images.wrist/file_index": [0, 0, 0],
        "videos/observation.images.wrist/from_timestamp": [0.0, 2.0, 4.0],
        "videos/observation.images.wrist/to_timestamp": [2.0, 4.0, 6.0],
    }
    table = pa.table(rows)
    pq.write_table(table, meta_dir / "file-000.parquet")

    video_path = video_dir / "file-000.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x64:rate=4",
        "-t",
        "6",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create test video: {proc.stderr}")

    return hf_home
