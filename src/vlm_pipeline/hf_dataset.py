from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import pyarrow as pa
import pyarrow.parquet as pq

from .episode_index import EpisodeRef


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    info_path: Path
    episodes_parquet_paths: list[Path]


class HFDatasetAdapter:
    def __init__(self, dataset_id: str, hf_home: Path, token: str | None = None) -> None:
        self.dataset_id = dataset_id
        self.hf_home = Path(hf_home)
        self.token = token

    def _download_patterns(self, patterns: list[str]) -> None:
        self.hf_home.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "repo_id": self.dataset_id,
            "repo_type": "dataset",
            "local_dir": str(self.hf_home),
            "allow_patterns": patterns,
            "token": self.token,
        }
        try:
            snapshot_download(local_dir_use_symlinks=False, **kwargs)
        except TypeError:
            snapshot_download(**kwargs)

    def ensure_metadata(self) -> DatasetInfo:
        info_path = self.hf_home / "meta" / "info.json"
        if not info_path.exists():
            self._download_patterns(["meta/info.json"])
        parquet_paths = sorted((self.hf_home / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not parquet_paths:
            self._download_patterns(["meta/episodes/chunk-*/file-*.parquet"])
            parquet_paths = sorted((self.hf_home / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not info_path.exists() or not parquet_paths:
            raise FileNotFoundError(
                f"Failed to resolve metadata for {self.dataset_id} under {self.hf_home}"
            )
        return DatasetInfo(
            dataset_id=self.dataset_id,
            info_path=info_path,
            episodes_parquet_paths=parquet_paths,
        )

    def load_info(self) -> dict[str, Any]:
        meta = self.ensure_metadata()
        return json.loads(meta.info_path.read_text(encoding="utf-8"))

    def available_camera_keys(self) -> list[str]:
        info = self.load_info()
        keys: list[str] = []
        for feature_name, feature_meta in info.get("features", {}).items():
            if isinstance(feature_meta, dict) and feature_meta.get("dtype") == "video":
                keys.append(feature_name)
        if keys:
            return sorted(keys)

        meta = self.ensure_metadata()
        first_table = pq.read_table(meta.episodes_parquet_paths[0])
        prefix = "videos/"
        suffix = "/chunk_index"
        for col in first_table.column_names:
            if col.startswith(prefix) and col.endswith(suffix):
                keys.append(col[len(prefix) : -len(suffix)])
        return sorted(set(keys))

    def _episode_table(self) -> pa.Table:
        meta = self.ensure_metadata()
        tables = [pq.read_table(path) for path in meta.episodes_parquet_paths]
        if len(tables) == 1:
            return tables[0]
        return pa.concat_tables(tables)

    def load_episode_rows(self) -> list[dict[str, Any]]:
        table = self._episode_table()
        rows = table.to_pylist()
        rows.sort(key=lambda row: int(row["episode_index"]))
        return rows

    def _video_path_from_row(self, row: dict[str, Any], camera_key: str) -> Path:
        chunk = int(row[f"videos/{camera_key}/chunk_index"])
        file_idx = int(row[f"videos/{camera_key}/file_index"])
        return (
            self.hf_home
            / "videos"
            / camera_key
            / f"chunk-{chunk:03d}"
            / f"file-{file_idx:03d}.mp4"
        )

    def _ensure_video_for_row(self, row: dict[str, Any], camera_key: str) -> Path:
        video_path = self._video_path_from_row(row, camera_key)
        if video_path.exists():
            return video_path
        pattern = (
            f"videos/{camera_key}/"
            f"chunk-{int(row[f'videos/{camera_key}/chunk_index']):03d}/"
            f"file-{int(row[f'videos/{camera_key}/file_index']):03d}.mp4"
        )
        self._download_patterns([pattern])
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found after download: {video_path}")
        return video_path

    def resolve_episode_refs(self, camera_key: str, episode_indices: list[int]) -> list[EpisodeRef]:
        rows_by_index = {int(row["episode_index"]): row for row in self.load_episode_rows()}
        refs: list[EpisodeRef] = []
        for episode_idx in episode_indices:
            if episode_idx not in rows_by_index:
                raise ValueError(f"Episode index {episode_idx} not found")
            row = rows_by_index[episode_idx]
            video_path = self._ensure_video_for_row(row, camera_key)
            tasks = row.get("tasks") or []
            task_text = tasks[0] if tasks else ""
            refs.append(
                EpisodeRef(
                    episode_index=episode_idx,
                    task_text=task_text,
                    camera_key=camera_key,
                    video_path=video_path,
                    start_s=float(row[f"videos/{camera_key}/from_timestamp"]),
                    end_s=float(row[f"videos/{camera_key}/to_timestamp"]),
                    num_frames=int(row.get("length", 0)),
                )
            )
        return refs
