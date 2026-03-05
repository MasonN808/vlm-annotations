from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from .episode_index import EpisodeRef


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp_s: float
    path: Path

    def to_json(self, root: Path) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "path": str(self.path.relative_to(root)),
        }


def build_ffmpeg_command(
    input_video: Path,
    output_pattern: Path,
    start_s: float,
    end_s: float,
    fps: float,
    max_frames: int,
) -> list[str]:
    duration_s = max(0.0, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(input_video),
        "-t",
        f"{duration_s:.6f}",
        "-vf",
        f"fps={fps:.6f}",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "2",
        str(output_pattern),
    ]
    return cmd


def sample_episode_frames(
    episode: EpisodeRef,
    output_dir: Path,
    fps: float,
    max_frames: int,
) -> list[FrameRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_command(
        input_video=episode.video_path,
        output_pattern=output_dir / "frame_%04d.jpg",
        start_s=episode.start_s,
        end_s=episode.end_s,
        fps=fps,
        max_frames=max_frames,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()}")

    frame_paths = sorted(output_dir.glob("frame_*.jpg"))
    records: list[FrameRecord] = []
    for idx, path in enumerate(frame_paths):
        records.append(
            FrameRecord(
                frame_id=idx,
                timestamp_s=episode.start_s + (idx / fps),
                path=path,
            )
        )
    return records
