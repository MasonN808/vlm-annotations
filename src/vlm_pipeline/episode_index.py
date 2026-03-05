from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path


@dataclass(frozen=True)
class EpisodeRef:
    episode_index: int
    task_text: str
    camera_key: str
    video_path: Path
    start_s: float
    end_s: float
    num_frames: int


TASK_PATTERN = re.compile(
    r"As the\s+(?P<robot>\w+)\s+robot,\s+pick up the\s+(?P<object>\w+)\s+cube",
    re.IGNORECASE,
)


def parse_task_entities(task_text: str) -> tuple[str | None, str | None]:
    match = TASK_PATTERN.search(task_text)
    if not match:
        return None, None
    return match.group("robot").lower(), match.group("object").lower()


def sanitize_model_alias(model_id: str) -> str:
    alias = re.sub(r"[^a-zA-Z0-9._-]", "_", model_id.strip())
    return alias.strip("._-") or "model"


def pick_episode_indices(
    available_indices: list[int],
    explicit_indices: list[int] | None,
    max_episodes: int,
) -> list[int]:
    if explicit_indices:
        wanted = sorted(set(explicit_indices))
        missing = [idx for idx in wanted if idx not in set(available_indices)]
        if missing:
            raise ValueError(f"Unknown episode indices requested: {missing}")
        return wanted
    return sorted(available_indices)[:max_episodes]
