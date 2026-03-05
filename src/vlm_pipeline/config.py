from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATASET_ID = "DorianAtSchool/pick_place"
DEFAULT_CAMERA_KEY = "observation.images.wrist"
DEFAULT_MAX_EPISODES = 5
DEFAULT_FPS = 1.0
DEFAULT_MAX_FRAMES = 16
DEFAULT_MAX_TOKENS = 1024
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_API_KEY = "EMPTY"
DEFAULT_MODEL_IDS = (
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
)


def default_hf_home() -> Path:
    return Path(os.getenv("HF_HOME", ".cache/hf"))


def default_vllm_base_url() -> str:
    return os.getenv("VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL)


def default_vllm_api_key() -> str:
    return os.getenv("VLLM_API_KEY", DEFAULT_VLLM_API_KEY)


def default_hf_token() -> str | None:
    return os.getenv("HUGGINGFACE_HUB_TOKEN")


def schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "annotation_v1.json"
