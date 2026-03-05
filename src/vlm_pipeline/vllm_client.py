from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import httpx


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    raw: dict[str, Any]
    usage: dict[str, Any]
    latency_s: float


class VLLMClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatCompletionResult:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Strongly bias vLLM/Qwen towards machine-parseable output.
        payload["response_format"] = {"type": "json_object"}
        if "qwen" in model_id.lower():
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        start = time.perf_counter()
        try:
            data = self._post_completion(payload=payload, headers=headers)
        except RuntimeError as exc:
            message = str(exc)
            # Some server/model combinations may not support response_format
            # or chat_template kwargs; retry with a minimal payload.
            if "response_format" in message or "chat_template_kwargs" in message:
                payload.pop("response_format", None)
                payload.pop("extra_body", None)
                data = self._post_completion(payload=payload, headers=headers)
            else:
                raise
        latency_s = time.perf_counter() - start

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM chat completion returned no choices")

        message_content = choices[0].get("message", {}).get("content", "")
        if isinstance(message_content, list):
            chunks: list[str] = []
            for item in message_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                else:
                    chunks.append(str(item))
            text = "\n".join(chunks)
        else:
            text = str(message_content)

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ChatCompletionResult(content=text, raw=data, usage=usage, latency_s=latency_s)

    def _post_completion(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        if response.status_code >= 400:
            raise RuntimeError(
                f"vLLM chat completion failed: HTTP {response.status_code}: {response.text[:500]}"
            )
        return response.json()
