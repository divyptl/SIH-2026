"""Thin async wrapper around the OpenRouter SDK.

Note: this module must NOT be named ``openrouter.py``. A top-level module with
that name shadows the installed SDK package and makes ``from openrouter import
OpenRouter`` import itself.
"""

from __future__ import annotations

import json
import re
from typing import Any

from openrouter import OpenRouter

from config import Settings

# Models often wrap JSON in a ``` fence even when asked not to.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class OpenRouterError(RuntimeError):
    """Raised when the upstream call fails or returns something unusable."""


class OpenRouterClient:
    """Issues chat completions, with or without images attached."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openrouter_api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set. Add it to backend/.env to enable analysis."
            )
        self._settings = settings

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        image_data_uris: list[str] | None = None,
        json_object: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> tuple[str, dict[str, Any]]:
        """Run one completion and return ``(text, usage)``.

        Images are passed as ``image_url`` content parts holding data URIs,
        which is what OpenRouter expects for inline uploads.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for uri in image_data_uris or []:
            content.append({"type": "image_url", "image_url": {"url": uri}})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "http_referer": self._settings.app_referer,
            "x_open_router_title": self._settings.app_title,
        }

        if json_object:
            # Deliberately json_object rather than json_schema: on the free
            # vision models this project targets, strict json_schema mode makes
            # some providers emit their internal serialisation format instead of
            # the requested object. The expected shape is pinned in the system
            # prompt and validated on the way out instead.
            kwargs["response_format"] = {"type": "json_object"}

        try:
            async with OpenRouter(
                api_key=self._settings.openrouter_api_key,
                timeout_ms=int(self._settings.request_timeout_s * 1000),
            ) as client:
                result = await client.chat.send_async(**kwargs)
        except Exception as exc:  # SDK raises a family of transport/API errors
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        choices = getattr(result, "choices", None)
        if not choices:
            raise OpenRouterError("OpenRouter returned no choices.")

        text = getattr(choices[0].message, "content", None)
        if not isinstance(text, str) or not text.strip():
            raise OpenRouterError("OpenRouter returned an empty message.")

        usage_obj = getattr(result, "usage", None)
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
            "cost": getattr(usage_obj, "cost", None),
        }
        return text, usage


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Structured outputs are requested, but not every provider honours them, so
    fall back to unwrapping a code fence and then to the first balanced object
    in the text before giving up.
    """
    candidates: list[str] = [text.strip()]

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise OpenRouterError(f"Model did not return JSON. Got: {text[:300]}")
