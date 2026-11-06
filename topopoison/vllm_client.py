from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 360.0,
        retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def _request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{(base_url or self.base_url).rstrip('/')}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="GET" if data is None else "POST",
        )
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt + 1 == self.retries:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("vLLM request failed")

    def models(self) -> list[str]:
        response = self._request("/models")
        return [str(item["id"]) for item in response.get("data", []) if "id" in item]

    def version(self) -> str:
        server_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return str(self._request("/version", base_url=server_root).get("version") or "")

    def check(self) -> None:
        models = self.models()
        if self.model not in models:
            raise RuntimeError(
                f"vLLM serves {models or 'no models'}, but the configured model is {self.model}"
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
    ) -> tuple[str, dict[str, Any]]:
        response = self._request(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            },
        )
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM returned no choices: {response}")
        output = str(choices[0].get("message", {}).get("content") or "")
        return output, {
            "usage": response.get("usage") or {},
            "finish_reason": choices[0].get("finish_reason"),
            "model": response.get("model", self.model),
        }
