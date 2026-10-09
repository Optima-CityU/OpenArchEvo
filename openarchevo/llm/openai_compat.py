"""Clients for OpenAI-compatible chat-completion and embedding endpoints.

Most providers (including Gemini, DashScope / Qwen and vLLM servers) expose an
OpenAI-compatible HTTP API. Base URLs and keys are read from environment
variables whose *names* are configured in ``configs/search.yaml``; no key is ever
stored in a config file.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import httpx

from openarchevo.llm.base import Proposal

OUTPUT_INSTRUCTION = (
    "\n\n## Response Format\nRespond with a single JSON object with exactly three string fields: "
    '"name" (model identifier), "motivation" (design rationale) and "code" (the complete Python '
    "source of the module, including all imports)."
)


def _env(name: str, required: bool = True) -> Optional[str]:
    value = os.environ.get(name)
    if required and not value:
        raise EnvironmentError(f"Environment variable {name} is not set.")
    return value


def post_json(client: httpx.Client, url: str, api_key: Optional[str], body: dict, retries: int = 3) -> dict:
    """POST with exponential backoff on rate limits, server errors and network failures."""
    for attempt in range(retries + 1):
        try:
            resp = client.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"})
            if resp.status_code != 429 and resp.status_code < 500:
                resp.raise_for_status()
                return resp.json()
            error: Exception = httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
        except httpx.TransportError as exc:
            error = exc
        if attempt < retries:
            time.sleep(2.0 ** attempt)
    raise error


def parse_proposal(text: str) -> Proposal:
    """Parse a JSON proposal, tolerating Markdown fences around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(cleaned)
    fields = {k: str(data.get(k, "")).strip() for k in ("name", "motivation", "code")}
    if not fields["code"] or not fields["motivation"]:
        raise ValueError("Proposal is missing 'code' or 'motivation'.")
    return Proposal(name=fields["name"], rationale=fields["motivation"], code=data["code"])


class OpenAICompatibleLLM:
    """Program generator backed by ``POST {base_url}/chat/completions``."""

    def __init__(self, model: str, base_url_env: str = "OPENAI_BASE_URL", api_key_env: str = "OPENAI_API_KEY",
                 max_tokens: int = 32768, temperature: Optional[float] = None, timeout: float = 600.0,
                 json_mode: bool = True, retries: int = 3):
        self.model = model
        self.base_url = (_env(base_url_env) or "").rstrip("/")
        self.api_key = _env(api_key_env)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = json_mode
        self.retries = retries
        self._client = httpx.Client(timeout=timeout)

    def propose(self, system_prompt: str, user_prompt: str) -> Proposal:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt + OUTPUT_INSTRUCTION},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        out = post_json(self._client, f"{self.base_url}/chat/completions", self.api_key, body, self.retries)
        return parse_proposal(out["choices"][0]["message"]["content"])


class OpenAICompatibleEmbedder:
    """Rationale embedder backed by ``POST {base_url}/embeddings``."""

    def __init__(self, model: str, base_url_env: str = "EMBEDDING_BASE_URL", api_key_env: str = "EMBEDDING_API_KEY",
                 dimensions: Optional[int] = None, timeout: float = 60.0):
        self.model = model
        self.base_url = (_env(base_url_env) or "").rstrip("/")
        self.api_key = _env(api_key_env)
        self.dimensions = dimensions
        self._client = httpx.Client(timeout=timeout)

    def embed(self, text: str) -> list[float]:
        body: dict = {"model": self.model, "input": text}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        out = post_json(self._client, f"{self.base_url}/embeddings", self.api_key, body)
        return [float(x) for x in out["data"][0]["embedding"]]
