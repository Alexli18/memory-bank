"""Thin httpx wrapper for Ollama API (embed + chat)."""

from __future__ import annotations

from typing import Any

import httpx


class OllamaError(Exception):
    """Base error for Ollama operations."""


class OllamaNotRunningError(OllamaError):
    """Ollama server is not reachable."""


class OllamaModelNotFoundError(OllamaError):
    """Requested model is not available."""


class OllamaTimeoutError(OllamaError):
    """Ollama request timed out."""


class OllamaClient:
    """Client for Ollama REST API.

    Args:
        base_url: Ollama server URL (default: http://localhost:11434).
        embed_model: Model for embeddings (default: nomic-embed-text).
        chat_model: Model for chat/summarization (default: gemma3:4b).
        embed_timeout: Timeout in seconds for embed requests.
        chat_timeout: Timeout in seconds for chat requests.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        chat_model: str = "gemma3:4b",
        embed_timeout: float = 120.0,
        chat_timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.embed_timeout = embed_timeout
        self.chat_timeout = chat_timeout

    def is_running(self) -> bool:
        """Check if Ollama server is reachable via GET /api/tags."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Embed one or more texts via POST /api/embed.

        Args:
            texts: Single string or list of strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """
        if isinstance(texts, str):
            texts = [texts]

        payload = {
            "model": self.embed_model,
            "input": texts,
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.embed_timeout,
            )
        except httpx.ConnectError as e:
            raise OllamaNotRunningError(
                f"Cannot connect to Ollama at {self.base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError("Ollama embed request timed out") from e

        if resp.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Model '{self.embed_model}' not found. Run: ollama pull {self.embed_model}"
            )
        resp.raise_for_status()

        data = resp.json()
        embeddings: list[list[float]] = data["embeddings"]
        return embeddings

    def chat(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        as_json: bool = False,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> dict[str, Any] | str:
        """Send a chat request via POST /api/chat with stream=false.

        Args:
            user_prompt: The user message.
            system_prompt: Optional system message.
            as_json: If True, request JSON format output.
            temperature: Sampling temperature (default 0.0 for determinism).
            seed: Random seed for determinism (default 42).

        Returns:
            Parsed dict if as_json=True, otherwise raw string content.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "top_k": 1,
            },
        }
        if as_json:
            payload["format"] = "json"

        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.chat_timeout,
            )
        except httpx.ConnectError as e:
            raise OllamaNotRunningError(
                f"Cannot connect to Ollama at {self.base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError("Ollama chat request timed out") from e

        if resp.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Model '{self.chat_model}' not found. Run: ollama pull {self.chat_model}"
            )
        resp.raise_for_status()

        data = resp.json()
        content = data["message"]["content"]

        if as_json:
            import json

            parsed: dict[str, Any] = json.loads(content)
            return parsed

        result: str = content
        return result


def client_from_config(config: dict[str, Any]) -> OllamaClient:
    """Create an OllamaClient from a config dict (as returned by storage.read_config)."""
    ollama_cfg = config.get("ollama", {})
    return OllamaClient(
        base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
        embed_model=ollama_cfg.get("embed_model", "nomic-embed-text"),
        chat_model=ollama_cfg.get("chat_model", "gemma3:4b"),
    )
