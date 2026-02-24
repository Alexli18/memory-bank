"""Tests for mb.ollama_client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from mb.ollama_client import (
    OllamaClient,
    OllamaModelNotFoundError,
    OllamaNotRunningError,
    client_from_config,
)


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
) -> MagicMock:
    """Create a mock httpx.Response with common attributes."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------


@patch("httpx.get")
def test_is_running_returns_true_on_200(mock_get: MagicMock) -> None:
    """is_running returns True when Ollama responds with HTTP 200."""
    mock_get.return_value = _mock_response(status_code=200)

    client = OllamaClient()
    assert client.is_running() is True

    mock_get.assert_called_once_with(
        "http://localhost:11434/api/tags", timeout=5.0,
    )


@patch("httpx.get")
def test_is_running_returns_false_on_connect_error(mock_get: MagicMock) -> None:
    """is_running returns False when Ollama is unreachable."""
    mock_get.side_effect = httpx.ConnectError("Connection refused")

    client = OllamaClient()
    assert client.is_running() is False


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


@patch("httpx.post")
def test_embed_single_string_returns_embeddings(mock_post: MagicMock) -> None:
    """embed() accepts a single string and returns a list of embedding vectors."""
    embeddings = [[0.1, 0.2, 0.3]]
    mock_post.return_value = _mock_response(
        status_code=200, json_data={"embeddings": embeddings},
    )

    client = OllamaClient()
    result = client.embed("hello world")

    assert result == embeddings
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["input"] == ["hello world"]


@patch("httpx.post")
def test_embed_list_of_strings_returns_embeddings(mock_post: MagicMock) -> None:
    """embed() accepts a list of strings and returns matching embedding vectors."""
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    mock_post.return_value = _mock_response(
        status_code=200, json_data={"embeddings": embeddings},
    )

    client = OllamaClient()
    result = client.embed(["one", "two"])

    assert result == embeddings
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["input"] == ["one", "two"]


@patch("httpx.post")
def test_embed_connect_error_raises_not_running(mock_post: MagicMock) -> None:
    """embed() raises OllamaNotRunningError when server is unreachable."""
    mock_post.side_effect = httpx.ConnectError("Connection refused")

    client = OllamaClient()
    with pytest.raises(OllamaNotRunningError, match="Cannot connect"):
        client.embed("test")


@patch("httpx.post")
def test_embed_404_raises_model_not_found(mock_post: MagicMock) -> None:
    """embed() raises OllamaModelNotFoundError on HTTP 404."""
    mock_post.return_value = _mock_response(status_code=404)

    client = OllamaClient()
    with pytest.raises(OllamaModelNotFoundError, match="not found"):
        client.embed("test")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@patch("httpx.post")
def test_chat_returns_string_content(mock_post: MagicMock) -> None:
    """chat() returns raw string content by default."""
    mock_post.return_value = _mock_response(
        status_code=200,
        json_data={"message": {"content": "Hello, I am a bot."}},
    )

    client = OllamaClient()
    result = client.chat("Hi")

    assert result == "Hello, I am a bot."
    assert isinstance(result, str)


@patch("httpx.post")
def test_chat_as_json_returns_dict(mock_post: MagicMock) -> None:
    """chat() with as_json=True parses response content as JSON dict."""
    json_content = '{"answer": "42", "confidence": 0.99}'
    mock_post.return_value = _mock_response(
        status_code=200,
        json_data={"message": {"content": json_content}},
    )

    client = OllamaClient()
    result = client.chat("What is the answer?", as_json=True)

    assert isinstance(result, dict)
    assert result["answer"] == "42"
    assert result["confidence"] == 0.99

    # Verify the payload included format=json
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["format"] == "json"


@patch("httpx.post")
def test_chat_connect_error_raises_not_running(mock_post: MagicMock) -> None:
    """chat() raises OllamaNotRunningError when server is unreachable."""
    mock_post.side_effect = httpx.ConnectError("Connection refused")

    client = OllamaClient()
    with pytest.raises(OllamaNotRunningError, match="Cannot connect"):
        client.chat("Hi")


# ---------------------------------------------------------------------------
# client_from_config
# ---------------------------------------------------------------------------


def test_client_from_config_constructs_with_correct_values() -> None:
    """client_from_config reads ollama section and passes values to OllamaClient."""
    config = {
        "ollama": {
            "base_url": "http://myhost:9999",
            "embed_model": "custom-embed",
            "chat_model": "custom-chat",
        },
    }

    client = client_from_config(config)

    assert client.base_url == "http://myhost:9999"
    assert client.embed_model == "custom-embed"
    assert client.chat_model == "custom-chat"


def test_client_from_config_uses_defaults_when_empty() -> None:
    """client_from_config falls back to defaults when config has no ollama section."""
    client = client_from_config({})

    assert client.base_url == "http://localhost:11434"
    assert client.embed_model == "nomic-embed-text"
    assert client.chat_model == "gemma3:4b"
