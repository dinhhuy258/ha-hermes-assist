"""Tests for the Hermes API client."""

import json
import logging
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
    AiohttpClientMockResponse,
)
from yarl import URL

from custom_components.hermes_assist.api import (
    HermesAuthError,
    HermesClient,
    HermesConnectionError,
    HermesResponseError,
    HermesStreamError,
    is_valid_session_id,
    normalize_url,
)

from .conftest import API_KEY, CHAT_URL, HERMES_URL, MODELS_URL

MESSAGES = [{"role": "user", "content": "Hello"}]
SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "X-Hermes-Session-Id": "session-1",
}
DONE = "data: [DONE]\n\n"


def chunk(
    delta: dict[str, Any] | None = None, finish: str | None = None, **extra: Any
) -> str:
    """Return one SSE frame with a chat.completion.chunk payload."""
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        **extra,
    }
    return f"data: {json.dumps(payload)}\n\n"


def event(name: str, data: dict[str, Any]) -> str:
    """Return one named SSE frame."""
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


class _BrokenBody:
    """An SSE body that loses the connection after its lines."""

    def __init__(self, text: str) -> None:
        self._lines = [line + b"\n" for line in text.encode().split(b"\n")]

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for line in self._lines:
            yield line
        raise aiohttp.ClientPayloadError("Response payload is not completed")


class BrokenResponse(AiohttpClientMockResponse):
    """A mocked SSE response whose body breaks after the given text."""

    def __init__(self, text: str) -> None:
        super().__init__("post", URL(CHAT_URL), headers=SSE_HEADERS)
        self._body = _BrokenBody(text)

    @property
    def content(self):
        return self._body


@pytest.fixture
def client(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> HermesClient:
    """Return a client that talks to the mocked Hermes server."""
    return HermesClient(async_get_clientsession(hass), HERMES_URL, API_KEY)


async def collect(client: HermesClient, **kwargs: Any) -> tuple[list[str], str | None]:
    """Stream one answer and return its texts and session id."""
    kwargs.setdefault("timeout", 30)
    async with client.stream_chat(MESSAGES, **kwargs) as stream:
        session_id = stream.session_id
        texts = [text async for text in stream]
    return texts, session_id


async def test_stream_yields_visible_content_only(
    client: HermesClient, aioclient_mock: AiohttpClientMocker
) -> None:
    body = (
        ": keepalive\n\n"
        + chunk({"role": "assistant"})
        + chunk({"content": "\n  "})
        + chunk({"content": " Hello"})
        + chunk({"reasoning_content": "Thinking about it."})
        + event(
            "hermes.tool.progress",
            {"tool": "search", "choices": [{"delta": {"content": "tool text"}}]},
        )
        + event("hermes.status", {"status": "running"})
        + event("approval.request", {"id": "approval-1"})
        + chunk({"content": ", world."})
        + chunk(finish="stop")
        + DONE
        + chunk({"content": "after done"})
    )
    aioclient_mock.post(CHAT_URL, text=body, headers=SSE_HEADERS)

    texts, session_id = await collect(client, session_key="homeassistant:assist")

    assert texts == ["Hello", ", world."]
    assert session_id == "session-1"
    method, url, payload, headers = aioclient_mock.mock_calls[0]
    assert method == "POST"
    assert str(url) == CHAT_URL
    assert payload == {"messages": MESSAGES, "stream": True}
    assert headers["Authorization"] == f"Bearer {API_KEY}"
    assert headers["X-Hermes-Session-Key"] == "homeassistant:assist"
    assert "X-Hermes-Session-Id" not in headers


async def test_session_id_is_available_before_the_first_delta(
    client: HermesClient, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(
        CHAT_URL, text=chunk({"content": "Hi"}) + DONE, headers=SSE_HEADERS
    )

    async with client.stream_chat(MESSAGES, timeout=30) as stream:
        assert stream.session_id == "session-1"
        assert stream.content_received is False
        assert [text async for text in stream] == ["Hi"]
        assert stream.content_received is True


async def test_stream_sends_model_and_session_id(
    client: HermesClient, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(
        CHAT_URL, text=chunk({"content": "Hi"}) + DONE, headers=SSE_HEADERS
    )

    await collect(client, model="hermes-agent", session_id="session-0")

    _, _, payload, headers = aioclient_mock.mock_calls[0]
    assert payload["model"] == "hermes-agent"
    assert headers["X-Hermes-Session-Id"] == "session-0"
    assert "X-Hermes-Session-Key" not in headers


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Type": "text/event-stream"},
        {"Content-Type": "text/event-stream", "X-Hermes-Session-Id": "x" * 129},
    ],
    ids=["missing", "too_long"],
)
async def test_missing_or_invalid_session_header(
    client: HermesClient,
    aioclient_mock: AiohttpClientMocker,
    headers: dict[str, str],
) -> None:
    aioclient_mock.post(CHAT_URL, text=chunk({"content": "Hi"}) + DONE, headers=headers)

    texts, session_id = await collect(client)

    assert texts == ["Hi"]
    assert session_id is None


@pytest.mark.parametrize("status", [401, 403])
async def test_rejected_key(
    client: HermesClient, aioclient_mock: AiohttpClientMocker, status: int
) -> None:
    aioclient_mock.post(
        CHAT_URL, status=status, json={"error": {"message": "Invalid API key"}}
    )

    with pytest.raises(HermesAuthError):
        await collect(client)


async def test_server_error_logs_a_truncated_body_and_no_key(
    client: HermesClient,
    aioclient_mock: AiohttpClientMocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="custom_components.hermes_assist")
    aioclient_mock.post(CHAT_URL, status=500, text="x" * 500)

    with pytest.raises(HermesResponseError):
        await collect(client)

    assert "x" * 200 in caplog.text
    assert "x" * 201 not in caplog.text
    assert API_KEY not in caplog.text


@pytest.mark.parametrize(
    "error",
    [aiohttp.ClientConnectionError("Connection refused"), TimeoutError()],
    ids=["refused", "timeout"],
)
async def test_connection_errors(
    client: HermesClient, aioclient_mock: AiohttpClientMocker, error: Exception
) -> None:
    aioclient_mock.post(CHAT_URL, exc=error)

    with pytest.raises(HermesConnectionError):
        await collect(client)


async def test_error_finish_before_content(
    client: HermesClient, aioclient_mock: AiohttpClientMocker
) -> None:
    body = (
        chunk({"role": "assistant"})
        + chunk(
            finish="error",
            error={"message": "Model overloaded", "type": "server_error"},
        )
        + DONE
    )
    aioclient_mock.post(CHAT_URL, text=body, headers=SSE_HEADERS)

    with pytest.raises(HermesResponseError):
        await collect(client)


async def test_error_finish_after_content_is_ignored(
    client: HermesClient, aioclient_mock: AiohttpClientMocker
) -> None:
    body = (
        chunk({"content": "Partial answer."})
        + chunk(finish="error", error={"message": "Model overloaded"})
        + DONE
    )
    aioclient_mock.post(CHAT_URL, text=body, headers=SSE_HEADERS)

    texts, _ = await collect(client)

    assert texts == ["Partial answer."]


async def test_close_without_done_ends_the_stream(
    client: HermesClient, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(
        CHAT_URL, text=chunk({"content": "Hello."}), headers=SSE_HEADERS
    )

    texts, _ = await collect(client)

    assert texts == ["Hello."]


async def test_broken_stream_after_content(
    client: HermesClient, aioclient_mock: AiohttpClientMocker
) -> None:
    async def respond(method: str, url: URL, data: Any) -> BrokenResponse:
        return BrokenResponse(chunk({"content": "Hello"}))

    aioclient_mock.post(CHAT_URL, side_effect=respond)
    texts: list[str] = []

    with pytest.raises(HermesStreamError):
        async with client.stream_chat(MESSAGES, timeout=30) as stream:
            async for text in stream:
                texts.append(text)

    assert texts == ["Hello"]


@pytest.mark.parametrize("status", [400, 404])
async def test_rejected_session_id_is_retried_once_without_it(
    client: HermesClient, aioclient_mock: AiohttpClientMocker, status: int
) -> None:
    responses = [
        AiohttpClientMockResponse(
            "post", URL(CHAT_URL), status=status, text="Unknown session"
        ),
        AiohttpClientMockResponse(
            "post",
            URL(CHAT_URL),
            text=chunk({"content": "Hi"}) + DONE,
            headers={**SSE_HEADERS, "X-Hermes-Session-Id": "session-2"},
        ),
    ]

    async def respond(method: str, url: URL, data: Any) -> AiohttpClientMockResponse:
        return responses.pop(0)

    aioclient_mock.post(CHAT_URL, side_effect=respond)

    texts, session_id = await collect(client, session_id="stale-session")

    assert texts == ["Hi"]
    assert session_id == "session-2"
    first, second = aioclient_mock.mock_calls
    assert first[3]["X-Hermes-Session-Id"] == "stale-session"
    assert "X-Hermes-Session-Id" not in second[3]


async def test_not_found_without_session_id_is_not_retried(
    client: HermesClient, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.post(CHAT_URL, status=404, text="Not found")

    with pytest.raises(HermesResponseError):
        await collect(client)

    assert aioclient_mock.call_count == 1


async def test_validate_success(
    client: HermesClient, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(MODELS_URL, json={"object": "list", "data": []})

    await client.async_validate()

    assert aioclient_mock.mock_calls[0][3]["Authorization"] == f"Bearer {API_KEY}"


@pytest.mark.parametrize(
    ("mock_kwargs", "error"),
    [
        ({"status": 401}, HermesAuthError),
        ({"status": 403}, HermesAuthError),
        ({"status": 500}, HermesConnectionError),
        ({"exc": aiohttp.ClientConnectionError()}, HermesConnectionError),
        ({"exc": TimeoutError()}, HermesConnectionError),
    ],
    ids=["401", "403", "500", "refused", "timeout"],
)
async def test_validate_errors(
    client: HermesClient,
    aioclient_mock: AiohttpClientMocker,
    mock_kwargs: dict[str, Any],
    error: type[Exception],
) -> None:
    aioclient_mock.get(MODELS_URL, **mock_kwargs)

    with pytest.raises(error):
        await client.async_validate()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://hermes.example.com", "https://hermes.example.com"),
        ("https://hermes.example.com/", "https://hermes.example.com"),
        ("https://hermes.example.com/v1", "https://hermes.example.com"),
        ("https://hermes.example.com/v1/", "https://hermes.example.com"),
        ("  https://hermes.example.com:8642/v1  ", "https://hermes.example.com:8642"),
        ("https://hermes.example.com/api/v1", "https://hermes.example.com/api"),
    ],
)
def test_normalize_url(url: str, expected: str) -> None:
    assert normalize_url(url) == expected


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("session-1", True),
        ("a" * 128, True),
        ("", False),
        ("a" * 129, False),
        ("a\nb", False),
        ("a\rb", False),
        ("a\0b", False),
        (None, False),
        (5, False),
    ],
)
def test_is_valid_session_id(value: object, valid: bool) -> None:
    assert is_valid_session_id(value) is valid


def test_repr_hides_the_key(client: HermesClient) -> None:
    assert HERMES_URL in repr(client)
    assert API_KEY not in repr(client)
