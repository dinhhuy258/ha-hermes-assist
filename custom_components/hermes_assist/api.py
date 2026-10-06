"""Client for the Hermes Agent API server."""

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import aclosing, asynccontextmanager
import json
import logging
from typing import Any

import aiohttp

from .const import CONNECT_TIMEOUT, MAX_SESSION_ID_LENGTH, VALIDATE_TIMEOUT

_LOGGER = logging.getLogger(__name__)

HEADER_SESSION_ID = "X-Hermes-Session-Id"
HEADER_SESSION_KEY = "X-Hermes-Session-Key"
ERROR_BODY_LIMIT = 200
_FORBIDDEN_ID_CHARACTERS = ("\r", "\n", "\0")


class HermesError(Exception):
    """Base class for Hermes errors."""


class HermesAuthError(HermesError):
    """Hermes rejected the API key."""


class HermesConnectionError(HermesError):
    """Hermes could not be reached."""


class HermesStreamError(HermesConnectionError):
    """The answer stream broke before it finished."""


class HermesTimeoutError(HermesError):
    """The answer did not finish within the request timeout."""


class HermesResponseError(HermesError):
    """Hermes answered with an error."""


def normalize_url(url: str) -> str:
    """Strip whitespace, trailing slashes and a trailing /v1 from a base URL."""
    url = url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url.removesuffix("/v1").rstrip("/")
    return url


def is_valid_session_id(value: object) -> bool:
    """Return whether a value is safe to send as X-Hermes-Session-Id."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_SESSION_ID_LENGTH
        and not any(char in value for char in _FORBIDDEN_ID_CHARACTERS)
    )


async def _iter_sse(content: AsyncIterable[bytes]) -> AsyncIterator[tuple[str, str]]:
    """Yield (event, data) pairs from a server-sent events body."""
    event = "message"
    data: list[str] = []
    async for raw in content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


class HermesStream:
    """The streamed answer of one chat completion request."""

    def __init__(self, response: aiohttp.ClientResponse) -> None:
        """Read the session id from the headers, before any content arrives."""
        self._response = response
        header = response.headers.get(HEADER_SESSION_ID)
        self.session_id: str | None = header if is_valid_session_id(header) else None
        self.content_received = False

    async def __aiter__(self) -> AsyncIterator[str]:
        """Yield the visible answer text, delta by delta."""
        try:
            async with aclosing(_iter_sse(self._response.content)) as events:
                async for event, data in events:
                    if event != "message":
                        # hermes.tool.progress, hermes.status, approval.request.
                        continue
                    if data.strip() == "[DONE]":
                        return
                    for text in self._handle_chunk(data):
                        yield text
        except TimeoutError as err:
            raise HermesTimeoutError(
                "Hermes did not finish the answer in time"
            ) from err
        except (aiohttp.ClientError, ValueError) as err:
            raise HermesStreamError(f"The Hermes stream broke: {err}") from err

    def _handle_chunk(self, data: str) -> list[str]:
        """Return the visible text of one chunk and check its finish reason."""
        try:
            chunk = json.loads(data)
        except ValueError:
            _LOGGER.debug("Skipping a stream chunk that is not JSON")
            return []
        if not isinstance(chunk, dict):
            return []
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return []
        choice = choices[0]
        if not isinstance(choice, dict):
            return []

        texts: list[str] = []
        delta = choice.get("delta")
        # reasoning_content and the role delta carry no "content" and are skipped.
        if isinstance(delta, dict) and isinstance(text := delta.get("content"), str):
            if not self.content_received:
                text = text.lstrip()
            if text:
                self.content_received = True
                texts.append(text)

        reason = choice.get("finish_reason")
        if reason and reason != "stop":
            self._check_finish(reason, chunk.get("error"))
        return texts

    def _check_finish(self, reason: str, error: object) -> None:
        """Fail on an error finish that arrives before any content."""
        message = error.get("message") if isinstance(error, dict) else None
        if self.content_received:
            _LOGGER.debug(
                "Hermes finished with %s after the answer started: %s",
                reason,
                message,
            )
            return
        raise HermesResponseError(f"Hermes finished with {reason}: {message}")


class HermesClient:
    """Talk to the OpenAI-compatible API server of Hermes Agent."""

    def __init__(self, session: aiohttp.ClientSession, url: str, api_key: str) -> None:
        """Store the HTTP session, the base URL and the API key."""
        self._session = session
        self.url = url
        self._api_key = api_key

    def __repr__(self) -> str:
        """Describe the client without its headers or key."""
        return f"HermesClient(url={self.url!r})"

    def _headers(
        self, session_key: str | None = None, session_id: str | None = None
    ) -> dict[str, str]:
        """Return the request headers."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if session_key:
            headers[HEADER_SESSION_KEY] = session_key
        if session_id:
            headers[HEADER_SESSION_ID] = session_id
        return headers

    async def async_validate(self) -> None:
        """Check the URL and the key with GET /v1/models."""
        try:
            async with self._session.get(
                f"{self.url}/v1/models",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=VALIDATE_TIMEOUT),
            ) as response:
                if response.status in (401, 403):
                    raise HermesAuthError("Hermes rejected the API key")
                response.raise_for_status()
        except (aiohttp.ClientError, TimeoutError) as err:
            raise HermesConnectionError(f"Could not reach Hermes: {err}") from err

    @asynccontextmanager
    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        session_key: str | None = None,
        session_id: str | None = None,
        timeout: float,
    ) -> AsyncIterator[HermesStream]:
        """Open a streaming chat completion and yield its answer stream.

        A stored session id that Hermes rejects with 400 or 404 is dropped and
        the request is retried once without it.
        """
        payload: dict[str, Any] = {"messages": messages, "stream": True}
        if model:
            payload["model"] = model
        client_timeout = aiohttp.ClientTimeout(total=timeout, connect=CONNECT_TIMEOUT)

        response = await self._async_post(
            payload, session_key, session_id, client_timeout
        )
        if session_id and response.status in (400, 404):
            _LOGGER.warning(
                "Hermes rejected the stored session id (HTTP %s); "
                "starting a new session",
                response.status,
            )
            response.release()
            response = await self._async_post(
                payload, session_key, None, client_timeout
            )
        try:
            await self._async_raise_for_status(response)
            yield HermesStream(response)
        finally:
            response.release()

    async def _async_post(
        self,
        payload: dict[str, Any],
        session_key: str | None,
        session_id: str | None,
        timeout: aiohttp.ClientTimeout,
    ) -> aiohttp.ClientResponse:
        """Send the chat completion request and return the open response."""
        try:
            return await self._session.post(
                f"{self.url}/v1/chat/completions",
                json=payload,
                headers=self._headers(session_key, session_id),
                timeout=timeout,
            )
        except TimeoutError as err:
            raise HermesConnectionError("Timed out connecting to Hermes") from err
        except aiohttp.ClientError as err:
            raise HermesConnectionError(f"Could not connect to Hermes: {err}") from err

    async def _async_raise_for_status(self, response: aiohttp.ClientResponse) -> None:
        """Raise for an error status; log a truncated body, never the headers."""
        if response.status in (401, 403):
            raise HermesAuthError("Hermes rejected the API key")
        if response.status >= 400:
            try:
                body = await response.text()
            except aiohttp.ClientError, TimeoutError, ValueError:
                body = ""
            _LOGGER.debug(
                "Hermes returned HTTP %s: %s",
                response.status,
                body[:ERROR_BODY_LIMIT],
            )
            raise HermesResponseError(f"Hermes returned HTTP {response.status}")
