"""Shared fixtures and helpers for the Hermes Assist tests."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from homeassistant.config_entries import ConfigEntryState, ConfigSubentryDataWithId
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.hermes_assist.api import HermesClient, ToolStarted
from custom_components.hermes_assist.const import (
    CONF_API_KEY,
    CONF_URL,
    DEFAULT_OPTIONS,
    DOMAIN,
    NAME,
    SUBENTRY_TYPE_CONVERSATION,
)

HERMES_URL = "https://hermes.example.com"
MODELS_URL = f"{HERMES_URL}/v1/models"
CHAT_URL = f"{HERMES_URL}/v1/chat/completions"
API_KEY = "test-key"
ENTITY_ID = "conversation.hermes_assist"
SUBENTRY_ID = "agent-1"
MODELS_RESPONSE: dict[str, Any] = {
    "object": "list",
    "data": [{"id": "hermes-agent", "object": "model"}],
}


class FakeStream:
    """A Hermes answer stream that the test feeds by hand."""

    def __init__(self, session_id: str | None) -> None:
        """Start empty; push, finish and fail add to the stream."""
        self.session_id = session_id
        self.content_received = False
        self.closed = False
        self._items: asyncio.Queue[str | ToolStarted | Exception | None] = (
            asyncio.Queue()
        )

    def push(self, *items: str | ToolStarted) -> None:
        """Add content deltas and tool starts."""
        for item in items:
            self._items.put_nowait(item)

    def finish(self) -> None:
        """End the stream normally."""
        self._items.put_nowait(None)

    def fail(self, error: Exception) -> None:
        """End the stream with an error."""
        self._items.put_nowait(error)

    async def __aiter__(self) -> AsyncIterator[str]:
        """Yield the pushed deltas until the stream ends."""
        async for item in self.items():
            if isinstance(item, str):
                yield item

    async def items(self) -> AsyncIterator[str | ToolStarted]:
        """Yield the pushed deltas and tool starts until the stream ends."""
        while (item := await self._items.get()) is not None:
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str):
                self.content_received = True
            yield item


class FakeHermes:
    """Replace HermesClient.stream_chat and record every request."""

    def __init__(self) -> None:
        """Start without prepared streams."""
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self._streams: list[FakeStream] = []

    def stream(self, session_id: str | None = "session-1") -> FakeStream:
        """Prepare the stream that answers the next request."""
        stream = FakeStream(session_id)
        self._streams.append(stream)
        return stream

    @asynccontextmanager
    async def stream_chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[FakeStream]:
        """Record the request and hand out the next prepared stream."""
        self.calls.append({"messages": messages, **kwargs})
        if self.error is not None:
            raise self.error
        stream = self._streams.pop(0)
        try:
            yield stream
        finally:
            stream.closed = True


def subentry_data(
    title: str = NAME, subentry_id: str = SUBENTRY_ID, **changes: Any
) -> ConfigSubentryDataWithId:
    """Return the stored form of one conversation agent."""
    return {
        "subentry_id": subentry_id,
        "subentry_type": SUBENTRY_TYPE_CONVERSATION,
        "title": title,
        "unique_id": None,
        "data": {**DEFAULT_OPTIONS, **changes},
    }


async def set_options(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    changes: dict[str, Any],
    subentry_id: str = SUBENTRY_ID,
) -> None:
    """Change an agent's options; the update listener reloads the entry."""
    subentry = entry.subentries[subentry_id]
    hass.config_entries.async_update_subentry(
        entry, subentry, data={**subentry.data, **changes}
    )
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED


@pytest.fixture(autouse=True)
async def auto_setup(hass: HomeAssistant, enable_custom_integrations: None) -> None:
    """Load custom_components/ and the core component the flows rely on."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def mock_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return a config entry with one default agent, added but not set up."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=NAME,
        version=2,
        minor_version=2,
        data={CONF_URL: HERMES_URL, CONF_API_KEY: API_KEY},
        subentries_data=[subentry_data()],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def setup_entry(
    hass: HomeAssistant,
    mock_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> MockConfigEntry:
    """Return the config entry after a successful setup."""
    aioclient_mock.get(MODELS_URL, json=MODELS_RESPONSE)
    assert await hass.config_entries.async_setup(mock_entry.entry_id)
    await hass.async_block_till_done()
    return mock_entry


@pytest.fixture
def hermes(monkeypatch: pytest.MonkeyPatch) -> FakeHermes:
    """Answer chat requests from hand-fed streams instead of HTTP."""
    fake = FakeHermes()
    monkeypatch.setattr(HermesClient, "stream_chat", fake.stream_chat)
    return fake
