"""Tests for the Hermes session store."""

from datetime import timedelta
from typing import Any

from freezegun.api import FrozenDateTimeFactory
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
import pytest
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.hermes_assist.const import (
    MAX_PER_CONVERSATION_SESSIONS,
    SESSION_MODE_IDLE,
    SESSION_MODE_KEEP,
    SESSION_MODE_PER_CONVERSATION,
)
from custom_components.hermes_assist.session import (
    SessionStore,
    async_remove_sessions,
    storage_key,
)

ENTRY_ID = "entry-1"
KEY = storage_key(ENTRY_ID)


def stored(session_id: Any, last_used: Any) -> dict[str, Any]:
    """Return a storage file as Store writes it."""
    return {
        "version": 1,
        "minor_version": 1,
        "key": KEY,
        "data": {"session_id": session_id, "last_used": last_used},
    }


async def load_store(
    hass: HomeAssistant, mode: str, idle_timeout: float = 1800
) -> SessionStore:
    """Create and load a session store."""
    store = SessionStore(hass, ENTRY_ID, mode, idle_timeout)
    await store.async_load()
    return store


async def flush(hass: HomeAssistant) -> None:
    """Run the delayed save."""
    async_fire_time_changed(hass, fire_all=True)
    await hass.async_block_till_done()


def test_storage_key() -> None:
    assert KEY == "hermes_assist.entry-1"


async def test_keep_mode_survives_time_and_restart(
    hass: HomeAssistant,
    hass_storage: dict[str, Any],
    freezer: FrozenDateTimeFactory,
) -> None:
    store = await load_store(hass, SESSION_MODE_KEEP)
    assert store.get("conversation-1") is None

    store.set("conversation-1", "session-1")
    freezer.tick(timedelta(days=30))

    assert store.get("conversation-2") == "session-1"
    await flush(hass)
    assert hass_storage[KEY]["data"]["session_id"] == "session-1"

    restarted = await load_store(hass, SESSION_MODE_KEEP)
    assert restarted.get(None) == "session-1"


async def test_idle_mode_expires_after_the_idle_timeout(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    store = await load_store(hass, SESSION_MODE_IDLE, idle_timeout=1800)
    store.set(None, "session-1")

    freezer.tick(timedelta(seconds=1799))
    assert store.get(None) == "session-1"

    store.touch(None)
    freezer.tick(timedelta(seconds=1799))
    assert store.get(None) == "session-1"

    freezer.tick(timedelta(seconds=1))
    assert store.get(None) is None


@pytest.mark.parametrize(("age", "expected"), [(1000, "session-1"), (2000, None)])
async def test_idle_mode_uses_the_stored_last_used(
    hass: HomeAssistant,
    hass_storage: dict[str, Any],
    freezer: FrozenDateTimeFactory,
    age: int,
    expected: str | None,
) -> None:
    hass_storage[KEY] = stored("session-1", dt_util.utcnow().timestamp() - age)

    store = await load_store(hass, SESSION_MODE_IDLE, idle_timeout=1800)

    assert store.get(None) == expected


async def test_per_conversation_mode_maps_conversations_in_memory(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    store = await load_store(hass, SESSION_MODE_PER_CONVERSATION)

    store.set("conversation-a", "session-a")
    store.set("conversation-b", "session-b")

    assert store.get("conversation-a") == "session-a"
    assert store.get("conversation-b") == "session-b"
    assert store.get("conversation-c") is None
    assert store.get(None) is None
    await flush(hass)
    assert KEY not in hass_storage


async def test_per_conversation_mode_evicts_the_oldest(hass: HomeAssistant) -> None:
    store = await load_store(hass, SESSION_MODE_PER_CONVERSATION)
    for index in range(MAX_PER_CONVERSATION_SESSIONS):
        store.set(f"conversation-{index}", f"session-{index}")

    store.touch("conversation-0")
    store.set("conversation-new", "session-new")

    assert store.get("conversation-0") == "session-0"
    assert store.get("conversation-1") is None
    assert store.get("conversation-new") == "session-new"


@pytest.mark.parametrize(
    "mode", [SESSION_MODE_KEEP, SESSION_MODE_IDLE, SESSION_MODE_PER_CONVERSATION]
)
async def test_reset_drops_the_session(hass: HomeAssistant, mode: str) -> None:
    store = await load_store(hass, mode)
    store.set("conversation-1", "session-1")

    store.reset("conversation-1")

    assert store.get("conversation-1") is None


async def test_reset_is_saved(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    store = await load_store(hass, SESSION_MODE_KEEP)
    store.set(None, "session-1")
    await flush(hass)
    assert hass_storage[KEY]["data"]["session_id"] == "session-1"

    store.reset(None)
    await flush(hass)

    assert hass_storage[KEY]["data"]["session_id"] is None


@pytest.mark.parametrize("session_id", ["bad\nid", "x" * 129, 42])
async def test_invalid_stored_ids_are_ignored(
    hass: HomeAssistant, hass_storage: dict[str, Any], session_id: Any
) -> None:
    hass_storage[KEY] = stored(session_id, dt_util.utcnow().timestamp())

    store = await load_store(hass, SESSION_MODE_KEEP)

    assert store.get(None) is None


async def test_async_clear_removes_the_file(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    store = await load_store(hass, SESSION_MODE_KEEP)
    store.set(None, "session-1")
    await flush(hass)
    assert KEY in hass_storage

    await store.async_clear()

    assert KEY not in hass_storage
    assert store.get(None) is None


async def test_async_remove_sessions(
    hass: HomeAssistant, hass_storage: dict[str, Any]
) -> None:
    hass_storage[KEY] = stored("session-1", 0.0)

    await async_remove_sessions(hass, ENTRY_ID)

    assert KEY not in hass_storage
