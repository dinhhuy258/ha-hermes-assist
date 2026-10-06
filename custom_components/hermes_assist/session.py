"""Hermes session state for one config entry."""

from collections import OrderedDict
from typing import TypedDict

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .api import is_valid_session_id
from .const import (
    DOMAIN,
    MAX_PER_CONVERSATION_SESSIONS,
    SESSION_MODE_IDLE,
    SESSION_MODE_PER_CONVERSATION,
    STORAGE_SAVE_DELAY,
    STORAGE_VERSION,
)


class StoredSession(TypedDict):
    """Content of .storage/hermes_assist.<entry_id>."""

    session_id: str | None
    last_used: float | None


def storage_key(entry_id: str) -> str:
    """Return the storage key of a config entry."""
    return f"{DOMAIN}.{entry_id}"


def _now() -> float:
    """Return the current time as a UTC timestamp."""
    return dt_util.utcnow().timestamp()


class SessionStore:
    """Resolve and remember the Hermes session id for each turn.

    keep: one stored id that never expires.
    idle: one stored id that expires after idle_timeout seconds without use.
    per_conversation: one id per Home Assistant conversation id, in memory only.
    """

    def __init__(
        self, hass: HomeAssistant, entry_id: str, mode: str, idle_timeout: float
    ) -> None:
        """Initialize the store; call async_load before use."""
        self._store: Store[StoredSession] = Store(
            hass, STORAGE_VERSION, storage_key(entry_id)
        )
        self._mode = mode
        self._idle_timeout = idle_timeout
        self._session_id: str | None = None
        self._last_used: float | None = None
        self._conversations: OrderedDict[str, str] = OrderedDict()

    @property
    def mode(self) -> str:
        """Return the session mode."""
        return self._mode

    async def async_load(self) -> None:
        """Load the stored session; invalid content is ignored."""
        if self._mode == SESSION_MODE_PER_CONVERSATION:
            return
        data = await self._store.async_load()
        if not isinstance(data, dict):
            return
        session_id = data.get("session_id")
        last_used = data.get("last_used")
        if is_valid_session_id(session_id):
            self._session_id = session_id
            self._last_used = (
                float(last_used) if isinstance(last_used, int | float) else None
            )

    def get(self, conversation_id: str | None) -> str | None:
        """Return the session id to send for this conversation, if any."""
        if self._mode == SESSION_MODE_PER_CONVERSATION:
            if conversation_id is None:
                return None
            return self._conversations.get(conversation_id)
        if self._session_id is None:
            return None
        if self._mode == SESSION_MODE_IDLE and (
            self._last_used is None or _now() - self._last_used >= self._idle_timeout
        ):
            return None
        return self._session_id

    def set(self, conversation_id: str | None, session_id: str | None) -> None:
        """Remember the session id Hermes returned; None forgets the session."""
        if self._mode == SESSION_MODE_PER_CONVERSATION:
            if conversation_id is None:
                return
            if session_id is None:
                self._conversations.pop(conversation_id, None)
                return
            self._conversations[conversation_id] = session_id
            self._conversations.move_to_end(conversation_id)
            while len(self._conversations) > MAX_PER_CONVERSATION_SESSIONS:
                self._conversations.popitem(last=False)
            return
        self._session_id = session_id
        self._last_used = _now() if session_id is not None else None
        self._schedule_save()

    def touch(self, conversation_id: str | None) -> None:
        """Mark the session as used now."""
        if self._mode == SESSION_MODE_PER_CONVERSATION:
            if conversation_id is not None and conversation_id in self._conversations:
                self._conversations.move_to_end(conversation_id)
            return
        if self._session_id is None:
            return
        self._last_used = _now()
        self._schedule_save()

    def reset(self, conversation_id: str | None) -> None:
        """Forget the session so the next request starts a new one."""
        self.set(conversation_id, None)

    async def async_clear(self) -> None:
        """Forget every session and delete the storage file."""
        self._session_id = None
        self._last_used = None
        self._conversations.clear()
        await self._store.async_remove()

    def _schedule_save(self) -> None:
        """Save the stored session after a short delay."""
        self._store.async_delay_save(self._data, STORAGE_SAVE_DELAY)

    def _data(self) -> StoredSession:
        """Return the content to save."""
        return {"session_id": self._session_id, "last_used": self._last_used}


async def async_remove_sessions(hass: HomeAssistant, entry_id: str) -> None:
    """Delete the storage file of a config entry."""
    await Store[StoredSession](
        hass, STORAGE_VERSION, storage_key(entry_id)
    ).async_remove()


async def async_clear_sessions(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Forget the stored sessions of a config entry, loaded or not.

    A loaded entry clears through its own store so that a pending delayed save
    cannot write the old session back.
    """
    if entry.state is ConfigEntryState.LOADED:
        await entry.runtime_data.store.async_clear()
        return
    await async_remove_sessions(hass, entry.entry_id)
