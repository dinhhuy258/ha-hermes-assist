"""The hermes_assist.ask service for automations and scripts."""

import asyncio
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

from .api import (
    HermesAuthError,
    HermesConnectionError,
    HermesError,
    HermesTimeoutError,
    is_valid_session_id,
)
from .const import (
    ATTR_CONFIG_ENTRY_ID,
    ATTR_SESSION_ID,
    ATTR_TEXT,
    CONF_MODEL,
    CONF_SESSION_KEY,
    CONF_TIMEOUT,
    DEFAULT_SESSION_KEY,
    DEFAULT_TIMEOUT,
    DOMAIN,
    ERROR_AUTH,
    ERROR_CANNOT_CONNECT,
    ERROR_GENERIC,
    ERROR_TOO_LONG,
    SERVICE_ASK,
)

if TYPE_CHECKING:
    from . import HermesAssistConfigEntry


def _session_id(value: Any) -> str:
    """Validate a session id given by the caller."""
    value = cv.string(value)
    if not is_valid_session_id(value):
        raise vol.Invalid("Invalid Hermes session id")
    return value


ASK_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_TEXT): cv.string,
        vol.Optional(ATTR_SESSION_ID): _session_id,
        vol.Optional(ATTR_CONFIG_ENTRY_ID): cv.string,
    }
)


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register hermes_assist.ask."""

    async def async_ask(call: ServiceCall) -> ServiceResponse:
        """Send text to Hermes and return the whole answer.

        The service does not stream and does not hand off; it waits up to the
        configured timeout. A given session_id is used for this call only and
        leaves the stored session unchanged.
        """
        entry = _get_entry(hass, call.data.get(ATTR_CONFIG_ENTRY_ID))
        data = entry.runtime_data
        options = entry.options
        override: str | None = call.data.get(ATTR_SESSION_ID)
        timeout = float(options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
        try:
            async with (
                asyncio.timeout(timeout),
                data.client.stream_chat(
                    [{"role": "user", "content": call.data[ATTR_TEXT]}],
                    model=options.get(CONF_MODEL) or None,
                    session_key=options.get(CONF_SESSION_KEY, DEFAULT_SESSION_KEY)
                    or None,
                    session_id=override or data.store.get(None),
                    timeout=timeout,
                ) as stream,
            ):
                session_id = stream.session_id
                if override is None:
                    data.store.set(None, session_id)
                text = "".join([part async for part in stream])
        except (TimeoutError, HermesTimeoutError) as err:
            raise HomeAssistantError(ERROR_TOO_LONG) from err
        except HermesAuthError as err:
            data.request_reauth(hass, entry)
            raise HomeAssistantError(ERROR_AUTH) from err
        except HermesConnectionError as err:
            raise HomeAssistantError(ERROR_CANNOT_CONNECT) from err
        except HermesError as err:
            raise HomeAssistantError(ERROR_GENERIC) from err

        if override is None:
            data.store.touch(None)
        return {"text": text.strip(), "session_id": session_id}

    hass.services.async_register(
        DOMAIN,
        SERVICE_ASK,
        async_ask,
        schema=ASK_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )


def _get_entry(hass: HomeAssistant, entry_id: str | None) -> HermesAssistConfigEntry:
    """Return the entry to ask; it may be left out when only one is loaded."""
    if entry_id:
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(f"No Hermes Assist entry with ID {entry_id}")
        if entry.state is not ConfigEntryState.LOADED:
            raise ServiceValidationError(
                f"Hermes Assist entry {entry.title} is not loaded"
            )
        return entry
    entries = hass.config_entries.async_loaded_entries(DOMAIN)
    if not entries:
        raise ServiceValidationError("No Hermes Assist entry is loaded")
    if len(entries) > 1:
        raise ServiceValidationError(
            "More than one Hermes Assist entry is loaded; set config_entry_id"
        )
    return entries[0]
