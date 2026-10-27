"""The hermes_assist.ask service for automations and scripts."""

import asyncio
from typing import TYPE_CHECKING, Any

from homeassistant.components import camera
from homeassistant.config_entries import ConfigEntryState, ConfigSubentry
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv, entity_registry as er
import voluptuous as vol

from .api import (
    HermesAuthError,
    HermesConnectionError,
    HermesError,
    HermesTimeoutError,
    is_valid_session_id,
)
from .const import (
    ATTR_AGENT_ID,
    ATTR_CAMERA_ENTITY_ID,
    ATTR_SESSION_ID,
    ATTR_SYSTEM_PROMPT,
    ATTR_TEXT,
    CONF_MODEL,
    CONF_REASONING_EFFORT,
    CONF_SESSION_KEY,
    CONF_TIMEOUT,
    DEFAULT_SESSION_KEY,
    DEFAULT_TIMEOUT,
    DOMAIN,
    SERVICE_ASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .context import build_messages
from .messages import (
    ERROR_AUTH,
    ERROR_CANNOT_CONNECT,
    ERROR_GENERIC,
    ERROR_TOO_LONG,
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
        vol.Optional(ATTR_AGENT_ID): cv.entity_id,
        vol.Optional(ATTR_SYSTEM_PROMPT): cv.string,
        vol.Optional(ATTR_CAMERA_ENTITY_ID): cv.entity_domain(camera.DOMAIN),
    }
)


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register hermes_assist.ask."""

    async def async_ask(call: ServiceCall) -> ServiceResponse:
        """Send text to Hermes and return the whole answer.

        The agent prompt and its exposed entities are sent like for a voice
        request; system_prompt replaces the agent prompt for this call. The
        service does not stream and does not hand off; it waits up to the
        configured timeout. A given session_id is used for this call only and
        leaves the stored session unchanged.
        """
        entry, subentry = _get_agent(hass, call.data.get(ATTR_AGENT_ID))
        data = entry.runtime_data
        store = data.stores[subentry.subentry_id]
        options = subentry.data
        override: str | None = call.data.get(ATTR_SESSION_ID)
        timeout = float(options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
        image: tuple[str, bytes] | None = None
        if camera_id := call.data.get(ATTR_CAMERA_ENTITY_ID):
            try:
                snapshot = await camera.async_get_image(hass, camera_id)
            except HomeAssistantError as err:
                raise HomeAssistantError(
                    f"Could not get an image from {camera_id}: {err}"
                ) from err
            image = (snapshot.content_type, snapshot.content)
        try:
            async with (
                asyncio.timeout(timeout),
                data.client.stream_chat(
                    build_messages(
                        hass,
                        options,
                        call.data[ATTR_TEXT],
                        prompt=call.data.get(ATTR_SYSTEM_PROMPT),
                        image=image,
                    ),
                    model=options.get(CONF_MODEL) or None,
                    reasoning_effort=options.get(CONF_REASONING_EFFORT) or None,
                    session_key=options.get(CONF_SESSION_KEY, DEFAULT_SESSION_KEY)
                    or None,
                    session_id=override or store.get(None),
                    timeout=timeout,
                ) as stream,
            ):
                session_id = stream.session_id
                if override is None:
                    store.set(None, session_id)
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
            store.touch(None)
        return {"text": text.strip(), "session_id": session_id}

    hass.services.async_register(
        DOMAIN,
        SERVICE_ASK,
        async_ask,
        schema=ASK_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )


def _get_agent(
    hass: HomeAssistant, agent_id: str | None
) -> tuple[HermesAssistConfigEntry, ConfigSubentry]:
    """Return the entry and agent to ask; agent_id may be left out with one agent."""
    if agent_id:
        registry_entry = er.async_get(hass).async_get(agent_id)
        if (
            registry_entry is None
            or registry_entry.platform != DOMAIN
            or registry_entry.config_entry_id is None
            or registry_entry.config_subentry_id is None
        ):
            raise ServiceValidationError(f"{agent_id} is not a Hermes Assist agent")
        entry = hass.config_entries.async_get_entry(registry_entry.config_entry_id)
        if entry is None or entry.state is not ConfigEntryState.LOADED:
            raise ServiceValidationError(
                f"Hermes Assist agent {agent_id} is not loaded"
            )
        return entry, entry.subentries[registry_entry.config_subentry_id]

    agents = [
        (entry, subentry)
        for entry in hass.config_entries.async_loaded_entries(DOMAIN)
        for subentry in entry.get_subentries_of_type(SUBENTRY_TYPE_CONVERSATION)
    ]
    if not agents:
        raise ServiceValidationError("No Hermes Assist agent is loaded")
    if len(agents) > 1:
        raise ServiceValidationError(
            "More than one Hermes Assist agent exists; set agent_id"
        )
    return agents[0]
