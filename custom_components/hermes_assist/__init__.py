"""The Hermes Assist integration."""

from dataclasses import dataclass
import logging
from types import MappingProxyType

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import (
    config_validation as cv,
    device_registry as dr,
    entity_registry as er,
)
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.typing import ConfigType

from .api import HermesAuthError, HermesClient, HermesError
from .const import (
    CONF_API_KEY,
    CONF_SESSION_IDLE_TIMEOUT,
    CONF_SESSION_MODE,
    CONF_URL,
    DEFAULT_OPTIONS,
    DEFAULT_SESSION_IDLE_TIMEOUT,
    DEFAULT_SESSION_MODE,
    DOMAIN,
    SUBENTRY_TYPE_CONVERSATION,
)
from .services import async_setup_services
from .session import SessionStore, async_move_sessions, async_remove_sessions

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.CONVERSATION]
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


@dataclass
class HermesAssistData:
    """Runtime objects of one config entry."""

    client: HermesClient
    # Session store of every conversation agent, by subentry id.
    stores: dict[str, SessionStore]
    reauth_requested: bool = False

    def request_reauth(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Start a reauth flow once; a new key reloads the entry and resets this."""
        if self.reauth_requested:
            return
        self.reauth_requested = True
        entry.async_start_reauth(hass)


type HermesAssistConfigEntry = ConfigEntry[HermesAssistData]


def conversation_subentries(entry: ConfigEntry) -> list[ConfigSubentry]:
    """Return the conversation agents of an entry."""
    return entry.get_subentries_of_type(SUBENTRY_TYPE_CONVERSATION)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the hermes_assist.ask service."""
    async_setup_services(hass)
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: HermesAssistConfigEntry
) -> bool:
    """Validate the key, load the session stores and set up the platforms."""
    client = HermesClient(
        async_get_clientsession(hass), entry.data[CONF_URL], entry.data[CONF_API_KEY]
    )
    try:
        await client.async_validate()
    except HermesAuthError as err:
        raise ConfigEntryAuthFailed("Hermes rejected the API key") from err
    except HermesError as err:
        raise ConfigEntryNotReady(f"Could not reach Hermes at {client.url}") from err

    stores: dict[str, SessionStore] = {}
    for subentry in conversation_subentries(entry):
        store = SessionStore(
            hass,
            subentry.subentry_id,
            subentry.data.get(CONF_SESSION_MODE, DEFAULT_SESSION_MODE),
            float(
                subentry.data.get(
                    CONF_SESSION_IDLE_TIMEOUT, DEFAULT_SESSION_IDLE_TIMEOUT
                )
            ),
        )
        await store.async_load()
        stores[subentry.subentry_id] = store
    entry.runtime_data = HermesAssistData(client=client, stores=stores)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_entry_updated))
    return True


async def _async_entry_updated(
    hass: HomeAssistant, entry: HermesAssistConfigEntry
) -> None:
    """Reload after a change; the sessions of removed agents are deleted first."""
    for subentry_id, store in entry.runtime_data.stores.items():
        if subentry_id not in entry.subentries:
            await store.async_clear()
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(
    hass: HomeAssistant, entry: HermesAssistConfigEntry
) -> bool:
    """Unload the platforms; background tasks of the entry are cancelled."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Delete the stored sessions of a removed entry."""
    for subentry in conversation_subentries(entry):
        await async_remove_sessions(hass, subentry.subentry_id)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Move the options of a version 1 entry into a conversation subentry.

    The entity, its device and its session file follow the subentry so that
    the agent keeps its entity id and its Hermes session. Home Assistant
    refuses a version newer than the config flow before calling this.
    """
    if entry.version == 1:
        subentry = ConfigSubentry(
            data=MappingProxyType({**DEFAULT_OPTIONS, **entry.options}),
            subentry_type=SUBENTRY_TYPE_CONVERSATION,
            title=entry.title,
            unique_id=None,
        )
        hass.config_entries.async_add_subentry(entry, subentry)

        entity_registry = er.async_get(hass)
        entity_id = entity_registry.async_get_entity_id(
            Platform.CONVERSATION, DOMAIN, entry.entry_id
        )
        if entity_id is not None:
            entity_registry.async_update_entity(
                entity_id,
                config_subentry_id=subentry.subentry_id,
                new_unique_id=subentry.subentry_id,
            )

        device_registry = dr.async_get(hass)
        device = device_registry.async_get_device_by_identifier(
            (DOMAIN, entry.entry_id), entry.entry_id
        )
        if device is not None:
            device_registry.async_update_device(
                device.id,
                new_identifiers={(DOMAIN, subentry.subentry_id)},
                new_config_subentry_id=subentry.subentry_id,
            )

        await async_move_sessions(hass, entry.entry_id, subentry.subentry_id)
        hass.config_entries.async_update_entry(
            entry, options={}, version=2, minor_version=1
        )
        _LOGGER.info("Migrated the Hermes Assist options into agent %s", entry.title)
    return True
