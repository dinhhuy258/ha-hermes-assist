"""The Hermes Assist integration."""

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.typing import ConfigType

from .api import HermesAuthError, HermesClient, HermesError
from .const import (
    CONF_API_KEY,
    CONF_SESSION_IDLE_TIMEOUT,
    CONF_SESSION_MODE,
    CONF_URL,
    DEFAULT_SESSION_IDLE_TIMEOUT,
    DEFAULT_SESSION_MODE,
    DOMAIN,
)
from .services import async_setup_services
from .session import SessionStore, async_remove_sessions

PLATFORMS: list[Platform] = [Platform.CONVERSATION]
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


@dataclass
class HermesAssistData:
    """Runtime objects of one config entry."""

    client: HermesClient
    store: SessionStore
    reauth_requested: bool = False

    def request_reauth(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Start a reauth flow once; a new key reloads the entry and resets this."""
        if self.reauth_requested:
            return
        self.reauth_requested = True
        entry.async_start_reauth(hass)


type HermesAssistConfigEntry = ConfigEntry[HermesAssistData]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the hermes_assist.ask service."""
    async_setup_services(hass)
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: HermesAssistConfigEntry
) -> bool:
    """Validate the key, load the session store and set up the platforms."""
    client = HermesClient(
        async_get_clientsession(hass), entry.data[CONF_URL], entry.data[CONF_API_KEY]
    )
    try:
        await client.async_validate()
    except HermesAuthError as err:
        raise ConfigEntryAuthFailed("Hermes rejected the API key") from err
    except HermesError as err:
        raise ConfigEntryNotReady(f"Could not reach Hermes at {client.url}") from err

    store = SessionStore(
        hass,
        entry.entry_id,
        entry.options.get(CONF_SESSION_MODE, DEFAULT_SESSION_MODE),
        float(
            entry.options.get(CONF_SESSION_IDLE_TIMEOUT, DEFAULT_SESSION_IDLE_TIMEOUT)
        ),
    )
    await store.async_load()
    entry.runtime_data = HermesAssistData(client=client, store=store)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: HermesAssistConfigEntry
) -> bool:
    """Unload the platforms; background tasks of the entry are cancelled."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Delete the stored session of a removed entry."""
    await async_remove_sessions(hass, entry.entry_id)
