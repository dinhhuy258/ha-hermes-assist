"""Tests for the reset session button."""

from typing import Any

from homeassistant.components.button import DOMAIN as BUTTON_DOMAIN, SERVICE_PRESS
from homeassistant.const import ATTR_ENTITY_ID, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.hermes_assist.const import DOMAIN
from custom_components.hermes_assist.session import storage_key

from .conftest import ENTITY_ID, SUBENTRY_ID, FakeHermes
from .test_config_flow import flush_storage
from .test_conversation import converse

BUTTON_ID = "button.hermes_assist_reset_session"


async def test_button_belongs_to_the_agent_device(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    registry = er.async_get(hass)
    button = registry.async_get(BUTTON_ID)
    agent = registry.async_get(ENTITY_ID)

    assert button is not None
    assert agent is not None
    assert button.unique_id == f"{SUBENTRY_ID}-reset_session"
    assert button.entity_category is EntityCategory.CONFIG
    assert button.config_subentry_id == SUBENTRY_ID
    assert button.device_id == agent.device_id
    device = dr.async_get(hass).async_get(agent.device_id)
    assert device is not None
    assert device.identifiers == {(DOMAIN, SUBENTRY_ID)}
    assert hass.states.get(BUTTON_ID).name == "Hermes Assist Reset session"


async def test_pressing_the_button_clears_the_sessions(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    hass_storage: dict[str, Any],
) -> None:
    key = storage_key(SUBENTRY_ID)
    setup_entry.runtime_data.stores[SUBENTRY_ID].set(None, "session-1")
    await flush_storage(hass)
    assert hass_storage[key]["data"]["session_id"] == "session-1"
    stream = hermes.stream("session-2")
    stream.push("Okay.")
    stream.finish()

    await hass.services.async_call(
        BUTTON_DOMAIN, SERVICE_PRESS, {ATTR_ENTITY_ID: BUTTON_ID}, blocking=True
    )
    await converse(hass, "What number?")

    assert hermes.calls[0]["session_id"] is None
