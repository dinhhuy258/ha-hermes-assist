"""Tests for the prompt context helpers."""

from unittest.mock import patch

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.hermes_assist.context import exposed_entities_cache
from custom_components.hermes_assist.prompts import EXPOSED_ENTITIES_INTRO

from .test_conversation import expose_fixtures

BUILD = "custom_components.hermes_assist.context._build_exposed_entities"


async def test_exposed_entities_text_is_cached(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    await expose_fixtures(hass, setup_entry)
    cache = exposed_entities_cache(hass)

    first = cache.text()
    with patch(BUILD, wraps=None) as build:
        second = cache.text()

    assert first == second
    assert first.startswith(EXPOSED_ENTITIES_INTRO)
    assert "- light.f1_living_room_light: Light" in first
    build.assert_not_called()


async def test_cache_is_rebuilt_after_a_registry_change(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    await expose_fixtures(hass, setup_entry)
    cache = exposed_entities_cache(hass)
    assert "(aliases: living room light, main light)" in cache.text()

    er.async_get(hass).async_update_entity(
        "light.f1_living_room_light", aliases=["big light"]
    )
    await hass.async_block_till_done()

    assert "(aliases: big light)" in cache.text()


async def test_cache_is_rebuilt_after_an_area_rename(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    await expose_fixtures(hass, setup_entry)
    cache = exposed_entities_cache(hass)
    assert "in Living room" in cache.text()

    area = ar.async_get(hass).async_get_area_by_name("Living room")
    assert area is not None
    ar.async_get(hass).async_update(area.id, name="Lounge")
    await hass.async_block_till_done()

    assert "in Lounge" in cache.text()


async def test_cache_is_rebuilt_after_exposure_changes(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    await expose_fixtures(hass, setup_entry)
    cache = exposed_entities_cache(hass)
    assert "light.hidden" not in cache.text()

    async_expose_entity(hass, conversation.DOMAIN, "light.hidden", True)
    await hass.async_block_till_done()

    assert "- light.hidden: Hidden" in cache.text()


async def test_cache_is_rebuilt_after_a_state_is_added_or_renamed(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    await expose_fixtures(hass, setup_entry)
    cache = exposed_entities_cache(hass)
    assert "switch.new" not in cache.text()

    hass.states.async_set("switch.new", "off", {"friendly_name": "New switch"})
    async_expose_entity(hass, conversation.DOMAIN, "switch.new", True)
    await hass.async_block_till_done()
    assert "- switch.new: New switch" in cache.text()

    hass.states.async_set("switch.new", "on", {"friendly_name": "New switch"})
    await hass.async_block_till_done()
    with patch(BUILD) as build:
        cache.text()
    build.assert_not_called()

    hass.states.async_set("switch.new", "on", {"friendly_name": "Renamed"})
    await hass.async_block_till_done()
    assert "- switch.new: Renamed" in cache.text()

    hass.states.async_remove("switch.new")
    await hass.async_block_till_done()
    assert "switch.new" not in cache.text()


async def test_no_exposed_entities_gives_empty_text(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    assert exposed_entities_cache(hass).text() == ""
