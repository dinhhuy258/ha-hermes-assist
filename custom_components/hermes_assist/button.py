"""Button that clears the stored Hermes sessions of an agent."""

import logging

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import HermesAssistConfigEntry, conversation_subentries
from .device import agent_device_info

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HermesAssistConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add one reset button per agent subentry."""
    for subentry in conversation_subentries(entry):
        async_add_entities(
            [HermesResetSessionButton(entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class HermesResetSessionButton(ButtonEntity):
    """Forget every stored session of the agent; the next request starts fresh."""

    _attr_has_entity_name = True
    _attr_translation_key = "reset_session"
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(
        self, entry: HermesAssistConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the button of one agent."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = f"{subentry.subentry_id}-reset_session"
        self._attr_device_info = agent_device_info(subentry)

    async def async_press(self) -> None:
        """Clear the sessions of this agent."""
        _LOGGER.debug("Resetting the sessions of %s", self.subentry.title)
        await self.entry.runtime_data.stores[self.subentry.subentry_id].async_clear()
