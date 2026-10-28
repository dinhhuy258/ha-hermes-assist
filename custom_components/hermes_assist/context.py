"""Context added to the system prompt: exposed entities, device and area."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_listen_entity_updates,
    async_should_expose,
)
from homeassistant.const import EVENT_STATE_CHANGED, MATCH_ALL
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    HomeAssistant,
    callback,
)
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)
from homeassistant.helpers.template import Template
from homeassistant.util.hass_dict import HassKey

from .const import (
    CONF_LIST_EXPOSED_ENTITIES,
    CONF_PROMPT,
    DEFAULT_LIST_EXPOSED_ENTITIES,
    DOMAIN,
)
from .prompts import DEFAULT_PROMPT, EXPOSED_ENTITIES_INTRO

_LOGGER = logging.getLogger(__name__)

DATA_EXPOSED_ENTITIES: HassKey[ExposedEntitiesCache] = HassKey(
    f"{DOMAIN}_exposed_entities"
)
# Registry fields that change a line of the exposed-entity list.
_ENTITY_FIELDS = ("aliases", "area_id", "device_id", "name", "original_name")
_DEVICE_FIELDS = ("area_id", "name", "name_by_user")


def _build_exposed_entities(hass: HomeAssistant) -> str:
    """List the entities exposed to Assist: ID, name, aliases and area, no state."""
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    area_registry = ar.async_get(hass)
    lines: list[str] = []
    for state in sorted(hass.states.async_all(), key=lambda s: s.entity_id):
        if not async_should_expose(hass, conversation.DOMAIN, state.entity_id):
            continue
        line = f"- {state.entity_id}: {state.name}"
        area_id: str | None = None
        if entry := entity_registry.async_get(state.entity_id):
            if aliases := [
                alias.strip() for alias in entry.aliases if isinstance(alias, str)
            ]:
                line += f" (aliases: {', '.join(aliases)})"
            area_id = entry.area_id
            if (
                area_id is None
                and entry.device_id
                and (device := device_registry.async_get(entry.device_id))
            ):
                area_id = device.area_id
        if area_id and (area := area_registry.async_get_area(area_id)):
            line += f" in {area.name}"
        lines.append(line)
    if not lines:
        return ""
    return "\n".join([EXPOSED_ENTITIES_INTRO, *lines])


class ExposedEntitiesCache:
    """The exposed-entity block, rebuilt only after a change that affects it."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Start empty and listen for the events that invalidate the block."""
        self._hass = hass
        self._text: str | None = None
        self._unsubscribe: list[CALLBACK_TYPE] = [
            hass.bus.async_listen(ar.EVENT_AREA_REGISTRY_UPDATED, self.invalidate),
            hass.bus.async_listen(
                er.EVENT_ENTITY_REGISTRY_UPDATED,
                self.invalidate,
                event_filter=self._entity_registry_filter,
            ),
            hass.bus.async_listen(
                dr.EVENT_DEVICE_REGISTRY_UPDATED,
                self.invalidate,
                event_filter=self._device_registry_filter,
            ),
            hass.bus.async_listen(
                EVENT_STATE_CHANGED, self.invalidate, event_filter=self._state_filter
            ),
            async_listen_entity_updates(hass, conversation.DOMAIN, self.invalidate),
        ]

    @callback
    def text(self) -> str:
        """Return the block, building it when it is stale."""
        if self._text is None:
            self._text = _build_exposed_entities(self._hass)
            _LOGGER.debug("Rebuilt the exposed-entity list")
        return self._text

    @callback
    def invalidate(self, event: Event[Any] | None = None) -> None:
        """Forget the block; the next request rebuilds it."""
        self._text = None

    @callback
    def _entity_registry_filter(
        self, event_data: er.EventEntityRegistryUpdatedData
    ) -> bool:
        if event_data["action"] != "update":
            return True
        return any(field in event_data["changes"] for field in _ENTITY_FIELDS)

    @callback
    def _device_registry_filter(
        self, event_data: dr.EventDeviceRegistryUpdatedData
    ) -> bool:
        if event_data["action"] != "update":
            return True
        return any(field in event_data["changes"] for field in _DEVICE_FIELDS)

    @callback
    def _state_filter(self, event_data: EventStateChangedData) -> bool:
        old, new = event_data["old_state"], event_data["new_state"]
        if old is None or new is None:
            return True
        return old.name != new.name


@callback
def exposed_entities_cache(hass: HomeAssistant) -> ExposedEntitiesCache:
    """Return the cache of this Home Assistant instance, creating it once."""
    if DATA_EXPOSED_ENTITIES not in hass.data:
        hass.data[DATA_EXPOSED_ENTITIES] = ExposedEntitiesCache(hass)
    return hass.data[DATA_EXPOSED_ENTITIES]


def validate_prompt(hass: HomeAssistant, prompt: str) -> bool:
    """Return whether the prompt is a template with valid syntax."""
    try:
        Template(prompt, hass).ensure_valid()
    except TemplateError:
        return False
    return True


@callback
def render_prompt(hass: HomeAssistant, prompt: str) -> str:
    """Render the prompt as a template; on failure send it as written."""
    try:
        return str(Template(prompt, hass).async_render(parse_result=False))
    except TemplateError as err:
        _LOGGER.warning("Could not render the prompt, sending it as written: %s", err)
        return prompt


@callback
def device_context(hass: HomeAssistant, device_id: str | None) -> str:
    """Describe the voice device that heard the request, and its area."""
    if not device_id:
        return ""
    device = dr.async_get(hass).async_get(device_id)
    if device is None:
        return ""
    name = device.name_by_user or device.name
    area = ar.async_get(hass).async_get_area(device.area_id) if device.area_id else None
    if area is None:
        return f'The user is talking to the voice device "{name}".'
    return (
        f'The user is talking to the voice device "{name}" in the {area.name}. '
        f"'Here' means the {area.name}."
    )


@callback
def build_system_prompt(
    hass: HomeAssistant,
    options: Mapping[str, Any],
    *,
    device_id: str | None = None,
    language: str | None = None,
    extra_system_prompt: str | None = None,
) -> str:
    """Combine the rendered agent prompt with the context of one request.

    The exposed entities come right after the prompt so that the part of the
    system message that is the same for every request stays a prefix.
    """
    parts = [render_prompt(hass, options.get(CONF_PROMPT, DEFAULT_PROMPT))]
    if options.get(CONF_LIST_EXPOSED_ENTITIES, DEFAULT_LIST_EXPOSED_ENTITIES):
        parts.append(exposed_entities_cache(hass).text())
    parts.append(device_context(hass, device_id))
    if language and language != MATCH_ALL:
        parts.append(f"Reply in the language with code '{language}'.")
    if extra_system_prompt:
        parts.append(extra_system_prompt)
    return "\n\n".join(part for part in parts if part)


@callback
def build_messages(
    hass: HomeAssistant,
    options: Mapping[str, Any],
    text: str,
    *,
    prompt: str | None = None,
    device_id: str | None = None,
    language: str | None = None,
    extra_system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Return the system and user messages of one request."""
    messages: list[dict[str, Any]] = []
    if system := build_system_prompt(
        hass,
        options,
        device_id=device_id,
        language=language,
        extra_system_prompt=extra_system_prompt,
    ):
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})
    return messages
