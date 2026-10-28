"""Tests for the Hermes Assist AI Task entity."""

from typing import Any
from unittest.mock import patch

from homeassistant.components import ai_task
from homeassistant.components.camera import Image
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_SUPPORTED_FEATURES
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol

from custom_components.hermes_assist.api import (
    HermesAuthError,
    HermesConnectionError,
    HermesResponseError,
    HermesTimeoutError,
)
from custom_components.hermes_assist.const import (
    CONF_AI_TASK_PROMPT,
    CONF_LIST_EXPOSED_ENTITIES,
    CONF_MODEL,
    CONF_REASONING_EFFORT,
    CONF_SESSION_KEY,
    CONF_TIMEOUT,
    DOMAIN,
)
from custom_components.hermes_assist.prompts import DEFAULT_PROMPT

from .conftest import ENTITY_ID, SUBENTRY_ID, FakeHermes, set_options
from .test_conversation import expose_fixtures

TASK_ENTITY_ID = "ai_task.hermes_assist"
# The ai_task component looks the camera helper up on its module at call time.
GET_IMAGE = "homeassistant.components.camera.async_get_image"
FRONT_DOOR = {
    "media_content_id": "media-source://camera/camera.front_door",
    "media_content_type": "image/jpeg",
}


async def generate(hass: HomeAssistant, **data: Any) -> ai_task.GenDataTaskResult:
    """Run one generate_data task on the default agent."""
    return await ai_task.async_generate_data(
        hass, task_name="test", entity_id=TASK_ENTITY_ID, **data
    )


def answer(hermes: FakeHermes, text: str) -> None:
    """Prepare a Hermes answer for the next task."""
    stream = hermes.stream("task-session")
    stream.push(text)
    stream.finish()


async def test_task_entity_belongs_to_the_agent_device(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    registry = er.async_get(hass)
    task = registry.async_get(TASK_ENTITY_ID)
    agent = registry.async_get(ENTITY_ID)

    assert task is not None
    assert agent is not None
    assert task.unique_id == f"{SUBENTRY_ID}-ai_task"
    assert task.config_subentry_id == SUBENTRY_ID
    assert task.device_id == agent.device_id
    device = dr.async_get(hass).async_get(agent.device_id)
    assert device is not None
    assert device.identifiers == {(DOMAIN, SUBENTRY_ID)}
    state = hass.states.get(TASK_ENTITY_ID)
    assert state is not None
    assert state.name == "Hermes Assist"
    assert state.attributes[ATTR_SUPPORTED_FEATURES] == (
        ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )


async def test_generate_data_returns_the_text(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    answer(hermes, "  Three meetings today. ")

    result = await generate(hass, instructions="Summarize today's calendar.")

    assert result.data == "Three meetings today."
    assert isinstance(result.conversation_id, str)
    call = hermes.calls[0]
    system, user = call["messages"]
    assert system["role"] == "system"
    assert system["content"].startswith(
        "You are a Home Assistant expert and help users with their tasks."
    )
    assert DEFAULT_PROMPT not in system["content"]
    assert user == {"role": "user", "content": "Summarize today's calendar."}
    assert "session_id" not in call
    assert call["session_key"] == "homeassistant:assist"
    assert call["model"] is None
    assert call["reasoning_effort"] is None
    assert call["timeout"] == 600.0
    # A task never touches the stored session of the agent.
    assert setup_entry.runtime_data.stores[SUBENTRY_ID].get(None) is None


async def test_generate_data_uses_the_agent_options(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(
        hass,
        setup_entry,
        {
            CONF_MODEL: "hermes-agent",
            CONF_REASONING_EFFORT: "low",
            CONF_SESSION_KEY: "homeassistant:tasks",
            CONF_TIMEOUT: 30,
        },
    )
    answer(hermes, "Done.")

    await generate(hass, instructions="Hello")

    call = hermes.calls[0]
    assert call["model"] == "hermes-agent"
    assert call["reasoning_effort"] == "low"
    assert call["session_key"] == "homeassistant:tasks"
    assert call["timeout"] == 30.0


@pytest.mark.parametrize("listed", [False, True], ids=["off", "on"])
async def test_exposed_entities_follow_the_agent_option(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    listed: bool,
) -> None:
    await set_options(hass, setup_entry, {CONF_LIST_EXPOSED_ENTITIES: listed})
    await expose_fixtures(hass, setup_entry)
    answer(hermes, "Okay.")

    await generate(hass, instructions="Hello")

    system = hermes.calls[0]["messages"][0]["content"]
    assert ("- light.f1_living_room_light: Light" in system) is listed
    assert "light.hidden" not in system


async def test_the_ai_task_prompt_replaces_the_default_sentence(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(
        hass,
        setup_entry,
        {
            CONF_AI_TASK_PROMPT: "Trả lời bằng tiếng Việt trong {{ 1 + 1 }} câu.",
            CONF_LIST_EXPOSED_ENTITIES: True,
        },
    )
    await expose_fixtures(hass, setup_entry)
    answer(hermes, "Xong.")

    await generate(hass, instructions="Hello")

    system = hermes.calls[0]["messages"][0]["content"]
    # The rendered prompt takes the place of the default sentence; the date
    # and time line of Home Assistant and the entities follow it.
    assert system.startswith("Trả lời bằng tiếng Việt trong 2 câu.\nCurrent time is ")
    assert "You are a Home Assistant expert" not in system
    assert DEFAULT_PROMPT not in system
    assert system.index("Today's date is") < system.index(
        "- light.f1_living_room_light: Light"
    )


async def test_an_ai_task_prompt_that_fails_to_render_is_sent_as_written(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    await set_options(
        hass,
        setup_entry,
        {CONF_AI_TASK_PROMPT: "{{ states.sensor.none.state.upper() }}"},
    )
    answer(hermes, "Okay.")

    await generate(hass, instructions="Hello")

    system = hermes.calls[0]["messages"][0]["content"]
    assert system.startswith("{{ states.sensor.none.state.upper() }}\nCurrent time is ")
    assert "Could not render the prompt" in caplog.text


async def test_an_empty_answer_fails(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    answer(hermes, "   ")

    with pytest.raises(HomeAssistantError, match="no answer"):
        await generate(hass, instructions="Hello")


async def test_a_rejected_key_starts_reauth(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hermes.error = HermesAuthError("Hermes rejected the API key")

    with (
        patch.object(ConfigEntry, "async_start_reauth") as start_reauth,
        pytest.raises(HomeAssistantError, match="rejected the API key"),
    ):
        await generate(hass, instructions="Hello")

    start_reauth.assert_called_once()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (HermesConnectionError("Connection refused"), "Could not reach Hermes"),
        (HermesTimeoutError("Hermes took too long"), "longer than 600 seconds"),
        (HermesResponseError("Hermes returned HTTP 500"), "Hermes returned an error"),
    ],
    ids=["connection", "timeout", "server"],
)
async def test_errors(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    error: Exception,
    message: str,
) -> None:
    hermes.error = error

    with pytest.raises(HomeAssistantError, match=message):
        await generate(hass, instructions="Hello")


STRUCTURE = vol.Schema(
    {
        vol.Required("count", description="Number of meetings"): int,
        vol.Optional("note"): str,
    }
)


@pytest.mark.parametrize(
    "text",
    ['{"count": 3}', '```json\n{"count": 3}\n```', '  {"count": 3}  '],
    ids=["bare", "fenced", "padded"],
)
async def test_structure_asks_for_json_and_parses_the_answer(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes, text: str
) -> None:
    answer(hermes, text)

    result = await generate(
        hass, instructions="How many meetings today?", structure=STRUCTURE
    )

    assert result.data == {"count": 3}
    system = hermes.calls[0]["messages"][0]["content"]
    assert "JSON" in system
    assert '"count"' in system
    assert "Number of meetings" in system
    assert '"required": ["count"]' in system


async def test_an_answer_that_is_not_json_fails(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    answer(hermes, "Three meetings.")

    with pytest.raises(HomeAssistantError, match="structure"):
        await generate(hass, instructions="How many?", structure=STRUCTURE)


async def test_without_a_structure_the_text_is_not_parsed(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    answer(hermes, '{"count": 3}')

    result = await generate(hass, instructions="How many?")

    assert result.data == '{"count": 3}'
    assert "JSON" not in hermes.calls[0]["messages"][0]["content"]


async def test_a_camera_attachment_is_sent_as_an_image(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    answer(hermes, "A parcel.")

    with patch(GET_IMAGE, return_value=Image("image/jpeg", b"\xff\xd8jpeg")) as get:
        result = await generate(
            hass, instructions="What is at the door?", attachments=[FRONT_DOOR]
        )

    get.assert_awaited_once_with(hass, "camera.front_door")
    assert result.data == "A parcel."
    assert hermes.calls[0]["messages"][1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is at the door?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9hqcGVn"},
            },
        ],
    }


async def test_a_non_image_attachment_is_rejected(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    with (
        patch(GET_IMAGE, return_value=Image("application/pdf", b"%PDF")),
        pytest.raises(HomeAssistantError, match="application/pdf"),
    ):
        await generate(hass, instructions="Read this", attachments=[FRONT_DOOR])

    assert hermes.calls == []


async def test_a_camera_that_fails_fails_the_task(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    with (
        patch(GET_IMAGE, side_effect=HomeAssistantError("Unable to get image")),
        pytest.raises(HomeAssistantError, match="Unable to get image"),
    ):
        await generate(hass, instructions="Look", attachments=[FRONT_DOOR])

    assert hermes.calls == []
