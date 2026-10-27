"""Tests for the Hermes Assist conversation agent and the ask service."""

import asyncio
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import patch

from homeassistant.components import conversation
from homeassistant.components.assist_satellite import AssistSatelliteEntityFeature
from homeassistant.components.camera import Image
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.config_entries import ConfigEntry, ConfigEntryState, ConfigSubentry
from homeassistant.core import Context, HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    intent,
)
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)
import voluptuous as vol
import yaml

from custom_components.hermes_assist.api import (
    HermesAuthError,
    HermesConnectionError,
    HermesResponseError,
    HermesStreamError,
    HermesTimeoutError,
)
from custom_components.hermes_assist.const import (
    ATTR_AGENT_ID,
    ATTR_CAMERA_ENTITY_ID,
    ATTR_SESSION_ID,
    ATTR_SYSTEM_PROMPT,
    ATTR_TEXT,
    CONF_HANDOFF_AFTER,
    CONF_LIST_EXPOSED_ENTITIES,
    CONF_MESSAGE_GENERIC,
    CONF_MESSAGE_HANDOFF,
    CONF_MESSAGES,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_SESSION_MODE,
    CONF_TIMEOUT,
    DEFAULT_OPTIONS,
    DOMAIN,
    NAME,
    SERVICE_ASK,
    SESSION_MODE_PER_CONVERSATION,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.hermes_assist.messages import (
    ERROR_CANNOT_CONNECT,
    ERROR_GENERIC,
    ERROR_TOO_LONG,
    HANDOFF_MESSAGE,
)
from custom_components.hermes_assist.prompts import (
    DEFAULT_PROMPT,
    EXPOSED_ENTITIES_INTRO,
)

from .conftest import ENTITY_ID, SUBENTRY_ID, FakeHermes, set_options

SATELLITE = "assist_satellite.kitchen"
ANNOUNCE = AssistSatelliteEntityFeature.ANNOUNCE
START_CONVERSATION = AssistSatelliteEntityFeature.START_CONVERSATION
NOTIFY = (
    "custom_components.hermes_assist.conversation.persistent_notification.async_create"
)
PIPELINES = (
    "custom_components.hermes_assist.conversation.assist_pipeline.async_get_pipelines"
)
MEDIA_ID = "custom_components.hermes_assist.conversation.tts.generate_media_source_id"


def pipeline(
    language: str,
    tts_engine: str | None = "tts.cloud",
    tts_voice: str | None = None,
    conversation_engine: str = ENTITY_ID,
) -> SimpleNamespace:
    """Return a stand-in for an Assist pipeline."""
    return SimpleNamespace(
        name=f"Pipeline {language}",
        conversation_engine=conversation_engine,
        language=language,
        tts_engine=tts_engine,
        tts_language=f"{language}-XX",
        tts_voice=tts_voice,
    )


async def converse(
    hass: HomeAssistant, text: str, **kwargs: Any
) -> conversation.ConversationResult:
    """Send one turn to the Hermes agent."""
    return await conversation.async_converse(
        hass,
        text,
        kwargs.pop("conversation_id", None),
        Context(),
        agent_id=ENTITY_ID,
        **kwargs,
    )


def speech(result: conversation.ConversationResult) -> str:
    """Return the spoken text of a result."""
    return result.response.speech["plain"]["speech"]


def satellite(
    hass: HomeAssistant, features: int, state: str = "idle"
) -> dict[str, list[ServiceCall]]:
    """Create a satellite state and mock its announce and start_conversation.

    Call this after the entry is set up: setting up the entry loads
    assist_satellite, which registers the real services.
    """
    hass.states.async_set(SATELLITE, state, {"supported_features": features})
    return {
        "announce": async_mock_service(hass, "assist_satellite", "announce"),
        "start_conversation": async_mock_service(
            hass, "assist_satellite", "start_conversation"
        ),
    }


@pytest.fixture
async def handoff_entry(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> MockConfigEntry:
    """Return an entry that hands off after 50 ms."""
    await set_options(hass, setup_entry, {CONF_HANDOFF_AFTER: 0.05})
    return setup_entry


async def test_streams_the_answer_into_the_chat_log(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    stream = hermes.stream("session-1")
    stream.push("**Hello**", " there, ", "living_room", "\n- lights on.")
    stream.finish()

    result = await converse(hass, "Hello")

    assert speech(result) == "Hello there, living room\nlights on."
    assert result.continue_conversation is False
    assert stream.closed
    call = hermes.calls[0]
    assert call["messages"][-1] == {"role": "user", "content": "Hello"}
    assert call["session_key"] == "homeassistant:assist"
    assert call["session_id"] is None
    assert call["model"] is None
    assert call["timeout"] == 600


async def test_question_keeps_the_microphone_open(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    stream = hermes.stream()
    stream.push("Which room?")
    stream.finish()

    result = await converse(hass, "Turn on the lights")

    assert result.continue_conversation is True


async def test_session_id_is_reused_and_overwritten(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    for session_id, text in (
        ("session-1", "One."),
        ("session-2", "Two."),
        ("session-2", "Three."),
    ):
        stream = hermes.stream(session_id)
        stream.push(text)
        stream.finish()

    await converse(hass, "First")
    await converse(hass, "Second")
    await converse(hass, "Third")

    assert [call["session_id"] for call in hermes.calls] == [
        None,
        "session-1",
        "session-2",
    ]


async def test_missing_session_header_warns_once(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for _ in range(3):
        stream = hermes.stream(None)
        stream.push("Okay.")
        stream.finish()

    for _ in range(3):
        await converse(hass, "Hello")

    assert [call["session_id"] for call in hermes.calls] == [None, None, None]
    assert caplog.text.count("X-Hermes-Session-Id") == 1


async def test_model_option_is_sent(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(hass, setup_entry, {CONF_MODEL: "hermes-agent"})
    stream = hermes.stream()
    stream.push("Okay.")
    stream.finish()

    await converse(hass, "Hello")

    assert hermes.calls[0]["model"] == "hermes-agent"


async def test_reasoning_effort_option_is_sent(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(hass, setup_entry, {CONF_REASONING_EFFORT: "low"})
    for _ in range(2):
        stream = hermes.stream()
        stream.push("Okay.")
        stream.finish()

    await converse(hass, "Hello")
    await ask(hass, **{ATTR_TEXT: "Hello"})

    assert hermes.calls[0]["reasoning_effort"] == "low"
    assert hermes.calls[1]["reasoning_effort"] == "low"


async def test_reasoning_effort_is_off_by_default(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    stream = hermes.stream()
    stream.push("Okay.")
    stream.finish()

    await converse(hass, "Hello")

    assert hermes.calls[0]["reasoning_effort"] is None


async def test_prompt_has_device_area_language_and_extra_prompt(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    area = ar.async_get(hass).async_create("Kitchen")
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=setup_entry.entry_id,
        identifiers={("test", "kitchen-satellite")},
        name="Kitchen Satellite",
    )
    dr.async_get(hass).async_update_device(device.id, area_id=area.id)
    stream = hermes.stream()
    stream.push("Done.")
    stream.finish()

    await converse(
        hass,
        "Turn on the lights",
        device_id=device.id,
        extra_system_prompt="The user set a timer earlier.",
    )

    system = hermes.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith(DEFAULT_PROMPT)
    assert (
        'The user is talking to the voice device "Kitchen Satellite" in the Kitchen.'
        in system["content"]
    )
    assert "'Here' means the Kitchen." in system["content"]
    assert "Reply in the language with code 'en'." in system["content"]
    assert system["content"].endswith("The user set a timer earlier.")


async def expose_fixtures(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Create two exposed entities, one with a device area, and one hidden one."""
    area = ar.async_get(hass).async_create("Living room")
    registry = er.async_get(hass)
    light = registry.async_get_or_create(
        "light", "test", "light-1", suggested_object_id="f1_living_room_light"
    )
    registry.async_update_entity(
        light.entity_id, area_id=area.id, aliases=["living room light", "main light"]
    )
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "fan-1")},
        name="Ceiling fan",
    )
    dr.async_get(hass).async_update_device(device.id, area_id=area.id)
    fan = registry.async_get_or_create(
        "fan",
        "test",
        "fan-1",
        suggested_object_id="f1_living_room_fan",
        device_id=device.id,
    )
    hass.states.async_set(light.entity_id, "off", {"friendly_name": "Light"})
    hass.states.async_set(fan.entity_id, "off", {"friendly_name": "Fan"})
    hass.states.async_set("switch.plain", "on", {"friendly_name": "Plain switch"})
    hass.states.async_set("light.hidden", "on", {"friendly_name": "Hidden"})
    for entity_id, exposed in (
        (light.entity_id, True),
        (fan.entity_id, True),
        ("switch.plain", True),
        ("light.hidden", False),
    ):
        async_expose_entity(hass, conversation.DOMAIN, entity_id, exposed)


async def test_exposed_entities_are_listed_when_enabled(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(hass, setup_entry, {CONF_LIST_EXPOSED_ENTITIES: True})
    await expose_fixtures(hass, setup_entry)
    stream = hermes.stream()
    stream.push("Done.")
    stream.finish()

    await converse(hass, "Turn on the living room light")

    system = hermes.calls[0]["messages"][0]["content"]
    assert system.startswith(DEFAULT_PROMPT)
    assert (
        "\n- light.f1_living_room_light: Light"
        " (aliases: living room light, main light) in Living room\n"
    ) in system
    assert "\n- fan.f1_living_room_fan: Fan in Living room\n" in system
    assert "\n- switch.plain: Plain switch" in system
    assert "light.hidden" not in system
    assert "conversation.hermes_assist" not in system
    assert system.index(EXPOSED_ENTITIES_INTRO) < system.index(
        "Reply in the language with code 'en'."
    )


async def test_exposed_entities_are_off_by_default(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await expose_fixtures(hass, setup_entry)
    stream = hermes.stream()
    stream.push("Done.")
    stream.finish()

    await converse(hass, "Turn on the living room light")

    system = hermes.calls[0]["messages"][0]["content"]
    assert EXPOSED_ENTITIES_INTRO not in system
    assert "light.f1_living_room_light" not in system


async def test_exposed_entities_follow_registry_changes(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(hass, setup_entry, {CONF_LIST_EXPOSED_ENTITIES: True})
    await expose_fixtures(hass, setup_entry)
    for _ in range(2):
        stream = hermes.stream()
        stream.push("Done.")
        stream.finish()

    await converse(hass, "Hello")
    er.async_get(hass).async_update_entity(
        "light.f1_living_room_light", aliases=["big light"]
    )
    await hass.async_block_till_done()
    await converse(hass, "Hello again")

    assert (
        "(aliases: living room light, main light)"
        in (hermes.calls[0]["messages"][0]["content"])
    )
    assert "(aliases: big light)" in hermes.calls[1]["messages"][0]["content"]


async def test_prompt_is_rendered_as_a_template(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hass.states.async_set("sensor.outside", "21", {"friendly_name": "Outside"})
    await set_options(
        hass,
        setup_entry,
        {CONF_PROMPT: "Outside it is {{ states('sensor.outside') }} degrees."},
    )
    stream = hermes.stream()
    stream.push("Okay.")
    stream.finish()

    await converse(hass, "Hello")

    system = hermes.calls[0]["messages"][0]["content"]
    assert system.startswith("Outside it is 21 degrees.")
    assert "Reply in the language with code 'en'." in system


async def test_prompt_that_fails_to_render_is_sent_raw(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompt = "Today: {{ states.sensor.missing.attributes.x.y }}"
    await set_options(hass, setup_entry, {CONF_PROMPT: prompt})
    stream = hermes.stream()
    stream.push("Okay.")
    stream.finish()

    await converse(hass, "Hello")

    assert hermes.calls[0]["messages"][0]["content"].startswith(prompt)
    assert "Could not render the prompt" in caplog.text


async def test_prompt_result_is_not_parsed(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(hass, setup_entry, {CONF_PROMPT: "{{ 1 + 1 }}"})
    stream = hermes.stream()
    stream.push("Okay.")
    stream.finish()

    await converse(hass, "Hello")

    assert hermes.calls[0]["messages"][0]["content"].startswith("2\n\n")


@pytest.mark.parametrize(
    ("error", "logged"),
    [
        (HermesConnectionError("Connection refused"), "Could not reach Hermes"),
        (HermesResponseError("Hermes returned HTTP 500"), "Hermes returned an error"),
        (HermesAuthError("Hermes rejected the API key"), "rejected the API key"),
        (HermesTimeoutError("Hermes took too long"), "longer than the timeout"),
    ],
    ids=["connection", "server", "auth", "timeout"],
)
async def test_every_error_speaks_the_error_message(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    logged: str,
) -> None:
    hermes.error = error

    result = await converse(hass, "Hello")

    assert speech(result) == ERROR_GENERIC
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert logged in caplog.text


async def test_rejected_key_starts_reauth_once(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hermes.error = HermesAuthError("Hermes rejected the API key")

    with patch.object(ConfigEntry, "async_start_reauth") as start_reauth:
        await converse(hass, "Hello")
        await converse(hass, "Hello again")

    assert start_reauth.call_count == 1


async def test_error_finish_before_content_is_spoken(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hermes.stream().fail(HermesResponseError("Hermes finished with error"))

    result = await converse(hass, "Hello")

    assert speech(result) == ERROR_GENERIC


async def test_empty_answer(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hermes.stream().finish()

    result = await converse(hass, "Hello")

    assert speech(result) == ERROR_GENERIC


async def test_broken_stream_keeps_what_was_spoken(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = hermes.stream()
    stream.push("It is sunny.")
    stream.fail(HermesStreamError("Connection lost"))

    result = await converse(hass, "Weather?")

    assert speech(result) == "It is sunny."
    assert "stream broke" in caplog.text


async def test_slow_answer_is_handed_off_and_announced(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    calls = satellite(hass, ANNOUNCE)
    stream = hermes.stream()

    result = await converse(hass, "Summarize my email", satellite_id=SATELLITE)
    assert speech(result) == HANDOFF_MESSAGE

    stream.push(
        "**Done.** See [the docs](https://example.com/docs) ",
        "for `light_kitchen`.",
    )
    stream.finish()
    await hass.async_block_till_done(wait_background_tasks=True)

    assert [call.data for call in calls["announce"]] == [
        {"entity_id": SATELLITE, "message": "Done. See the docs for light kitchen."}
    ]
    assert calls["start_conversation"] == []


async def test_late_question_starts_a_conversation_on_the_same_session(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(
        hass, handoff_entry, {CONF_SESSION_MODE: SESSION_MODE_PER_CONVERSATION}
    )
    calls = satellite(hass, ANNOUNCE | START_CONVERSATION)
    question = "You have three new emails. Should I read them?"
    stream = hermes.stream("session-1")

    await converse(hass, "Check my email", satellite_id=SATELLITE)
    stream.push(question)
    stream.finish()
    await hass.async_block_till_done(wait_background_tasks=True)

    assert [call.data for call in calls["start_conversation"]] == [
        {
            "entity_id": SATELLITE,
            "start_message": question,
            "extra_system_prompt": question,
        }
    ]
    assert calls["announce"] == []

    follow_up = hermes.stream("session-1")
    follow_up.push("Reading them now.")
    follow_up.finish()
    await converse(
        hass, "Yes please", satellite_id=SATELLITE, extra_system_prompt=question
    )

    assert hermes.calls[1]["session_id"] == "session-1"


async def test_delivery_waits_until_the_satellite_is_idle(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    calls = satellite(hass, ANNOUNCE, state="responding")
    stream = hermes.stream()

    await converse(hass, "Slow question", satellite_id=SATELLITE)
    stream.push("All done.")
    stream.finish()
    await asyncio.sleep(0.05)
    assert calls["announce"] == []

    hass.states.async_set(SATELLITE, "idle", {"supported_features": ANNOUNCE})
    await hass.async_block_till_done(wait_background_tasks=True)

    assert calls["announce"][0].data["message"] == "All done."


async def test_busy_satellite_falls_back_to_a_notification(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    calls = satellite(hass, ANNOUNCE, state="processing")
    stream = hermes.stream()

    with (
        patch(
            "custom_components.hermes_assist.conversation.SATELLITE_IDLE_TIMEOUT", 0.05
        ),
        patch(NOTIFY) as notify,
    ):
        await converse(hass, "Slow question", satellite_id=SATELLITE)
        stream.push("All done.")
        stream.finish()
        await hass.async_block_till_done(wait_background_tasks=True)

    notify.assert_called_once_with(hass, "All done.", title=NAME)
    assert calls["announce"] == []


async def test_spoken_messages_are_configurable(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(
        hass,
        handoff_entry,
        {
            CONF_MESSAGES: {
                CONF_MESSAGE_HANDOFF: "Đang làm, xong sẽ báo lại.",
                CONF_MESSAGE_GENERIC: "",
            }
        },
    )
    hermes.stream()

    handoff = await converse(hass, "Mở nhạc")
    hermes.error = HermesAuthError("Hermes rejected the API key")
    error = await converse(hass, "Mở nhạc")

    assert speech(handoff) == "Đang làm, xong sẽ báo lại."
    assert speech(error) == ERROR_GENERIC


async def test_late_answer_uses_the_tts_of_the_matching_pipeline(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    calls = satellite(hass, ANNOUNCE | START_CONVERSATION)
    pipelines = [
        pipeline("en", conversation_engine="conversation.other"),
        pipeline("en", tts_engine=None),
        pipeline("en"),
        pipeline("vi", tts_voice="Charon"),
    ]
    first = hermes.stream()
    second = hermes.stream()

    with (
        patch(PIPELINES, return_value=pipelines),
        patch(MEDIA_ID, return_value="media-source://tts/cloud") as media_id,
    ):
        await converse(hass, "Mở nhạc", satellite_id=SATELLITE, language="vi")
        first.push("Đã mở nhạc.")
        first.finish()
        await hass.async_block_till_done(wait_background_tasks=True)
        await converse(hass, "Check", satellite_id=SATELLITE, language="vi")
        second.push("Mở tiếp không?")
        second.finish()
        await hass.async_block_till_done(wait_background_tasks=True)

    media_id.assert_called_with(
        hass,
        "Mở tiếp không?",
        engine="tts.cloud",
        language="vi-XX",
        options={"voice": "Charon"},
    )
    assert [call.data for call in calls["announce"]] == [
        {
            "entity_id": SATELLITE,
            "message": "Đã mở nhạc.",
            "media_id": "media-source://tts/cloud",
        }
    ]
    assert calls["start_conversation"][0].data["start_media_id"] == (
        "media-source://tts/cloud"
    )


async def test_late_answer_falls_back_to_the_satellite_tts(
    hass: HomeAssistant,
    handoff_entry: MockConfigEntry,
    hermes: FakeHermes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    await set_options(
        hass,
        handoff_entry,
        {CONF_MESSAGES: {CONF_MESSAGE_GENERIC: "Xin lỗi, Hermes gặp lỗi."}},
    )
    calls = satellite(hass, ANNOUNCE)
    stream = hermes.stream()

    with (
        patch(PIPELINES, return_value=[pipeline("en")]),
        patch(MEDIA_ID, side_effect=HomeAssistantError("Invalid TTS provider")),
    ):
        await converse(hass, "Mở nhạc", satellite_id=SATELLITE, language="vi")
        stream.fail(HermesConnectionError("Connection refused"))
        await hass.async_block_till_done(wait_background_tasks=True)

    assert [call.data for call in calls["announce"]] == [
        {"entity_id": SATELLITE, "message": "Xin lỗi, Hermes gặp lỗi."}
    ]
    assert "Could not use the text to speech of pipeline Pipeline en" in caplog.text


async def test_handoff_without_a_satellite_notifies(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    stream = hermes.stream()

    with patch(NOTIFY) as notify:
        await converse(hass, "Slow question")
        stream.push("All done.")
        stream.finish()
        await hass.async_block_till_done(wait_background_tasks=True)

    notify.assert_called_once_with(hass, "All done.", title=NAME)


async def test_handoff_that_exceeds_the_timeout(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(hass, handoff_entry, {CONF_TIMEOUT: 0.1})
    stream = hermes.stream()

    with patch(NOTIFY) as notify:
        result = await converse(hass, "Very slow question")
        await hass.async_block_till_done(wait_background_tasks=True)

    assert speech(result) == HANDOFF_MESSAGE
    notify.assert_called_once_with(hass, ERROR_GENERIC, title=NAME)
    assert stream.closed


async def test_handoff_disabled_waits_for_the_answer(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(hass, setup_entry, {CONF_HANDOFF_AFTER: 0})
    stream = hermes.stream()

    def answer() -> None:
        stream.push("Finally.")
        stream.finish()

    hass.loop.call_later(0.2, answer)
    result = await converse(hass, "Slow question")

    assert speech(result) == "Finally."


@pytest.mark.parametrize("first_delta", [None, "It is "], ids=["before", "after"])
async def test_cancelled_turn_leaves_no_background_work(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    first_delta: str | None,
) -> None:
    calls = satellite(hass, ANNOUNCE)
    stream = hermes.stream()
    if first_delta:
        stream.push(first_delta)

    turn = hass.async_create_task(converse(hass, "Slow", satellite_id=SATELLITE))
    await asyncio.sleep(0.05)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    stream.push("Too late.")
    stream.finish()
    await hass.async_block_till_done(wait_background_tasks=True)

    assert stream.closed
    assert calls["announce"] == []
    assert calls["start_conversation"] == []


async def test_unload_cancels_handed_off_requests(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    calls = satellite(hass, ANNOUNCE)
    stream = hermes.stream()

    await converse(hass, "Slow question", satellite_id=SATELLITE)
    assert await hass.config_entries.async_unload(handoff_entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)

    assert handoff_entry.state is ConfigEntryState.NOT_LOADED
    assert stream.closed
    assert calls["announce"] == []


async def test_per_conversation_mode_keeps_one_session_per_conversation(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(
        hass, setup_entry, {CONF_SESSION_MODE: SESSION_MODE_PER_CONVERSATION}
    )
    for session_id in ("session-a", "session-b", "session-a"):
        stream = hermes.stream(session_id)
        stream.push("Okay.")
        stream.finish()

    first = await converse(hass, "One")
    await converse(hass, "Two")
    await converse(hass, "Three", conversation_id=first.conversation_id)

    assert [call["session_id"] for call in hermes.calls] == [None, None, "session-a"]


async def test_each_agent_has_its_own_prompt_and_session(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hass.config_entries.async_add_subentry(
        setup_entry,
        ConfigSubentry(
            data=MappingProxyType(
                {**DEFAULT_OPTIONS, CONF_PROMPT: "Trả lời tiếng Việt."}
            ),
            subentry_id="agent-2",
            subentry_type=SUBENTRY_TYPE_CONVERSATION,
            title="Hermes Vietnamese",
            unique_id=None,
        ),
    )
    await hass.async_block_till_done()
    for session_id in ("session-en", "session-vi", "session-en", "session-vi"):
        stream = hermes.stream(session_id)
        stream.push("Okay.")
        stream.finish()

    await converse(hass, "Hello")
    await conversation.async_converse(
        hass, "Xin chào", None, Context(), agent_id="conversation.hermes_vietnamese"
    )
    await converse(hass, "Again")
    await conversation.async_converse(
        hass, "Nữa", None, Context(), agent_id="conversation.hermes_vietnamese"
    )

    assert hermes.calls[0]["messages"][0]["content"].startswith(DEFAULT_PROMPT)
    assert hermes.calls[1]["messages"][0]["content"].startswith("Trả lời tiếng Việt.")
    assert [call["session_id"] for call in hermes.calls] == [
        None,
        None,
        "session-en",
        "session-vi",
    ]
    device = dr.async_get(hass).async_get_device_by_identifier(
        (DOMAIN, "agent-2"), setup_entry.entry_id
    )
    assert device is not None
    assert device.name == "Hermes Vietnamese"


SERVICES_YAML = (
    Path(__file__).parent.parent
    / "custom_components"
    / "hermes_assist"
    / "services.yaml"
)


async def ask(hass: HomeAssistant, **data: Any) -> dict[str, Any]:
    """Call hermes_assist.ask and return its response."""
    response = await hass.services.async_call(
        DOMAIN, SERVICE_ASK, data, blocking=True, return_response=True
    )
    assert response is not None
    return dict(response)


async def test_ask_returns_the_text_and_stores_the_session(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    stream = hermes.stream("session-9")
    stream.push("  Three meetings today.")
    stream.finish()

    response = await ask(hass, **{ATTR_TEXT: "What is on my calendar?"})

    assert response == {"text": "Three meetings today.", "session_id": "session-9"}
    assert hermes.calls[0]["messages"] == [
        {"role": "system", "content": DEFAULT_PROMPT},
        {"role": "user", "content": "What is on my calendar?"},
    ]

    voice = hermes.stream("session-9")
    voice.push("Two tomorrow.")
    voice.finish()
    await converse(hass, "And tomorrow?")
    assert hermes.calls[1]["session_id"] == "session-9"


async def test_ask_with_a_session_id_leaves_the_stored_session_alone(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    for session_id in ("session-1", "session-x", "session-1"):
        stream = hermes.stream(session_id)
        stream.push("Okay.")
        stream.finish()

    await converse(hass, "Hello")
    response = await ask(
        hass, **{ATTR_TEXT: "Status report", ATTR_SESSION_ID: "manual-session"}
    )
    await converse(hass, "Hello again")

    assert response["session_id"] == "session-x"
    assert [call["session_id"] for call in hermes.calls] == [
        None,
        "manual-session",
        "session-1",
    ]


async def test_ask_sends_the_rendered_agent_prompt_and_exposed_entities(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(
        hass,
        setup_entry,
        {CONF_PROMPT: "Count: {{ 2 + 3 }}.", CONF_LIST_EXPOSED_ENTITIES: True},
    )
    await expose_fixtures(hass, setup_entry)
    stream = hermes.stream()
    stream.push("Okay.")
    stream.finish()

    await ask(hass, **{ATTR_TEXT: "Hello"})

    system = hermes.calls[0]["messages"][0]["content"]
    assert system.startswith("Count: 5.\n\n")
    assert "- light.f1_living_room_light: Light" in system
    assert "voice device" not in system
    assert "Reply in the language" not in system


async def test_ask_system_prompt_replaces_the_agent_prompt(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(hass, setup_entry, {CONF_LIST_EXPOSED_ENTITIES: True})
    await expose_fixtures(hass, setup_entry)
    stream = hermes.stream()
    stream.push("Okay.")
    stream.finish()

    await ask(
        hass,
        **{
            ATTR_TEXT: "Hello",
            ATTR_SYSTEM_PROMPT: "Answer in JSON. {{ not rendered }}",
        },
    )

    system = hermes.calls[0]["messages"][0]["content"]
    assert system.startswith("Answer in JSON. {{ not rendered }}\n\n")
    assert DEFAULT_PROMPT not in system
    assert "- light.f1_living_room_light: Light" in system


async def test_ask_with_an_empty_agent_prompt_sends_no_system_message(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(hass, setup_entry, {CONF_PROMPT: ""})
    stream = hermes.stream()
    stream.push("Okay.")
    stream.finish()

    await ask(hass, **{ATTR_TEXT: "Hello"})

    assert hermes.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]


GET_IMAGE = "custom_components.hermes_assist.services.camera.async_get_image"


async def test_ask_sends_a_camera_snapshot(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hass.states.async_set("camera.front_door", "idle")
    stream = hermes.stream()
    stream.push("A parcel.")
    stream.finish()

    with patch(GET_IMAGE, return_value=Image("image/jpeg", b"\xff\xd8jpeg")) as get:
        response = await ask(
            hass,
            **{
                ATTR_TEXT: "What is at the door?",
                ATTR_CAMERA_ENTITY_ID: "camera.front_door",
            },
        )

    get.assert_awaited_once_with(hass, "camera.front_door")
    assert response["text"] == "A parcel."
    user = hermes.calls[0]["messages"][-1]
    assert user["role"] == "user"
    assert user["content"] == [
        {"type": "text", "text": "What is at the door?"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9hqcGVn"},
        },
    ]


async def test_ask_with_a_camera_that_fails(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hass.states.async_set("camera.front_door", "idle")

    with (
        patch(GET_IMAGE, side_effect=HomeAssistantError("Unable to get image")),
        pytest.raises(HomeAssistantError, match=r"camera\.front_door"),
    ):
        await ask(
            hass, **{ATTR_TEXT: "Look", ATTR_CAMERA_ENTITY_ID: "camera.front_door"}
        )

    assert hermes.calls == []


async def test_ask_rejects_a_non_camera_entity(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    with pytest.raises((vol.Invalid, ServiceValidationError)):
        await ask(hass, **{ATTR_TEXT: "Look", ATTR_CAMERA_ENTITY_ID: "light.kitchen"})


async def test_ask_with_a_rejected_key_starts_reauth(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hermes.error = HermesAuthError("Hermes rejected the API key")

    with (
        patch.object(ConfigEntry, "async_start_reauth") as start_reauth,
        pytest.raises(HomeAssistantError, match="rejected the API key"),
    ):
        await ask(hass, **{ATTR_TEXT: "Hello"})

    start_reauth.assert_called_once()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (HermesConnectionError("Connection refused"), ERROR_CANNOT_CONNECT),
        (HermesTimeoutError("Hermes took too long"), ERROR_TOO_LONG),
        (HermesResponseError("Hermes returned HTTP 500"), ERROR_GENERIC),
    ],
    ids=["connection", "timeout", "server"],
)
async def test_ask_errors(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    error: Exception,
    message: str,
) -> None:
    hermes.error = error

    with pytest.raises(HomeAssistantError) as err:
        await ask(hass, **{ATTR_TEXT: "Hello"})

    assert str(err.value) == message


async def test_ask_with_an_unknown_agent(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    with pytest.raises(ServiceValidationError, match=r"conversation\.missing"):
        await ask(hass, **{ATTR_TEXT: "Hello", ATTR_AGENT_ID: "conversation.missing"})


async def test_ask_needs_an_agent_when_several_exist(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hass.config_entries.async_add_subentry(
        setup_entry,
        ConfigSubentry(
            data=MappingProxyType(dict(DEFAULT_OPTIONS)),
            subentry_id="agent-2",
            subentry_type=SUBENTRY_TYPE_CONVERSATION,
            title="Hermes Vietnamese",
            unique_id=None,
        ),
    )
    await hass.async_block_till_done()
    stream = hermes.stream("session-vi")
    stream.push("Được.")
    stream.finish()

    with pytest.raises(ServiceValidationError, match="agent_id"):
        await ask(hass, **{ATTR_TEXT: "Hello"})

    response = await ask(
        hass, **{ATTR_TEXT: "Xin chào", ATTR_AGENT_ID: "conversation.hermes_vietnamese"}
    )

    assert response["session_id"] == "session-vi"
    assert setup_entry.runtime_data.stores["agent-2"].get(None) == "session-vi"
    assert setup_entry.runtime_data.stores[SUBENTRY_ID].get(None) is None


async def test_ask_with_an_unloaded_agent(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    assert await hass.config_entries.async_unload(setup_entry.entry_id)

    with pytest.raises(ServiceValidationError, match="not loaded"):
        await ask(hass, **{ATTR_TEXT: "Hello", ATTR_AGENT_ID: ENTITY_ID})
    with pytest.raises(ServiceValidationError, match="No Hermes"):
        await ask(hass, **{ATTR_TEXT: "Hello"})


async def test_ask_rejects_an_unsafe_session_id(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    with pytest.raises((vol.Invalid, ServiceValidationError)):
        await ask(hass, **{ATTR_TEXT: "Hello", ATTR_SESSION_ID: "bad\nid"})


def test_services_yaml_lists_the_ask_fields() -> None:
    services = yaml.safe_load(SERVICES_YAML.read_text())

    assert set(services[SERVICE_ASK]["fields"]) == {
        ATTR_TEXT,
        ATTR_SESSION_ID,
        ATTR_AGENT_ID,
        ATTR_SYSTEM_PROMPT,
        ATTR_CAMERA_ENTITY_ID,
    }
    assert services[SERVICE_ASK]["fields"][ATTR_TEXT]["required"] is True
