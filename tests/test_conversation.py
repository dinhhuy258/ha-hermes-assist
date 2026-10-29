"""Tests for the Hermes Assist conversation agent."""

import asyncio
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import patch

from homeassistant.components import conversation
from homeassistant.components.assist_satellite import AssistSatelliteEntityFeature
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.config_entries import ConfigEntry, ConfigEntryState, ConfigSubentry
from homeassistant.core import Context, HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
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

from custom_components.hermes_assist.api import (
    HermesAuthError,
    HermesConnectionError,
    HermesResponseError,
    HermesStreamError,
    HermesTimeoutError,
    ToolStarted,
)
from custom_components.hermes_assist.const import (
    CONF_HANDOFF_AFTER,
    CONF_LIST_EXPOSED_ENTITIES,
    CONF_MESSAGE_GENERIC,
    CONF_MESSAGE_HANDOFF,
    CONF_MESSAGES,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_SESSION_MODE,
    CONF_SPEAK_EVERY_TOOL,
    CONF_TIMEOUT,
    DEFAULT_OPTIONS,
    DOMAIN,
    NAME,
    SESSION_MODE_PER_CONVERSATION,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.hermes_assist.messages import (
    ERROR_GENERIC,
    HANDOFF_MESSAGE,
)
from custom_components.hermes_assist.prompts import (
    DEFAULT_PROMPT,
    EXPOSED_ENTITIES_INTRO,
)

from .conftest import ENTITY_ID, FakeHermes, set_options

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
    stream = hermes.stream()
    stream.push("Okay.")
    stream.finish()

    await converse(hass, "Hello")

    assert hermes.calls[0]["reasoning_effort"] == "low"


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


@pytest.fixture
def assistant_log(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str | None, list[str]]]:
    """Record each streamed assistant message: its text and its tool names."""
    added: list[tuple[str | None, list[str]]] = []
    add = conversation.ChatLog.async_add_assistant_content

    def _add(self: conversation.ChatLog, content: Any, /, **kwargs: Any) -> Any:
        if isinstance(content, conversation.AssistantContent):
            tools = [tool.tool_name for tool in content.tool_calls or []]
            added.append((content.content, tools))
        return add(self, content, **kwargs)

    monkeypatch.setattr(conversation.ChatLog, "async_add_assistant_content", _add)
    return added


async def test_tool_without_text_is_silent(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    assistant_log: list[tuple[str | None, list[str]]],
) -> None:
    stream = hermes.stream()
    stream.push(ToolStarted("calendar"), "You have a dentist appointment.")
    stream.finish()

    result = await converse(hass, "What's on tomorrow?")

    assert speech(result) == "You have a dentist appointment."
    assert assistant_log == [("You have a dentist appointment.", [])]


async def test_text_before_the_first_tool_is_said(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    assistant_log: list[tuple[str | None, list[str]]],
) -> None:
    stream = hermes.stream()
    stream.push(
        ToolStarted("calendar", "**Let me** check your calendar."),
        ToolStarted("weather", "Now the weather."),
        "**Tomorrow** you have a dentist appointment.",
    )
    stream.finish()

    result = await converse(hass, "What's on tomorrow?")

    assert speech(result) == "Tomorrow you have a dentist appointment."
    assert assistant_log == [
        ("Let me check your calendar.\n\n", ["calendar", "weather"]),
        ("Tomorrow you have a dentist appointment.", []),
    ]


async def test_text_before_every_tool_is_said_when_asked(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    assistant_log: list[tuple[str | None, list[str]]],
) -> None:
    """With the option on, a repeated sentence and tool talk are still skipped."""
    await set_options(hass, setup_entry, {CONF_SPEAK_EVERY_TOOL: True})
    stream = hermes.stream()
    stream.push(
        ToolStarted("search", "Looking for song A."),
        ToolStarted("search", "Not found, trying another source."),
        ToolStarted("search", "Not found, trying another source."),
        ToolStarted("play", "Calling play_media."),
        ToolStarted("play"),
        "Playing song A.",
    )
    stream.finish()

    result = await converse(hass, "Play song A")

    assert speech(result) == "Playing song A."
    assert assistant_log == [
        (
            "Looking for song A.\n\nNot found, trying another source.\n\n",
            ["search", "search", "search", "play", "play"],
        ),
        ("Playing song A.", []),
    ]


async def test_silent_first_tool_then_a_sentence_is_said_when_asked(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    assistant_log: list[tuple[str | None, list[str]]],
) -> None:
    await set_options(hass, setup_entry, {CONF_SPEAK_EVERY_TOOL: True})
    stream = hermes.stream()
    stream.push(
        ToolStarted("search"),
        ToolStarted("search", "Trying another source."),
        "Playing song A.",
    )
    stream.finish()

    await converse(hass, "Play song A")

    assert assistant_log == [
        ("Trying another source.\n\n", ["search"]),
        ("Playing song A.", []),
    ]


async def test_handoff_time_counts_from_the_last_spoken_sentence(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    assistant_log: list[tuple[str | None, list[str]]],
) -> None:
    """Hand off after 0.4 s; sentences at 0.2 s and 0.5 s move it to 0.9 s."""
    await set_options(
        hass, setup_entry, {CONF_HANDOFF_AFTER: 0.4, CONF_SPEAK_EVERY_TOOL: True}
    )
    stream = hermes.stream()

    def answer() -> None:
        stream.push("Done.")
        stream.finish()

    hass.loop.call_later(0.2, stream.push, ToolStarted("search", "Looking."))
    hass.loop.call_later(
        0.5, stream.push, ToolStarted("search", "Trying another source.")
    )
    hass.loop.call_later(0.8, answer)
    result = await converse(hass, "Play song A")

    assert speech(result) == "Done."
    assert [text for text, _ in assistant_log] == [
        "Looking.\n\nTrying another source.\n\n",
        "Done.",
    ]


@pytest.mark.parametrize(
    ("text", "said"),
    [
        (f"Okay. {'Checking the calendar. ' * 6}", "Okay.\n\n"),
        ("x" * 121, None),
    ],
    ids=["first sentence", "too long"],
)
async def test_long_text_before_a_tool(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    assistant_log: list[tuple[str | None, list[str]]],
    text: str,
    said: str | None,
) -> None:
    stream = hermes.stream()
    stream.push(ToolStarted("calendar", text), "You are free.")
    stream.finish()

    await converse(hass, "What's on tomorrow?")

    expected = [] if said is None else [(said, ["calendar"])]
    assert assistant_log == [*expected, ("You are free.", [])]


@pytest.mark.parametrize(
    "text",
    [
        "Calling search_music now.",
        "Checking light.kitchen.",
        'Searching with {"query": "A"}.',
        'Searching for "song A".',
        "Searching for “song A”.",
        "Running `search`.",
        "Let me use the tool.",
        "Để tao dùng công cụ tìm.",
        "Calling the function.",
        "Querying the API.",
        "Running a query.",
    ],
    ids=[
        "snake_case",
        "dotted",
        "json",
        "quoted",
        "curly quoted",
        "backticks",
        "tool",
        "cong cu",
        "function",
        "api",
        "query",
    ],
)
async def test_tool_talk_before_a_tool_is_not_said(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    assistant_log: list[tuple[str | None, list[str]]],
    text: str,
) -> None:
    stream = hermes.stream()
    stream.push(ToolStarted("search", text), "Found it.")
    stream.finish()

    await converse(hass, "Find song A")

    assert assistant_log == [("Found it.", [])]


async def test_answer_without_tools_has_no_progress_sentence(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    assistant_log: list[tuple[str | None, list[str]]],
) -> None:
    stream = hermes.stream()
    stream.push("Hello.")
    stream.finish()

    result = await converse(hass, "Hi")

    assert speech(result) == "Hello."
    assert assistant_log == [("Hello.", [])]


async def test_handoff_time_counts_from_the_progress_sentence(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    assistant_log: list[tuple[str | None, list[str]]],
) -> None:
    """Hand off after 0.4 s; the progress sentence at 0.2 s moves it to 0.6 s."""
    await set_options(hass, setup_entry, {CONF_HANDOFF_AFTER: 0.4})
    stream = hermes.stream()

    def answer() -> None:
        stream.push("Done.")
        stream.finish()

    hass.loop.call_later(
        0.2, stream.push, ToolStarted("backup", "Starting the backup.")
    )
    hass.loop.call_later(0.5, answer)
    result = await converse(hass, "Run a backup")

    assert speech(result) == "Done."
    assert [text for text, _ in assistant_log] == ["Starting the backup.\n\n", "Done."]


async def test_silent_tool_keeps_the_handoff_time(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
) -> None:
    """Hand off after 0.4 s; a tool at 0.2 s with nothing to say changes nothing."""
    await set_options(hass, setup_entry, {CONF_HANDOFF_AFTER: 0.4})
    stream = hermes.stream()

    def answer() -> None:
        stream.push("Done.")
        stream.finish()

    hass.loop.call_later(0.2, stream.push, ToolStarted("backup"))
    hass.loop.call_later(0.5, answer)
    result = await converse(hass, "Run a backup")

    assert speech(result) == HANDOFF_MESSAGE
    await hass.async_block_till_done(wait_background_tasks=True)


@pytest.mark.parametrize("speak_every_tool", [False, True], ids=["first", "every"])
async def test_handoff_after_the_progress_sentence(
    hass: HomeAssistant,
    handoff_entry: MockConfigEntry,
    hermes: FakeHermes,
    assistant_log: list[tuple[str | None, list[str]]],
    speak_every_tool: bool,
) -> None:
    """Nothing is said after the hand-off; the answer is delivered later."""
    await set_options(hass, handoff_entry, {CONF_SPEAK_EVERY_TOOL: speak_every_tool})
    calls = satellite(hass, ANNOUNCE)
    stream = hermes.stream()
    stream.push(ToolStarted("backup", "Để mình sao lưu."))

    result = await converse(hass, "Sao lưu đi", satellite_id=SATELLITE)
    assert speech(result) == HANDOFF_MESSAGE
    assert [text for text, _ in assistant_log] == [
        "Để mình sao lưu.\n\n",
        HANDOFF_MESSAGE,
    ]

    stream.push(ToolStarted("verify", "Now checking."), "Backup done.")
    stream.finish()
    await hass.async_block_till_done(wait_background_tasks=True)

    assert [text for text, _ in assistant_log] == [
        "Để mình sao lưu.\n\n",
        HANDOFF_MESSAGE,
    ]
    assert [call.data for call in calls["announce"]] == [
        {"entity_id": SATELLITE, "message": "Backup done."}
    ]


@pytest.mark.parametrize(
    ("end", "log"),
    [
        (None, "Hermes returned no answer"),
        (HermesConnectionError("Connection refused"), "Could not reach Hermes"),
    ],
    ids=["no answer", "error"],
)
async def test_failure_after_the_progress_sentence_is_said(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    caplog: pytest.LogCaptureFixture,
    end: Exception | None,
    log: str,
) -> None:
    stream = hermes.stream()
    stream.push(ToolStarted("search", "Let me search."))
    if end is None:
        stream.finish()
    else:
        stream.fail(end)

    result = await converse(hass, "Search the web")

    assert speech(result) == ERROR_GENERIC
    assert log in caplog.text


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
