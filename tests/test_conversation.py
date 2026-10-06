"""Tests for the Hermes Assist conversation agent and the ask service."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

from homeassistant.components import conversation
from homeassistant.components.assist_satellite import AssistSatelliteEntityFeature
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import Context, HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import area_registry as ar, device_registry as dr, intent
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
)
from custom_components.hermes_assist.const import (
    ATTR_CONFIG_ENTRY_ID,
    ATTR_SESSION_ID,
    ATTR_TEXT,
    CONF_HANDOFF_AFTER,
    CONF_MODEL,
    CONF_SESSION_MODE,
    CONF_TIMEOUT,
    DOMAIN,
    SERVICE_ASK,
    SESSION_MODE_PER_CONVERSATION,
)
from custom_components.hermes_assist.messages import (
    ERROR_AUTH,
    ERROR_CANNOT_CONNECT,
    ERROR_EMPTY,
    ERROR_GENERIC,
    ERROR_TOO_LONG,
    HANDOFF_MESSAGE,
    NOTIFICATION_TITLE,
    RESET_MESSAGE,
)
from custom_components.hermes_assist.prompts import (
    DEFAULT_PROMPT,
)

from .conftest import ENTITY_ID, FakeHermes, set_options

SATELLITE = "assist_satellite.kitchen"
ANNOUNCE = AssistSatelliteEntityFeature.ANNOUNCE
START_CONVERSATION = AssistSatelliteEntityFeature.START_CONVERSATION
NOTIFY = (
    "custom_components.hermes_assist.conversation.persistent_notification.async_create"
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


async def test_reset_phrase_starts_a_new_session(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    for session_id in ("session-1", "session-2"):
        stream = hermes.stream(session_id)
        stream.push("Okay.")
        stream.finish()

    await converse(hass, "Remember the number seven")
    result = await converse(hass, "Start over!")
    await converse(hass, "What number?")

    assert speech(result) == RESET_MESSAGE
    assert len(hermes.calls) == 2
    assert hermes.calls[1]["session_id"] is None


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (HermesConnectionError("Connection refused"), ERROR_CANNOT_CONNECT),
        (HermesResponseError("Hermes returned HTTP 500"), ERROR_GENERIC),
        (HermesAuthError("Hermes rejected the API key"), ERROR_AUTH),
    ],
    ids=["connection", "server", "auth"],
)
async def test_errors_are_spoken(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hermes: FakeHermes,
    error: Exception,
    message: str,
) -> None:
    hermes.error = error

    result = await converse(hass, "Hello")

    assert speech(result) == message
    assert result.response.response_type is intent.IntentResponseType.ERROR


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

    assert speech(result) == ERROR_EMPTY


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

    notify.assert_called_once_with(hass, "All done.", title=NOTIFICATION_TITLE)
    assert calls["announce"] == []


async def test_handoff_without_a_satellite_notifies(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    stream = hermes.stream()

    with patch(NOTIFY) as notify:
        await converse(hass, "Slow question")
        stream.push("All done.")
        stream.finish()
        await hass.async_block_till_done(wait_background_tasks=True)

    notify.assert_called_once_with(hass, "All done.", title=NOTIFICATION_TITLE)


async def test_handoff_that_exceeds_the_timeout(
    hass: HomeAssistant, handoff_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    await set_options(hass, handoff_entry, {CONF_TIMEOUT: 0.1})
    stream = hermes.stream()

    with patch(NOTIFY) as notify:
        result = await converse(hass, "Very slow question")
        await hass.async_block_till_done(wait_background_tasks=True)

    assert speech(result) == HANDOFF_MESSAGE
    notify.assert_called_once_with(hass, ERROR_TOO_LONG, title=NOTIFICATION_TITLE)
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
        {"role": "user", "content": "What is on my calendar?"}
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


async def test_ask_connection_error(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    hermes.error = HermesConnectionError("Connection refused")

    with pytest.raises(HomeAssistantError) as err:
        await ask(hass, **{ATTR_TEXT: "Hello"})

    assert str(err.value) == ERROR_CANNOT_CONNECT


async def test_ask_with_an_unknown_entry(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hermes: FakeHermes
) -> None:
    with pytest.raises(ServiceValidationError):
        await ask(hass, **{ATTR_TEXT: "Hello", ATTR_CONFIG_ENTRY_ID: "missing"})


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
        ATTR_CONFIG_ENTRY_ID,
    }
    assert services[SERVICE_ASK]["fields"][ATTR_TEXT]["required"] is True
