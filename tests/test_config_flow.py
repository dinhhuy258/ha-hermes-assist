"""Tests for the Hermes Assist config flow, agent subentry flow and entry setup."""

import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import aiohttp
from homeassistant.config_entries import (
    SOURCE_REAUTH,
    SOURCE_USER,
    ConfigEntryState,
    ConfigSubentry,
)
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType, InvalidData
from homeassistant.helpers import device_registry as dr, entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker
import voluptuous as vol

from custom_components.hermes_assist.const import (
    ATTR_AGENT_ID,
    ATTR_CAMERA_ENTITY_ID,
    ATTR_SESSION_ID,
    ATTR_SYSTEM_PROMPT,
    ATTR_TEXT,
    CONF_API_KEY,
    CONF_HANDOFF_AFTER,
    CONF_LIST_EXPOSED_ENTITIES,
    CONF_MESSAGE_HANDOFF,
    CONF_MESSAGES,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_SESSION_IDLE_TIMEOUT,
    CONF_SESSION_KEY,
    CONF_SESSION_MODE,
    CONF_SPEAK_PROGRESS,
    CONF_TIMEOUT,
    CONF_URL,
    DEFAULT_MESSAGES,
    DEFAULT_OPTIONS,
    DOMAIN,
    NAME,
    REASONING_EFFORTS,
    SESSION_MODE_KEEP,
    SESSION_MODES,
    SUBENTRY_TYPE_CONVERSATION,
)
from custom_components.hermes_assist.session import storage_key

from .conftest import (
    API_KEY,
    ENTITY_ID,
    HERMES_URL,
    MODELS_RESPONSE,
    MODELS_URL,
    SUBENTRY_ID,
    subentry_data,
)

NEW_URL = "https://hermes-new.example.com"


def model_options(schema: vol.Schema) -> list[str]:
    """Return the dropdown options of the model field of an agent form."""
    for key, selector in schema.schema.items():
        if key == CONF_MODEL:
            return [
                option if isinstance(option, str) else option["value"]
                for option in selector.config["options"]
            ]
    raise AssertionError("model field missing")


def agent_input(**changes: Any) -> dict[str, Any]:
    """Return a complete agent form submission."""
    data = {CONF_NAME: NAME, **DEFAULT_OPTIONS, **changes}
    if not data[CONF_REASONING_EFFORT]:
        # The frontend leaves a cleared dropdown out of the submission.
        del data[CONF_REASONING_EFFORT]
    return data


async def flush_storage(hass: HomeAssistant) -> None:
    """Run pending delayed saves."""
    async_fire_time_changed(hass, fire_all=True)
    await hass.async_block_till_done()


async def test_user_flow_creates_entry(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(MODELS_URL, json=MODELS_RESPONSE)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: f" {HERMES_URL}/v1/ ", CONF_API_KEY: API_KEY}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == NAME
    assert result["data"] == {CONF_URL: HERMES_URL, CONF_API_KEY: API_KEY}
    assert result["options"] == {}
    assert result["subentries"] == [
        {
            "subentry_type": SUBENTRY_TYPE_CONVERSATION,
            "title": NAME,
            "unique_id": None,
            "data": DEFAULT_OPTIONS,
        }
    ]
    entry = result["result"]
    (subentry,) = entry.subentries.values()
    assert hass.states.get(ENTITY_ID) is not None
    assert entry.runtime_data.stores.keys() == {subentry.subentry_id}


@pytest.mark.parametrize(
    ("mock_kwargs", "error"),
    [
        ({"status": 401}, "invalid_auth"),
        ({"status": 403}, "invalid_auth"),
        ({"status": 500}, "cannot_connect"),
        ({"exc": aiohttp.ClientConnectionError()}, "cannot_connect"),
    ],
    ids=["401", "403", "500", "refused"],
)
async def test_user_flow_errors_then_recovers(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    mock_kwargs: dict[str, Any],
    error: str,
) -> None:
    aioclient_mock.get(MODELS_URL, **mock_kwargs)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: HERMES_URL, CONF_API_KEY: API_KEY}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": error}

    aioclient_mock.clear_requests()
    aioclient_mock.get(MODELS_URL, json=MODELS_RESPONSE)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: HERMES_URL, CONF_API_KEY: API_KEY}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_user_flow_unknown_error(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    with patch(
        "custom_components.hermes_assist.config_flow.HermesClient.async_validate",
        side_effect=RuntimeError("boom"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_URL: HERMES_URL, CONF_API_KEY: API_KEY}
        )

    assert result["errors"] == {"base": "unknown"}


@pytest.mark.parametrize("url", ["hermes.example.com", "ftp://hermes.example.com"])
async def test_user_flow_rejects_invalid_url(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, url: str
) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: url, CONF_API_KEY: API_KEY}
    )

    assert result["errors"] == {CONF_URL: "invalid_url"}
    assert aioclient_mock.call_count == 0


async def test_user_flow_aborts_on_duplicate_url(
    hass: HomeAssistant, mock_entry: MockConfigEntry
) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: f"{HERMES_URL}/v1", CONF_API_KEY: API_KEY}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reconfigure_keeps_the_key_when_left_empty(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.get(f"{NEW_URL}/v1/models", json=MODELS_RESPONSE)
    result = await setup_entry.start_reconfigure_flow(hass)
    assert result["step_id"] == "reconfigure"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: f"{NEW_URL}/"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert setup_entry.data == {CONF_URL: NEW_URL, CONF_API_KEY: API_KEY}
    assert setup_entry.state is ConfigEntryState.LOADED


async def test_reconfigure_replaces_the_key(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    result = await setup_entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: HERMES_URL, CONF_API_KEY: "new-key"}
    )
    await hass.async_block_till_done()

    assert result["reason"] == "reconfigure_successful"
    assert setup_entry.data[CONF_API_KEY] == "new-key"


async def test_reconfigure_with_a_wrong_key_shows_an_error(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    result = await setup_entry.start_reconfigure_flow(hass)
    aioclient_mock.clear_requests()
    aioclient_mock.get(MODELS_URL, status=401)

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_URL: HERMES_URL, CONF_API_KEY: "wrong-key"}
    )

    assert result["errors"] == {"base": "invalid_auth"}
    assert setup_entry.data[CONF_API_KEY] == API_KEY


async def test_setup_with_a_rejected_key_starts_reauth(
    hass: HomeAssistant,
    mock_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.get(MODELS_URL, status=401)
    await hass.config_entries.async_setup(mock_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_entry.state is ConfigEntryState.SETUP_ERROR

    (flow,) = mock_entry.async_get_active_flows(hass, {SOURCE_REAUTH})
    assert flow["step_id"] == "reauth_confirm"

    result = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {CONF_API_KEY: "still-wrong"}
    )
    assert result["errors"] == {"base": "invalid_auth"}

    aioclient_mock.clear_requests()
    aioclient_mock.get(MODELS_URL, json=MODELS_RESPONSE)
    result = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {CONF_API_KEY: "new-key"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert mock_entry.data == {CONF_URL: HERMES_URL, CONF_API_KEY: "new-key"}
    assert mock_entry.state is ConfigEntryState.LOADED


async def test_setup_retries_when_unreachable(
    hass: HomeAssistant,
    mock_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.get(MODELS_URL, exc=TimeoutError())

    await hass.config_entries.async_setup(mock_entry.entry_id)

    assert mock_entry.state is ConfigEntryState.SETUP_RETRY


async def test_unload_and_remove_delete_the_sessions(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    key = storage_key(SUBENTRY_ID)
    assert await hass.config_entries.async_unload(setup_entry.entry_id)
    assert setup_entry.state is ConfigEntryState.NOT_LOADED

    hass_storage[key] = {
        "version": 1,
        "minor_version": 1,
        "key": key,
        "data": {"session_id": "session-1", "last_used": 0.0},
    }
    await hass.config_entries.async_remove(setup_entry.entry_id)
    await hass.async_block_till_done()

    assert key not in hass_storage


async def test_removing_an_agent_deletes_its_sessions_and_entity(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
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
    assert hass.states.get("conversation.hermes_vietnamese") is not None
    setup_entry.runtime_data.stores["agent-2"].set(None, "session-2")
    await flush_storage(hass)
    assert storage_key("agent-2") in hass_storage

    hass.config_entries.async_remove_subentry(setup_entry, "agent-2")
    await hass.async_block_till_done()

    assert storage_key("agent-2") not in hass_storage
    assert hass.states.get("conversation.hermes_vietnamese") is None
    assert "agent-2" not in setup_entry.runtime_data.stores
    assert setup_entry.state is ConfigEntryState.LOADED


async def test_migrates_a_version_1_entry_into_an_agent(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    hass_storage: dict[str, Any],
) -> None:
    aioclient_mock.get(MODELS_URL, json=MODELS_RESPONSE)
    options = {
        **DEFAULT_OPTIONS,
        CONF_PROMPT: "Be brief.",
        CONF_SESSION_MODE: SESSION_MODE_KEEP,
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=NAME,
        version=1,
        data={CONF_URL: HERMES_URL, CONF_API_KEY: API_KEY},
        options=options,
    )
    entry.add_to_hass(hass)
    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)
    device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, entry.entry_id)},
        name=NAME,
        manufacturer="Nous Research",
        model="Hermes Agent",
        entry_type=dr.DeviceEntryType.SERVICE,
    )
    entity_registry.async_get_or_create(
        "conversation",
        DOMAIN,
        entry.entry_id,
        config_entry=entry,
        device_id=device.id,
        suggested_object_id="hermes_assist",
    )
    old_key = storage_key(entry.entry_id)
    hass_storage[old_key] = {
        "version": 1,
        "minor_version": 1,
        "key": old_key,
        "data": {"session_id": "session-1", "last_used": 0.0},
    }

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.version == 2
    assert entry.options == {}
    (subentry,) = entry.subentries.values()
    assert subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION
    assert subentry.title == NAME
    assert dict(subentry.data) == options
    registry_entry = entity_registry.async_get(ENTITY_ID)
    assert registry_entry is not None
    assert registry_entry.unique_id == subentry.subentry_id
    assert registry_entry.config_subentry_id == subentry.subentry_id
    assert hass.states.get(ENTITY_ID) is not None
    device = device_registry.async_get(device.id)
    assert device is not None
    assert device.identifiers == {(DOMAIN, subentry.subentry_id)}
    assert device.config_entry_id == entry.entry_id
    assert device.config_subentry_id == subentry.subentry_id
    assert (
        device_registry.async_get_device_by_identifier(
            (DOMAIN, subentry.subentry_id), entry.entry_id
        )
        == device
    )
    assert old_key not in hass_storage
    assert hass_storage[storage_key(subentry.subentry_id)]["data"]["session_id"] == (
        "session-1"
    )
    assert entry.runtime_data.stores[subentry.subentry_id].get(None) == "session-1"


async def test_migrates_a_version_1_entry_that_never_loaded(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(MODELS_URL, json=MODELS_RESPONSE)
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=NAME,
        version=1,
        data={CONF_URL: HERMES_URL, CONF_API_KEY: API_KEY},
        options={},
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    (subentry,) = entry.subentries.values()
    assert dict(subentry.data) == DEFAULT_OPTIONS
    assert hass.states.get(ENTITY_ID) is not None


async def test_migrates_minor_version_1_option_names(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    aioclient_mock.get(MODELS_URL, json=MODELS_RESPONSE)
    old_options = {
        key: value
        for key, value in DEFAULT_OPTIONS.items()
        if key != CONF_LIST_EXPOSED_ENTITIES
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=NAME,
        version=2,
        minor_version=1,
        data={CONF_URL: HERMES_URL, CONF_API_KEY: API_KEY},
        subentries_data=[
            subentry_data(
                **{
                    **old_options,
                    "exposed_entities": True,
                    "reset_phrases": "start over",
                    CONF_MESSAGES: {
                        **old_options[CONF_MESSAGES],
                        "reset_message": "Okay.",
                        "auth_message": "Bad key.",
                        "too_long_message": "Too slow.",
                    },
                }
            ),
            subentry_data("Bare", "agent-2", **old_options),
        ],
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.minor_version == 2
    first = entry.subentries[SUBENTRY_ID].data
    assert first[CONF_LIST_EXPOSED_ENTITIES] is True
    assert "exposed_entities" not in first
    assert "reset_phrases" not in first
    assert first[CONF_MESSAGES] == DEFAULT_MESSAGES
    assert entry.subentries["agent-2"].data[CONF_LIST_EXPOSED_ENTITIES] is False


async def test_a_future_version_is_not_migrated(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=NAME,
        version=3,
        data={CONF_URL: HERMES_URL, CONF_API_KEY: API_KEY},
    )
    entry.add_to_hass(hass)

    assert not await hass.config_entries.async_setup(entry.entry_id)

    assert entry.state is ConfigEntryState.MIGRATION_ERROR


async def start_add_agent(hass: HomeAssistant, entry: MockConfigEntry) -> Any:
    """Open the add-agent subentry form."""
    return await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_CONVERSATION), context={"source": SOURCE_USER}
    )


async def test_add_agent_creates_an_entity_with_its_own_options(
    hass: HomeAssistant, setup_entry: MockConfigEntry, hass_storage: dict[str, Any]
) -> None:
    result = await start_add_agent(hass, setup_entry)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Hermes Vietnamese",
            CONF_PROMPT: "Trả lời tiếng Việt.",
            CONF_MODEL: "hermes-agent",
            CONF_SESSION_KEY: "homeassistant:vi",
            CONF_SESSION_MODE: SESSION_MODE_KEEP,
            CONF_SESSION_IDLE_TIMEOUT: 900,
            CONF_HANDOFF_AFTER: 10,
            CONF_SPEAK_PROGRESS: True,
            CONF_TIMEOUT: 300,
            CONF_LIST_EXPOSED_ENTITIES: True,
            CONF_REASONING_EFFORT: "high",
            CONF_MESSAGES: {CONF_MESSAGE_HANDOFF: " Đang làm, xong sẽ báo lại. "},
        },
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Hermes Vietnamese"
    assert result["data"] == {
        CONF_PROMPT: "Trả lời tiếng Việt.",
        CONF_MODEL: "hermes-agent",
        CONF_SESSION_KEY: "homeassistant:vi",
        CONF_SESSION_MODE: SESSION_MODE_KEEP,
        CONF_SESSION_IDLE_TIMEOUT: 900,
        CONF_HANDOFF_AFTER: 10,
        CONF_SPEAK_PROGRESS: True,
        CONF_TIMEOUT: 300,
        CONF_LIST_EXPOSED_ENTITIES: True,
        CONF_REASONING_EFFORT: "high",
        CONF_MESSAGES: {
            **dict.fromkeys(DEFAULT_MESSAGES, ""),
            CONF_MESSAGE_HANDOFF: "Đang làm, xong sẽ báo lại.",
        },
    }
    assert isinstance(result["data"][CONF_TIMEOUT], int)
    assert len(setup_entry.subentries) == 2
    assert setup_entry.state is ConfigEntryState.LOADED
    assert hass.states.get("conversation.hermes_vietnamese") is not None
    assert hass.states.get(ENTITY_ID) is not None
    assert set(setup_entry.runtime_data.stores) == set(setup_entry.subentries)


async def test_add_agent_stores_cleared_text_fields_as_empty(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    result = await start_add_agent(hass, setup_entry)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Bare",
            CONF_SESSION_MODE: DEFAULT_OPTIONS[CONF_SESSION_MODE],
            CONF_SESSION_IDLE_TIMEOUT: 1800,
            CONF_HANDOFF_AFTER: 15,
            CONF_TIMEOUT: 600,
        },
    )
    await hass.async_block_till_done()

    assert result["data"][CONF_PROMPT] == ""
    assert result["data"][CONF_MODEL] == ""
    assert result["data"][CONF_SESSION_KEY] == ""
    assert result["data"][CONF_MESSAGES] == dict.fromkeys(DEFAULT_MESSAGES, "")
    assert result["data"][CONF_LIST_EXPOSED_ENTITIES] is False
    assert result["data"][CONF_REASONING_EFFORT] == ""
    assert result["data"][CONF_SPEAK_PROGRESS] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (CONF_SESSION_IDLE_TIMEOUT, 59),
        (CONF_SESSION_IDLE_TIMEOUT, 86401),
        (CONF_HANDOFF_AFTER, -1),
        (CONF_HANDOFF_AFTER, 30),
        (CONF_TIMEOUT, 4),
        (CONF_TIMEOUT, 1801),
        (CONF_SESSION_MODE, "forever"),
        (CONF_REASONING_EFFORT, "extreme"),
    ],
)
async def test_add_agent_rejects_out_of_range_values(
    hass: HomeAssistant, setup_entry: MockConfigEntry, field: str, value: Any
) -> None:
    result = await start_add_agent(hass, setup_entry)

    with pytest.raises(InvalidData):
        await hass.config_entries.subentries.async_configure(
            result["flow_id"], agent_input(**{field: value})
        )


async def test_add_agent_rejects_a_broken_template(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    result = await start_add_agent(hass, setup_entry)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], agent_input(**{CONF_PROMPT: "Hello {{ states("})
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_PROMPT: "invalid_template"}
    assert len(setup_entry.subentries) == 1


async def test_reconfigure_agent_rejects_a_broken_template(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    result = await setup_entry.start_subentry_reconfigure_flow(hass, SUBENTRY_ID)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], agent_input(**{CONF_PROMPT: "{% if %}"})
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_PROMPT: "invalid_template"}
    assert (
        setup_entry.subentries[SUBENTRY_ID].data[CONF_PROMPT]
        == (DEFAULT_OPTIONS[CONF_PROMPT])
    )


async def test_agent_form_offers_the_models_of_the_server(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.clear_requests()
    aioclient_mock.get(
        MODELS_URL,
        json={"object": "list", "data": [{"id": "hermes-agent"}, {"id": "gpt-5"}]},
    )

    result = await start_add_agent(hass, setup_entry)

    assert model_options(result["data_schema"]) == ["hermes-agent", "gpt-5"]

    result = await setup_entry.start_subentry_reconfigure_flow(hass, SUBENTRY_ID)
    assert model_options(result["data_schema"]) == ["hermes-agent", "gpt-5"]


async def test_agent_form_accepts_a_model_that_is_not_listed(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    result = await start_add_agent(hass, setup_entry)

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], agent_input(**{CONF_MODEL: "my-custom-model"})
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_MODEL] == "my-custom-model"


async def test_agent_form_opens_when_the_model_list_fails(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    aioclient_mock.clear_requests()
    aioclient_mock.get(MODELS_URL, status=500)

    result = await start_add_agent(hass, setup_entry)

    assert result["type"] is FlowResultType.FORM
    assert model_options(result["data_schema"]) == []


async def test_agent_form_of_an_unloaded_entry_fetches_the_models(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    assert await hass.config_entries.async_unload(setup_entry.entry_id)
    aioclient_mock.clear_requests()
    aioclient_mock.get(MODELS_URL, json={"data": [{"id": "only-model"}]})

    result = await setup_entry.start_subentry_reconfigure_flow(hass, SUBENTRY_ID)

    assert model_options(result["data_schema"]) == ["only-model"]


async def test_reconfigure_agent_saves_all_fields(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    result = await setup_entry.start_subentry_reconfigure_flow(hass, SUBENTRY_ID)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        agent_input(
            **{
                CONF_NAME: "Hermes English",
                CONF_PROMPT: "Be brief.",
                CONF_HANDOFF_AFTER: 10,
            }
        ),
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    subentry = setup_entry.subentries[SUBENTRY_ID]
    assert subentry.title == "Hermes English"
    assert dict(subentry.data) == {
        **DEFAULT_OPTIONS,
        CONF_PROMPT: "Be brief.",
        CONF_HANDOFF_AFTER: 10,
    }
    assert setup_entry.state is ConfigEntryState.LOADED
    assert hass.states.get(ENTITY_ID) is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (CONF_SESSION_KEY, "homeassistant:other"),
        (CONF_SESSION_MODE, SESSION_MODE_KEEP),
    ],
)
async def test_changing_the_session_scope_clears_sessions(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
    field: str,
    value: str,
) -> None:
    key = storage_key(SUBENTRY_ID)
    setup_entry.runtime_data.stores[SUBENTRY_ID].set(None, "session-1")
    await flush_storage(hass)
    assert hass_storage[key]["data"]["session_id"] == "session-1"

    result = await setup_entry.start_subentry_reconfigure_flow(hass, SUBENTRY_ID)
    await hass.config_entries.subentries.async_configure(
        result["flow_id"], agent_input(**{field: value})
    )
    await hass.async_block_till_done()

    assert key not in hass_storage
    assert setup_entry.runtime_data.stores[SUBENTRY_ID].get(None) is None


async def test_other_option_changes_keep_sessions(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    key = storage_key(SUBENTRY_ID)
    setup_entry.runtime_data.stores[SUBENTRY_ID].set(None, "session-1")
    await flush_storage(hass)

    result = await setup_entry.start_subentry_reconfigure_flow(hass, SUBENTRY_ID)
    await hass.config_entries.subentries.async_configure(
        result["flow_id"], agent_input(**{CONF_HANDOFF_AFTER: 5})
    )
    await hass.async_block_till_done()

    assert hass_storage[key]["data"]["session_id"] == "session-1"
    assert setup_entry.runtime_data.stores[SUBENTRY_ID].get(None) == "session-1"


async def test_changing_the_session_scope_of_an_unloaded_agent_clears_sessions(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    key = storage_key(SUBENTRY_ID)
    setup_entry.runtime_data.stores[SUBENTRY_ID].set(None, "session-1")
    await flush_storage(hass)
    assert await hass.config_entries.async_unload(setup_entry.entry_id)

    result = await setup_entry.start_subentry_reconfigure_flow(hass, SUBENTRY_ID)
    await hass.config_entries.subentries.async_configure(
        result["flow_id"], agent_input(**{CONF_SESSION_MODE: SESSION_MODE_KEEP})
    )
    await hass.async_block_till_done()

    assert key not in hass_storage


COMPONENT = Path(__file__).parent.parent / "custom_components" / DOMAIN


def _texts(node: Any) -> list[str]:
    """Return every string in a translation tree."""
    if isinstance(node, dict):
        return [text for value in node.values() for text in _texts(value)]
    return [node] if isinstance(node, str) else []


def test_translation_placeholders_are_identifiers() -> None:
    """Braces are placeholders in Home Assistant strings, as hassfest checks."""
    strings = json.loads((COMPONENT / "strings.json").read_text())

    for text in _texts(strings):
        for placeholder in re.findall(r"\{([^{}]*)\}", text):
            assert re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", placeholder), text


def test_translations_cover_every_form_field() -> None:
    strings = json.loads((COMPONENT / "strings.json").read_text())
    english = json.loads((COMPONENT / "translations" / "en.json").read_text())

    assert english == strings
    steps = strings["config"]["step"]
    assert set(steps["user"]["data"]) == {CONF_URL, CONF_API_KEY}
    assert set(steps["reconfigure"]["data"]) == {CONF_URL, CONF_API_KEY}
    assert set(steps["reauth_confirm"]["data"]) == {CONF_API_KEY}
    assert "{url}" in steps["reauth_confirm"]["description"]
    assert set(strings["config"]["error"]) == {
        "cannot_connect",
        "invalid_auth",
        "invalid_url",
        "unknown",
    }
    assert set(strings["config"]["abort"]) == {
        "already_configured",
        "reauth_successful",
        "reconfigure_successful",
    }
    assert "options" not in strings
    agent = strings["config_subentries"][SUBENTRY_TYPE_CONVERSATION]
    assert set(agent["initiate_flow"]) == {"user", "reconfigure"}
    fields = {CONF_NAME, *DEFAULT_OPTIONS} - {CONF_MESSAGES}
    for step in ("user", "reconfigure"):
        assert set(agent["step"][step]["data"]) == fields
        assert set(agent["step"][step]["data_description"]) == fields
        messages = agent["step"][step]["sections"][CONF_MESSAGES]
        assert set(messages["data"]) == set(DEFAULT_MESSAGES)
        assert set(messages["data_description"]) == set(DEFAULT_MESSAGES)
    assert set(agent["abort"]) == {"reconfigure_successful"}
    assert set(agent["error"]) == {"invalid_template"}
    assert set(strings["selector"]["session_mode"]["options"]) == set(SESSION_MODES)
    assert set(strings["selector"]["reasoning_effort"]["options"]) == set(
        REASONING_EFFORTS
    )
    assert set(strings["services"]["ask"]["fields"]) == {
        ATTR_TEXT,
        ATTR_SESSION_ID,
        ATTR_AGENT_ID,
        ATTR_SYSTEM_PROMPT,
        ATTR_CAMERA_ENTITY_ID,
    }
