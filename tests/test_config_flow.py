"""Tests for the Hermes Assist config flow, options flow and entry setup."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import aiohttp
from homeassistant.config_entries import SOURCE_REAUTH, SOURCE_USER, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType, InvalidData
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.hermes_assist.const import (
    ATTR_CONFIG_ENTRY_ID,
    ATTR_SESSION_ID,
    ATTR_TEXT,
    CONF_API_KEY,
    CONF_HANDOFF_AFTER,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_RESET_PHRASES,
    CONF_SESSION_IDLE_TIMEOUT,
    CONF_SESSION_KEY,
    CONF_SESSION_MODE,
    CONF_TIMEOUT,
    CONF_URL,
    DEFAULT_OPTIONS,
    DOMAIN,
    NAME,
    SESSION_MODE_KEEP,
    SESSION_MODES,
)
from custom_components.hermes_assist.session import storage_key

from .conftest import API_KEY, HERMES_URL, MODELS_RESPONSE, MODELS_URL

NEW_URL = "https://hermes-new.example.com"


def options_input(**changes: Any) -> dict[str, Any]:
    """Return a complete options form submission."""
    return {**DEFAULT_OPTIONS, **changes}


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
    assert result["options"] == DEFAULT_OPTIONS


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
    key = storage_key(setup_entry.entry_id)
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


async def test_options_flow_saves_all_fields(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    result = await hass.config_entries.options.async_init(setup_entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_PROMPT: "Be brief.",
            CONF_MODEL: "hermes-agent",
            CONF_SESSION_KEY: "homeassistant:kitchen",
            CONF_SESSION_MODE: SESSION_MODE_KEEP,
            CONF_SESSION_IDLE_TIMEOUT: 900,
            CONF_RESET_PHRASES: "forget it",
            CONF_HANDOFF_AFTER: 10,
            CONF_TIMEOUT: 300,
        },
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert setup_entry.options == {
        CONF_PROMPT: "Be brief.",
        CONF_MODEL: "hermes-agent",
        CONF_SESSION_KEY: "homeassistant:kitchen",
        CONF_SESSION_MODE: SESSION_MODE_KEEP,
        CONF_SESSION_IDLE_TIMEOUT: 900,
        CONF_RESET_PHRASES: "forget it",
        CONF_HANDOFF_AFTER: 10,
        CONF_TIMEOUT: 300,
    }
    assert isinstance(setup_entry.options[CONF_TIMEOUT], int)
    assert setup_entry.state is ConfigEntryState.LOADED


async def test_options_flow_stores_cleared_text_fields_as_empty(
    hass: HomeAssistant, setup_entry: MockConfigEntry
) -> None:
    result = await hass.config_entries.options.async_init(setup_entry.entry_id)

    await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_SESSION_MODE: DEFAULT_OPTIONS[CONF_SESSION_MODE],
            CONF_SESSION_IDLE_TIMEOUT: 1800,
            CONF_HANDOFF_AFTER: 15,
            CONF_TIMEOUT: 600,
        },
    )
    await hass.async_block_till_done()

    assert setup_entry.options[CONF_PROMPT] == ""
    assert setup_entry.options[CONF_MODEL] == ""
    assert setup_entry.options[CONF_SESSION_KEY] == ""
    assert setup_entry.options[CONF_RESET_PHRASES] == ""


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
    ],
)
async def test_options_flow_rejects_out_of_range_values(
    hass: HomeAssistant, setup_entry: MockConfigEntry, field: str, value: Any
) -> None:
    result = await hass.config_entries.options.async_init(setup_entry.entry_id)

    with pytest.raises(InvalidData):
        await hass.config_entries.options.async_configure(
            result["flow_id"], options_input(**{field: value})
        )


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
    key = storage_key(setup_entry.entry_id)
    setup_entry.runtime_data.store.set(None, "session-1")
    await flush_storage(hass)
    assert hass_storage[key]["data"]["session_id"] == "session-1"

    result = await hass.config_entries.options.async_init(setup_entry.entry_id)
    await hass.config_entries.options.async_configure(
        result["flow_id"], options_input(**{field: value})
    )
    await hass.async_block_till_done()

    assert key not in hass_storage
    assert setup_entry.runtime_data.store.get(None) is None


async def test_other_option_changes_keep_sessions(
    hass: HomeAssistant,
    setup_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    key = storage_key(setup_entry.entry_id)
    setup_entry.runtime_data.store.set(None, "session-1")
    await flush_storage(hass)

    result = await hass.config_entries.options.async_init(setup_entry.entry_id)
    await hass.config_entries.options.async_configure(
        result["flow_id"], options_input(**{CONF_HANDOFF_AFTER: 5})
    )
    await hass.async_block_till_done()

    assert hass_storage[key]["data"]["session_id"] == "session-1"
    assert setup_entry.runtime_data.store.get(None) == "session-1"


COMPONENT = Path(__file__).parent.parent / "custom_components" / DOMAIN


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
    assert set(strings["options"]["step"]["init"]["data"]) == set(DEFAULT_OPTIONS)
    assert set(strings["options"]["step"]["init"]["data_description"]) == set(
        DEFAULT_OPTIONS
    )
    assert set(strings["selector"]["session_mode"]["options"]) == set(SESSION_MODES)
    assert set(strings["services"]["ask"]["fields"]) == {
        ATTR_TEXT,
        ATTR_SESSION_ID,
        ATTR_CONFIG_ENTRY_ID,
    }
