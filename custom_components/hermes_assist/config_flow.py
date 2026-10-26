"""Config flow for Hermes Assist and the subentry flow of its agents."""

from collections.abc import Mapping
import logging
from typing import Any

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import section
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
import voluptuous as vol
from yarl import URL

from .api import HermesAuthError, HermesClient, HermesError, normalize_url
from .const import (
    CONF_API_KEY,
    CONF_EXPOSED_ENTITIES,
    CONF_HANDOFF_AFTER,
    CONF_MESSAGES,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_RESET_PHRASES,
    CONF_SESSION_IDLE_TIMEOUT,
    CONF_SESSION_KEY,
    CONF_SESSION_MODE,
    CONF_TIMEOUT,
    CONF_URL,
    DEFAULT_MESSAGES,
    DEFAULT_OPTIONS,
    DOMAIN,
    MAX_HANDOFF_AFTER,
    MAX_SESSION_IDLE_TIMEOUT,
    MAX_TIMEOUT,
    MIN_HANDOFF_AFTER,
    MIN_SESSION_IDLE_TIMEOUT,
    MIN_TIMEOUT,
    NAME,
    SESSION_MODES,
    SUBENTRY_TYPE_CONVERSATION,
    TIMEOUT_STEP,
)
from .session import async_clear_sessions

_LOGGER = logging.getLogger(__name__)

_URL_SELECTOR = TextSelector(TextSelectorConfig(type=TextSelectorType.URL))
_KEY_SELECTOR = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))

STEP_USER_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL): _URL_SELECTOR,
        vol.Required(CONF_API_KEY): _KEY_SELECTOR,
    }
)
STEP_RECONFIGURE_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL): _URL_SELECTOR,
        vol.Optional(CONF_API_KEY): _KEY_SELECTOR,
    }
)
STEP_REAUTH_SCHEMA = vol.Schema({vol.Required(CONF_API_KEY): _KEY_SELECTOR})


def _seconds(minimum: int, maximum: int, step: int = 1) -> NumberSelector:
    """Return a number box in seconds."""
    return NumberSelector(
        NumberSelectorConfig(
            min=minimum,
            max=maximum,
            step=step,
            unit_of_measurement="s",
            mode=NumberSelectorMode.BOX,
        )
    )


AGENT_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME): TextSelector(),
        vol.Optional(CONF_PROMPT): TextSelector(TextSelectorConfig(multiline=True)),
        vol.Optional(CONF_MODEL): TextSelector(),
        vol.Optional(CONF_SESSION_KEY): TextSelector(),
        vol.Required(CONF_SESSION_MODE): SelectSelector(
            SelectSelectorConfig(
                options=SESSION_MODES,
                mode=SelectSelectorMode.DROPDOWN,
                translation_key=CONF_SESSION_MODE,
            )
        ),
        vol.Required(CONF_SESSION_IDLE_TIMEOUT): _seconds(
            MIN_SESSION_IDLE_TIMEOUT, MAX_SESSION_IDLE_TIMEOUT
        ),
        vol.Optional(CONF_RESET_PHRASES): TextSelector(),
        vol.Required(CONF_HANDOFF_AFTER): _seconds(
            MIN_HANDOFF_AFTER, MAX_HANDOFF_AFTER
        ),
        vol.Required(CONF_TIMEOUT): _seconds(MIN_TIMEOUT, MAX_TIMEOUT, TIMEOUT_STEP),
        vol.Optional(CONF_EXPOSED_ENTITIES): BooleanSelector(),
        vol.Optional(CONF_MESSAGES): section(
            vol.Schema({vol.Optional(key): TextSelector() for key in DEFAULT_MESSAGES}),
            {"collapsed": True},
        ),
    }
)


def _is_valid_url(url: str) -> bool:
    """Return whether a normalized URL is an http or https URL with a host."""
    return url.startswith(("http://", "https://")) and bool(URL(url).host)


async def _async_validate(
    hass: HomeAssistant, url: str, api_key: str
) -> dict[str, str]:
    """Call GET /v1/models and return the form errors."""
    client = HermesClient(async_get_clientsession(hass), url, api_key)
    try:
        await client.async_validate()
    except HermesAuthError:
        return {"base": "invalid_auth"}
    except HermesError:
        return {"base": "cannot_connect"}
    except Exception:
        _LOGGER.exception("Unexpected error while validating Hermes")
        return {"base": "unknown"}
    return {}


def _options_from_input(user_input: dict[str, Any]) -> dict[str, Any]:
    """Return the agent options to store; cleared text fields are stored as empty."""
    return {
        CONF_PROMPT: user_input.get(CONF_PROMPT, "").strip(),
        CONF_MODEL: user_input.get(CONF_MODEL, "").strip(),
        CONF_SESSION_KEY: user_input.get(CONF_SESSION_KEY, "").strip(),
        CONF_SESSION_MODE: user_input[CONF_SESSION_MODE],
        CONF_SESSION_IDLE_TIMEOUT: int(user_input[CONF_SESSION_IDLE_TIMEOUT]),
        CONF_RESET_PHRASES: user_input.get(CONF_RESET_PHRASES, "").strip(),
        CONF_HANDOFF_AFTER: int(user_input[CONF_HANDOFF_AFTER]),
        CONF_TIMEOUT: int(user_input[CONF_TIMEOUT]),
        CONF_EXPOSED_ENTITIES: bool(user_input.get(CONF_EXPOSED_ENTITIES, False)),
        CONF_MESSAGES: {
            key: user_input.get(CONF_MESSAGES, {}).get(key, "").strip()
            for key in DEFAULT_MESSAGES
        },
    }


class HermesAssistConfigFlow(ConfigFlow, domain=DOMAIN):
    """Set up a Hermes Agent API server."""

    VERSION = 2
    MINOR_VERSION = 1

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return the subentry flows; each conversation agent is one subentry."""
        return {SUBENTRY_TYPE_CONVERSATION: HermesAgentSubentryFlow}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for the URL and the API key; the entry starts with one agent."""
        errors: dict[str, str] = {}
        if user_input is not None:
            url = normalize_url(user_input[CONF_URL])
            if not _is_valid_url(url):
                errors[CONF_URL] = "invalid_url"
            else:
                self._async_abort_entries_match({CONF_URL: url})
                errors = await _async_validate(self.hass, url, user_input[CONF_API_KEY])
                if not errors:
                    return self.async_create_entry(
                        title=NAME,
                        data={CONF_URL: url, CONF_API_KEY: user_input[CONF_API_KEY]},
                        subentries=[
                            {
                                "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                                "title": NAME,
                                "unique_id": None,
                                "data": dict(DEFAULT_OPTIONS),
                            }
                        ],
                    )
        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_SCHEMA, user_input
            ),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the URL, and the key when one is entered."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            url = normalize_url(user_input[CONF_URL])
            api_key = user_input.get(CONF_API_KEY) or entry.data[CONF_API_KEY]
            if not _is_valid_url(url):
                errors[CONF_URL] = "invalid_url"
            else:
                if url != entry.data[CONF_URL]:
                    self._async_abort_entries_match({CONF_URL: url})
                errors = await _async_validate(self.hass, url, api_key)
                if not errors:
                    return self.async_update_reload_and_abort(
                        entry, data_updates={CONF_URL: url, CONF_API_KEY: api_key}
                    )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                STEP_RECONFIGURE_SCHEMA, {CONF_URL: entry.data[CONF_URL]}
            ),
            errors=errors,
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Start reauthentication after Hermes rejected the key."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for a new API key."""
        entry = self._get_reauth_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = await _async_validate(
                self.hass, entry.data[CONF_URL], user_input[CONF_API_KEY]
            )
            if not errors:
                return self.async_update_reload_and_abort(
                    entry, data_updates={CONF_API_KEY: user_input[CONF_API_KEY]}
                )
        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=STEP_REAUTH_SCHEMA,
            errors=errors,
            description_placeholders={"url": entry.data[CONF_URL]},
        )


class HermesAgentSubentryFlow(ConfigSubentryFlow):
    """Add or reconfigure one conversation agent."""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Create an agent with its name and options."""
        if user_input is not None:
            return self.async_create_entry(
                title=user_input[CONF_NAME].strip() or NAME,
                data=_options_from_input(user_input),
            )
        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                AGENT_SCHEMA, {CONF_NAME: NAME, **DEFAULT_OPTIONS}
            ),
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Change the name and options; the entry reloads through its listener."""
        entry = self._get_entry()
        subentry = self._get_reconfigure_subentry()
        current = {**DEFAULT_OPTIONS, **subentry.data}
        if user_input is not None:
            options = _options_from_input(user_input)
            if (
                options[CONF_SESSION_KEY] != current[CONF_SESSION_KEY]
                or options[CONF_SESSION_MODE] != current[CONF_SESSION_MODE]
            ):
                # Never continue a session under a different memory scope.
                await async_clear_sessions(self.hass, entry, subentry.subentry_id)
            return self.async_update_and_abort(
                entry,
                subentry,
                title=user_input[CONF_NAME].strip() or subentry.title,
                data=options,
            )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                AGENT_SCHEMA, {CONF_NAME: subentry.title, **current}
            ),
        )
