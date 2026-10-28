"""Constants for the Hermes Assist integration."""

from typing import Any, Final

from .messages import (
    ERROR_GENERIC,
    HANDOFF_MESSAGE,
)
from .prompts import DEFAULT_PROMPT

DOMAIN: Final = "hermes_assist"
NAME: Final = "Hermes Assist"

CONF_URL: Final = "url"
CONF_API_KEY: Final = "api_key"
CONF_PROMPT: Final = "prompt"
CONF_MODEL: Final = "model"
CONF_SESSION_KEY: Final = "session_key"
CONF_SESSION_MODE: Final = "session_mode"
CONF_SESSION_IDLE_TIMEOUT: Final = "session_idle_timeout"
CONF_HANDOFF_AFTER: Final = "handoff_after"
CONF_TIMEOUT: Final = "timeout"
CONF_LIST_EXPOSED_ENTITIES: Final = "list_exposed_entities"
CONF_SPEAK_EVERY_TOOL: Final = "speak_every_tool"
CONF_REASONING_EFFORT: Final = "reasoning_effort"
CONF_MESSAGES: Final = "messages"
CONF_MESSAGE_HANDOFF: Final = "handoff_message"
CONF_MESSAGE_GENERIC: Final = "error_message"

SESSION_MODE_KEEP: Final = "keep"
SESSION_MODE_IDLE: Final = "idle"
SESSION_MODE_PER_CONVERSATION: Final = "per_conversation"
SESSION_MODES: Final = [
    SESSION_MODE_KEEP,
    SESSION_MODE_IDLE,
    SESSION_MODE_PER_CONVERSATION,
]

# The values Hermes accepts for model_options.reasoning_effort.
REASONING_EFFORTS: Final = [
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
]

DEFAULT_MODEL: Final = ""
DEFAULT_SESSION_KEY: Final = "homeassistant:assist"
DEFAULT_SESSION_MODE: Final = SESSION_MODE_IDLE
DEFAULT_SESSION_IDLE_TIMEOUT: Final = 1800
DEFAULT_HANDOFF_AFTER: Final = 15
DEFAULT_TIMEOUT: Final = 600
DEFAULT_LIST_EXPOSED_ENTITIES: Final = False
DEFAULT_SPEAK_EVERY_TOOL: Final = False
# Empty means the Hermes default.
DEFAULT_REASONING_EFFORT: Final = ""
# An empty or missing message falls back to its default.
DEFAULT_MESSAGES: Final[dict[str, str]] = {
    CONF_MESSAGE_HANDOFF: HANDOFF_MESSAGE,
    CONF_MESSAGE_GENERIC: ERROR_GENERIC,
}

MIN_SESSION_IDLE_TIMEOUT: Final = 60
MAX_SESSION_IDLE_TIMEOUT: Final = 86400
MIN_HANDOFF_AFTER: Final = 0
MAX_HANDOFF_AFTER: Final = 29
MIN_TIMEOUT: Final = 5
MAX_TIMEOUT: Final = 1800
TIMEOUT_STEP: Final = 5

DEFAULT_OPTIONS: Final[dict[str, Any]] = {
    CONF_PROMPT: DEFAULT_PROMPT,
    CONF_MODEL: DEFAULT_MODEL,
    CONF_SESSION_KEY: DEFAULT_SESSION_KEY,
    CONF_SESSION_MODE: DEFAULT_SESSION_MODE,
    CONF_SESSION_IDLE_TIMEOUT: DEFAULT_SESSION_IDLE_TIMEOUT,
    CONF_HANDOFF_AFTER: DEFAULT_HANDOFF_AFTER,
    CONF_TIMEOUT: DEFAULT_TIMEOUT,
    CONF_LIST_EXPOSED_ENTITIES: DEFAULT_LIST_EXPOSED_ENTITIES,
    CONF_SPEAK_EVERY_TOOL: DEFAULT_SPEAK_EVERY_TOOL,
    CONF_REASONING_EFFORT: DEFAULT_REASONING_EFFORT,
    CONF_MESSAGES: DEFAULT_MESSAGES,
}

STORAGE_VERSION: Final = 1
STORAGE_SAVE_DELAY: Final = 1
MAX_PER_CONVERSATION_SESSIONS: Final = 50
MAX_SESSION_ID_LENGTH: Final = 128
SATELLITE_IDLE_TIMEOUT: Final = 60
VALIDATE_TIMEOUT: Final = 10
CONNECT_TIMEOUT: Final = 10

SUBENTRY_TYPE_CONVERSATION: Final = "conversation"
