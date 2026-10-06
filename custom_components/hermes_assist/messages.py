"""Text spoken to the user or shown in Home Assistant."""

from typing import Final

from .const import NAME

HANDOFF_MESSAGE: Final = "I'm working on that. I'll let you know when it's done."
RESET_MESSAGE: Final = "Okay, starting a new conversation."
ERROR_CANNOT_CONNECT: Final = "Sorry, I couldn't reach Hermes in time."
ERROR_AUTH: Final = "Hermes rejected the API key."
ERROR_GENERIC: Final = "Sorry, Hermes had a problem."
ERROR_EMPTY: Final = "Sorry, Hermes didn't answer."
ERROR_TOO_LONG: Final = "Sorry, Hermes took too long."
NOTIFICATION_TITLE: Final = NAME
