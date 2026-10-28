"""Default text spoken to the user, and the error texts of the ask action.

An agent can replace HANDOFF_MESSAGE and ERROR_GENERIC; the other errors are
only raised by the ask action.
"""

from typing import Final

HANDOFF_MESSAGE: Final = "I'm working on that. I'll let you know when it's done."
ERROR_CANNOT_CONNECT: Final = "Sorry, I couldn't reach Hermes in time."
ERROR_AUTH: Final = "Hermes rejected the API key."
ERROR_GENERIC: Final = "Sorry, Hermes had a problem."
ERROR_EMPTY: Final = "Sorry, Hermes didn't answer."
ERROR_TOO_LONG: Final = "Sorry, Hermes took too long."
