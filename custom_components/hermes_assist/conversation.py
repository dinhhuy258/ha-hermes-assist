"""Conversation agent that streams Hermes Agent answers into Assist."""

import asyncio
from collections.abc import AsyncGenerator
import logging
import re
from typing import Literal

from homeassistant.components import conversation, persistent_notification
from homeassistant.components.assist_satellite import AssistSatelliteEntityFeature
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_SUPPORTED_FEATURES,
    MATCH_ALL,
    STATE_IDLE,
    STATE_UNAVAILABLE,
)
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry as ar, device_registry as dr, intent
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event

from . import HermesAssistConfigEntry
from .api import HermesAuthError, HermesConnectionError, HermesError, HermesTimeoutError
from .const import (
    CONF_HANDOFF_AFTER,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_RESET_PHRASES,
    CONF_SESSION_KEY,
    CONF_TIMEOUT,
    DEFAULT_HANDOFF_AFTER,
    DEFAULT_PROMPT,
    DEFAULT_RESET_PHRASES,
    DEFAULT_SESSION_KEY,
    DEFAULT_TIMEOUT,
    DOMAIN,
    ERROR_AUTH,
    ERROR_CANNOT_CONNECT,
    ERROR_EMPTY,
    ERROR_GENERIC,
    ERROR_TOO_LONG,
    HANDOFF_MESSAGE,
    NOTIFICATION_TITLE,
    RESET_MESSAGE,
    SATELLITE_IDLE_TIMEOUT,
)

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0

# Marks the end of an answer in the producer queue.
_DONE = object()
# A question mark, ASCII or full-width, keeps the late answer conversational.
_QUESTION_ENDINGS = ("?", "\uff1f")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_LINE_MARKUP = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*|>[ \t]*|[-*+][ \t]+)", re.MULTILINE)
_INLINE_MARKUP = re.compile(r"[*`]")
_WORD_UNDERSCORE = re.compile(r"(?<=[^\W_])_+(?=[^\W_])")


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HermesAssistConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add the Hermes conversation entity."""
    async_add_entities([HermesAssistEntity(entry)])


class _SymbolStripper:
    """Remove markdown symbols from text that arrives in pieces.

    Headings, quotes and list bullets are only removed at the start of a line,
    so the stripper remembers whether the previous piece ended a line.
    """

    def __init__(self) -> None:
        self._at_line_start = True

    def __call__(self, text: str) -> str:
        if self._at_line_start:
            stripped = _LINE_MARKUP.sub("", text)
        else:
            # The prefix stops "^" from matching where the piece starts mid-line.
            stripped = _LINE_MARKUP.sub("", f"x{text}")[1:]
        self._at_line_start = text.endswith("\n")
        stripped = _INLINE_MARKUP.sub("", stripped)
        stripped = _WORD_UNDERSCORE.sub(" ", stripped)
        return stripped.replace("_", "")


def _speakable(text: str) -> str:
    """Turn a complete answer into plain text for text to speech."""
    text = _LINK.sub(r"\1", text)
    return " ".join(_SymbolStripper()(text).split())


def _normalize(text: str) -> str:
    """Lowercase and drop punctuation, since speech to text output varies."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def _is_reset(text: str, phrases: str) -> bool:
    """Return whether the utterance is one of the comma-separated reset phrases."""
    wanted = {_normalize(phrase) for phrase in phrases.split(",")} - {""}
    return _normalize(text) in wanted


async def _async_collect(
    queue: asyncio.Queue[object],
) -> tuple[str, HermesError | None]:
    """Read the rest of an answer; return its text and the error that ended it."""
    parts: list[str] = []
    while isinstance(item := await queue.get(), str):
        parts.append(item)
    return "".join(parts), (item if isinstance(item, HermesError) else None)


class HermesAssistEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Hermes Agent as a streaming Assist conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, entry: HermesAssistConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Nous Research",
            model="Hermes Agent",
            entry_type=DeviceEntryType.SERVICE,
        )
        # Satellite entity id -> (question delivered late, its conversation id).
        self._follow_ups: dict[str, tuple[str, str]] = {}
        self._warned_missing_session = False

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Hermes handles every language its model does."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Register the agent."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister the agent."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Stream the Hermes answer, or hand it off when the first delta is late."""
        data = self.entry.runtime_data
        options = self.entry.options
        conversation_id = chat_log.conversation_id

        reset_phrases = options.get(CONF_RESET_PHRASES, DEFAULT_RESET_PHRASES)
        if _is_reset(user_input.text, reset_phrases):
            data.store.reset(conversation_id)
            if user_input.satellite_id:
                self._follow_ups.pop(user_input.satellite_id, None)
            return self._reply(user_input, chat_log, RESET_MESSAGE)

        self._link_follow_up(user_input, conversation_id)
        deadline = self.hass.loop.time() + float(
            options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
        )
        queue: asyncio.Queue[object] = asyncio.Queue()
        producer = self.entry.async_create_background_task(
            self.hass,
            self._async_produce(
                self._messages(user_input),
                data.store.get(conversation_id),
                conversation_id,
                queue,
            ),
            f"{DOMAIN} answer {conversation_id}",
        )
        handoff_after = float(options.get(CONF_HANDOFF_AFTER, DEFAULT_HANDOFF_AFTER))

        try:
            try:
                async with asyncio.timeout(handoff_after or None):
                    first = await queue.get()
            except TimeoutError:
                self._hand_off(producer, queue, deadline, user_input, conversation_id)
                return self._reply(user_input, chat_log, HANDOFF_MESSAGE)
            if isinstance(first, HermesError):
                return self._error(user_input, chat_log, self._error_message(first))
            if isinstance(first, str):
                await self._async_stream(user_input, chat_log, first, queue)
        except asyncio.CancelledError:
            # The pipeline gave up on this turn: close the stream, announce nothing.
            producer.cancel()
            raise

        data.store.touch(conversation_id)
        last = chat_log.content[-1]
        if not isinstance(last, conversation.AssistantContent) or not last.content:
            return self._error(user_input, chat_log, ERROR_EMPTY)
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    def _messages(
        self, user_input: conversation.ConversationInput
    ) -> list[dict[str, str]]:
        """Return the system and user messages of this turn."""
        messages: list[dict[str, str]] = []
        if prompt := self._build_prompt(user_input):
            messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": user_input.text})
        return messages

    def _build_prompt(self, user_input: conversation.ConversationInput) -> str:
        """Combine the configured prompt with the context of this turn."""
        parts = [
            self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
            self._device_context(user_input),
        ]
        if user_input.language and user_input.language != MATCH_ALL:
            parts.append(f"Reply in the language with code '{user_input.language}'.")
        if user_input.extra_system_prompt:
            parts.append(user_input.extra_system_prompt)
        return "\n\n".join(part for part in parts if part)

    def _device_context(self, user_input: conversation.ConversationInput) -> str:
        """Describe the voice device that heard the request, and its area."""
        if not user_input.device_id:
            return ""
        device = dr.async_get(self.hass).async_get(user_input.device_id)
        if device is None:
            return ""
        name = device.name_by_user or device.name
        area = (
            ar.async_get(self.hass).async_get_area(device.area_id)
            if device.area_id
            else None
        )
        if area is None:
            return f'The user is talking to the voice device "{name}".'
        return (
            f'The user is talking to the voice device "{name}" in the {area.name}. '
            f"'Here' means the {area.name}."
        )

    def _link_follow_up(
        self, user_input: conversation.ConversationInput, conversation_id: str
    ) -> None:
        """Continue the Hermes session of a question that was delivered late.

        start_conversation opens a new Home Assistant conversation and passes
        the question as extra_system_prompt, which identifies the follow-up.
        """
        if not user_input.satellite_id:
            return
        follow_up = self._follow_ups.pop(user_input.satellite_id, None)
        if follow_up is None:
            return
        question, previous_id = follow_up
        if user_input.extra_system_prompt != question or previous_id == conversation_id:
            return
        store = self.entry.runtime_data.store
        if (session_id := store.get(previous_id)) is not None:
            store.set(conversation_id, session_id)

    async def _async_produce(
        self,
        messages: list[dict[str, str]],
        session_id: str | None,
        conversation_id: str,
        queue: asyncio.Queue[object],
    ) -> None:
        """Read the Hermes stream into the queue, ending with _DONE or an error."""
        data = self.entry.runtime_data
        options = self.entry.options
        try:
            async with data.client.stream_chat(
                messages,
                model=options.get(CONF_MODEL) or None,
                session_key=options.get(CONF_SESSION_KEY, DEFAULT_SESSION_KEY) or None,
                session_id=session_id,
                timeout=float(options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)),
            ) as stream:
                self._remember_session(conversation_id, stream.session_id)
                async for text in stream:
                    queue.put_nowait(text)
        except HermesError as err:
            queue.put_nowait(err)
        except Exception as err:
            _LOGGER.exception("Unexpected error while reading the Hermes answer")
            queue.put_nowait(HermesError(str(err)))
        else:
            queue.put_nowait(_DONE)

    def _remember_session(self, conversation_id: str, session_id: str | None) -> None:
        """Store the session id from the response headers; a missing header changes nothing."""
        if session_id is None:
            if not self._warned_missing_session:
                self._warned_missing_session = True
                _LOGGER.warning(
                    "Hermes sent no X-Hermes-Session-Id header; "
                    "every request will start a new Hermes session"
                )
            return
        self.entry.runtime_data.store.set(conversation_id, session_id)

    async def _async_stream(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        first: str,
        queue: asyncio.Queue[object],
    ) -> None:
        """Feed the queued deltas into the chat log so TTS can start early."""
        strip = _SymbolStripper()

        async def deltas() -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
            yield {"role": "assistant"}
            item: object = first
            while isinstance(item, str):
                yield {"content": strip(item)}
                item = await queue.get()
            if isinstance(item, HermesError):
                _LOGGER.warning(
                    "The Hermes stream broke after the answer started: %s", item
                )

        async for _content in chat_log.async_add_delta_content_stream(
            user_input.agent_id, deltas()
        ):
            pass

    def _hand_off(
        self,
        producer: asyncio.Task[None],
        queue: asyncio.Queue[object],
        deadline: float,
        user_input: conversation.ConversationInput,
        conversation_id: str,
    ) -> None:
        """Let the request finish in the background and deliver it later."""
        _LOGGER.debug("Hermes is still working; handing off %s", conversation_id)
        self.entry.async_create_background_task(
            self.hass,
            self._async_finish_later(
                producer, queue, deadline, user_input.satellite_id, conversation_id
            ),
            f"{DOMAIN} late answer {conversation_id}",
        )

    async def _async_finish_later(
        self,
        producer: asyncio.Task[None],
        queue: asyncio.Queue[object],
        deadline: float,
        satellite_id: str | None,
        conversation_id: str,
    ) -> None:
        """Wait for a handed-off answer within the timeout, then deliver it."""
        try:
            async with asyncio.timeout_at(deadline):
                text, error = await _async_collect(queue)
        except TimeoutError:
            producer.cancel()
            message = ERROR_TOO_LONG
        else:
            message = self._late_message(text, error)
        self.entry.runtime_data.store.touch(conversation_id)
        await self._async_deliver(satellite_id, conversation_id, message)

    def _late_message(self, text: str, error: HermesError | None) -> str:
        """Return what to deliver for a handed-off answer."""
        if isinstance(error, HermesTimeoutError):
            return ERROR_TOO_LONG
        if speakable := _speakable(text):
            if error is not None:
                _LOGGER.warning(
                    "The Hermes stream broke before the answer finished: %s", error
                )
            return speakable
        if error is not None:
            return self._error_message(error)
        return ERROR_EMPTY

    def _error_message(self, error: HermesError) -> str:
        """Return the spoken message for an error; ask for a new key once."""
        if isinstance(error, HermesAuthError):
            self.entry.runtime_data.request_reauth(self.hass, self.entry)
            return ERROR_AUTH
        if isinstance(error, HermesTimeoutError):
            return ERROR_TOO_LONG
        if isinstance(error, HermesConnectionError):
            _LOGGER.warning("Could not reach Hermes: %s", error)
            return ERROR_CANNOT_CONNECT
        _LOGGER.warning("Hermes returned an error: %s", error)
        return ERROR_GENERIC

    async def _async_deliver(
        self, satellite_id: str | None, conversation_id: str, message: str
    ) -> None:
        """Speak a late answer on the satellite that asked, else notify."""
        if satellite_id is None or not await self._async_wait_for_idle(satellite_id):
            persistent_notification.async_create(
                self.hass, message, title=NOTIFICATION_TITLE
            )
            return

        state = self.hass.states.get(satellite_id)
        features = state.attributes.get(ATTR_SUPPORTED_FEATURES, 0) if state else 0

        if features & AssistSatelliteEntityFeature.START_CONVERSATION and (
            message.rstrip().endswith(_QUESTION_ENDINGS)
        ):
            # Record first: the satellite may listen before the call returns.
            self._follow_ups[satellite_id] = (message, conversation_id)
            try:
                await self.hass.services.async_call(
                    "assist_satellite",
                    "start_conversation",
                    {
                        ATTR_ENTITY_ID: satellite_id,
                        "start_message": message,
                        "extra_system_prompt": message,
                    },
                    blocking=True,
                )
            except HomeAssistantError as err:
                self._follow_ups.pop(satellite_id, None)
                _LOGGER.warning(
                    "Could not start a conversation on %s: %s", satellite_id, err
                )
            else:
                return

        if features & AssistSatelliteEntityFeature.ANNOUNCE:
            try:
                await self.hass.services.async_call(
                    "assist_satellite",
                    "announce",
                    {ATTR_ENTITY_ID: satellite_id, "message": message},
                    blocking=True,
                )
            except HomeAssistantError as err:
                _LOGGER.warning("Could not announce on %s: %s", satellite_id, err)
            else:
                return

        persistent_notification.async_create(
            self.hass, message, title=NOTIFICATION_TITLE
        )

    async def _async_wait_for_idle(self, satellite_id: str) -> bool:
        """Wait until the satellite is idle so the delivery interrupts nothing."""
        state = self.hass.states.get(satellite_id)
        if state is None or state.state == STATE_UNAVAILABLE:
            return False
        if state.state == STATE_IDLE:
            return True

        idle = asyncio.Event()

        @callback
        def _async_state_changed(event: Event[EventStateChangedData]) -> None:
            new_state = event.data["new_state"]
            if new_state is not None and new_state.state == STATE_IDLE:
                idle.set()

        unsubscribe = async_track_state_change_event(
            self.hass, [satellite_id], _async_state_changed
        )
        try:
            async with asyncio.timeout(SATELLITE_IDLE_TIMEOUT):
                await idle.wait()
        except TimeoutError:
            _LOGGER.info(
                "%s stayed busy for %s seconds; sending a notification instead",
                satellite_id,
                SATELLITE_IDLE_TIMEOUT,
            )
            return False
        finally:
            unsubscribe()
        return True

    @staticmethod
    def _reply(
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        text: str,
    ) -> conversation.ConversationResult:
        """Answer with a fixed sentence."""
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=user_input.agent_id, content=text)
        )
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    @staticmethod
    def _error(
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        message: str,
    ) -> conversation.ConversationResult:
        """Answer with an error response."""
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
        return conversation.ConversationResult(
            response=response, conversation_id=chat_log.conversation_id
        )
