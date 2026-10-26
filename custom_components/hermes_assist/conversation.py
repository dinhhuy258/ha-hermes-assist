"""Conversation agent that streams Hermes Agent answers into Assist."""

import asyncio
from collections.abc import AsyncGenerator, Mapping
import logging
import re
from typing import Any, Literal

from homeassistant.components import (
    assist_pipeline,
    conversation,
    persistent_notification,
    tts,
)
from homeassistant.components.assist_satellite import AssistSatelliteEntityFeature
from homeassistant.components.homeassistant.exposed_entities import (
    async_should_expose,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_SUPPORTED_FEATURES,
    MATCH_ALL,
    STATE_IDLE,
    STATE_UNAVAILABLE,
)
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    intent,
)
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event

from . import HermesAssistConfigEntry, conversation_subentries
from .api import HermesAuthError, HermesConnectionError, HermesError, HermesTimeoutError
from .const import (
    CONF_EXPOSED_ENTITIES,
    CONF_HANDOFF_AFTER,
    CONF_MESSAGE_AUTH,
    CONF_MESSAGE_CANNOT_CONNECT,
    CONF_MESSAGE_EMPTY,
    CONF_MESSAGE_GENERIC,
    CONF_MESSAGE_HANDOFF,
    CONF_MESSAGE_RESET,
    CONF_MESSAGE_TOO_LONG,
    CONF_MESSAGES,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_RESET_PHRASES,
    CONF_SESSION_KEY,
    CONF_TIMEOUT,
    DEFAULT_EXPOSED_ENTITIES,
    DEFAULT_HANDOFF_AFTER,
    DEFAULT_MESSAGES,
    DEFAULT_RESET_PHRASES,
    DEFAULT_SESSION_KEY,
    DEFAULT_TIMEOUT,
    DOMAIN,
    NAME,
    SATELLITE_IDLE_TIMEOUT,
)
from .prompts import DEFAULT_PROMPT, EXPOSED_ENTITIES_INTRO
from .session import SessionStore

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
    """Add one conversation entity per agent subentry."""
    for subentry in conversation_subentries(entry):
        async_add_entities(
            [HermesAssistEntity(entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


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


def _same_language(first: str | None, second: str | None) -> bool:
    """Return whether two language codes share their base language."""
    if not first or not second:
        return False
    return (
        first.replace("_", "-").split("-")[0].lower()
        == second.replace("_", "-").split("-")[0].lower()
    )


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

    def __init__(
        self, entry: HermesAssistConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the agent of one subentry."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Nous Research",
            model="Hermes Agent",
            entry_type=DeviceEntryType.SERVICE,
        )
        # Satellite entity id -> (question delivered late, its conversation id).
        self._follow_ups: dict[str, tuple[str, str]] = {}
        self._warned_missing_session = False

    @property
    def _options(self) -> Mapping[str, Any]:
        """Return the options of this agent."""
        return self.subentry.data

    def _message(self, key: str) -> str:
        """Return a spoken message of this agent, or its default when it is empty."""
        messages = self._options.get(CONF_MESSAGES) or {}
        return messages.get(key) or DEFAULT_MESSAGES[key]

    @property
    def _store(self) -> SessionStore:
        """Return the session store of this agent."""
        return self.entry.runtime_data.stores[self.subentry.subentry_id]

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
        options = self._options
        conversation_id = chat_log.conversation_id

        reset_phrases = options.get(CONF_RESET_PHRASES, DEFAULT_RESET_PHRASES)
        if _is_reset(user_input.text, reset_phrases):
            self._store.reset(conversation_id)
            if user_input.satellite_id:
                self._follow_ups.pop(user_input.satellite_id, None)
            return self._reply(user_input, chat_log, self._message(CONF_MESSAGE_RESET))

        self._link_follow_up(user_input, conversation_id)
        deadline = self.hass.loop.time() + float(
            options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
        )
        queue: asyncio.Queue[object] = asyncio.Queue()
        producer = self.entry.async_create_background_task(
            self.hass,
            self._async_produce(
                self._messages(user_input),
                self._store.get(conversation_id),
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
                return self._reply(
                    user_input, chat_log, self._message(CONF_MESSAGE_HANDOFF)
                )
            if isinstance(first, HermesError):
                return self._error(user_input, chat_log, self._error_message(first))
            if isinstance(first, str):
                await self._async_stream(user_input, chat_log, first, queue)
        except asyncio.CancelledError:
            # The pipeline gave up on this turn: close the stream, announce nothing.
            producer.cancel()
            raise

        self._store.touch(conversation_id)
        last = chat_log.content[-1]
        if not isinstance(last, conversation.AssistantContent) or not last.content:
            return self._error(user_input, chat_log, self._message(CONF_MESSAGE_EMPTY))
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
        """Combine the configured prompt with the context of this turn.

        The exposed entities come right after the prompt so that the part of
        the system message that is the same for every request stays a prefix.
        """
        parts = [
            self._options.get(CONF_PROMPT, DEFAULT_PROMPT),
            self._exposed_entities_context(),
            self._device_context(user_input),
        ]
        if user_input.language and user_input.language != MATCH_ALL:
            parts.append(f"Reply in the language with code '{user_input.language}'.")
        if user_input.extra_system_prompt:
            parts.append(user_input.extra_system_prompt)
        return "\n\n".join(part for part in parts if part)

    def _exposed_entities_context(self) -> str:
        """List the entities exposed to Assist: ID, name, aliases and area, no state."""
        if not self._options.get(CONF_EXPOSED_ENTITIES, DEFAULT_EXPOSED_ENTITIES):
            return ""
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        lines: list[str] = []
        for state in sorted(self.hass.states.async_all(), key=lambda s: s.entity_id):
            if not async_should_expose(self.hass, conversation.DOMAIN, state.entity_id):
                continue
            line = f"- {state.entity_id}: {state.name}"
            area_id: str | None = None
            if entry := entity_registry.async_get(state.entity_id):
                # Skip the computed-name entry: it repeats the name already listed.
                if aliases := [
                    alias.strip() for alias in entry.aliases if isinstance(alias, str)
                ]:
                    line += f" (aliases: {', '.join(aliases)})"
                area_id = entry.area_id
                if (
                    area_id is None
                    and entry.device_id
                    and (device := device_registry.async_get(entry.device_id))
                ):
                    area_id = device.area_id
            if area_id and (area := area_registry.async_get_area(area_id)):
                line += f" in {area.name}"
            lines.append(line)
        if not lines:
            return ""
        return "\n".join([EXPOSED_ENTITIES_INTRO, *lines])

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
        store = self._store
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
        options = self._options
        try:
            async with self.entry.runtime_data.client.stream_chat(
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
        self._store.set(conversation_id, session_id)

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
                producer,
                queue,
                deadline,
                user_input.satellite_id,
                conversation_id,
                user_input.language,
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
        language: str | None,
    ) -> None:
        """Wait for a handed-off answer within the timeout, then deliver it."""
        try:
            async with asyncio.timeout_at(deadline):
                text, error = await _async_collect(queue)
        except TimeoutError:
            producer.cancel()
            message = self._message(CONF_MESSAGE_TOO_LONG)
        else:
            message = self._late_message(text, error)
        self._store.touch(conversation_id)
        await self._async_deliver(satellite_id, conversation_id, message, language)

    def _late_message(self, text: str, error: HermesError | None) -> str:
        """Return what to deliver for a handed-off answer."""
        if isinstance(error, HermesTimeoutError):
            return self._message(CONF_MESSAGE_TOO_LONG)
        if speakable := _speakable(text):
            if error is not None:
                _LOGGER.warning(
                    "The Hermes stream broke before the answer finished: %s", error
                )
            return speakable
        if error is not None:
            return self._error_message(error)
        return self._message(CONF_MESSAGE_EMPTY)

    def _error_message(self, error: HermesError) -> str:
        """Return the spoken message for an error; ask for a new key once."""
        if isinstance(error, HermesAuthError):
            self.entry.runtime_data.request_reauth(self.hass, self.entry)
            return self._message(CONF_MESSAGE_AUTH)
        if isinstance(error, HermesTimeoutError):
            return self._message(CONF_MESSAGE_TOO_LONG)
        if isinstance(error, HermesConnectionError):
            _LOGGER.warning("Could not reach Hermes: %s", error)
            return self._message(CONF_MESSAGE_CANNOT_CONNECT)
        _LOGGER.warning("Hermes returned an error: %s", error)
        return self._message(CONF_MESSAGE_GENERIC)

    def _media_id(self, message: str, language: str | None) -> str | None:
        """Return the message spoken by the TTS of a pipeline that uses this agent.

        The satellite speaks an announcement with the TTS of its own pipeline,
        which may be set up for another language. A pipeline in the language of
        the request is preferred. None means the satellite chooses the TTS.
        """
        pipelines = [
            pipeline
            for pipeline in assist_pipeline.async_get_pipelines(self.hass)
            if pipeline.conversation_engine == self.entity_id and pipeline.tts_engine
        ]
        if not pipelines:
            return None
        pipeline = next(
            (p for p in pipelines if _same_language(p.language, language)),
            pipelines[0],
        )
        options = {tts.ATTR_VOICE: pipeline.tts_voice} if pipeline.tts_voice else None
        try:
            return tts.generate_media_source_id(
                self.hass,
                message,
                engine=pipeline.tts_engine,
                language=pipeline.tts_language,
                options=options,
            )
        except HomeAssistantError as err:
            _LOGGER.warning(
                "Could not use the text to speech of pipeline %s: %s",
                pipeline.name,
                err,
            )
            return None

    async def _async_deliver(
        self,
        satellite_id: str | None,
        conversation_id: str,
        message: str,
        language: str | None,
    ) -> None:
        """Speak a late answer on the satellite that asked, else notify."""
        if satellite_id is None or not await self._async_wait_for_idle(satellite_id):
            persistent_notification.async_create(self.hass, message, title=NAME)
            return

        state = self.hass.states.get(satellite_id)
        features = state.attributes.get(ATTR_SUPPORTED_FEATURES, 0) if state else 0
        media_id = self._media_id(message, language)

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
                        **({"start_media_id": media_id} if media_id else {}),
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
                    {
                        ATTR_ENTITY_ID: satellite_id,
                        "message": message,
                        **({"media_id": media_id} if media_id else {}),
                    },
                    blocking=True,
                )
            except HomeAssistantError as err:
                _LOGGER.warning("Could not announce on %s: %s", satellite_id, err)
            else:
                return

        persistent_notification.async_create(self.hass, message, title=NAME)

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
