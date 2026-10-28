"""Conversation agent that streams Hermes Agent answers into Assist."""

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
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
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event

from . import HermesAssistConfigEntry, conversation_subentries
from .api import (
    HermesAuthError,
    HermesConnectionError,
    HermesError,
    HermesTimeoutError,
    ToolStarted,
)
from .const import (
    CONF_HANDOFF_AFTER,
    CONF_MESSAGE_GENERIC,
    CONF_MESSAGE_HANDOFF,
    CONF_MESSAGES,
    CONF_MODEL,
    CONF_REASONING_EFFORT,
    CONF_SESSION_KEY,
    CONF_SPEAK_EVERY_TOOL,
    CONF_TIMEOUT,
    DEFAULT_HANDOFF_AFTER,
    DEFAULT_MESSAGES,
    DEFAULT_SESSION_KEY,
    DEFAULT_TIMEOUT,
    DOMAIN,
    NAME,
    SATELLITE_IDLE_TIMEOUT,
)
from .context import build_messages
from .device import agent_device_info
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
_SENTENCE_END = re.compile(r"(?<=[.!?\u2026])\s")
# Longer text before a tool is narration, not a progress sentence.
_MAX_PROGRESS_LENGTH = 120
# Text about tool work rather than for the user: snake_case or dotted names,
# JSON, quoted arguments, or words that name the machinery.
_TOOL_TALK = re.compile(
    r"\w_\w"
    r"|[^\W\d_]\.[^\W\d_]"
    r"|[{}]"
    r"|\"[^\"]*\"|“[^”]*”|`[^`]*`"
    r"|\b(?:tools?|functions?|api|query|queries)\b"
    r"|công cụ",
    re.IGNORECASE,
)


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


def _progress_sentence(text: str) -> str:
    """Return the short sentence to say from the text Hermes wrote before a tool.

    Nothing when the text talks about tool work; else the whole text when it
    is short, else its first sentence when that is short, else nothing.
    """
    if _TOOL_TALK.search(text):
        return ""
    text = _speakable(text)
    if len(text) <= _MAX_PROGRESS_LENGTH:
        return text
    first = _SENTENCE_END.split(text, maxsplit=1)[0]
    return first if len(first) <= _MAX_PROGRESS_LENGTH else ""


def _same_language(first: str | None, second: str | None) -> bool:
    """Return whether two language codes share their base language."""
    if not first or not second:
        return False
    return (
        first.replace("_", "-").split("-")[0].lower()
        == second.replace("_", "-").split("-")[0].lower()
    )


async def _async_collect(
    queue: asyncio.Queue[object],
) -> tuple[str, HermesError | None]:
    """Read the rest of an answer; return its text and the error that ended it."""
    parts: list[str] = []
    while isinstance(item := await queue.get(), str | ToolStarted):
        if isinstance(item, str):
            parts.append(item)
    return "".join(parts), (item if isinstance(item, HermesError) else None)


def _tool_input(tool: ToolStarted) -> llm.ToolInput:
    """Record a tool that Hermes runs; Home Assistant does not run it."""
    return llm.ToolInput(tool_name=tool.name, tool_args={}, external=True)


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
        self._attr_device_info = agent_device_info(subentry)
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
        speak_every_tool = bool(options.get(CONF_SPEAK_EVERY_TOOL, False))
        handed_off = False

        def hand_off() -> None:
            nonlocal handed_off
            handed_off = True
            self._hand_off(producer, queue, deadline, user_input, conversation_id)

        try:
            try:
                async with asyncio.timeout(handoff_after or None):
                    # A tool with nothing to say before it keeps the wait going.
                    while isinstance(first := await queue.get(), ToolStarted) and not (
                        progress := _progress_sentence(first.text)
                    ):
                        pass
            except TimeoutError:
                hand_off()
                return self._reply(
                    user_input, chat_log, self._message(CONF_MESSAGE_HANDOFF)
                )
            if isinstance(first, HermesError):
                return self._error(user_input, chat_log, self._error_message(first))
            if isinstance(first, ToolStarted):
                await self._async_stream_progress(
                    user_input,
                    chat_log,
                    first,
                    progress,
                    queue,
                    handoff_after,
                    hand_off,
                    speak_every_tool,
                )
            elif isinstance(first, str):
                await self._async_stream(user_input, chat_log, first, queue)
        except asyncio.CancelledError:
            # The pipeline gave up on this turn: close the stream, announce
            # nothing. A handed-off request keeps running.
            if not handed_off:
                producer.cancel()
            raise

        self._store.touch(conversation_id)
        last = chat_log.content[-1]
        if not isinstance(last, conversation.AssistantContent) or not last.content:
            _LOGGER.warning("Hermes returned no answer")
            return self._error(
                user_input, chat_log, self._message(CONF_MESSAGE_GENERIC)
            )
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    def _messages(
        self, user_input: conversation.ConversationInput
    ) -> list[dict[str, Any]]:
        """Return the system and user messages of this turn."""
        return build_messages(
            self.hass,
            self._options,
            user_input.text,
            device_id=user_input.device_id,
            language=user_input.language,
            extra_system_prompt=user_input.extra_system_prompt,
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
        messages: list[dict[str, Any]],
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
                reasoning_effort=options.get(CONF_REASONING_EFFORT) or None,
                session_key=options.get(CONF_SESSION_KEY, DEFAULT_SESSION_KEY) or None,
                session_id=session_id,
                timeout=float(options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)),
            ) as stream:
                self._remember_session(conversation_id, stream.session_id)
                async for item in stream.items():
                    queue.put_nowait(item)
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

    async def _async_stream_progress(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        first: ToolStarted,
        progress: str,
        queue: asyncio.Queue[object],
        handoff_after: float,
        hand_off: Callable[[], None],
        speak_every_tool: bool,
    ) -> None:
        """Say what Hermes wrote before its tool, then the answer.

        Text followed by a tool call makes the pipeline start text to speech
        at once instead of waiting for the answer. The hand-off time counts
        from the last spoken sentence, so the hand-off message is said only
        when Hermes is still working handoff_after seconds later. Text before
        later tools is said only with speak_every_tool, and not when it
        repeats the previous sentence. Nothing is said after the hand-off.
        """
        strip = _SymbolStripper()
        loop = self.hass.loop
        handoff_at = loop.time() + handoff_after if handoff_after else None

        async def deltas() -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
            nonlocal handoff_at
            yield {"role": "assistant"}
            # The space keeps the next sentence apart when the pipeline joins them.
            yield {"content": f"{progress} "}
            yield {"tool_calls": [_tool_input(first)]}
            spoken = progress
            answered = False
            while True:
                try:
                    async with asyncio.timeout_at(None if answered else handoff_at):
                        item = await queue.get()
                except TimeoutError:
                    _LOGGER.debug("Hermes is still working after the progress sentence")
                    hand_off()
                    yield {"role": "assistant"}
                    yield {"content": self._message(CONF_MESSAGE_HANDOFF)}
                    return
                if isinstance(item, ToolStarted):
                    sentence = _progress_sentence(item.text) if speak_every_tool else ""
                    if sentence and sentence != spoken:
                        spoken = sentence
                        yield {"content": f"{sentence} "}
                        if handoff_at is not None:
                            handoff_at = loop.time() + handoff_after
                    yield {"tool_calls": [_tool_input(item)]}
                elif isinstance(item, str):
                    if not answered:
                        answered = True
                        yield {"role": "assistant"}
                    yield {"content": strip(item)}
                else:
                    break
            if answered:
                if isinstance(item, HermesError):
                    _LOGGER.warning(
                        "The Hermes stream broke after the answer started: %s", item
                    )
                return
            # Speech has started, so a failure is said instead of returned as an error.
            if isinstance(item, HermesError):
                message = self._error_message(item)
            else:
                _LOGGER.warning("Hermes returned no answer")
                message = self._message(CONF_MESSAGE_GENERIC)
            yield {"role": "assistant"}
            yield {"content": message}

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
            _LOGGER.warning("Hermes took longer than the timeout")
            message = self._message(CONF_MESSAGE_GENERIC)
        else:
            message = self._late_message(text, error)
        self._store.touch(conversation_id)
        await self._async_deliver(satellite_id, conversation_id, message, language)

    def _late_message(self, text: str, error: HermesError | None) -> str:
        """Return what to deliver for a handed-off answer."""
        if isinstance(error, HermesTimeoutError):
            return self._error_message(error)
        if speakable := _speakable(text):
            if error is not None:
                _LOGGER.warning(
                    "The Hermes stream broke before the answer finished: %s", error
                )
            return speakable
        if error is not None:
            return self._error_message(error)
        _LOGGER.warning("Hermes returned no answer")
        return self._message(CONF_MESSAGE_GENERIC)

    def _error_message(self, error: HermesError) -> str:
        """Log the cause of an error and return the one spoken error message.

        A rejected API key also asks for a new key, once.
        """
        if isinstance(error, HermesAuthError):
            _LOGGER.warning("Hermes rejected the API key")
            self.entry.runtime_data.request_reauth(self.hass, self.entry)
        elif isinstance(error, HermesTimeoutError):
            _LOGGER.warning("Hermes took longer than the timeout: %s", error)
        elif isinstance(error, HermesConnectionError):
            _LOGGER.warning("Could not reach Hermes: %s", error)
        else:
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
