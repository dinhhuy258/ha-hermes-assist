"""AI Task entity that runs generate_data tasks with the agent's Hermes settings."""

import asyncio
import base64
import json
from json import JSONDecodeError
import logging
import re
from typing import Any

from homeassistant.components import ai_task, conversation
from homeassistant.components.ai_task.const import DEFAULT_SYSTEM_PROMPT
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads
from probatio import to_openapi

from . import HermesAssistConfigEntry, conversation_subentries
from .api import (
    HermesAuthError,
    HermesConnectionError,
    HermesError,
    HermesTimeoutError,
)
from .const import (
    CONF_AI_TASK_PROMPT,
    CONF_LIST_EXPOSED_ENTITIES,
    CONF_MODEL,
    CONF_REASONING_EFFORT,
    CONF_SESSION_KEY,
    CONF_TIMEOUT,
    DEFAULT_AI_TASK_PROMPT,
    DEFAULT_LIST_EXPOSED_ENTITIES,
    DEFAULT_SESSION_KEY,
    DEFAULT_TIMEOUT,
)
from .context import exposed_entities_cache, render_prompt
from .device import agent_device_info

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0

STRUCTURE_PROMPT = (
    "Respond with only a JSON object that matches this JSON schema, "
    "without markdown or any text around it:"
)
# Code fences that some models put around a JSON answer.
_FENCE = re.compile(r"^```[\w-]*\s*|\s*```$")


def _parse_json(text: str) -> Any:
    """Return the JSON of a structured answer; code fences are removed first."""
    try:
        return json_loads(_FENCE.sub("", text))
    except JSONDecodeError as err:
        _LOGGER.warning("Hermes did not answer with JSON: %s", text)
        raise HomeAssistantError(
            "Hermes did not answer with the requested structure"
        ) from err


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HermesAssistConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add one AI Task entity per agent subentry."""
    for subentry in conversation_subentries(entry):
        async_add_entities(
            [HermesTaskEntity(entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class HermesTaskEntity(ai_task.AITaskEntity):
    """Answer generate_data tasks with the model, session key and timeout of an agent.

    Every task starts a new Hermes session; nothing is stored. The system
    message starts with the AI task prompt of the agent, or with the default
    sentence of Home Assistant when that prompt is empty, never with the voice
    prompt; the exposed entities follow when the agent lists them.
    """

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = (
        ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )

    def __init__(
        self, entry: HermesAssistConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the AI Task entity of one agent."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = f"{subentry.subentry_id}-ai_task"
        self._attr_device_info = agent_device_info(subentry)

    async def _async_generate_data(
        self, task: ai_task.GenDataTask, chat_log: conversation.ChatLog
    ) -> ai_task.GenDataTaskResult:
        """Send the task to Hermes and return the whole answer."""
        options = self.subentry.data
        timeout = float(options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT))
        messages = [
            {"role": "system", "content": self._system_prompt(task, chat_log)},
            await self._user_message(task),
        ]
        data = self.entry.runtime_data
        _LOGGER.debug("Running AI task %s on %s", task.name, self.subentry.title)
        try:
            async with (
                asyncio.timeout(timeout),
                data.client.stream_chat(
                    messages,
                    model=options.get(CONF_MODEL) or None,
                    reasoning_effort=options.get(CONF_REASONING_EFFORT) or None,
                    session_key=options.get(CONF_SESSION_KEY, DEFAULT_SESSION_KEY)
                    or None,
                    timeout=timeout,
                ) as stream,
            ):
                text = "".join([part async for part in stream]).strip()
        except (TimeoutError, HermesTimeoutError) as err:
            raise HomeAssistantError(
                f"Hermes took longer than {timeout:g} seconds"
            ) from err
        except HermesAuthError as err:
            data.request_reauth(self.hass, self.entry)
            raise HomeAssistantError("Hermes rejected the API key") from err
        except HermesConnectionError as err:
            raise HomeAssistantError(f"Could not reach Hermes: {err}") from err
        except HermesError as err:
            raise HomeAssistantError(f"Hermes returned an error: {err}") from err
        if not text:
            raise HomeAssistantError("Hermes returned no answer")

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=self.entity_id, content=text)
        )
        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=text if task.structure is None else _parse_json(text),
        )

    def _system_prompt(
        self, task: ai_task.GenDataTask, chat_log: conversation.ChatLog
    ) -> str:
        """Return the system message: prompt, exposed entities and schema.

        AITaskEntity put the default sentence of Home Assistant, a date and
        time line and the prompt of an llm_api when one is given, at the start
        of the chat log. The AI task prompt of the agent replaces the sentence
        and keeps the rest.
        """
        first = chat_log.content[0]
        head = first.content if isinstance(first, conversation.SystemContent) else ""
        options = self.subentry.data
        if prompt := options.get(CONF_AI_TASK_PROMPT, DEFAULT_AI_TASK_PROMPT):
            head = render_prompt(self.hass, prompt) + head.removeprefix(
                DEFAULT_SYSTEM_PROMPT
            )
        parts: list[str] = [head.strip()]
        if options.get(CONF_LIST_EXPOSED_ENTITIES, DEFAULT_LIST_EXPOSED_ENTITIES):
            parts.append(exposed_entities_cache(self.hass).text())
        if task.structure is not None:
            schema = to_openapi(
                task.structure,
                custom_serializer=chat_log.llm_api.custom_serializer
                if chat_log.llm_api
                else llm.selector_serializer,
            )
            parts.append(f"{STRUCTURE_PROMPT}\n{json.dumps(schema)}")
        return "\n\n".join(part for part in parts if part)

    async def _user_message(self, task: ai_task.GenDataTask) -> dict[str, Any]:
        """Return the instructions with every attachment as an image data URL."""
        if not task.attachments:
            return {"role": "user", "content": task.instructions}
        content: list[dict[str, Any]] = [{"type": "text", "text": task.instructions}]
        for attachment in task.attachments:
            if not attachment.mime_type.startswith("image/"):
                raise HomeAssistantError(
                    f"Hermes Assist sends only images; {attachment.media_content_id}"
                    f" is {attachment.mime_type}"
                )
            data = await self.hass.async_add_executor_job(attachment.path.read_bytes)
            url = (
                f"data:{attachment.mime_type};base64,{base64.b64encode(data).decode()}"
            )
            content.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": "user", "content": content}
