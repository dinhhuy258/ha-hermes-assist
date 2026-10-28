# Hermes Assist

Hermes Assist is a Home Assistant custom integration that makes a [Hermes Agent](https://github.com/NousResearch/hermes-agent) API server the conversation agent of an Assist pipeline.
Hermes Assist speaks only the final answer of Hermes, not the text Hermes writes before it calls a tool.

## Features

- Answers through the streaming Hermes `/v1/chat/completions` endpoint; the text Hermes writes before a tool call is kept out of the answer.
- Hermes-owned session continuity through `X-Hermes-Session-Id`, with three session modes.
- Hand-off for long requests: the satellite says a short acknowledgement and the answer is delivered later.
- Progress sentence: the satellite says the short sentence Hermes writes before its first tool, in the language Hermes uses.
- The prompt is a Home Assistant template, rendered on every request; device, area and language context is added after it.
- Optional list of the entities exposed to Assist (ID, name, aliases, area) in the system prompt, so the agent can act without looking entities up first.
- Spoken messages, such as the hand-off and error sentences, set per agent so each agent can speak its own language.
- Markdown removed from spoken answers.
- The `hermes_assist.ask` action for automations and scripts, with the agent prompt, an optional replacement prompt and an optional camera snapshot.
- A **Reset session** button on each agent device.

## Requirements

- Home Assistant 2026.9 or newer.
- A Hermes Agent API server with `API_SERVER_KEY` set.
- An Assist pipeline with speech to text and text to speech, for example Wyoming Whisper and Wyoming Piper.

## Installation

1. In HACS, open the menu, choose **Custom repositories** and add `https://github.com/dinhhuy258/ha-hermes-assist` with the type **Integration**.
2. Install **Hermes Assist** and restart Home Assistant.
3. Go to **Settings > Devices & services > Add integration** and choose **Hermes Assist**.
4. Enter the base URL of the Hermes API server, for example `https://hermes.example.com`, and the value of `API_SERVER_KEY`.

A trailing `/v1` in the URL is removed, and each URL can be added once.

## Using it in Assist

Setting up the integration creates one conversation agent, `conversation.hermes_assist`, with the default options.

1. Go to **Settings > Voice assistants** and open or create a pipeline.
2. Select the Hermes agent as the conversation agent.
3. Select a text to speech engine.

## Conversation agents

Each agent is a subentry of the Hermes Assist entry: it has its own name, prompt, model, session and timing, and it appears as one conversation entity with one device.
Add more agents from the integration page with **Add conversation agent**, for example one agent per language or per room, and pick each one in its own pipeline.
Reconfigure an agent from the three-dot menu of its row; deleting an agent removes its entity and its stored session.

| Option | Default | Description |
|---|---|---|
| Name | `Hermes Assist` | Name of the conversation entity and its device. |
| Prompt | Smart-speaker prompt | System prompt sent with every request. Rendered as a Home Assistant template, so `{{ states('sensor.x') }}` and the other template functions work. A prompt with a syntax error is rejected when saving; one that fails at runtime is sent as written and logged. |
| Model | Empty | Chosen from the models of the server, or typed. Sent as `model` only when set. |
| Session key | `homeassistant:assist` | Sent as `X-Hermes-Session-Key` to select the Hermes memory scope. Agents that share a key share Hermes memory. |
| Session mode | `idle` | `keep`, `idle` or `per_conversation`. |
| Session idle timeout | 1800 s | Used only in `idle` mode, 60 to 86400 seconds. |
| Hand off after | 15 s | 0 to 29 seconds; 0 disables the hand-off. |
| Timeout | 600 s | 5 to 1800 seconds for one request, including a handed-off request. |
| List exposed entities in the prompt | Off | Adds the entities exposed to Assist (ID, name, aliases, area) to the system prompt, one bullet per entity. No states are sent. |
| Reasoning effort | Empty | One of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra`, sent as `model_options.reasoning_effort`. Empty uses the Hermes default. |
| Spoken messages | English sentences | The hand-off sentence and the one error sentence said for every failure; the cause of a failure is logged. Write them in the language of the pipeline that uses the agent. An empty field uses the English default. |

Changing the session key or the session mode of an agent clears its stored session.

## Session modes

- `keep`: one Hermes session that never expires.
- `idle`: one Hermes session that is replaced after the idle timeout without requests.
- `per_conversation`: one Hermes session per Home Assistant conversation, kept in memory and lost on restart.

The session id of each agent lives in `.storage/hermes_assist.<subentry_id>`.
Hermes does not expire sessions that are addressed by id, so the idle timeout is the only expiry.

Each agent device has a **Reset session** button that forgets every stored session of the agent, from the UI or from an automation with `button.press`.

## Long requests

Hermes Assist reads the whole Hermes stream before it passes the answer on, because any text can still be followed by a tool call.
The text after the last tool call is the answer.

When the answer is not ready within **Hand off after**, the satellite says "I'm working on that. I'll let you know when it's done." and the request keeps running in the background.
Each agent can replace this sentence under **Spoken messages**.

When Hermes starts its first tool, the satellite says the sentence Hermes wrote just before it, such as "Let me check your calendar." or "Để mình xem lịch của bạn.", and the hand-off time counts from that sentence instead of from the request.
With a hand-off time of 15 seconds and a first tool at 5 seconds, the satellite says the sentence at 5 seconds and the hand-off sentence at 20 seconds, only if Hermes is still working then.
The default prompt asks Hermes for this sentence.
Updating replaces the earlier default sentence that told Hermes to stay silent before a tool; a custom prompt without that sentence needs a line that asks for one short sentence before the first tool.

- Only the text before the first tool is said, with markdown removed. Text longer than 120 characters is cut to its first sentence, and nothing is said when that is still longer.
- Tool names, tool arguments and the progress labels of Hermes are never said.
- When Hermes writes nothing before its first tool, nothing is said and the hand-off time still counts from the request.
- Requests without a tool are answered as before.
- The sentence is said early only when the text to speech engine of the pipeline supports streaming; otherwise the satellite says it together with the answer.

When the answer is ready, Hermes Assist waits up to 60 seconds for the satellite to be idle and then delivers it:

1. With `assist_satellite.start_conversation` when the answer ends with a question and the satellite supports it, so the reply continues the same Hermes session.
2. With `assist_satellite.announce` when the satellite supports it.
3. As a persistent notification otherwise.

A spoken answer uses the text to speech engine, language and voice of a pipeline that uses this agent, preferably one in the language of the request.
When no such pipeline exists, the satellite uses the text to speech of its own pipeline.
After `start_conversation`, the satellite listens with its own pipeline, so the reply goes through the speech to text of that pipeline.

If you speak again while a request is still streaming, the earlier request is dropped and nothing is announced later.

## Action: `hermes_assist.ask`

The action sends text to Hermes and returns the whole answer without streaming or hand-off.
It sends the prompt of the agent and, when the option is on, the exposed entities, like a voice request; no device or language sentence is added.

```yaml
action: hermes_assist.ask
data:
  text: "Summarize today's calendar in two sentences."
response_variable: hermes
```

The response is `{"text": "...", "session_id": "..."}`.

| Field | Description |
|---|---|
| `text` | The request. Required. |
| `system_prompt` | Replaces the prompt of the agent for this call. Sent as written, not rendered as a template; render it in the automation first if needed. |
| `camera_entity_id` | A camera whose current snapshot is sent with the text, so Hermes can describe what it sees. |
| `session_id` | A Hermes session to use for this call only. The stored session is left unchanged. |
| `agent_id` | The conversation entity, for example `conversation.hermes_assist`. Required when more than one agent exists. |

```yaml
action: hermes_assist.ask
data:
  text: "Is there a parcel at the door? Answer yes or no."
  system_prompt: "You describe camera images in one short sentence."
  camera_entity_id: camera.front_door
response_variable: hermes
```

## Privacy and security

- The API key is stored in the config entry and is never logged.
- Hermes receives the system prompt, the device and area name and the user text; no entity states are sent.
- With **List exposed entities in the prompt** on, Hermes also receives the ID, name, aliases and area of every entity exposed to Assist.
- With `camera_entity_id`, Hermes receives the camera snapshot.
- A template in the prompt can read any state or attribute; what it renders is sent to Hermes.
- Anyone who can call Home Assistant actions can call `hermes_assist.ask`.
- TLS certificate verification is always on.

## Troubleshooting

Enable debug logging to see the request flow:

```yaml
logger:
  logs:
    custom_components.hermes_assist: debug
```

## Development

The tests need Python 3.14 and a C++ compiler, because two Home Assistant test dependencies build from source.

```bash
python3.14 -m venv .venv
. .venv/bin/activate
pip install -r requirements_test.txt
ruff check .
ruff format --check .
pytest
```

## License

MIT, see [LICENSE](LICENSE).
