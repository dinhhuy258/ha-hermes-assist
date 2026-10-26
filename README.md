# Hermes Assist

Hermes Assist is a Home Assistant custom integration that makes a [Hermes Agent](https://github.com/NousResearch/hermes-agent) API server the conversation agent of an Assist pipeline.
Hermes Assist speaks only the final answer of Hermes, not the text Hermes writes before it calls a tool.

## Features

- Answers through the streaming Hermes `/v1/chat/completions` endpoint, without the text Hermes writes before a tool call.
- Hermes-owned session continuity through `X-Hermes-Session-Id`, with three session modes.
- Hand-off for long requests: the satellite says a short acknowledgement and the answer is delivered later.
- Reset phrases that start a new Hermes session.
- Device and area context in the system prompt, and the pipeline language passed to Hermes.
- Spoken messages, such as the hand-off and error sentences, set per agent so each agent can speak its own language.
- Markdown removed from spoken answers.
- The `hermes_assist.ask` action for automations and scripts.

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
| Prompt | Smart-speaker prompt | System prompt sent with every voice request. |
| Model | Empty | Sent as `model` only when set. |
| Session key | `homeassistant:assist` | Sent as `X-Hermes-Session-Key` to select the Hermes memory scope. Agents that share a key share Hermes memory. |
| Session mode | `idle` | `keep`, `idle` or `per_conversation`. |
| Session idle timeout | 1800 s | Used only in `idle` mode, 60 to 86400 seconds. |
| Reset phrases | `start over, new conversation, reset conversation` | Comma-separated phrases that start a new session. |
| Hand off after | 15 s | 0 to 29 seconds; 0 disables the hand-off. |
| Timeout | 600 s | 5 to 1800 seconds for one request, including a handed-off request. |
| Spoken messages | English sentences | The hand-off, new session and error sentences. Write them in the language of the pipeline that uses the agent. An empty field uses the English default. |

Changing the session key or the session mode of an agent clears its stored session.

## Session modes

- `keep`: one Hermes session that never expires.
- `idle`: one Hermes session that is replaced after the idle timeout without requests.
- `per_conversation`: one Hermes session per Home Assistant conversation, kept in memory and lost on restart.

The session id of each agent lives in `.storage/hermes_assist.<subentry_id>`.
Hermes does not expire sessions that are addressed by id, so the idle timeout is the only expiry.

## Long requests

Hermes Assist reads the whole Hermes stream before it passes the answer on, because any text can still be followed by a tool call.
The text after the last tool call is the answer.

When the answer is not ready within **Hand off after**, the satellite says "I'm working on that. I'll let you know when it's done." and the request keeps running in the background.
Each agent can replace this sentence under **Spoken messages**.

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

```yaml
action: hermes_assist.ask
data:
  text: "Summarize today's calendar in two sentences."
response_variable: hermes
```

The response is `{"text": "...", "session_id": "..."}`.
Pass `session_id` to use a specific Hermes session for one call, and `agent_id` (the conversation entity, for example `conversation.hermes_assist`) when more than one agent exists.

## Privacy and security

- The API key is stored in the config entry and is never logged.
- Hermes receives the system prompt, the device and area name and the user text; no entity states are sent.
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
