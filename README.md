# Hermes Assist

Hermes Assist is a Home Assistant custom integration that makes a [Hermes Agent](https://github.com/NousResearch/hermes-agent) API server the conversation agent of an Assist pipeline.
Answers stream into the pipeline, so text to speech can start before Hermes has finished.

## Features

- Streaming answers through the Hermes `/v1/chat/completions` endpoint.
- Hermes-owned session continuity through `X-Hermes-Session-Id`, with three session modes.
- Hand-off for long requests: the satellite says a short acknowledgement and the answer is delivered later.
- Reset phrases that start a new Hermes session.
- Device and area context in the system prompt, and the pipeline language passed to Hermes.
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

1. Go to **Settings > Voice assistants** and open or create a pipeline.
2. Select **Hermes Assist** as the conversation agent.
3. Select a streaming text to speech engine to hear long answers sooner.

## Options

| Option | Default | Description |
|---|---|---|
| Prompt | Smart-speaker prompt | System prompt sent with every voice request. |
| Model | Empty | Sent as `model` only when set. |
| Session key | `homeassistant:assist` | Sent as `X-Hermes-Session-Key` to select the Hermes memory scope. |
| Session mode | `idle` | `keep`, `idle` or `per_conversation`. |
| Session idle timeout | 1800 s | Used only in `idle` mode, 60 to 86400 seconds. |
| Reset phrases | `start over, new conversation, reset conversation` | Comma-separated phrases that start a new session. |
| Hand off after | 15 s | 0 to 29 seconds; 0 disables the hand-off. |
| Timeout | 600 s | 5 to 1800 seconds for one request, including a handed-off request. |

Changing the session key or the session mode clears the stored session.

## Session modes

- `keep`: one Hermes session that never expires.
- `idle`: one Hermes session that is replaced after the idle timeout without requests.
- `per_conversation`: one Hermes session per Home Assistant conversation, kept in memory and lost on restart.

The session id lives in `.storage/hermes_assist.<entry_id>`.
Hermes does not expire sessions that are addressed by id, so the idle timeout is the only expiry.

## Long requests

When Hermes has not started answering within **Hand off after**, the satellite says "I'm working on that. I'll let you know when it's done." and the request keeps running in the background.

When the answer is ready, Hermes Assist waits up to 60 seconds for the satellite to be idle and then delivers it:

1. With `assist_satellite.start_conversation` when the answer ends with a question and the satellite supports it, so the reply continues the same Hermes session.
2. With `assist_satellite.announce` when the satellite supports it.
3. As a persistent notification otherwise.

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
Pass `session_id` to use a specific Hermes session for one call, and `config_entry_id` when more than one Hermes Assist entry is set up.

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
