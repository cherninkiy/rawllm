# RawLLM

Minimal POC of "dumb orchestrator – smart model". The LLM evolves itself by writing plugins (`add_plugin` / `run_plugin`). Inspired by MemPalace (96.6% on LongMemEval) and Claude Code’s TAOR loop. The core is immutable (<150 loc). HTTP transport is a plugin.

## Idea

Instead of hard-coding complexity into the orchestrator, the model is given two tools:
- `add_plugin(name, code)` — write and save a new plugin (or overwrite an existing one)
- `run_plugin(name, input_data)` — execute an already-loaded plugin by name

The core (orchestrator) is **immutable** and deliberately "dumb" (~150 lines). All intelligence — creating new capabilities, coordinating agents, managing memory, parsing data — lives in the LLM’s reasoning and in the plugins it generates. The model decides when to write a plugin and can hot-reload them at any time.

## Inspiration

- **MemPalace** — the approach that dominated LongMemEval (96.6%) without complex RAG, simply by giving the model raw data and freedom to decide.
- **Claude Code** — the TAOR (Think‑Act‑Observe‑Repeat) architecture and the "dumb orchestrator, smart model" principle.
- **Critique of over-engineered RAG pipelines** — give the model a clean context and let it decide.

## Status

✅ **Implemented** — core, HTTP plugin, tests and CI.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env with your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Start the orchestrator (HTTP server on port 8080)
python run.py

# 4. Send a request
curl -X POST http://localhost:8080/ \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a plugin that returns the current time", "context": {}}'
```

The server port can be overridden via the `HTTP_PORT` environment variable.

## Docker sandbox (WSL)

For more isolated plugin execution in WSL, use the Docker backend. In this
mode, untrusted plugins run under a separate user and container filesystem
boundary rather than sharing the orchestrator process privileges.

- `rawllm-core` - orchestrator process user on host side
- `rawllm-plugin` - plugin subprocess user inside sandbox container

### 1) Build sandbox image

```bash
docker build -t rawllm/plugin-sandbox:latest -f docker/sandbox/Dockerfile .
```

### 2) Enable docker backend

```bash
echo "SANDBOX_BACKEND=docker" >> .env
echo "SANDBOX_DOCKER_IMAGE=rawllm/plugin-sandbox:latest" >> .env
```

### 3) Run tests in WSL (Docker required)

```bash
pytest -q
```

When docker backend is enabled, plugin execution uses isolated volumes:
- `rawllm_workspace` (rw)
- `rawllm_core_repo` (ro snapshot)
- `rawllm_plugin_store` (ro snapshot)

## Running with Free / Lightweight LLMs

`run.py` supports any OpenAI-compatible provider via the `LLM_PROVIDER`
environment variable (default: `anthropic`). All security and versioning
features are automatically active.

### Supported providers

| `LLM_PROVIDER` | API key env var | Default model |
|---|---|---|
| `anthropic` *(default)* | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20241022` |
| `groq` | `GROQ_API_KEY` | `llama3-70b-8192` |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| `openrouter` | `OPEN_ROUTER_API_KEY` | `qwen/qwen3-coder:free` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `ollama` | *(none required)* | `llama3.2:3b` |
| `ollama-qwen-coder` | *(none required)* | `qwen2.5-coder:7b` |

Override any default with `LLM_MODEL` and `LLM_BASE_URL`.

### Groq (free tier)
```bash
echo "GROQ_API_KEY=gsk_..." >> .env
LLM_PROVIDER=groq python run.py
```

### Google Gemini
```bash
echo "GEMINI_API_KEY=AIza..." >> .env
LLM_PROVIDER=gemini python run.py
```

### OpenRouter (free models)
```bash
echo "OPEN_ROUTER_API_KEY=sk-or-..." >> .env
LLM_PROVIDER=openrouter python run.py
# Use a specific free model:
LLM_PROVIDER=openrouter LLM_MODEL=google/gemma-3-27b-it:free python run.py
```

### DeepSeek
```bash
echo "DEEPSEEK_API_KEY=sk-..." >> .env
LLM_PROVIDER=deepseek python run.py
```

### Ollama (fully local, no API key)
```bash
# 1. Install Ollama: https://ollama.com/
ollama pull llama3.2:3b   # or any model you prefer

# 2. Run
LLM_PROVIDER=ollama python run.py
# Custom model:
LLM_PROVIDER=ollama LLM_MODEL=mistral python run.py
```

### Local Qwen Coder 7B for container testing
```bash
# 1. Pull the local coding model in WSL / host environment
ollama pull qwen2.5-coder:7b

# 2. Run RawLLM against the dedicated provider alias
LLM_PROVIDER=ollama-qwen-coder python run.py
```

If the orchestrator itself runs in a container and Ollama stays on the host,
override the endpoint explicitly:

```bash
LLM_PROVIDER=ollama-qwen-coder \
LLM_BASE_URL=http://host.docker.internal:11434/v1 \
python run.py
```

## CLI (`rawllm`)

Install the package in editable mode to get the `rawllm` command:

```bash
pip install -e .
```

Or invoke directly without installing:

```bash
python cli.py <command>
```

### Orchestrator lifecycle
```bash
rawllm run                        # use default provider (anthropic)
rawllm run --provider groq        # use a specific provider
```

### Plugin management
```bash
rawllm plugin list
rawllm plugin show my_plugin
rawllm plugin add my_plugin path/to/code.py
rawllm plugin rollback my_plugin
```

## Plugin authoring contract

Use module-level docstring in every plugin as a prompt for RawLLM. The
docstring should describe plugin role, input/output contract, operational
constraints, and failure behavior. See [plugins/http.py](plugins/http.py) as a
reference template.

### Dependency approval
```bash
rawllm deps pending               # list modules awaiting approval
rawllm deps approve requests      # approve a module
rawllm deps reject requests       # reject a module
```

### Metrics & analytics
```bash
rawllm metrics show                          # all plugins, table format
rawllm metrics show --plugin my_plugin       # one plugin
rawllm metrics show --format json            # JSON output
rawllm metrics evolution my_plugin           # chronological timeline
```

### Configuration
```bash
rawllm config show
rawllm config set LLM_PROVIDER groq
rawllm config set ALLOWED_REQUIREMENTS "json,datetime,requests"
```

## ⚠️ Security Warning

> The statement "plugins run with the same privileges as the orchestrator"
> applies to trusted in-process plugins and the legacy subprocess backend.
> When `SANDBOX_BACKEND=docker` is enabled, untrusted plugins run in a separate
> container with reduced privileges and isolated mounted volumes.
> **Do not load plugins from untrusted sources in a production environment.**
> This project is a research POC — run it only inside a hardened isolated
> environment (sandbox, Docker, VM), and review Docker runtime permissions.

## Architecture

```
rawllm/  (dumb-orchestrator-poc)
├── core/
│   ├── llm/                    # LLM abstraction subpackage
│   │   ├── protocol.py         # LLMClientProtocol structural Protocol
│   │   ├── registry.py         # LLM_PROVIDERS — single source of truth
│   │   ├── factory.py          # get_llm_client(provider) factory
│   │   └── clients/
│   │       ├── anthropic.py    # AnthropicClient
│   │       └── openai_compat.py# OpenAICompatibleClient (Groq, Gemini, Ollama, …)
│   ├── plugin_manager.py       # Plugin loading, hot-reload, versioning, sandbox
│   ├── tool_executor.py        # Tool-call routing + dependency gating
│   ├── taor_loop.py            # Think → Act → Observe → Repeat loop
│   ├── config.py               # Settings: trusted_plugins, allowed_requirements
│   ├── metrics.py              # Event logging to metrics.jsonl
│   ├── sandbox_wrapper.py      # Isolated subprocess wrapper for untrusted plugins
│   └── utils.py                # Shared utilities + extract_imports
├── plugins/
│   └── http.py                 # HTTP transport plugin (port set via HTTP_PORT)
├── plugins_store/              # Versioned plugin storage (created automatically)
│   ├── current/                # Symlinks to active versions
│   └── archive/{name}/         # Previous versions with metrics snapshots
├── cli.py                      # CLI entry point (rawllm)
├── system_prompt.txt           # LLM system prompt
└── run.py                      # Unified entry point (Anthropic / Groq / Gemini / Ollama / …)
```

## License

MIT — use the ideas freely, fork, and improve.

