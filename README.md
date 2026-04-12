# dumb-orchestrator-poc
Minimal POC of "dumb orchestrator – smart model". LLM evolves itself by writing plugins (add_plugin/run_plugin). Inspired by MemPalace (96.6% on LongMemEval) and Claude Code's TAOR. Core is immutable (&lt;150 loc). HTTP transport as a plugin.


# Dumb Orchestrator – Smart Model

**Глупый оркестратор, умная модель**  
Минималистичный POC агента, который эволюционирует через плагины, создаваемые самой LLM.

## Идея

Вместо того чтобы зашивать сложность в код, мы даём модели всего два инструмента:
- `add_plugin(name, code)` — написать и сохранить новый плагин
- `run_plugin(name, input_data)` — вызвать существующий плагин

Ядро (оркестратор) — неизменяемый, «глупый» (~150 строк). Вся интеллектуальная работа, включая создание новых возможностей, координацию агентов, память, парсинг данных, — ложится на LLM. Модель сама решает, когда и какой плагин написать, и может перезаписывать их на лету (горячая замена).

## Вдохновение

- **MemPalace / Милла Йовович** — подход, который разнёс LongMemEval (96.6%) без сложного RAG, просто дав модели сырые данные и свободу.
- **Claude Code** — архитектура TAOR (Think‑Act‑Observe‑Repeat) и принцип «глупый оркестратор».
- **Критика переусложнённых RAG‑пайплайнов** — даём модели чистый контекст и право решать.

## Статус

✅ **Реализовано** — ядро, HTTP‑плагин, тесты и CI доступны в ветке [`copilot/core-architecture`](https://github.com/cherninkiy/dumb-orchestrator-poc/tree/copilot/core-architecture).

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Создать .env с ключом Anthropic
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Запустить оркестратор (HTTP-сервер на порту 8080)
python run.py

# 4. Отправить запрос
curl -X POST http://localhost:8080/ \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Напиши плагин, который возвращает текущее время", "context": {}}'
```

Порт сервера можно переопределить через переменную окружения `HTTP_PORT`.

## Running with Free / Lightweight LLMs

`run.py` supports any OpenAI-compatible provider via the `LLM_PROVIDER`
environment variable (default: `anthropic`).  All Part-A security and versioning
enhancements are automatically active.

### Supported providers

| `LLM_PROVIDER` | API key env var | Default model |
|---|---|---|
| `anthropic` *(default)* | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20241022` |
| `groq` | `GROQ_API_KEY` | `llama3-70b-8192` |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| `openrouter` | `OPENROUTER_API_KEY` | `mistralai/mistral-7b-instruct:free` |
| `ollama` | *(none required)* | `llama3.2:3b` |

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
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
LLM_PROVIDER=openrouter python run.py
# Use a specific free model:
LLM_PROVIDER=openrouter LLM_MODEL=google/gemma-3-27b-it:free python run.py
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



> **Плагины выполняются с теми же привилегиями, что и сам оркестратор.**  
> Код плагина имеет полный доступ к файловой системе, сети и переменным окружения процесса.  
> **Не запускайте плагины из ненадёжных источников в продакшн-среде.**  
> Этот проект является исследовательским POC — запускайте его только в изолированном окружении (sandbox, Docker, VM).

> **Plugins run with the same privileges as the orchestrator.**  
> Plugin code has unrestricted access to the filesystem, network, and process environment.  
> **Do not load plugins from untrusted sources in a production environment.**  
> This project is a research POC — run it only inside an isolated environment (sandbox, Docker, VM).

## Архитектура

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
│   ├── plugin_manager.py       # Загрузка/горячая замена плагинов, версионирование, sandbox
│   ├── tool_executor.py        # Маршрутизация вызовов инструментов + проверка зависимостей
│   ├── taor_loop.py            # Цикл Think→Act→Observe→Repeat
│   ├── config.py               # Настройки: trusted_plugins, allowed_requirements
│   ├── metrics.py              # Логирование событий в metrics.jsonl
│   ├── sandbox_wrapper.py      # Обёртка для изолированного запуска плагинов в subprocess
│   └── utils.py                # Вспомогательные утилиты + extract_imports
├── plugins/
│   └── http.py                 # HTTP-транспорт (порт задаётся через HTTP_PORT)
├── plugins_store/              # Версионированное хранилище плагинов (создаётся автоматически)
│   ├── current/                # Символические ссылки на активные версии
│   └── archive/{name}/         # Архив предыдущих версий с метриками
├── cli.py                      # CLI entry point (rawllm)
├── system_prompt.txt           # Системный промпт для LLM
└── run.py                      # Единая точка входа (Anthropic / Groq / Gemini / Ollama / …)
```

## Лицензия

MIT — свободно используйте идеи, форкайте, улучшайте.

---

*Pull Request: [#1 feat: implement Dumb Orchestrator – Smart Model POC](https://github.com/cherninkiy/dumb-orchestrator-poc/pull/1)*