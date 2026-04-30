# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Phase 1 intelligence modules** – added `core/context_repository.py` with `ContextPromptRepository`, `PromptTemplate`, and `SemanticIndex`; added `core/reflection.py` with `ErrorAnalyzer`, `CorrectionGenerator`, and `ReflectionLoop`.
- **Initial agent recommender module** – added `core/agents/recommender.py` with `TrajectoryEncoder`, `AgentRecommender`, and `CommitteeBuilder` to support Phase 1 recommendation workflows.
- **Test coverage for new modules** – added `tests/test_context_repository.py` and `tests/test_reflection.py`.

### Changed
- **Context repository indexing flow** – `store_prompt()` now supports deferred index rebuilds via `rebuild_index=False`, and default template initialization performs a single rebuild after batch insert.
- **Metrics enrichment for recommendations** – `log_execution()` now supports optional `task_type`, and `core.metrics` includes normalized execution history helpers for recommendation training.

### Fixed
- **Runtime correction wrapper** – fixed `_wrap_with_error_handling()` indentation in `core/reflection.py` so wrapped code is valid Python under `try`.
- **Error pattern matching efficiency** – error patterns are normalized once (lowercased) and matched without repeated per-iteration `.lower()` calls.
- **Timestamp import style** – replaced inline `__import__("datetime")` call with standard module-level `datetime` import.
- **Lint and ignore file cleanup** – removed markdown fence artifacts from `.gitignore`, removed unused locals/imports, and cleaned trailing-whitespace lines to keep CI (`flake8`) green.

---

---

## [0.2.0] – 2026-04-21

### Added
- **Resource-aware runtime** – operator passes available ports, workspace path, and external services at startup via `--ports`, `--workspace`, `--services` CLI flags (also readable from env vars `RAWLLM_PORTS`, `RAWLLM_WORKSPACE`, `RAWLLM_SERVICES`).
- **Plugin resource manifests** – `add_plugin` accepts an optional `manifest` object with `requires` and `publishes` sections; the runtime validates, assigns, and persists resource assignments in `plugins_store/resource_assignments.json`.
- **Environment injection** – assigned resources are exposed inside each plugin's sandbox as `PORT_<N>`, `WORKSPACE_PATH`, and `<SERVICE>_URI` environment variables.
- **Startup prompt builder** (`core/prompt_builder.py`) – assembles the initial LLM message from available resources and an optional `--prompt` task string.
- **Autonomous bootstrap** – orchestrator starts with no preloaded plugins and fires an initial request so the model decides what interface to create.
- **`resources` CLI group** – `rawllm resources list` and `rawllm resources show <name>` expose persisted plugin resource assignments.
- **`startup_prompt` parameter on `TAORLoop`** – fallback prompt used when `process_request()` is called without a user message.

### Changed
- `TAORLoop.__init__` now requires `startup_prompt` as a positional argument after `system_prompt` (**breaking**).
- `run.py` no longer calls `plugin_manager.load_plugins()` on startup; the model bootstraps its own plugins each session (**breaking**).
- `unload_plugin("http")` is no longer blocked — hard-coded `PROTECTED_PLUGINS` guard removed.
- `system_prompt.txt` rewritten to reflect the dumb-orchestrator model and document manifest format and env variable conventions.
- CI workflow now triggers on `main` and `dev` branches (removed stale `core/architecture` branch filter).

### Fixed
- `process_request(user_prompt="")` no longer fell through to `startup_prompt` due to falsy `or`-check; replaced with explicit `is not None` guard.
- Reversed port range (e.g. `8002-8000`) now raises `"Invalid port range"` instead of the generic `"Invalid port value"`.
- Non-contiguous ports (e.g. `8000, 8005, 8080`) are now displayed individually in the startup prompt instead of as a misleading `8000-8080` range.
- `add_plugin` manifest resolution and `_resource_assignments` write are now atomic under `self._lock`, preventing TOCTOU when two `add_plugin_async` calls race for the same port.
- `_get_plugin_env()` returns only resource-specific overrides (`PORT_*`, `WORKSPACE_PATH`, `*_URI`) instead of a full `os.environ` copy, preventing API key leakage into untrusted sandboxes.
- `_save_resource_assignments()` on the `add_plugin` success path now runs under `self._lock`.
- `run.py` workspace parsing now uses `config._parse_workspace()` consistently with the rest of config parsing.
- `PLUGINS_DIR` in `cli.py` is now resolved at call time via `_get_plugins_dir()` so `.env`-configured values take effect after `_load_dotenv_if_present()` runs.
- Ports outside the valid TCP/UDP range (1–65535) now raise `ValueError` immediately.

[0.2.0]: https://github.com/cherninkiy/rawllm/compare/v0.1.0...v0.2.0

## [0.1.0] – 2026-04-20

### Added
- **TAOR loop** – synchronous and asynchronous Think-Act-Observe-Repeat cycle with configurable iteration limit.
- **Multi-provider LLM support** – Anthropic, Groq, Gemini, OpenRouter, DeepSeek, Ollama, and Ollama-Qwen-Coder via a unified `LLMClientProtocol`.
- **Plugin manager** – hot-reload, versioning, archiving, and `rollback_plugin`; thread-safe with `RLock`.
- **Parallel tool execution** – `run_plugins_parallel` dispatches independent plugin calls concurrently via `asyncio.gather`.
- **Sandbox backends** – subprocess (default) and Docker (RO root-fs, `--network none`, non-root user isolation).
- **Dependency gating** – AST-based import parsing with an allow-list; unapproved imports are written to `pending_requirements.txt` instead of executing.
- **Metrics** – append-only JSONL log (`metrics.jsonl`) with aggregation commands in the CLI.
- **CLI** (`rawllm`) – `run`, `plugin`, `deps`, `metrics`, and `config` sub-commands.
- **CI** – GitHub Actions pipeline: pytest (coverage ≥ 90 %), flake8, mypy.
- **Documentation** – full README with quickstart, Docker instructions, all providers, CLI reference, and architecture diagram; CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, SUPPORT.

### Notes
- `rawllm plugin rollback --version <N>` is not yet implemented; the command currently rolls back to the most recently archived version regardless of the `--version` flag.

[0.1.0]: https://github.com/cherninkiy/rawllm/releases/tag/v0.1.0
