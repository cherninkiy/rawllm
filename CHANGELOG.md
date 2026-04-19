# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

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

[0.1.0]: https://github.com/cherninkiy/dumb-orchestrator-poc/releases/tag/v0.1.0
