# Security Policy

## Supported Versions

This project is a research proof-of-concept. Only the latest commit on the default branch (`main`) and the active development branch receive security attention.

| Branch | Supported |
|---|---|
| `main` | :white_check_mark: |
| `core/architecture` | :white_check_mark: |
| Older branches | :x: |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

To report a vulnerability, please:

1. Open a [private security advisory](https://github.com/cherninkiy/dumb-orchestrator-poc/security/advisories/new) on GitHub (preferred).
2. Or send an email to the repository owner found on their [GitHub profile](https://github.com/cherninkiy).

Include as much of the following information as possible:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Affected versions / branches
- Any suggested fix or mitigation

You can expect an acknowledgement within **72 hours** and a status update within **7 days**.

## Known Security Considerations

RawLLM supports two plugin sandbox backends:

1. `SANDBOX_BACKEND=subprocess` (legacy)
2. `SANDBOX_BACKEND=docker` (recommended)

When docker backend is enabled, untrusted plugins run as `rawllm-plugin` with:

- read-only root filesystem
- network disabled (`--network none`)
- dropped capabilities + `no-new-privileges`
- isolated volumes only:
  - workspace (rw)
  - core_repo snapshot (ro)
  - plugin_store snapshot (ro)

The orchestrator process remains under `rawllm-core`.

Residual risk: this project is still a research POC and should be deployed only
in controlled environments. Validate your container runtime hardening settings
and image supply chain before production use.
