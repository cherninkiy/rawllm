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

> **Plugins run with the same privileges as the orchestrator process.**
> This is by design for the POC. Do **not** run this system in a production environment
> or with untrusted input without adding additional OS-level sandboxing (e.g., Docker, seccomp, nsjail).

The built-in subprocess sandbox (`core/sandbox_wrapper.py`) provides process isolation but does **not** restrict filesystem or network access within the subprocess.
