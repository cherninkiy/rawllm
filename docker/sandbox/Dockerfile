FROM python:3.11-slim

RUN groupadd --gid 10001 rawllm-core \
    && useradd --uid 10001 --gid 10001 --create-home rawllm-core \
    && groupadd --gid 10002 rawllm-plugin \
    && useradd --uid 10002 --gid 10002 --create-home rawllm-plugin

WORKDIR /workspace
USER rawllm-core
