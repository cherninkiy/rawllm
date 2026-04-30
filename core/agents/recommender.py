"""Lightweight agent recommendation based on trajectory history.

This module intentionally starts with a compact heuristic model that mimics the
planned RecVAE API surface while remaining dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _tokenize(text: str) -> set[str]:
    return {part.strip(".,!?;:()[]{}\"'").lower() for part in text.split() if part.strip()}


@dataclass
class AgentPerformance:
    """Aggregated performance for an agent and task category."""

    agent_id: str
    task_type: str
    success_rate: float
    avg_success_score: float
    avg_latency_ms: float
    sample_count: int


class TrajectoryEncoder:
    """Simple trajectory encoder/decoder.

    Produces a compact latent vector:
    [avg_success_score, avg_latency_norm, average_step_depth, event_count_norm].
    """

    def encode_trajectory(self, events_list: list[dict[str, Any]]) -> list[float]:
        if not events_list:
            return [0.0, 0.0, 0.0, 0.0]

        count = len(events_list)
        avg_success = sum(float(e.get("success_score", 0.0)) for e in events_list) / count
        avg_latency = sum(float(e.get("execution_time_ms", 0.0)) for e in events_list) / count
        avg_step = (
            sum(float(e.get("step_number", 0.0) or 0.0) for e in events_list) / count
        )

        return [
            max(0.0, min(1.0, avg_success)),
            max(0.0, min(1.0, avg_latency / 10000.0)),
            max(0.0, min(1.0, avg_step / 20.0)),
            max(0.0, min(1.0, count / 20.0)),
        ]

    def decode_vector(self, latent_vector: list[float]) -> dict[str, float]:
        padded = (latent_vector + [0.0, 0.0, 0.0, 0.0])[:4]
        return {
            "expected_success_score": max(0.0, min(1.0, float(padded[0]))),
            "expected_latency_ms": max(0.0, float(padded[1])) * 10000.0,
            "expected_step_depth": max(0.0, float(padded[2])) * 20.0,
            "trajectory_density": max(0.0, min(1.0, float(padded[3]))),
        }


class AgentRecommender:
    """Recommend agents using historical success and task similarity."""

    def __init__(self) -> None:
        self._stats: dict[tuple[str, str], AgentPerformance] = {}
        self._trained = False

    def train_on_history(self, metrics_events: list[dict[str, Any]]) -> None:
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for event in metrics_events:
            agent_id = str(event.get("agent_id", "unknown"))
            task_type = str(event.get("task_type", "general"))
            key = (agent_id, task_type)
            grouped.setdefault(
                key,
                {
                    "count": 0.0,
                    "success_sum": 0.0,
                    "success_bool_sum": 0.0,
                    "latency_sum": 0.0,
                },
            )
            g = grouped[key]
            score = float(event.get("success_score", 1.0 if event.get("success") else 0.0))
            g["count"] += 1.0
            g["success_sum"] += score
            g["success_bool_sum"] += 1.0 if score >= 0.5 else 0.0
            g["latency_sum"] += float(event.get("execution_time_ms", 0.0))

        self._stats = {}
        for (agent_id, task_type), g in grouped.items():
            count = max(1, int(g["count"]))
            self._stats[(agent_id, task_type)] = AgentPerformance(
                agent_id=agent_id,
                task_type=task_type,
                success_rate=g["success_bool_sum"] / count,
                avg_success_score=g["success_sum"] / count,
                avg_latency_ms=g["latency_sum"] / count,
                sample_count=count,
            )
        self._trained = True

    def recommend_agents(
        self,
        task_description: str,
        context: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not self._trained or not self._stats:
            return []

        context = context or {}
        target_task_type = str(context.get("task_type", "general"))
        target_tokens = _tokenize(f"{target_task_type} {task_description}")

        scored: list[tuple[float, AgentPerformance]] = []
        for perf in self._stats.values():
            type_tokens = _tokenize(perf.task_type)
            overlap = len(target_tokens & type_tokens)
            similarity = overlap / max(1, len(target_tokens))

            # Higher success and lower latency are preferred; sample_count adds confidence.
            confidence = min(1.0, perf.sample_count / 10.0)
            latency_bonus = 1.0 / (1.0 + perf.avg_latency_ms / 1000.0)
            final_score = (
                perf.avg_success_score * 0.6
                + perf.success_rate * 0.2
                + similarity * 0.1
                + latency_bonus * 0.1
            ) * (0.7 + 0.3 * confidence)
            scored.append((final_score, perf))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "agent_id": perf.agent_id,
                "task_type": perf.task_type,
                "score": round(score, 4),
                "success_rate": round(perf.success_rate, 4),
                "avg_success_score": round(perf.avg_success_score, 4),
                "avg_latency_ms": round(perf.avg_latency_ms, 2),
                "sample_count": perf.sample_count,
            }
            for score, perf in scored[: max(0, top_k)]
        ]

    def get_agent_success_rate(self, agent_id: str, task_type: str = "general") -> float:
        perf = self._stats.get((agent_id, task_type))
        if perf:
            return perf.success_rate

        # Fallback to overall rate for the agent across all task types.
        matches = [p for p in self._stats.values() if p.agent_id == agent_id]
        if not matches:
            return 0.0
        return sum(p.success_rate for p in matches) / len(matches)


class CommitteeBuilder:
    """Build and optimize an execution committee from recommendations."""

    def build_committee(
        self,
        recommendations: list[dict[str, Any]],
        constraints: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        constraints = constraints or {}
        max_members = int(constraints.get("max_members", 3))
        min_score = float(constraints.get("min_score", 0.0))
        filtered = [item for item in recommendations if float(item.get("score", 0.0)) >= min_score]
        return filtered[: max(0, max_members)]

    def optimize_committee_composition(
        self,
        candidate_agents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Encourage task-type diversity while keeping high-ranked agents first.
        chosen: list[dict[str, Any]] = []
        seen_task_types: set[str] = set()
        for agent in candidate_agents:
            task_type = str(agent.get("task_type", "general"))
            if task_type not in seen_task_types:
                chosen.append(agent)
                seen_task_types.add(task_type)

        # Add remaining candidates to fill committee capacity.
        for agent in candidate_agents:
            if agent not in chosen:
                chosen.append(agent)
        return chosen
