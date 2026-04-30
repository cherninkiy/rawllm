"""Tests for core.agents.recommender."""

from core.agents.recommender import AgentRecommender, CommitteeBuilder, TrajectoryEncoder


def test_trajectory_encoder_roundtrip() -> None:
    encoder = TrajectoryEncoder()
    events = [
        {"success_score": 0.9, "execution_time_ms": 100, "step_number": 1},
        {"success_score": 0.7, "execution_time_ms": 300, "step_number": 2},
    ]
    vector = encoder.encode_trajectory(events)
    decoded = encoder.decode_vector(vector)
    assert len(vector) == 4
    assert decoded["expected_success_score"] > 0.0
    assert decoded["expected_latency_ms"] > 0.0


def test_recommender_prefers_higher_score_agent() -> None:
    recommender = AgentRecommender()
    history = [
        {"agent_id": "coder_a", "task_type": "python", "success_score": 0.9, "execution_time_ms": 150},
        {"agent_id": "coder_a", "task_type": "python", "success_score": 0.8, "execution_time_ms": 180},
        {"agent_id": "coder_b", "task_type": "python", "success_score": 0.4, "execution_time_ms": 120},
    ]
    recommender.train_on_history(history)
    recs = recommender.recommend_agents("write parser", {"task_type": "python"}, top_k=2)
    assert len(recs) == 2
    assert recs[0]["agent_id"] == "coder_a"
    assert recs[0]["score"] >= recs[1]["score"]


def test_get_agent_success_rate_with_fallback() -> None:
    recommender = AgentRecommender()
    history = [
        {"agent_id": "planner", "task_type": "analysis", "success_score": 1.0},
        {"agent_id": "planner", "task_type": "debug", "success_score": 0.0},
    ]
    recommender.train_on_history(history)
    assert recommender.get_agent_success_rate("planner", "analysis") == 1.0
    assert 0.0 <= recommender.get_agent_success_rate("planner", "unknown-task") <= 1.0


def test_committee_builder_respects_constraints_and_diversity() -> None:
    builder = CommitteeBuilder()
    recommendations = [
        {"agent_id": "a", "task_type": "python", "score": 0.9},
        {"agent_id": "b", "task_type": "python", "score": 0.8},
        {"agent_id": "c", "task_type": "debug", "score": 0.85},
    ]
    committee = builder.build_committee(
        recommendations,
        constraints={"max_members": 2, "min_score": 0.81},
    )
    assert len(committee) == 2
    optimized = builder.optimize_committee_composition(committee)
    assert optimized[0]["task_type"] in {"python", "debug"}
