# 📋 RawLLM to HolobiontLLM Development Roadmap

This document describes the roadmap for transforming RawLLM into a multi-agent system with self-learning capabilities, aligned with the HolobiontLLM concept.

## 🎯 Strategic Goal

Transform RawLLM from a monolithic orchestrator system into an **evolving multi-agent ecosystem** where:
- Agents form dynamic communication graphs
- The system strategically plans actions via MCTS
- Accumulated experience is used for self-learning (self-play)

---

## 🧱 Phase 1: Core Functionality Extension

**Goal:** Create a foundation for multi-agent interactions with basic intelligent functions.

### 1.1 Enhanced Tool Usage

#### Task 1.1.1: Tool Reranking Implementation
**Files:** `core/tool_executor.py`, `core/llm/protocol.py`
**Description:**
- Add post-processing stage for LLM-selected tools
- Implement scoring mechanism for ranking tools by relevance
- Add confidence threshold for filtering low-priority calls

**API Changes:**
```python
class ToolExecutor:
    def rerank_tools(
        self, 
        tool_calls: List[ToolCall], 
        context: dict
    ) -> List[ToolCall]:
        """Rerank tools based on context and history."""
```

**Priority:** 🔴 High  
**Complexity:** Medium  
**Dependencies:** None
**Status:** ✅ Completed

---

#### Task 1.1.2: Tool Reject Option
**Files:** `core/tool_executor.py`, `core/taor_loop.py`
**Description:**
- Give the system the right to refuse tool execution
- Implement "soft rejection" mechanism with explanation
- Log rejection cases for subsequent analysis

**API Changes:**
```python
class ToolExecutor:
    def reject_tool_call(
        self, 
        tool_call: ToolCall, 
        reason: str
    ) -> RejectionResult:
        """Reject tool execution with justification."""
```

**Priority:** 🔴 High  
**Complexity:** Low  
**Dependencies:** 1.1.1
**Status:** ✅ Completed

---

### 1.2 Self-Reflection and Self-Correction Tools

#### Task 1.2.1: ToolReflection Cycle
**Files:** `core/tool_executor.py`, `core/metrics.py`, new: `core/reflection.py`
**Description:**
- Track tool execution errors
- Automatic generation of corrected requests
- Feedback from API/sandbox for error analysis

**Components:**
```
core/reflection.py
├── ErrorAnalyzer
│   ├── analyze_error(tool_call, result, traceback)
│   └── categorize_error(error_type)
├── CorrectionGenerator
│   ├── generate_correction(error_analysis, original_call)
│   └── validate_correction(proposed_call)
└── ReflectionLoop
    ├── run_reflection_cycle(history, error_context)
    └── log_reflection_event(reflection_data)
```

**Priority:** 🟡 Medium  
**Complexity:** High  
**Dependencies:** 1.1
**Status:** ✅ Completed

---

### 1.3 Context Prompt Repository

#### Task 1.3.1: ContextPromptRepository Subsystem
**Files:** new: `core/context_repository.py`, modification: `core/prompt_builder.py`
**Description:**
- Store prompt templates for various task types
- Extract relevant context via semantic search
- Integrate with ProConSuL-like logic

**Components:**
```
core/context_repository.py
├── ContextPromptRepository
│   ├── store_prompt(template_id, prompt_template, metadata)
│   ├── retrieve_prompts(query, top_k=5)
│   ├── search_by_semantics(embedding_query)
│   └── get_context_for_task(task_type, context_hints)
├── PromptTemplate
│   ├── template: str
│   ├── variables: List[str]
│   └── render(**kwargs) -> str
└── SemanticIndex
    ├── build_index(prompts)
    └── similarity_search(query_vector)
```

**Integration with prompt_builder.py:**
```python
def build_startup_prompt(
    available_resources: dict | None = None,
    user_task: str | None = None,
    context_repository: ContextPromptRepository | None = None,
) -> str:
```

**Priority:** 🟡 Medium  
**Complexity:** Medium  
**Dependencies:** None
**Status:** ✅ Completed

---

### 1.4 Metrics and Evaluation System

#### Task 1.4.1: Extended Event Logging
**Files:** `core/metrics.py`, `core/tool_executor.py`
**Description:**
- Add `success_score` field (0-1) for each event
- Log execution trajectories (sequence of tool calls)
- Evaluate success of multi-step operations

**API Changes:**
```python
def log_execution(...):
    # Add parameters:
    success_score: float,  # 0.0 - 1.0
    trajectory_id: str,    # ID of action sequence
    step_number: int,      # Step number in trajectory
```

**Priority:** 🔴 High  
**Complexity:** Low  
**Dependencies:** None
**Status:** ✅ Completed

---

#### Task 1.4.2: RecVAE for Agent Recommendation
**Files:** new: `core/agents/recommender.py`, modification: `core/metrics.py`
**Description:**
- Use success history to select agent committee composition
- Simple VAE architecture for encoding trajectories
- Recommend optimal agents for new tasks

**Components:**
```
core/agents/recommender.py
├── TrajectoryEncoder
│   ├── encode_trajectory(events_list) -> latent_vector
│   └── decode_vector(latent_vector) -> trajectory_pattern
├── AgentRecommender
│   ├── train_on_history(metrics_events)
│   ├── recommend_agents(task_description, context)
│   └── get_agent_success_rate(agent_id, task_type)
└── CommitteeBuilder
    ├── build_committee(recommendations, constraints)
    └── optimize_committee_composition(candidate_agents)
```

**Priority:** 🟢 Low (for Phase 1)  
**Complexity:** Very High  
**Dependencies:** 1.4.1
**Status:** ✅ Completed (MVP)

---

## 🌿 Phase 2: Advanced Methods (HolobiontLLM)

**Goal:** Implement key principles of the HolobiontLLM concept — strategic planning and self-learning.

### 2.1 Agent Communication Graph

#### Task 2.1.1: Dynamic Routing System
**Files:** new: `core/agent_graph.py`, modification: `core/taor_loop.py`
**Description:**
- Create dynamic agent graph (Planner, Coder, Critic, Executor)
- Route calls on-the-fly depending on task
- Support cyclic dependencies and feedback loops

**Components:**
```
core/agent_graph.py
├── AgentNode
│   ├── agent_id: str
│   ├── role: str  # planner, coder, critic, executor
│   ├── capabilities: List[str]
│   └── current_load: float
├── CommunicationGraph
│   ├── add_agent(agent_node)
│   ├── remove_agent(agent_id)
│   ├── route_request(source_id, target_ids, message)
│   ├── build_dynamic_graph(task_requirements)
│   └── get_optimal_path(start_agent, end_goal)
└── GraphRouter
    ├── dispatch_to_agents(request, graph_config)
    ├── collect_responses(timeout)
    └── aggregate_results(responses)
```

**Integration with taor_loop.py:**
```python
class TAORLoop:
    def __init__(self, ..., agent_graph: CommunicationGraph | None = None):
        self._agent_graph = agent_graph
        
    async def process_request_async(self, ...):
        if self._agent_graph:
            return await self._process_with_graph(...)
        else:
            return await self._process_single(...)
```

**Priority:** 🔴 High (key for Holobiont)  
**Complexity:** Very High  
**Dependencies:** Phase 1 complete
**Status:** ⏳ Planned

---

#### Task 2.1.2: Specialized Agents
**Files:** new: `core/agents/` package
**Description:**
- Planner: complex task decomposition
- Coder: code generation and refactoring
- Critic: result validation, error detection
- Executor: tool execution
- Meta-Agent: past error analysis, coordination

**Structure:**
```
core/agents/
├── __init__.py
├── base_agent.py       # AbstractAgent base class
├── planner.py          # PlannerAgent
├── coder.py            # CoderAgent
├── critic.py           # CriticAgent
├── executor.py         # ExecutorAgent
└── meta_agent.py       # MetaAgent (coordination + learning)
```

**Priority:** 🔴 High  
**Complexity:** High  
**Dependencies:** 2.1.1
**Status:** ⏳ Planned

---

### 2.2 MCTS as Action Planner

#### Task 2.2.1: MCTS Module
**Files:** new: `core/mcts_planner.py`
**Description:**
- Monte Carlo Tree Search for strategic planning
- Build tree of possible agent call sequences
- Evaluate branch promisingness based on simulations

**Components:**
```
core/mcts_planner.py
├── MCTSNode
│   ├── state: AgentGraphState
│   ├── action: AgentCall | None
│   ├── visits: int
│   ├── value: float
│   └── children: Dict[action, MCTSNode]
├── MCTSPlanner
│   ├── select(node) -> node
│   ├── expand(node) -> new_nodes
│   ├── simulate(state) -> reward
│   ├── backpropagate(path, reward)
│   └── plan(initial_state, n_iterations=1000) -> best_action_sequence
└── StateEvaluator
    ├── evaluate_state(state) -> float
    └── heuristic_value(partial_trajectory)
```

**Algorithm:**
```python
def mcts_plan(initial_state, n_iterations=1000):
    root = MCTSNode(initial_state)
    
    for _ in range(n_iterations):
        node = root
        state = initial_state.copy()
        
        # Selection
        while node.is_fully_expanded():
            node = node.select_best_child()
            state = state.apply_action(node.action)
        
        # Expansion
        if not state.is_terminal():
            actions = state.get_legal_actions()
            for action in actions:
                new_state = state.apply_action(action)
                node.add_child(action, new_state)
        
        # Simulation
        reward = simulate_random_rollout(state)
        
        # Backpropagation
        node.backpropagate(reward)
    
    return root.get_best_action()
```

**Priority:** 🟡 Medium  
**Complexity:** Very High  
**Dependencies:** 2.1
**Status:** ⏳ Planned

---

### 2.3 Training Loop

#### Task 2.3.1: Self-play Infrastructure
**Files:** new: `core/training/` package, modification: `core/mcts_planner.py`
**Description:**
- Generate training data from MCTS trajectories
- Update router/Meta-Agent based on successful trajectories
- Iterative improvement of agent selection strategy

**Components:**
```
core/training/
├── __init__.py
├── trajectory_collector.py
│   └── TrajectoryCollector
│       ├── record_mcts_trajectory(trajectory)
│       ├── label_trajectory(success_metric)
│       └── export_training_dataset()
├── model_updater.py
│   └── ModelUpdater
│       ├── update_router_policy(trajectories)
│       ├── update_agent_embeddings(success_patterns)
│       └── save_checkpoint(model_state)
└── self_play_loop.py
    └── SelfPlayLoop
        ├── run_episode() -> trajectory
        ├── evaluate_episode(trajectory) -> reward
        └── train_on_batch(trajectories)
```

**Priority:** 🟢 Low (most complex stage)  
**Complexity:** Extreme  
**Dependencies:** 2.2, all Phase 1 and 2
**Status:** ⏳ Planned

---

## 📊 Priority Summary Table

| # | Task | Priority | Complexity | Time Estimate |
|---|------|----------|------------|---------------|
| 1.1.1 | Tool Reranking | 🔴 High | Medium | 3-5 days |
| 1.1.2 | Reject Option | 🔴 High | Low | 1-2 days |
| 1.2.1 | ToolReflection Cycle | 🟡 Medium | High | 7-10 days |
| 1.3.1 | Context Repository | 🟡 Medium | Medium | 4-6 days |
| 1.4.1 | Extended Metrics | 🔴 High | Low | 2-3 days |
| 1.4.2 | RecVAE Recommender | 🟢 Low | Very High | 14-21 days |
| 2.1.1 | Agent Graph | 🔴 High | Very High | 10-14 days |
| 2.1.2 | Specialized Agents | 🔴 High | High | 7-10 days |
| 2.2.1 | MCTS Planner | 🟡 Medium | Very High | 14-21 days |
| 2.3.1 | Training Loop | 🟢 Low | Extreme | 21-30 days |

---

## 🗺️ Sprint Roadmap

### Sprint 1 (Weeks 1-2): Phase 1 Foundation
- ✅ 1.1.1 Tool Reranking — **Completed**
- ✅ 1.1.2 Reject Option — **Completed**
- ✅ 1.4.1 Extended Metrics — **Completed**

**Deliverable:** Basic tool reranking and rejection system, extended metrics with success_score and trajectory_id. Implemented in `core/tool_management.py` and `core/metrics.py`.

---

### Sprint 2 (Weeks 3-4): Phase 1 Intelligence
- ✅ 1.3.1 Context Repository — **Completed**
- ✅ 1.2.1 ToolReflection Cycle (start) — **Completed**

**Goal:** Context prompt repository and beginning of self-reflection cycle implementation.

**Deliverables:**
- `core/context_repository.py`: ContextPromptRepository with semantic search, PromptTemplate, SemanticIndex
- `core/reflection.py`: ErrorAnalyzer, CorrectionGenerator, ReflectionLoop for automatic error analysis and correction
- `tests/test_context_repository.py`: 21 tests for context repository functionality
- `tests/test_reflection.py`: 30 tests for reflection cycle functionality
- Default templates for code generation, debugging, analysis, documentation, and testing
- Error categorization for 8 error types (SyntaxError, RuntimeError, TimeoutError, etc.)
- Automatic correction generation with try-except wrapping for runtime errors

---

### Sprint 3 (Weeks 5-6): Phase 1 Completion
- ⏳ 1.2.1 ToolReflection Cycle (completion) — **Planned**
- ⏳ Start 2.1.1 Agent Graph (design) — **Planned**

**Goal:** Full ToolReflection implementation and agent communication graph design.

---

### Sprint 4-5 (Weeks 7-10): Phase 2 Core
- ⏳ 2.1.1 Agent Graph (implementation) — **Planned**
- ⏳ 2.1.2 Specialized Agents — **Planned**

**Goal:** Dynamic agent graph and specialized roles (Planner, Coder, Critic, Executor).

---

### Sprint 6-8 (Weeks 11-16): Planning and Learning
- ⏳ 2.2.1 MCTS Planner — **Planned**
- ⏳ 2.3.1 Training Loop (prototype) — **Planned**

**Goal:** MCTS for strategic planning and self-play learning prototype.

---

### Sprint 9+ (Weeks 17+): Optimization and Scaling
- ⏳ 1.4.2 RecVAE (if required) — **Planned**
- ⏳ Polishing, testing, documentation — **Planned**

---

## 🔧 Technical Requirements

### New Dependencies
```txt
# For semantic search in context repository
sentence-transformers>=2.2.0

# For VAE in RecVAE
torch>=2.0.0
scikit-learn>=1.0.0

# For graph visualization (optional)
networkx>=2.8.0
pyvis>=0.3.0
```

### Testing Requirements
- Test coverage ≥80% for all new modules
- Integration tests for multi-agent scenarios
- Load tests for MCTS (planning time verification)

### Monitoring and Observability
- Log all MCTS decisions
- Real-time agent success metrics
- Communication graph visualization

---

## 📈 Success Criteria

### Phase 1 Completion Criteria
- [x] All tools go through reranking
- [x] System can argumentatively reject a tool
- [x] ToolReflection cycle works automatically on errors
- [x] Context Repository provides relevant prompts
- [x] Metrics include success_score and trajectory_id

### Phase 2 Completion Criteria
- [ ] Dynamic graph of 3+ agents assembled and executes task
- [ ] MCTS finds optimal sequence of 5+ steps
- [ ] Self-play improves success rate by 20% after 100 episodes
- [ ] Meta-Agent analyzes errors and suggests corrections

---

## ⚠️ Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|---------|------------|
| MCTS complexity exceeds estimate | High | High | Start with simplified version (depth-limited) |
| Performance degradation | Medium | High | Result caching, asynchronicity |
| Overfitting in self-play | Medium | Medium | Regularization, diversity in simulations |
| Graph debugging complexity | High | Medium | Detailed logging, visualization |

---

## 📚 Additional Materials

- [MemPalace Paper](https://arxiv.org/abs/...) — long-context memory management
- [Claude Code TAOR](https://claude.ai/code) — reference architecture
- [MCTS Survey](https://arxiv.org/abs/...) — comprehensive MCTS review
- [Multi-Agent Systems](https://www.mas-book.org/) — foundational concepts

---

*Document created: 2025*  
*Version: 1.1*  
*Status: Phase 1 in progress (Sprint 2 completed)*
