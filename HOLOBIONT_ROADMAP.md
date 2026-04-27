# 📋 План развития RawLLM → HolobiontLLM

Этот документ описывает дорожную карту трансформации RawLLM в multi-agent систему с элементами самообучения, соответствующую концепции HolobiontLLM.

## 🎯 Стратегическая цель

Превратить RawLLM из单体ной orchestrator-системы в **эволюционирующую multi-agent экосистему**, где:
- Агенты формируют динамические коммуникационные графы
- Система стратегически планирует действия через MCTS
- Накопленный опыт используется для самообучения (self-play)

---

## 🧱 Фаза 1: Базовое расширение функциональности

**Цель:** Создать фундамент для multi-agent взаимодействий с базовыми интеллектуальными функциями.

### 1.1 Расширенное управление инструментами (Tool Usage)

#### Задача 1.1.1: Внедрение реранжинга инструментов
**Файлы:** `core/tool_executor.py`, `core/llm/protocol.py`
**Описание:**
- Добавить этап пост-обработки выбранных LLM инструментов
- Реализовать scoring-механизм для ранжирования инструментов по релевантности
- Добавить порог уверенности для отсеивания низкоприоритетных вызовов

**API изменения:**
```python
class ToolExecutor:
    def rerank_tools(
        self, 
        tool_calls: List[ToolCall], 
        context: dict
    ) -> List[ToolCall]:
        """Реранжинг инструментов на основе контекста и истории."""
```

**Приоритет:** 🔴 Высокий  
**Сложность:** Средняя  
**Зависимости:** Нет
**Статус:** ✅ Выполнено

---

#### Задача 1.1.2: Reject option для инструментов
**Файлы:** `core/tool_executor.py`, `core/taor_loop.py`
**Описание:**
- Дать системе право отказаться от выполнения инструмента
- Реализовать механизм "мягкого отказа" с объяснением причины
- Логировать случаи отказа для последующего анализа

**API изменения:**
```python
class ToolExecutor:
    def reject_tool_call(
        self, 
        tool_call: ToolCall, 
        reason: str
    ) -> RejectionResult:
        """Отказ от выполнения инструмента с обоснованием."""
```

**Приоритет:** 🔴 Высокий  
**Сложность:** Низкая  
**Зависимости:** 1.1.1
**Статус:** ✅ Выполнено

---

### 1.2 Инструменты саморефлексии и самокоррекции

#### Задача 1.2.1: Цикл ToolReflection
**Файлы:** `core/tool_executor.py`, `core/metrics.py`, новое: `core/reflection.py`
**Описание:**
- Отслеживание ошибок выполнения инструментов
- Автоматическая генерация исправленных запросов
- Обратная связь от API/песочницы для анализа ошибок

**Компоненты:**
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

**Приоритет:** 🟡 Средний  
**Сложность:** Высокая  
**Зависимости:** 1.1
**Статус:** ⏳ В работе

---

### 1.3 Депозиторий контекстных промптов (Context Prompting)

#### Задача 1.3.1: Подсистема ContextPromptRepository
**Файлы:** новое: `core/context_repository.py`, модификация: `core/prompt_builder.py`
**Описание:**
- Хранение шаблонов промптов для различных типов задач
- Извлечение релевантного контекста по семантическому поиску
- Интеграция с ProConSuL-подобной логикой

**Компоненты:**
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

**Интеграция с prompt_builder.py:**
```python
def build_startup_prompt(
    available_resources: dict | None = None,
    user_task: str | None = None,
    context_repository: ContextPromptRepository | None = None,
) -> str:
```

**Приоритет:** 🟡 Средний  
**Сложность:** Средняя  
**Зависимости:** Нет
**Статус:** ⏳ Запланировано

---

### 1.4 Система метрик и оценок

#### Задача 1.4.1: Расширение журнала событий
**Файлы:** `core/metrics.py`, `core/tool_executor.py`
**Описание:**
- Добавление поля `success_score` (0-1) для каждого события
- Логирование траекторий выполнения (sequence of tool calls)
- Оценка успешности multi-step операций

**API изменения:**
```python
def log_execution(...):
    # Добавить параметры:
    success_score: float,  # 0.0 - 1.0
    trajectory_id: str,    # ID последовательности действий
    step_number: int,      # Номер шага в траектории
```

**Приоритет:** 🔴 Высокий  
**Сложность:** Низкая  
**Зависимости:** Нет
**Статус:** ✅ Выполнено

---

#### Задача 1.4.2: RecVAE для рекомендации агентов
**Файлы:** новое: `core/agent_recommender.py`, модификация: `core/metrics.py`
**Описание:**
- Использование истории успехов для подбора состава комитета агентов
- Простая VAE-архитектура для кодирования траекторий
- Рекомендация оптимальных агентов для новых задач

**Компоненты:**
```
core/agent_recommender.py
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

**Приоритет:** 🟢 Низкий (для Фазы 1)  
**Сложность:** Очень высокая  
**Зависимости:** 1.4.1
**Статус:** ⏳ Запланировано

---

## 🌿 Фаза 2: Продвинутые методы (HolobiontLLM)

**Цель:** Реализация ключевых принципов концепции HolobiontLLM — стратегическое планирование и самообучение.

### 2.1 Коммуникационный граф агентов

#### Задача 2.1.1: Динамическая система маршрутизации
**Файлы:** новое: `core/agent_graph.py`, модификация: `core/taor_loop.py`
**Описание:**
- Создание динамического графа агентов (Planner, Coder, Critic, Executor)
- Маршрутизация вызовов на лету в зависимости от задачи
- Поддержка циклических зависимостей и обратной связи

**Компоненты:**
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

**Интеграция с taor_loop.py:**
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

**Приоритет:** 🔴 Высокий (ключевой для Holobiont)  
**Сложность:** Очень высокая  
**Зависимости:** Фаза 1 полностью
**Статус:** ⏳ Запланировано

---

#### Задача 2.1.2: Специализированные агенты
**Файлы:** новое: `core/agents/` пакет
**Описание:**
- Planner: декомпозиция сложных задач
- Coder: генерация и рефакторинг кода
- Critic: валидация результатов, поиск ошибок
- Executor: выполнение инструментов
- Meta-Agent: анализ прошлых ошибок, координация

**Структура:**
```
core/agents/
├── __init__.py
├── base_agent.py       # AbstractAgent基类
├── planner.py          # PlannerAgent
├── coder.py            # CoderAgent
├── critic.py           # CriticAgent
├── executor.py         # ExecutorAgent
└── meta_agent.py       # MetaAgent (координация + обучение)
```

**Приоритет:** 🔴 Высокий  
**Сложность:** Высокая  
**Зависимости:** 2.1.1
**Статус:** ⏳ Запланировано

---

### 2.2 MCTS как планировщик действий

#### Задача 2.2.1: Модуль MCTS
**Файлы:** новое: `core/mcts_planner.py`
**Описание:**
- Monte Carlo Tree Search для стратегического планирования
- Построение дерева возможных последовательностей агентов
- Оценка перспективности веток на основе симуляций

**Компоненты:**
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

**Алгоритм:**
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

**Приоритет:** 🟡 Средний  
**Сложность:** Очень высокая  
**Зависимости:** 2.1
**Статус:** ⏳ Запланировано

---

### 2.3 Цикл обучения (Training Loop)

#### Задача 2.3.1: Self-play инфраструктура
**Файлы:** новое: `core/training/` пакет, модификация: `core/mcts_planner.py`
**Описание:**
- Генерация тренировочных данных из траекторий MCTS
- Обновление роутера/Meta-Agent на основе успешных траекторий
- Итеративное улучшение стратегии выбора агентов

**Компоненты:**
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

**Приоритет:** 🟢 Низкий (самый сложный этап)  
**Сложность:** Экстремальная  
**Зависимости:** 2.2, вся Фаза 1 и 2
**Статус:** ⏳ Запланировано

---

## 📊 Сводная таблица приоритетов

| № | Задача | Приоритет | Сложность | Оценка времени |
|---|--------|-----------|-----------|----------------|
| 1.1.1 | Реранжинг инструментов | 🔴 Высокий | Средняя | 3-5 дней |
| 1.1.2 | Reject option | 🔴 Высокий | Низкая | 1-2 дня |
| 1.2.1 | ToolReflection цикл | 🟡 Средний | Высокая | 7-10 дней |
| 1.3.1 | Context Repository | 🟡 Средний | Средняя | 4-6 дней |
| 1.4.1 | Расширение метрик | 🔴 Высокий | Низкая | 2-3 дня |
| 1.4.2 | RecVAE рекомендатель | 🟢 Низкий | Очень высокая | 14-21 день |
| 2.1.1 | Agent Graph | 🔴 Высокий | Очень высокая | 10-14 дней |
| 2.1.2 | Специализированные агенты | 🔴 Высокий | Высокая | 7-10 дней |
| 2.2.1 | MCTS планировщик | 🟡 Средний | Очень высокая | 14-21 день |
| 2.3.1 | Training Loop | 🟢 Низкий | Экстремальная | 21-30 дней |

---

## 🗺️ Дорожная карта по спринтам

### Спринт 1 (Недели 1-2): Фундамент Фазы 1
- ✅ 1.1.1 Реранжинг инструментов — **Выполнено**
- ✅ 1.1.2 Reject option — **Выполнено**
- ✅ 1.4.1 Расширение метрик — **Выполнено**

**Результат:** Базовая система реранжинга и отказов от инструментов, расширенные метрики с success_score и trajectory_id. Реализовано в `core/tool_management.py` и `core/metrics.py`.

---

### Спринт 2 (Недели 3-4): Интеллект Фазы 1
- ⏳ 1.3.1 Context Repository — **Запланировано**
- ⏳ 1.2.1 ToolReflection цикл (начало) — **В работе**

**Цель:** Депозиторий контекстных промптов и начало реализации цикла саморефлексии.

---

### Спринт 3 (Недели 5-6): Завершение Фазы 1
- ⏳ 1.2.1 ToolReflection цикл (завершение) — **Запланировано**
- ⏳ Начало 2.1.1 Agent Graph (проектирование) — **Запланировано**

**Цель:** Полная реализация ToolReflection и проектирование коммуникационного графа агентов.

---

### Спринт 4-5 (Недели 7-10): Ядро Фазы 2
- ⏳ 2.1.1 Agent Graph (реализация) — **Запланировано**
- ⏳ 2.1.2 Специализированные агенты — **Запланировано**

**Цель:** Динамический граф агентов и специализированные роли (Planner, Coder, Critic, Executor).

---

### Спринт 6-8 (Недели 11-16): Планирование и обучение
- ⏳ 2.2.1 MCTS планировщик — **Запланировано**
- ⏳ 2.3.1 Training Loop (прототип) — **Запланировано**

**Цель:** MCTS для стратегического планирования и прототип self-play обучения.

---

### Спринт 9+ (Недели 17+): Оптимизация и масштабирование
- ⏳ 1.4.2 RecVAE (если требуется) — **Запланировано**
- ⏳ Полировка, тесты, документация — **Запланировано**

---

## 🔧 Технические требования

### Новые зависимости
```txt
# Для semantic search в context repository
sentence-transformers>=2.2.0

# Для VAE в RecVAE
torch>=2.0.0
scikit-learn>=1.0.0

# Для визуализации графов (опционально)
networkx>=2.8.0
pyvis>=0.3.0
```

### Требования к тестированию
- Покрытие тестами ≥80% для всех новых модулей
- Интеграционные тесты для multi-agent сценариев
- Нагрузочные тесты для MCTS (проверка времени планирования)

### Мониторинг и observability
- Логирование всех решений MCTS
- Метрики успешности агентов в реальном времени
- Визуализация коммуникационного графа

---

## 📈 Критерии успеха

### Критерии завершения Фазы 1
- [x] Все инструменты проходят через реранжинг
- [x] Система может аргументированно отказаться от инструмента
- [ ] Cycle ToolReflection работает автоматически при ошибках
- [ ] Context Repository предоставляет релевантные промпты
- [x] Метрики включают success_score и trajectory_id

### Критерии завершения Фазы 2
- [ ] Динамический граф из 3+ агентов собран и выполняет задачу
- [ ] MCTS находит оптимальную последовательность из 5+ шагов
- [ ] Self-play улучшает success rate на 20% после 100 эпизодов
- [ ] Meta-Agent анализирует ошибки и предлагает коррекции

---

## ⚠️ Риски и митигация

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Сложность MCTS превысит оценку | Высокая | Высокое | Начать с упрощённой версии (depth-limited) |
| Performance degradation | Средняя | Высокое | Кэширование результатов, асинхронность |
| Overfitting в self-play | Средняя | Среднее | Регуляризация, разнообразие в симуляциях |
| Сложность отладки графа | Высокая | Среднее | Детальное логирование, визуализация |

---

## 📚 Дополнительные материалы

- [MemPalace Paper](https://arxiv.org/abs/...) — long-context memory management
- [Claude Code TAOR](https://claude.ai/code) — reference architecture
- [MCTS Survey](https://arxiv.org/abs/...) — comprehensive MCTS review
- [Multi-Agent Systems](https://www.mas-book.org/) — foundational concepts

---

*Документ создан: 2025*  
*Версия: 1.0*  
*Статус: Планирование*
