"""Tests for the Context Prompt Repository (Sprint 2 - Task 1.3.1)."""

import pytest
from core.context_repository import (
    ContextPromptRepository,
    PromptTemplate,
    SemanticIndex,
    get_repository,
)


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_template_creation(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            template_id="test_template",
            template="Hello {name}, welcome to {place}!",
            description="A greeting template",
            variables=["name", "place"],
            tags=["greeting", "welcome"],
        )

        assert template.template_id == "test_template"
        assert template.description == "A greeting template"
        assert len(template.variables) == 2
        assert len(template.tags) == 2
        assert template.usage_count == 0
        assert template.success_rate == 0.5

    def test_template_rendering(self):
        """Test rendering a template with variables."""
        template = PromptTemplate(
            template_id="test",
            template="Calculate {a} + {b} = {result}",
            variables=["a", "b", "result"],
        )

        rendered = template.render(a=5, b=3, result=8)
        assert rendered == "Calculate 5 + 3 = 8"
        assert template.usage_count == 1

    def test_template_rendering_missing_variable(self):
        """Test that missing variables raise ValueError."""
        template = PromptTemplate(
            template_id="test",
            template="Hello {name}!",
            variables=["name"],
        )

        with pytest.raises(ValueError, match="Missing required variable"):
            template.render()

    def test_template_to_dict(self):
        """Test converting template to dictionary."""
        template = PromptTemplate(
            template_id="test",
            template="Test {value}",
            description="Test template",
            variables=["value"],
            tags=["test"],
            metadata={"key": "value"},
        )

        data = template.to_dict()
        assert data["template_id"] == "test"
        assert data["template"] == "Test {value}"
        assert data["description"] == "Test template"
        assert data["variables"] == ["value"]
        assert data["tags"] == ["test"]
        assert data["metadata"] == {"key": "value"}

    def test_template_from_dict(self):
        """Test creating template from dictionary."""
        data = {
            "template_id": "restored",
            "template": "Restored {item}",
            "description": "Restored template",
            "variables": ["item"],
            "tags": ["restore"],
            "metadata": {},
            "usage_count": 5,
            "success_rate": 0.8,
        }

        template = PromptTemplate.from_dict(data)
        assert template.template_id == "restored"
        assert template.usage_count == 5
        assert template.success_rate == 0.8


class TestSemanticIndex:
    """Tests for SemanticIndex class."""

    def test_build_index(self):
        """Test building a semantic index."""
        index = SemanticIndex()
        prompts = [
            PromptTemplate(
                template_id=f"template_{i}",
                template=f"Template {i} for testing",
                tags=["test", f"tag{i}"],
            )
            for i in range(5)
        ]

        index.build_index(prompts)
        assert index._built is True
        assert len(index._index) == 5

    def test_similarity_search(self):
        """Test semantic similarity search."""
        index = SemanticIndex()
        prompts = [
            PromptTemplate(
                template_id="python_code",
                template="Generate Python code for {task}",
                description="Python code generation",
                tags=["code", "python"],
            ),
            PromptTemplate(
                template_id="javascript_code",
                template="Generate JavaScript code for {task}",
                description="JavaScript code generation",
                tags=["code", "javascript"],
            ),
            PromptTemplate(
                template_id="debug_error",
                template="Debug this error: {error}",
                description="Error debugging",
                tags=["debug", "error"],
            ),
        ]

        index.build_index(prompts)

        # Search for Python-related templates
        results = index.similarity_search("python code generation", top_k=2)
        assert len(results) <= 2
        assert results[0][0] == "python_code"

    def test_similarity_search_empty_index(self):
        """Test search on unbuilt index returns empty."""
        index = SemanticIndex()
        results = index.similarity_search("test query")
        assert results == []

    def test_keyword_extraction(self):
        """Test keyword extraction removes stopwords."""
        index = SemanticIndex()
        text = "The quick brown fox jumps over the lazy dog"
        keywords = index._extract_keywords(text)

        # Should not contain common stopwords
        assert "the" not in keywords

        # Should contain meaningful words (at least some of them)
        assert len(keywords) > 0
        assert any(word in keywords for word in ["quick", "brown", "fox", "jumps", "lazy", "dog"])


class TestContextPromptRepository:
    """Tests for ContextPromptRepository class."""

    def test_store_prompt(self):
        """Test storing a prompt template."""
        repo = ContextPromptRepository()

        template = repo.store_prompt(
            template_id="test_store",
            prompt_template="Test {value}",
            description="Test storage",
            tags=["test"],
        )

        assert template.template_id == "test_store"
        assert "test_store" in repo.list_templates()

    def test_store_prompt_auto_detect_variables(self):
        """Test automatic variable detection when storing."""
        repo = ContextPromptRepository()

        template = repo.store_prompt(
            template_id="auto_vars",
            prompt_template="Hello {name}, you are {age} years old",
        )

        assert "name" in template.variables
        assert "age" in template.variables

    def test_retrieve_prompts(self):
        """Test retrieving prompts by query."""
        repo = ContextPromptRepository()
        repo.initialize_with_defaults()

        results = repo.retrieve_prompts("python code", top_k=3)
        assert len(results) > 0
        assert any("python" in t.tags for t in results)

    def test_get_context_for_task(self):
        """Test getting context for a specific task type."""
        repo = ContextPromptRepository()
        repo.initialize_with_defaults()

        context = repo.get_context_for_task(
            "debugging",
            {"language": "python"},
            top_k=2,
        )

        assert len(context) > 0
        assert "template_id" in context[0]
        assert "template" in context[0]

    def test_get_template(self):
        """Test getting a specific template by ID."""
        repo = ContextPromptRepository()
        repo.store_prompt(
            template_id="specific",
            prompt_template="Specific {test}",
        )

        template = repo.get_template("specific")
        assert template is not None
        assert template.template_id == "specific"

        # Non-existent template
        missing = repo.get_template("nonexistent")
        assert missing is None

    def test_update_success_rate(self):
        """Test updating template success rate."""
        repo = ContextPromptRepository()
        repo.store_prompt(
            template_id="rate_test",
            prompt_template="Test {x}",
        )

        # Initial rate should be 0.5
        template = repo.get_template("rate_test")
        assert template.success_rate == 0.5

        # Update with success
        repo.update_success_rate("rate_test", True)
        template = repo.get_template("rate_test")
        assert template.success_rate > 0.5

        # Update with failure
        repo.update_success_rate("rate_test", False)
        template = repo.get_template("rate_test")
        assert template.success_rate < 0.6  # EMA should reduce it

    def test_list_templates(self):
        """Test listing templates with optional tag filter."""
        repo = ContextPromptRepository()
        repo.store_prompt(
            template_id="tagged1",
            prompt_template="Test 1",
            tags=["alpha", "beta"],
        )
        repo.store_prompt(
            template_id="tagged2",
            prompt_template="Test 2",
            tags=["beta", "gamma"],
        )
        repo.store_prompt(
            template_id="untagged",
            prompt_template="Test 3",
        )

        # All templates
        all_templates = repo.list_templates()
        assert len(all_templates) == 3

        # Filtered by tag
        beta_templates = repo.list_templates(tags_filter=["beta"])
        assert len(beta_templates) == 2
        assert "tagged1" in beta_templates
        assert "tagged2" in beta_templates

    def test_export_import_templates(self):
        """Test exporting and importing templates."""
        repo1 = ContextPromptRepository()
        repo1.store_prompt(
            template_id="export_test",
            prompt_template="Export {this}",
            tags=["export"],
        )

        # Export
        exported = repo1.export_templates()
        assert "export_test" in exported

        # Import to new repository
        repo2 = ContextPromptRepository()
        count = repo2.import_templates(exported)
        assert count == 1

        # Verify imported template
        template = repo2.get_template("export_test")
        assert template is not None
        assert template.template == "Export {this}"

    def test_initialize_with_defaults(self):
        """Test initialization with default templates."""
        repo = ContextPromptRepository()
        repo.initialize_with_defaults()

        templates = repo.list_templates()
        assert len(templates) >= 5  # At least 5 default templates

        # Check specific default templates exist
        assert "code_generation_python" in templates
        assert "debug_error" in templates
        assert "code_review" in templates

    def test_singleton_get_repository(self):
        """Test singleton pattern for get_repository."""
        repo1 = get_repository()
        repo2 = get_repository()

        assert repo1 is repo2  # Same instance


class TestIntegration:
    """Integration tests for context repository."""

    def test_full_workflow(self):
        """Test complete workflow: store, search, retrieve, render."""
        repo = ContextPromptRepository()

        # Store custom template
        repo.store_prompt(
            template_id="custom_analysis",
            prompt_template="Analyze this {data_type} data: {data}\n\nGoals: {goals}",
            description="Custom data analysis template",
            tags=["analysis", "custom", "data"],
        )

        # Search for it
        results = repo.retrieve_prompts("data analysis custom", top_k=5)
        assert any(t.template_id == "custom_analysis" for t in results)

        # Get context for task
        context = repo.get_context_for_task("analysis", {"data_type": "numeric"})
        assert len(context) > 0

        # Render template
        template = repo.get_template("custom_analysis")
        rendered = template.render(
            data_type="numeric",
            data="[1, 2, 3, 4, 5]",
            goals="Find patterns and outliers",
        )

        assert "numeric" in rendered
        assert "[1, 2, 3, 4, 5]" in rendered
        assert "Find patterns and outliers" in rendered

    def test_success_rate_tracking(self):
        """Test tracking success rates through multiple updates."""
        repo = ContextPromptRepository()
        repo.store_prompt(
            template_id="tracked",
            prompt_template="Track {this}",
        )

        # Simulate multiple uses with varying success
        for _ in range(5):
            repo.update_success_rate("tracked", True)
        for _ in range(2):
            repo.update_success_rate("tracked", False)

        template = repo.get_template("tracked")
        # Should be above 0.5 due to more successes
        assert template.success_rate > 0.5
        assert template.success_rate < 1.0
