"""Context Prompt Repository for storing and retrieving prompt templates.

This module provides:
- Storage of prompt templates for various task types
- Semantic search for relevant context extraction
- Integration with prompt_builder for dynamic context assembly
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A template for generating prompts with variables."""

    template_id: str
    template: str
    description: str = ""
    variables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    usage_count: int = 0
    success_rate: float = 0.5  # Default prior

    def render(self, **kwargs: Any) -> str:
        """Render the template with provided variables.

        Args:
            **kwargs: Variable values to substitute in the template.

        Returns:
            Rendered prompt string.

        Raises:
            ValueError: If required variables are missing.
        """
        try:
            rendered = self.template.format(**kwargs)
            self.usage_count += 1
            return rendered
        except KeyError as e:
            missing_var = e.args[0]
            logger.warning(
                "Missing variable '%s' for template '%s'. Available: %s",
                missing_var,
                self.template_id,
                list(kwargs.keys()),
            )
            raise ValueError(
                f"Missing required variable '{missing_var}' for template '{self.template_id}'"
            ) from e

    def to_dict(self) -> dict[str, Any]:
        """Convert template to dictionary representation."""
        return {
            "template_id": self.template_id,
            "template": self.template,
            "description": self.description,
            "variables": self.variables,
            "metadata": self.metadata,
            "tags": self.tags,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptTemplate:
        """Create a PromptTemplate from a dictionary."""
        return cls(
            template_id=data.get("template_id", ""),
            template=data.get("template", ""),
            description=data.get("description", ""),
            variables=data.get("variables", []),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate", 0.5),
        )


class SemanticIndex:
    """Simple semantic index for prompt similarity search.

    Uses keyword-based similarity with optional embedding support.
    For production, integrate with sentence-transformers for better embeddings.
    """

    def __init__(self) -> None:
        """Initialize the semantic index."""
        self._index: dict[str, dict[str, float]] = {}
        self._keyword_index: dict[str, set[str]] = {}
        self._built = False

    def build_index(self, prompts: list[PromptTemplate]) -> None:
        """Build the semantic index from a list of prompts.

        Args:
            prompts: List of PromptTemplate objects to index.
        """
        self._index.clear()
        self._keyword_index.clear()

        for prompt in prompts:
            template_id = prompt.template_id
            
            # Extract keywords from template, description, and tags
            text_content = (
                f"{prompt.template} {prompt.description} {' '.join(prompt.tags)}"
            ).lower()
            
            # Simple keyword extraction (can be enhanced with TF-IDF or embeddings)
            keywords = self._extract_keywords(text_content)
            
            # Store keyword weights for this template
            self._index[template_id] = {}
            for kw in keywords:
                self._index[template_id][kw] = self._index[template_id].get(kw, 0) + 1
                
                # Build reverse index
                if kw not in self._keyword_index:
                    self._keyword_index[kw] = set()
                self._keyword_index[kw].add(template_id)
        
        self._built = True
        logger.info("Built semantic index with %d prompts", len(prompts))

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text.

        Simple implementation: split on whitespace and remove stopwords.
        Can be enhanced with NLP techniques.
        """
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "to", "of", "in", "for", "on", "with", "at",
            "by", "from", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "although", "though", "this",
            "that", "these", "those", "it", "its"
        }
        
        words = text.split()
        keywords = [
            word.strip(".,!?;:\"'()[]{}")
            for word in words
            if word.lower() not in stopwords and len(word) > 2
        ]
        return keywords

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Search for similar prompts based on query.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (template_id, score) tuples sorted by relevance.
        """
        if not self._built:
            logger.warning("Semantic index not built yet")
            return []

        query_keywords = set(self._extract_keywords(query.lower()))
        
        if not query_keywords:
            return []

        # Calculate similarity scores using Jaccard-like metric
        scores: dict[str, float] = {}
        
        for keyword in query_keywords:
            if keyword in self._keyword_index:
                for template_id in self._keyword_index[keyword]:
                    if template_id not in scores:
                        scores[template_id] = 0.0
                    
                    # Weight by keyword frequency in template
                    keyword_weight = self._index[template_id].get(keyword, 0)
                    scores[template_id] += keyword_weight

        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]


class ContextPromptRepository:
    """Repository for storing and retrieving prompt templates.

    Provides centralized storage for prompt templates with semantic search
    capabilities for finding relevant prompts based on task context.
    """

    def __init__(self) -> None:
        """Initialize the repository."""
        self._templates: dict[str, PromptTemplate] = {}
        self._semantic_index = SemanticIndex()
        self._initialized = False

    def store_prompt(
        self,
        template_id: str,
        prompt_template: str,
        description: str = "",
        variables: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> PromptTemplate:
        """Store a prompt template in the repository.

        Args:
            template_id: Unique identifier for the template.
            prompt_template: The template string with {variable} placeholders.
            description: Human-readable description of the template's purpose.
            variables: List of variable names used in the template.
            metadata: Additional metadata for the template.
            tags: Tags for categorization and search.

        Returns:
            The stored PromptTemplate object.
        """
        # Auto-detect variables if not provided
        if variables is None:
            import re
            variables = list(set(re.findall(r'\{(\w+)\}', prompt_template)))

        template = PromptTemplate(
            template_id=template_id,
            template=prompt_template,
            description=description,
            variables=variables or [],
            metadata=metadata or {},
            tags=tags or [],
        )

        self._templates[template_id] = template
        logger.info("Stored prompt template '%s' with variables: %s", template_id, variables)

        # Rebuild semantic index
        self._rebuild_index()

        return template

    def retrieve_prompts(
        self,
        query: str,
        top_k: int = 5,
        tags_filter: list[str] | None = None,
    ) -> list[PromptTemplate]:
        """Retrieve prompts relevant to a query.

        Args:
            query: Search query describing the task or context.
            top_k: Maximum number of results to return.
            tags_filter: Optional list of tags to filter by.

        Returns:
            List of relevant PromptTemplate objects.
        """
        # Use semantic search
        search_results = self._semantic_index.similarity_search(query, top_k=top_k * 2)
        
        results: list[PromptTemplate] = []
        
        for template_id, score in search_results:
            if template_id not in self._templates:
                continue
                
            template = self._templates[template_id]
            
            # Apply tag filter if specified
            if tags_filter:
                if not any(tag in template.tags for tag in tags_filter):
                    continue
            
            results.append(template)
            
            if len(results) >= top_k:
                break

        # Fallback: if no semantic matches, return all templates
        if not results and self._templates:
            results = list(self._templates.values())[:top_k]

        logger.debug("Retrieved %d prompts for query: %s", len(results), query)
        return results

    def get_context_for_task(
        self,
        task_type: str,
        context_hints: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Get relevant context prompts for a specific task type.

        Args:
            task_type: Type of task (e.g., 'code_generation', 'debugging', 'analysis').
            context_hints: Additional context information for better matching.
            top_k: Number of context prompts to return.

        Returns:
            List of dicts with 'template', 'variables', and 'context' keys.
        """
        # Build search query from task type and hints
        query_parts = [task_type]
        if context_hints:
            for key, value in context_hints.items():
                query_parts.append(f"{key}: {value}")
        
        query = " ".join(query_parts)
        
        templates = self.retrieve_prompts(query, top_k=top_k)
        
        context_list = []
        for template in templates:
            context_item = {
                "template_id": template.template_id,
                "template": template.template,
                "variables": template.variables,
                "description": template.description,
                "tags": template.tags,
            }
            context_list.append(context_item)

        logger.info(
            "Retrieved %d context prompts for task type '%s'",
            len(context_list),
            task_type,
        )
        return context_list

    def get_template(self, template_id: str) -> PromptTemplate | None:
        """Get a specific template by ID.

        Args:
            template_id: The unique template identifier.

        Returns:
            The PromptTemplate if found, None otherwise.
        """
        return self._templates.get(template_id)

    def update_success_rate(self, template_id: str, success: bool) -> None:
        """Update the success rate for a template.

        Args:
            template_id: The template to update.
            success: Whether the template usage was successful.
        """
        if template_id not in self._templates:
            logger.warning("Template '%s' not found for success rate update", template_id)
            return

        template = self._templates[template_id]
        
        # Exponential moving average
        alpha = 0.1
        current = template.success_rate
        new_value = 1.0 if success else 0.0
        template.success_rate = (1 - alpha) * current + alpha * new_value
        
        logger.debug(
            "Updated success rate for '%s': %.3f",
            template_id,
            template.success_rate,
        )

    def list_templates(self, tags_filter: list[str] | None = None) -> list[str]:
        """List all template IDs, optionally filtered by tags.

        Args:
            tags_filter: Optional list of tags to filter by.

        Returns:
            List of template IDs.
        """
        if not tags_filter:
            return list(self._templates.keys())
        
        return [
            tid for tid, template in self._templates.items()
            if any(tag in template.tags for tag in tags_filter)
        ]

    def export_templates(self) -> dict[str, Any]:
        """Export all templates as a dictionary.

        Returns:
            Dictionary of template_id -> template_dict mappings.
        """
        return {
            tid: template.to_dict()
            for tid, template in self._templates.items()
        }

    def import_templates(self, templates_data: dict[str, Any]) -> int:
        """Import templates from a dictionary.

        Args:
            templates_data: Dictionary of template data.

        Returns:
            Number of templates imported.
        """
        count = 0
        for template_id, data in templates_data.items():
            # Ensure template_id matches
            data["template_id"] = template_id
            template = PromptTemplate.from_dict(data)
            self._templates[template_id] = template
            count += 1
        
        self._rebuild_index()
        logger.info("Imported %d templates", count)
        return count

    def _rebuild_index(self) -> None:
        """Rebuild the semantic index from current templates."""
        prompts = list(self._templates.values())
        self._semantic_index.build_index(prompts)
        self._initialized = True

    def initialize_with_defaults(self) -> None:
        """Initialize the repository with default prompt templates."""
        default_templates = [
            # Code generation templates
            {
                "template_id": "code_generation_python",
                "template": (
                    "You are an expert Python developer. Generate clean, efficient, "
                    "and well-documented Python code for the following task:\n\n"
                    "Task: {task_description}\n\n"
                    "Requirements:\n{requirements}\n\n"
                    "Constraints:\n{constraints}\n\n"
                    "Please provide the complete implementation with error handling."
                ),
                "description": "Template for Python code generation tasks",
                "tags": ["code", "python", "generation"],
                "metadata": {"language": "python", "complexity": "medium"},
            },
            {
                "template_id": "code_review",
                "template": (
                    "Review the following code for quality, security, and best practices:\n\n"
                    "Code:\n```{language}\n{code}\n```\n\n"
                    "Focus areas: {focus_areas}\n\n"
                    "Provide specific recommendations with code examples where applicable."
                ),
                "description": "Template for code review tasks",
                "tags": ["code", "review", "analysis"],
                "metadata": {"type": "review"},
            },
            # Debugging templates
            {
                "template_id": "debug_error",
                "template": (
                    "Help debug the following error:\n\n"
                    "Error message: {error_message}\n\n"
                    "Code context:\n```{language}\n{code_snippet}\n```\n\n"
                    "Stack trace:\n{stack_trace}\n\n"
                    "Describe the likely cause and provide a fix."
                ),
                "description": "Template for debugging error messages",
                "tags": ["debug", "error", "troubleshooting"],
                "metadata": {"type": "debug"},
            },
            # Analysis templates
            {
                "template_id": "data_analysis",
                "template": (
                    "Analyze the following data and provide insights:\n\n"
                    "Data description: {data_description}\n\n"
                    "Analysis goals: {goals}\n\n"
                    "Key questions to answer:\n{questions}\n\n"
                    "Provide a structured analysis with findings and recommendations."
                ),
                "description": "Template for data analysis tasks",
                "tags": ["analysis", "data", "insights"],
                "metadata": {"type": "analysis"},
            },
            # Documentation templates
            {
                "template_id": "doc_generation",
                "template": (
                    "Generate documentation for the following code:\n\n"
                    "Code:\n```{language}\n{code}\n```\n\n"
                    "Documentation style: {style}\n\n"
                    "Include: {include_sections}\n\n"
                    "Generate comprehensive documentation suitable for {audience}."
                ),
                "description": "Template for generating code documentation",
                "tags": ["documentation", "writing", "code"],
                "metadata": {"type": "documentation"},
            },
            # Testing templates
            {
                "template_id": "test_generation",
                "template": (
                    "Generate comprehensive tests for the following code:\n\n"
                    "Code to test:\n```{language}\n{code}\n```\n\n"
                    "Testing framework: {framework}\n\n"
                    "Test scenarios to cover:\n{scenarios}\n\n"
                    "Include edge cases and error conditions."
                ),
                "description": "Template for generating test cases",
                "tags": ["testing", "code", "quality"],
                "metadata": {"type": "testing"},
            },
        ]

        for template_data in default_templates:
            self.store_prompt(
                template_id=template_data["template_id"],
                prompt_template=template_data["template"],
                description=template_data["description"],
                tags=template_data["tags"],
                metadata=template_data.get("metadata", {}),
            )

        logger.info("Initialized repository with %d default templates", len(default_templates))


# Singleton instance for easy access
_repository_instance: ContextPromptRepository | None = None


def get_repository() -> ContextPromptRepository:
    """Get the singleton repository instance.

    Returns:
        The ContextPromptRepository instance.
    """
    global _repository_instance
    if _repository_instance is None:
        _repository_instance = ContextPromptRepository()
        _repository_instance.initialize_with_defaults()
    return _repository_instance
