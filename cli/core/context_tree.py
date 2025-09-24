"""Hierarchical context management system with intelligent compression.

This module implements the hierarchical context tree from the blueprint,
providing intelligent compression and selective context inheritance for agents.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json

from pydantic import BaseModel, Field


class ContextWindow:
    """Individual context window with token management."""

    def __init__(self, max_tokens: int = 256000):
        """Initialize context window.

        Args:
            max_tokens: Maximum token capacity
        """
        self.max_tokens = max_tokens
        self.content: List[Dict[str, Any]] = []
        self.current_tokens = 0
        self.tokenizer = None  # Could integrate with tiktoken
        self.logger = logging.getLogger(__name__)

    def append(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Append content to context window.

        Args:
            content: Content to add
            metadata: Optional metadata

        Returns:
            True if content was added, False if rejected due to limits

        Raises:
            ValueError: If content exceeds token limit
        """
        # Estimate tokens (simplified - could use tiktoken)
        content_tokens = len(content.split()) * 1.3  # Rough approximation

        if self.current_tokens + content_tokens > self.max_tokens:
            raise ValueError(f"Content exceeds maximum token limit ({self.max_tokens})")

        context_item = {
            "content": content,
            "tokens": content_tokens,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        self.content.append(context_item)
        self.current_tokens += content_tokens

        return True

    def get_content(self, max_tokens: Optional[int] = None) -> str:
        """Get context content, optionally limited by tokens.

        Args:
            max_tokens: Maximum tokens to return

        Returns:
            Concatenated context content
        """
        if max_tokens:
            available_tokens = max_tokens
            result_content = []

            # Take items from the end (most recent first)
            for item in reversed(self.content):
                if available_tokens <= 0:
                    break

                item_tokens = item["tokens"]
                if item_tokens <= available_tokens:
                    result_content.insert(0, item["content"])
                    available_tokens -= item_tokens
                else:
                    # Truncate content to fit
                    truncated_content = self._truncate_content(item["content"], available_tokens)
                    result_content.insert(0, truncated_content)
                    break

            return "\n".join(result_content)
        else:
            return "\n".join([item["content"] for item in self.content])

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token limit.

        Args:
            content: Content to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated content
        """
        words = content.split()
        estimated_tokens = 0
        result_words = []

        for word in words:
            word_tokens = len(word) * 0.3  # Rough approximation
            if estimated_tokens + word_tokens > max_tokens:
                break
            result_words.append(word)
            estimated_tokens += word_tokens

        return " ".join(result_words) + "..."

    def clear(self):
        """Clear all content from context window."""
        self.content.clear()
        self.current_tokens = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get context window statistics."""
        return {
            "max_tokens": self.max_tokens,
            "current_tokens": self.current_tokens,
            "utilization": self.current_tokens / self.max_tokens,
            "item_count": len(self.content)
        }


class AdaptiveCompressor:
    """Intelligent compression for different content types."""

    def __init__(self):
        """Initialize adaptive compressor."""
        self.logger = logging.getLogger(__name__)

    def compress_paper_analysis(self, full_analysis: Dict[str, Any]) -> str:
        """Compress paper analysis result for context.

        Args:
            full_analysis: Full paper analysis result

        Returns:
            Compressed analysis summary
        """
        compressed = {
            "key_findings": full_analysis.get("findings", [])[:3],  # Top 3 findings
            "methodology": full_analysis.get("methodology_type", "Unknown"),
            "relevance_score": full_analysis.get("relevance", 0.0),
            "citations_to_explore": full_analysis.get("top_citations", [])[:5]
        }

        return json.dumps(compressed, indent=2)

    def compress_search_results(self, papers: List[Dict[str, Any]]) -> str:
        """Compress search results for context.

        Args:
            papers: List of paper search results

        Returns:
            Compressed search results summary
        """
        compressed_papers = []
        for paper in papers[:20]:  # Limit to top 20
            compressed_paper = {
                "title": paper.get("title", "Unknown"),
                "authors": paper.get("authors", [])[:3],  # First 3 authors
                "year": paper.get("year", "Unknown"),
                "relevance": paper.get("score", 0.0)
            }
            compressed_papers.append(compressed_paper)

        return json.dumps(compressed_papers, indent=2)

    def compress_agent_result(self, result: Dict[str, Any], target_tokens: int = 2000) -> str:
        """Compress agent result to target token count.

        Args:
            result: Full agent result
            target_tokens: Target token count

        Returns:
            Compressed result summary
        """
        if isinstance(result, str):
            # Simple string compression
            words = result.split()
            if len(words) <= target_tokens:
                return result

            # Truncate while preserving structure
            truncated = " ".join(words[:int(target_tokens * 0.8)]) + "..."
            return truncated

        # Structured result compression
        compressed = {
            "summary": result.get("summary", ""),
            "key_insights": result.get("insights", [])[:3],  # Top 3 insights
            "artifacts": len(result.get("artifacts", []))  # Count of artifacts
        }

        return json.dumps(compressed, indent=2)

    def extract_relevant_context(self, source_context: str, task: Any, target_tokens: int = 5000) -> str:
        """Extract relevant context for a specific task.

        Args:
            source_context: Full source context
            task: Task requiring context
            target_tokens: Target token count

        Returns:
            Relevant context summary
        """
        # Simple relevance extraction - could be enhanced with embeddings
        context_sections = source_context.split("\n\n")
        relevant_sections = []

        # Look for sections that might be relevant to the task
        task_keywords = set(task.description.lower().split())

        for section in context_sections:
            section_words = set(section.lower().split())
            overlap = task_keywords.intersection(section_words)

            if len(overlap) > 0:
                relevant_sections.append(section)

        # If no relevant sections found, return beginning of context
        if not relevant_sections:
            relevant_sections = context_sections[:2]

        # Combine and truncate
        combined_context = "\n\n".join(relevant_sections)

        if len(combined_context.split()) <= target_tokens:
            return combined_context

        # Truncate while preserving structure
        words = combined_context.split()
        return " ".join(words[:int(target_tokens * 0.8)]) + "..."


class ContextTree:
    """Hierarchical context management with intelligent compression.

    Implements the hybrid context tree from the blueprint:
    - Root: 20k tokens (orchestrator coordination)
    - Branches: Up to 256k tokens (agent specialization)
    """

    def __init__(self, context_limits: Dict[str, int]):
        """Initialize context tree.

        Args:
            context_limits: Token limits for each level (e.g., {"root": 20000, "branch": 80000})
        """
        self.context_limits = context_limits
        self.root_context = ContextWindow(context_limits.get("root", 20000))
        self.branch_contexts: Dict[str, ContextWindow] = {}
        self.compression_strategy = AdaptiveCompressor()
        self.logger = logging.getLogger(__name__)

    def spawn_subagent(self, agent_id: str, task: Any) -> ContextWindow:
        """Create isolated context for subagent with selective inheritance.

        Args:
            agent_id: Unique agent identifier
            task: Task to be executed by agent

        Returns:
            New ContextWindow for the subagent
        """
        # Create new context window for subagent
        max_tokens = self.context_limits.get("branch", 80000)
        subagent_context = ContextWindow(max_tokens)

        # Extract essential context from root
        essential_context = self.compression_strategy.extract_relevant_context(
            source_context=self.root_context.get_content(),
            task=task,
            target_tokens=5000  # Compressed to 5k tokens
        )

        # Add essential context and task description
        subagent_context.append(essential_context, {"type": "inherited_context"})
        subagent_context.append(task.description, {"type": "task_description"})

        self.branch_contexts[agent_id] = subagent_context
        self.logger.info(f"Spawned subagent context for {agent_id}")

        return subagent_context

    def propagate_result(self, agent_id: str, result: Any):
        """Compress subagent result before propagating to root.

        Args:
            agent_id: Agent identifier
            result: Agent execution result
        """
        if agent_id not in self.branch_contexts:
            self.logger.warning(f"Agent {agent_id} not found in branch contexts")
            return

        # Store full result separately (for trace)
        self._store_full_result(agent_id, result)

        # Compress result for root context
        if isinstance(result, dict) and "full_output" in result:
            compressed_summary = self.compression_strategy.compress_agent_result(
                result["full_output"],
                target_tokens=2000  # 2k token summary
            )
        else:
            compressed_summary = self.compression_strategy.compress_agent_result(
                result,
                target_tokens=2000
            )

        # Add compressed summary to root with trace pointer
        self.root_context.append(
            f"Agent {agent_id} result: {compressed_summary}",
            {
                "type": "agent_result",
                "agent_id": agent_id,
                "trace_id": result.get("trace_id"),
                "full_trace_available": True
            }
        )

        self.logger.info(f"Propagated compressed result from {agent_id} to root context")

    def _store_full_result(self, agent_id: str, result: Any):
        """Store full agent result for trace purposes.

        Args:
            agent_id: Agent identifier
            result: Full agent result
        """
        # This would typically store to a trace database or file system
        # For now, we'll store in memory (could be enhanced)
        full_result_key = f"full_result_{agent_id}_{datetime.utcnow().isoformat()}"
        # In a real implementation, this would be stored in Langfuse or similar
        self.logger.debug(f"Storing full result for {agent_id}")

    def get_root_context(self) -> str:
        """Get current root context.

        Returns:
            Root context content
        """
        return self.root_context.get_content()

    def get_branch_context(self, agent_id: str) -> Optional[str]:
        """Get specific branch context.

        Args:
            agent_id: Agent identifier

        Returns:
            Branch context content or None if not found
        """
        if agent_id in self.branch_contexts:
            return self.branch_contexts[agent_id].get_content()
        return None

    def get_context_stats(self) -> Dict[str, Any]:
        """Get context tree statistics.

        Returns:
            Dictionary with context statistics
        """
        stats = {
            "root": self.root_context.get_stats(),
            "branches": {}
        }

        for agent_id, context in self.branch_contexts.items():
            stats["branches"][agent_id] = context.get_stats()

        return stats

    def optimize_context(self):
        """Optimize context tree by cleaning up old or irrelevant content."""
        # Remove old branch contexts that are no longer needed
        current_time = datetime.utcnow()

        contexts_to_remove = []
        for agent_id, context in self.branch_contexts.items():
            # Check if context is old or underutilized
            if len(context.content) == 0:
                contexts_to_remove.append(agent_id)

        for agent_id in contexts_to_remove:
            del self.branch_contexts[agent_id]
            self.logger.info(f"Removed unused context for agent {agent_id}")

        # Optimize root context if needed
        if self.root_context.current_tokens > self.root_context.max_tokens * 0.9:
            self._optimize_root_context()

    def _optimize_root_context(self):
        """Optimize root context when approaching capacity."""
        # Remove oldest entries that aren't recent agent results
        optimized_content = []

        for item in self.root_context.content:
            metadata = item.get("metadata", {})
            item_type = metadata.get("type")

            # Keep recent agent results, remove old inherited context
            if item_type == "agent_result" or item == self.root_context.content[-1]:
                optimized_content.append(item)

        self.root_context.content = optimized_content
        self.root_context.current_tokens = sum(item["tokens"] for item in optimized_content)

        self.logger.info("Optimized root context to reduce token usage")


# Context management utilities
def create_research_context_tree() -> ContextTree:
    """Create context tree optimized for research workflows.

    Returns:
        ContextTree configured for research
    """
    return ContextTree({
        "root": 20000,    # Orchestrator coordination
        "branch": 80000   # Agent specialization
    })


def create_balanced_context_tree() -> ContextTree:
    """Create balanced context tree for mixed workloads.

    Returns:
        ContextTree with balanced limits
    """
    return ContextTree({
        "root": 30000,    # More coordination capacity
        "branch": 60000   # Moderate agent capacity
    })


def create_high_capacity_context_tree() -> ContextTree:
    """Create high-capacity context tree for complex tasks.

    Returns:
        ContextTree with high limits
    """
    return ContextTree({
        "root": 50000,    # High coordination capacity
        "branch": 120000  # High agent capacity
    })
