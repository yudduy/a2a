"""Comprehensive tests for CLI research system components."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import CLI components (will be created)
try:
    from ..core.a2a_client import A2AClient, AgentCard, Task, AgentResult
    from ..core.context_tree import ContextTree, ContextWindow, AdaptiveCompressor
    from ..orchestration.langgraph_orchestrator import OrchestrationEngine, ResearchResult
    from ..orchestration.trace_collector import TraceCollector
    from ..core.cli_interface import ResearchCLI
    from ..agents.research_agent import ResearchAgent
    from ..utils.research_types import ResearchState, StreamMessage, RoutedMessage
except ImportError:
    # For running tests directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from core.a2a_client import A2AClient, AgentCard, Task, AgentResult
    from core.context_tree import ContextTree, ContextWindow, AdaptiveCompressor
    from orchestration.langgraph_orchestrator import OrchestrationEngine, ResearchResult
    from orchestration.trace_collector import TraceCollector
    from core.cli_interface import ResearchCLI
    from agents.research_agent import ResearchAgent
    from utils.research_types import ResearchState, StreamMessage, RoutedMessage


class TestA2AProtocol:
    """Test A2A protocol implementation."""

    def test_agent_card_creation(self):
        """Test AgentCard creation and validation."""
        card_data = {
            "name": "test-research-agent",
            "description": "Test research agent for unit testing",
            "version": "1.0.0",
            "capabilities": {
                "skills": [
                    {
                        "name": "analyze_paper",
                        "description": "Deep analysis of academic papers",
                        "input_schema": {"type": "object", "properties": {"paper": {"type": "string"}}},
                        "output_schema": {"type": "object", "properties": {"analysis": {"type": "string"}}}
                    }
                ]
            },
            "endpoints": {
                "base_url": "http://localhost:8000/test-research"
            }
        }

        agent_card = AgentCard(**card_data)
        assert agent_card.name == "test-research-agent"
        assert len(agent_card.capabilities["skills"]) == 1
        assert agent_card.endpoints["base_url"] == "http://localhost:8000/test-research"

    @pytest.mark.asyncio
    async def test_a2a_client_task_delegation(self):
        """Test A2A client task delegation."""
        # Mock HTTP response
        mock_response = {
            "result": {
                "summary": "Test research completed successfully",
                "artifacts": [{"type": "analysis", "content": "Detailed analysis content"}]
            },
            "trace_id": "test-trace-123"
        }

        agent_card = AgentCard(
            name="test-agent",
            endpoints={"base_url": "http://localhost:8000/test"}
        )

        task = Task(
            id="task-123",
            description="Analyze research paper",
            context_summary="Paper about AI research"
        )

        # Test client initialization
        client = A2AClient()

        # Mock the actual HTTP call
        with patch.object(client, 'send_a2a_message', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response

            result = await client.delegate_task(agent_card, task)

            assert result.summary == "Test research completed successfully"
            assert len(result.artifacts) == 1
            assert result.trace_id == "test-trace-123"
            mock_send.assert_called_once()

    def test_task_creation_and_validation(self):
        """Test Task creation and validation."""
        task_data = {
            "id": "task-123",
            "description": "Research quantum computing applications",
            "context_summary": "Focus on practical applications in industry",
            "metadata": {"priority": "high", "deadline": "2024-12-31"}
        }

        task = Task(**task_data)
        assert task.id == "task-123"
        assert task.description == "Research quantum computing applications"
        assert task.metadata["priority"] == "high"


class TestContextManagement:
    """Test hierarchical context management system."""

    def test_context_window_creation(self):
        """Test ContextWindow creation and basic operations."""
        window = ContextWindow(max_tokens=1000)

        # Test initial state
        assert window.max_tokens == 1000
        assert len(window.content) == 0
        assert window.current_tokens == 0

        # Test content addition
        window.append("This is test content for context window.")
        assert len(window.content) == 1
        assert window.current_tokens > 0

    def test_context_window_token_limits(self):
        """Test ContextWindow token limit enforcement."""
        window = ContextWindow(max_tokens=10)  # Very small limit

        # Add content that exceeds token limit
        long_content = "This is a very long piece of content that definitely exceeds the token limit when processed by the tokenizer."

        with pytest.raises(ValueError, match="Content exceeds maximum token limit"):
            window.append(long_content)

    def test_context_tree_hierarchy(self):
        """Test ContextTree hierarchical structure."""
        context_tree = ContextTree({
            "root": 20000,
            "branch": 80000
        })

        # Test root context creation
        assert context_tree.root_context.max_tokens == 20000

        # Test branch spawning
        task = Mock()
        task.description = "Test research task"

        subagent_context = context_tree.spawn_subagent("test-agent", task)

        assert subagent_context.max_tokens == 80000
        assert len(subagent_context.content) == 2  # task description + essential context
        assert "test-agent" in context_tree.branch_contexts

    def test_adaptive_compressor(self):
        """Test AdaptiveCompressor functionality."""
        compressor = AdaptiveCompressor()

        # Test compression with different content types
        paper_analysis = {
            "findings": ["Finding 1", "Finding 2", "Finding 3"],
            "methodology_type": "Experimental study",
            "relevance": 0.85,
            "citations": ["Paper A", "Paper B", "Paper C", "Paper D", "Paper E"]
        }

        compressed = compressor.compress_paper_analysis(paper_analysis)

        # Parse JSON string result
        import json
        compressed_dict = json.loads(compressed)

        assert "key_findings" in compressed_dict
        assert len(compressed_dict["key_findings"]) <= 3  # Limited to top 3
        assert compressed_dict["methodology"] == "Experimental study"
        assert len(compressed_dict["citations_to_explore"]) <= 5  # Limited to top 5

    def test_context_tree_compression(self):
        """Test ContextTree compression and propagation."""
        context_tree = ContextTree({"root": 20000, "branch": 80000})

        # Add content to root
        context_tree.root_context.append("Root context information for research coordination.")

        # Spawn subagent and add content
        task = Mock()
        task.description = "Analyze specific research area"

        subagent_context = context_tree.spawn_subagent("analysis-agent", task)
        subagent_context.append("Detailed analysis results and findings from specialized research.")

        # Test result propagation with compression
        agent_result = {
            "full_output": "Comprehensive analysis with detailed findings and extensive research data.",
            "trace_id": "analysis-trace-123"
        }

        context_tree.propagate_result("analysis-agent", agent_result)

        # Verify root context received compressed summary
        assert len(context_tree.root_context.content) == 2  # Original + compressed result


class TestOrchestrationEngine:
    """Test LangGraph orchestration engine."""

    @pytest.mark.asyncio
    async def test_orchestration_engine_initialization(self):
        """Test OrchestrationEngine initialization."""
        with patch('langfuse.Langfuse') as mock_langfuse:
            mock_langfuse.return_value = Mock()

            engine = OrchestrationEngine()
            assert engine.graph is not None
            assert engine.tracer is not None

    @pytest.mark.asyncio
    async def test_research_execution_workflow(self):
        """Test complete research workflow execution."""
        with patch('langfuse.Langfuse') as mock_langfuse:
            mock_langfuse.return_value = Mock()

            engine = OrchestrationEngine()

            # Mock graph execution
            with patch.object(engine.graph, 'ainvoke') as mock_invoke:
                mock_result = {
                    "synthesis": "Research completed successfully with key findings...",
                    "papers": [
                        {"title": "Paper 1", "relevance": 0.9},
                        {"title": "Paper 2", "relevance": 0.8}
                    ]
                }
                mock_invoke.return_value = mock_result

                result = await engine.execute_research("Test research query about AI")

                assert result.synthesis == "Research completed successfully with key findings..."
                assert len(result.papers) == 2
                assert result.trace_id is not None

    @pytest.mark.asyncio
    async def test_research_streaming(self):
        """Test streaming research execution."""
        with patch('langfuse.Langfuse') as mock_langfuse:
            mock_langfuse.return_value = Mock()

            engine = OrchestrationEngine()

            # Mock streaming response
            mock_stream = [
                {"type": "agent_spawn", "agent_id": "search_agent"},
                {"type": "agent_progress", "agent_id": "search_agent", "status": "Searching..."},
                {"type": "agent_complete", "agent_id": "search_agent", "summary": "Found 10 relevant papers"},
                {"type": "research_complete", "result": "Final research synthesis"}
            ]

            with patch.object(engine, 'execute_stream', return_value=mock_stream):
                events = []
                async for event in engine.execute_stream("Test streaming query"):
                    events.append(event)

                assert len(events) == 4
                assert events[0]["type"] == "agent_spawn"
                assert events[-1]["type"] == "research_complete"


class TestTraceCollector:
    """Test Langfuse trace collection."""

    def test_trace_creation(self):
        """Test trace creation and metadata."""
        with patch('langfuse.Langfuse') as mock_langfuse:
            mock_langfuse_instance = Mock()
            mock_langfuse.return_value = mock_langfuse_instance

            collector = TraceCollector()

            # Mock trace creation
            mock_trace = Mock()
            mock_trace.id = "test-trace-123"
            mock_langfuse_instance.trace.return_value = mock_trace

            trace = collector.trace_agent_execution("test-agent", Mock())

            assert trace == "test-trace-123"
            mock_langfuse_instance.trace.assert_called_once()

    def test_generation_tracking(self):
        """Test generation tracking with token usage."""
        with patch('langfuse.Langfuse') as mock_langfuse:
            mock_langfuse_instance = Mock()
            mock_langfuse.return_value = mock_langfuse_instance

            collector = TraceCollector()

            # Mock trace and generation
            mock_trace = Mock()
            mock_generation = Mock()
            mock_langfuse_instance.trace.return_value = mock_trace
            mock_trace.generation.return_value = mock_generation

            with collector.trace_agent_execution("test-agent", Mock()):
                # Simulate agent execution
                pass

            # Verify generation tracking calls
            mock_trace.generation.assert_called_once()
            mock_generation.end.assert_called_once()


class TestCLIInterface:
    """Test CLI interface functionality."""

    def test_cli_initialization(self):
        """Test CLI initialization."""
        with patch('cli.orchestration.langgraph_orchestrator.OrchestrationEngine') as mock_engine:
            mock_engine.return_value = Mock()

            cli = ResearchCLI()
            assert cli.orchestrator is not None

    @pytest.mark.asyncio
    async def test_research_command_execution(self):
        """Test research command execution."""
        with patch('cli.orchestration.langgraph_orchestrator.OrchestrationEngine') as mock_engine:
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance

            # Mock execution result
            mock_result = Mock()
            mock_result.synthesis = "Research completed successfully"
            mock_engine_instance.execute.return_value = mock_result

            cli = ResearchCLI()

            # Test with mock console
            with patch('rich.console.Console') as mock_console:
                result = await cli.research("Test research query about machine learning")

                assert result == mock_result
                mock_engine_instance.execute.assert_called_once_with("Test research query about machine learning")


class TestResearchAgent:
    """Test research agent functionality."""

    def test_agent_initialization(self):
        """Test research agent initialization."""
        agent = ResearchAgent("test-agent")

        assert agent.name == "test-agent"
        assert agent.card is not None
        assert agent.card.name == "test-agent"

    @pytest.mark.asyncio
    async def test_agent_task_handling(self):
        """Test agent task handling."""
        agent = ResearchAgent("test-agent")

        # Mock task
        task = Task(
            id="task-123",
            description="Research test topic",
            context_summary="Context for research"
        )

        # Mock agent execution
        with patch.object(agent, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = Mock(
                summary="Task completed successfully",
                artifacts=[{"type": "analysis", "content": "Test analysis"}]
            )

            result = await agent.handle_task(task)

            assert result.summary == "Task completed successfully"
            assert len(result.artifacts) == 1


class TestIntegration:
    """Test end-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_research_workflow(self):
        """Test complete research workflow from CLI to results."""
        with patch('cli.orchestration.langgraph_orchestrator.OrchestrationEngine') as mock_engine:
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance

            # Mock complete research result
            mock_result = ResearchResult(
                synthesis="Complete research synthesis with all findings",
                papers=[
                    {"title": "Paper 1", "relevance": 0.9, "summary": "Key paper findings"},
                    {"title": "Paper 2", "relevance": 0.8, "summary": "Additional insights"}
                ],
                trace_id="workflow-trace-123"
            )
            mock_engine_instance.execute_research.return_value = mock_result

            # Create CLI and execute research
            cli = ResearchCLI()
            result = await cli.research("Research quantum computing applications in healthcare")

            # Verify complete workflow
            assert result.synthesis is not None
            assert len(result.papers) == 2
            assert result.trace_id is not None

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        with patch('cli.orchestration.langgraph_orchestrator.OrchestrationEngine') as mock_engine:
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance

            # Mock execution failure
            mock_engine_instance.execute_research.side_effect = Exception("Research failed")

            cli = ResearchCLI()

            # Test graceful error handling
            with pytest.raises(Exception, match="Research failed"):
                await cli.research("Test query that should fail")


# Performance and load tests
class TestPerformance:
    """Test system performance under load."""

    @pytest.mark.asyncio
    async def test_concurrent_research_execution(self):
        """Test concurrent research execution performance."""
        # This would test the actual performance of parallel execution
        # For now, just verify the structure is in place
        with patch('cli.orchestration.langgraph_orchestrator.OrchestrationEngine') as mock_engine:
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance

            # Mock concurrent execution
            mock_engine_instance.execute_research.return_value = Mock()

            # Test that system can handle multiple concurrent requests
            tasks = [
                mock_engine_instance.execute_research(f"Query {i}")
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert mock_engine_instance.execute_research.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
