"""Comprehensive integration tests for Sequential Multi-Agent Supervisor Architecture.

This module tests the complete integration of the Sequential Supervisor system including:
- Agent registry loading and management
- Automatic completion detection and handoffs
- Context sharing between sequential agents
- Running report incremental building
- LangGraph workflow integration
- Performance requirements validation

Test Categories:
1. End-to-end workflow integration
2. Agent registry and loading functionality
3. Completion detection accuracy
4. Context management and handoffs
5. Report building throughout execution
6. Error handling and resilience
7. Performance benchmarking
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from open_deep_research.supervisor.sequential_supervisor import SequentialSupervisor, SupervisorConfig
from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.agents.completion_detector import CompletionDetector, DetectionStrategy, CompletionResult
from open_deep_research.orchestration.report_builder import RunningReportBuilder
from open_deep_research.configuration import Configuration
from open_deep_research.state import (
    SequentialSupervisorState,
    SequentialAgentState,
    AgentExecutionReport,
    RunningReport
)


class TestSequentialSupervisorEndToEnd:
    """End-to-end integration tests for the Sequential Supervisor system."""
    
    def setup_method(self):
        """Set up test fixtures and temporary directories."""
        # Create temporary directory for test agents
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / ".open_deep_research" / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample agent files
        self._create_test_agents()
        
        # Initialize configuration
        self.config = Configuration()
        
        # Create supervisor config
        self.supervisor_config = SupervisorConfig(
            debug_mode=True,
            agent_timeout_seconds=30.0,
            max_agents_per_sequence=5
        )
        
        # Initialize agent registry
        self.agent_registry = AgentRegistry(project_root=str(self.temp_dir))
        
        # Initialize sequential supervisor
        with patch('open_deep_research.supervisor.sequential_supervisor.init_chat_model'):
            self.supervisor = SequentialSupervisor(
                agent_registry=self.agent_registry,
                config=self.supervisor_config,
                system_config=self.config
            )
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_agents(self):
        """Create test agent definition files."""
        # Academic Agent
        academic_agent = """# Academic Researcher Agent

## Description
Specialized in academic research, theoretical analysis, and scientific literature review.

## Expertise Areas
- Academic research methodologies
- Theoretical frameworks
- Scientific literature analysis
- Peer review processes

## Tools
- search
- scraper

## Completion Indicators
- Theoretical analysis complete
- Literature review finished
- Academic findings summarized

## Focus Questions
- What are the current theoretical frameworks?
- What does recent academic literature suggest?
- What are the scientific foundations?
"""
        (self.agents_dir / "academic_researcher.md").write_text(academic_agent)
        
        # Industry Agent
        industry_agent = """# Industry Analysis Agent

## Description
Focused on market dynamics, business applications, and industry trends.

## Expertise Areas
- Market analysis
- Business strategy
- Industry trends
- Competitive landscape

## Tools
- search
- scraper

## Completion Indicators
- Market analysis complete
- Industry research finished
- Business insights documented

## Focus Questions
- What are the current market trends?
- How are businesses applying this?
- What is the competitive landscape?
"""
        (self.agents_dir / "industry_analyst.md").write_text(industry_agent)
        
        # Technical Trends Agent
        technical_agent = """# Technical Trends Agent

## Description
Analyzes technical innovations, emerging technologies, and implementation patterns.

## Expertise Areas
- Technical architecture
- Emerging technologies
- Implementation strategies
- Innovation patterns

## Tools
- search
- scraper

## Completion Indicators
- Technical analysis complete
- Innovation trends documented
- Implementation patterns identified

## Focus Questions
- What are the emerging technical trends?
- How is this being implemented?
- What innovations are happening?
"""
        (self.agents_dir / "technical_trends.md").write_text(technical_agent)
    
    @pytest.mark.asyncio
    async def test_complete_sequential_workflow(self):
        """Test complete sequential workflow from start to finish."""
        # Create test state
        state = SequentialSupervisorState(
            research_topic="Impact of artificial intelligence on healthcare delivery",
            planned_sequence=["academic_researcher", "industry_analyst", "technical_trends"],
            sequence_position=0,
            handoff_ready=True,
            executed_agents=[],
            agent_insights={},
            agent_context={},
            agent_reports={},
            completion_signals={},
            supervisor_messages=[],
            sequence_modifications=[],
            sequence_start_time=None,
            running_report=None,
            last_agent_completed=None,
            current_agent=None
        )
        
        # Mock model responses for each agent
        mock_responses = [
            AIMessage(content="Academic research complete. AI in healthcare shows significant promise based on peer-reviewed literature. Key theoretical frameworks include human-AI collaboration models and evidence-based integration approaches. Research complete."),
            AIMessage(content="Industry analysis complete. Market adoption of AI in healthcare is accelerating, with major players investing heavily in diagnostic tools and patient care optimization. Market analysis complete."),
            AIMessage(content="Technical analysis complete. Implementation patterns show emphasis on interoperability, data privacy, and real-time processing capabilities. Technical analysis complete.")
        ]
        
        # Mock agent execution
        with patch.object(self.supervisor.model, 'bind_tools') as mock_bind_tools, \
             patch('open_deep_research.supervisor.sequential_supervisor.get_all_tools') as mock_get_tools, \
             patch('open_deep_research.supervisor.sequential_supervisor.think_tool'):
            
            mock_model = AsyncMock()
            mock_bind_tools.return_value = mock_model
            mock_get_tools.return_value = []
            
            # Set up response sequence
            mock_model.ainvoke = AsyncMock(side_effect=mock_responses)
            
            # Create workflow and execute
            workflow_graph = await self.supervisor.create_workflow_graph()
            workflow = workflow_graph.compile()
            
            # Execute workflow
            result = await workflow.ainvoke(state)
            
            # Verify workflow completion
            assert len(result.executed_agents) == 3
            assert all(agent in ["academic_researcher", "industry_analyst", "technical_trends"] 
                      for agent in result.executed_agents)
            assert result.running_report is not None
            assert result.running_report.total_agents_executed == 3
            assert len(result.running_report.all_insights) > 0
            
            # Verify agent execution order
            assert result.executed_agents[0] == "academic_researcher"
            assert result.executed_agents[1] == "industry_analyst"
            assert result.executed_agents[2] == "technical_trends"
    
    @pytest.mark.asyncio
    async def test_agent_registry_loading_integration(self):
        """Test agent registry loading and validation within supervisor."""
        # Verify agents are loaded correctly
        available_agents = self.supervisor.get_available_agents()
        assert len(available_agents) == 3
        assert "academic_researcher" in available_agents
        assert "industry_analyst" in available_agents
        assert "technical_trends" in available_agents
        
        # Test agent configuration retrieval
        academic_config = self.agent_registry.get_agent("academic_researcher")
        assert academic_config is not None
        assert "academic research methodologies" in academic_config.get("expertise_areas", [])
        assert "Theoretical analysis complete" in academic_config.get("completion_indicators", [])
        
        # Test sequence validation
        valid_sequence = ["academic_researcher", "industry_analyst"]
        validation_result = self.supervisor.validate_sequence(valid_sequence)
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
        
        # Test invalid sequence
        invalid_sequence = ["nonexistent_agent", "academic_researcher"]
        validation_result = self.supervisor.validate_sequence(invalid_sequence)
        assert validation_result["valid"] is False
        assert len(validation_result["errors"]) > 0


class TestAutomaticHandoffDetection:
    """Test automatic completion detection and handoff mechanisms."""
    
    def setup_method(self):
        """Set up completion detector tests."""
        self.detector = CompletionDetector(debug_mode=True)
        self.detector.set_completion_threshold(0.6)
    
    def test_completion_detection_with_explicit_indicators(self):
        """Test completion detection with explicit completion phrases."""
        test_cases = [
            {
                "content": "Based on my research analysis, I have completed the investigation. Research complete.",
                "expected_complete": True,
                "confidence_threshold": 0.8
            },
            {
                "content": "The analysis is still ongoing and requires more investigation.",
                "expected_complete": False,
                "confidence_threshold": 0.3
            },
            {
                "content": "Task accomplished! The comprehensive study shows significant findings.",
                "expected_complete": True,
                "confidence_threshold": 0.7
            },
            {
                "content": "I need to gather more data before providing conclusions.",
                "expected_complete": False,
                "confidence_threshold": 0.2
            }
        ]
        
        for case in test_cases:
            message = AIMessage(content=case["content"])
            result = self.detector.analyze_completion_patterns(
                message, 
                strategy=DetectionStrategy.COMBINED
            )
            
            assert result.is_complete == case["expected_complete"]
            assert result.confidence >= case["confidence_threshold"]
    
    def test_completion_detection_with_custom_indicators(self):
        """Test completion detection with custom agent-specific indicators."""
        custom_indicators = [
            "market analysis finished",
            "business insights documented",
            "competitive landscape mapped"
        ]
        
        message = AIMessage(content="After thorough investigation, the competitive landscape mapped and all business insights documented.")
        
        result = self.detector.analyze_completion_patterns(
            message,
            custom_indicators=custom_indicators,
            strategy=DetectionStrategy.COMBINED
        )
        
        assert result.is_complete is True
        assert result.confidence > 0.7
        assert any("competitive landscape mapped" in reason for reason in result.reasoning)


class TestContextSharingAndHandoffs:
    """Test context management and sharing between sequential agents."""
    
    def setup_method(self):
        """Set up context sharing tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / ".open_deep_research" / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create basic agent
        agent_content = """# Test Agent
## Description
Test agent for context sharing
## Expertise Areas
- Test expertise
## Tools
- search
"""
        (self.agents_dir / "test_agent.md").write_text(agent_content)
        
        self.registry = AgentRegistry(project_root=str(self.temp_dir))
        
        with patch('open_deep_research.supervisor.sequential_supervisor.init_chat_model'):
            self.supervisor = SequentialSupervisor(
                agent_registry=self.registry,
                config=SupervisorConfig()
            )
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_context_preparation_for_sequential_agents(self):
        """Test context preparation between sequential agents."""
        # Create supervisor state with previous agent context
        supervisor_state = SequentialSupervisorState(
            research_topic="Test research topic",
            planned_sequence=["test_agent"],
            sequence_position=1,
            executed_agents=["previous_agent"],
            agent_insights={"previous_agent": ["Key insight from previous agent", "Another important finding"]},
            agent_context={"previous_agent": {"execution_time": 45.0, "tool_calls_made": 3}},
            agent_reports={},
            completion_signals={},
            supervisor_messages=[],
            sequence_modifications=[],
            handoff_ready=True,
            sequence_start_time=datetime.utcnow(),
            running_report=None,
            last_agent_completed=None,
            current_agent=None
        )
        
        # Get agent config
        agent_config = self.registry.get_agent("test_agent")
        
        # Prepare agent context
        agent_state = await self.supervisor._prepare_agent_context(
            supervisor_state, "test_agent", agent_config
        )
        
        # Verify context preparation
        assert agent_state.agent_name == "test_agent"
        assert agent_state.research_topic == "Test research topic"
        assert agent_state.sequence_position == 1
        assert len(agent_state.previous_agent_insights) == 2
        assert "Key insight from previous agent" in agent_state.previous_agent_insights
        assert "execution_time" in agent_state.previous_agent_context
        assert agent_state.previous_agent_context["execution_time"] == 45.0


class TestRunningReportBuilding:
    """Test incremental running report building during sequential execution."""
    
    def setup_method(self):
        """Set up report building tests."""
        self.research_topic = "AI applications in renewable energy systems"
        self.sequence_name = "sequential_3_agents"
        self.planned_agents = ["academic_researcher", "industry_analyst", "technical_trends"]
    
    def test_report_initialization(self):
        """Test running report initialization."""
        report = RunningReportBuilder.initialize_report(
            research_topic=self.research_topic,
            sequence_name=self.sequence_name,
            planned_agents=self.planned_agents
        )
        
        assert report.research_topic == self.research_topic
        assert report.sequence_name == self.sequence_name
        assert report.planned_agents == self.planned_agents
        assert report.total_agents_executed == 0
        assert len(report.all_insights) == 0
        assert report.creation_timestamp is not None
    
    def test_incremental_agent_execution_addition(self):
        """Test adding agent execution results to running report."""
        # Initialize report
        report = RunningReportBuilder.initialize_report(
            research_topic=self.research_topic,
            sequence_name=self.sequence_name,
            planned_agents=self.planned_agents
        )
        
        # Create agent execution report
        agent_report = AgentExecutionReport(
            agent_name="academic_researcher",
            agent_type="research_agent",
            execution_start=datetime.utcnow(),
            execution_end=datetime.utcnow() + timedelta(seconds=30),
            execution_duration=30.0,
            insights=["Academic insight 1", "Academic insight 2"],
            research_content="Comprehensive academic research findings...",
            questions_addressed=["What does literature suggest?"],
            completion_confidence=0.85,
            insight_quality_score=0.9,
            research_depth_score=0.8,
            handoff_context={"execution_time": 30.0, "tool_calls_made": 3},
            suggested_next_questions=["How can this be applied industrially?"]
        )
        
        # Add to report
        updated_report = RunningReportBuilder.add_agent_execution(report, agent_report)
        
        # Verify update
        assert updated_report.total_agents_executed == 1
        assert len(updated_report.all_insights) == 2
        assert "Academic insight 1" in updated_report.all_insights
        assert "academic_researcher" in updated_report.agent_summaries
        assert updated_report.agent_summaries["academic_researcher"]["execution_duration"] == 30.0
    
    def test_executive_summary_updates(self):
        """Test executive summary updates during report building."""
        # Initialize and populate report
        report = RunningReportBuilder.initialize_report(
            research_topic=self.research_topic,
            sequence_name=self.sequence_name,
            planned_agents=self.planned_agents
        )
        
        # Add multiple agent reports
        for i, agent_name in enumerate(["academic_researcher", "industry_analyst"]):
            agent_report = AgentExecutionReport(
                agent_name=agent_name,
                agent_type="research_agent",
                execution_start=datetime.utcnow(),
                execution_end=datetime.utcnow() + timedelta(seconds=30),
                execution_duration=30.0,
                insights=[f"{agent_name} insight {j+1}" for j in range(2)],
                research_content=f"{agent_name} research content...",
                questions_addressed=[f"Question for {agent_name}"],
                completion_confidence=0.8,
                insight_quality_score=0.85,
                research_depth_score=0.8,
                handoff_context={"execution_time": 30.0},
                suggested_next_questions=[f"Next question for {agent_name}"]
            )
            report = RunningReportBuilder.add_agent_execution(report, agent_report)
        
        # Update executive summary
        updated_report = RunningReportBuilder.update_executive_summary(report)
        
        # Verify executive summary
        assert updated_report.executive_summary is not None
        assert len(updated_report.executive_summary) > 50
        assert self.research_topic.lower() in updated_report.executive_summary.lower()


class TestPerformanceRequirements:
    """Test performance requirements including handoff timing."""
    
    def setup_method(self):
        """Set up performance testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / ".open_deep_research" / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal test agent
        agent_content = """# Fast Agent
## Description
Minimal test agent for performance testing
## Expertise Areas
- Test
## Tools
- search
"""
        (self.agents_dir / "fast_agent.md").write_text(agent_content)
        
        self.registry = AgentRegistry(project_root=str(self.temp_dir))
        
        # Fast config for performance testing
        self.fast_config = SupervisorConfig(
            agent_timeout_seconds=10.0,
            debug_mode=False
        )
        
        with patch('open_deep_research.supervisor.sequential_supervisor.init_chat_model'):
            self.supervisor = SequentialSupervisor(
                agent_registry=self.registry,
                config=self.fast_config
            )
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_handoff_timing_requirement(self):
        """Test that handoff overhead is <3 seconds as required."""
        handoff_times = []
        
        # Test multiple handoffs
        for i in range(5):
            start_time = time.time()
            
            # Create minimal state
            state = SequentialSupervisorState(
                research_topic=f"Test topic {i}",
                planned_sequence=["fast_agent"],
                sequence_position=0,
                handoff_ready=True,
                executed_agents=[],
                agent_insights={},
                agent_context={},
                agent_reports={},
                completion_signals={},
                supervisor_messages=[],
                sequence_modifications=[],
                sequence_start_time=datetime.utcnow(),
                running_report=None,
                last_agent_completed=None,
                current_agent=None
            )
            
            # Mock fast agent execution
            with patch.object(self.supervisor.model, 'bind_tools') as mock_bind_tools, \
                 patch('open_deep_research.supervisor.sequential_supervisor.get_all_tools'), \
                 patch('open_deep_research.supervisor.sequential_supervisor.think_tool'):
                
                mock_model = AsyncMock()
                mock_bind_tools.return_value = mock_model
                mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Fast response. Task accomplished."))
                
                # Time the supervisor node execution (handoff logic)
                result = await self.supervisor.supervisor_node(state)
                
                handoff_time = time.time() - start_time
                handoff_times.append(handoff_time)
        
        # Verify handoff timing requirement
        avg_handoff_time = sum(handoff_times) / len(handoff_times)
        max_handoff_time = max(handoff_times)
        
        assert avg_handoff_time < 3.0, f"Average handoff time {avg_handoff_time:.2f}s exceeds 3s requirement"
        assert max_handoff_time < 5.0, f"Max handoff time {max_handoff_time:.2f}s is too high"
        
        print(f"Handoff performance: avg={avg_handoff_time:.3f}s, max={max_handoff_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_during_execution(self):
        """Test memory usage remains reasonable during execution."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute multiple sequences
        for i in range(3):
            state = SequentialSupervisorState(
                research_topic=f"Memory test topic {i}",
                planned_sequence=["fast_agent"],
                sequence_position=0,
                handoff_ready=True,
                executed_agents=[],
                agent_insights={},
                agent_context={},
                agent_reports={},
                completion_signals={},
                supervisor_messages=[],
                sequence_modifications=[],
                sequence_start_time=datetime.utcnow(),
                running_report=None,
                last_agent_completed=None,
                current_agent=None
            )
            
            with patch.object(self.supervisor.model, 'bind_tools') as mock_bind_tools, \
                 patch('open_deep_research.supervisor.sequential_supervisor.get_all_tools'), \
                 patch('open_deep_research.supervisor.sequential_supervisor.think_tool'):
                
                mock_model = AsyncMock()
                mock_bind_tools.return_value = mock_model
                mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Memory test response complete."))
                
                await self.supervisor.supervisor_node(state)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for test operations)
        assert memory_increase < 100, f"Memory increase of {memory_increase:.2f}MB is too high"
        
        print(f"Memory usage: initial={initial_memory:.2f}MB, final={final_memory:.2f}MB, increase={memory_increase:.2f}MB")


class TestErrorHandlingAndResilience:
    """Test error handling, recovery mechanisms, and system resilience."""
    
    def setup_method(self):
        """Set up error handling tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / ".open_deep_research" / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test agent
        (self.agents_dir / "error_test_agent.md").write_text("""# Error Test Agent
## Description
Agent for testing error handling
## Expertise Areas
- Error testing
## Tools
- search
""")
        
        self.registry = AgentRegistry(project_root=str(self.temp_dir))
        
        # Config with error handling enabled
        self.config = SupervisorConfig(
            continue_on_agent_failure=True,
            max_agent_retries=2,
            debug_mode=True
        )
        
        with patch('open_deep_research.supervisor.sequential_supervisor.init_chat_model'):
            self.supervisor = SequentialSupervisor(
                agent_registry=self.registry,
                config=self.config
            )
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_agent_timeout_handling(self):
        """Test handling of agent execution timeouts."""
        state = SequentialSupervisorState(
            research_topic="Timeout test",
            planned_sequence=["error_test_agent"],
            sequence_position=0,
            handoff_ready=False,
            current_agent="error_test_agent",
            executed_agents=[],
            agent_insights={},
            agent_context={},
            agent_reports={},
            completion_signals={},
            supervisor_messages=[],
            sequence_modifications=[],
            sequence_start_time=datetime.utcnow(),
            running_report=None,
            last_agent_completed=None
        )
        
        # Mock timeout scenario
        with patch.object(self.supervisor.model, 'bind_tools') as mock_bind_tools, \
             patch('open_deep_research.supervisor.sequential_supervisor.get_all_tools'), \
             patch('open_deep_research.supervisor.sequential_supervisor.think_tool'):
            
            mock_model = AsyncMock()
            mock_bind_tools.return_value = mock_model
            
            # Simulate timeout
            mock_model.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError("Agent execution timed out"))
            
            # Execute agent with timeout
            result = await self.supervisor.agent_executor_node(state)
            
            # Verify timeout handling
            assert isinstance(result, Command)
            assert result.goto == "supervisor"  # Should continue to supervisor
            assert len(state.sequence_modifications) > 0  # Should record error
            
            # Check error was recorded
            timeout_error = next((mod for mod in state.sequence_modifications 
                                if mod.get("type") == "agent_timeout"), None)
            assert timeout_error is not None
    
    @pytest.mark.asyncio
    async def test_missing_agent_handling(self):
        """Test handling when planned agent doesn't exist in registry."""
        state = SequentialSupervisorState(
            research_topic="Missing agent test",
            planned_sequence=["nonexistent_agent", "error_test_agent"],
            sequence_position=0,
            handoff_ready=True,
            executed_agents=[],
            agent_insights={},
            agent_context={},
            agent_reports={},
            completion_signals={},
            supervisor_messages=[],
            sequence_modifications=[],
            sequence_start_time=datetime.utcnow(),
            running_report=None,
            last_agent_completed=None,
            current_agent=None
        )
        
        # Execute supervisor node
        result = await self.supervisor.supervisor_node(state)
        
        # Should detect missing agent and route to error
        assert isinstance(result, Command)
        # First agent doesn't exist, so should go to error or skip to next
        assert result.goto in ["error", "supervisor"]
    
    def test_sequence_validation_error_detection(self):
        """Test sequence validation detects various error conditions."""
        # Empty sequence
        empty_result = self.supervisor.validate_sequence([])
        assert empty_result["valid"] is False
        assert "Empty sequence" in str(empty_result["errors"])
        
        # Too long sequence
        long_sequence = ["error_test_agent"] * 15
        long_result = self.supervisor.validate_sequence(long_sequence)
        assert long_result["valid"] is False
        assert "too long" in str(long_result["errors"]).lower()
        
        # Missing agents
        missing_result = self.supervisor.validate_sequence(["nonexistent1", "nonexistent2"])
        assert missing_result["valid"] is False
        assert "not found" in str(missing_result["errors"]).lower()
        
        # Valid sequence should pass
        valid_result = self.supervisor.validate_sequence(["error_test_agent"])
        assert valid_result["valid"] is True
        assert len(valid_result["errors"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])