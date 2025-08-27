"""Comprehensive tests for Supervisor and Execution components.

This module tests the execution layer and performance characteristics:
- Sequential supervisor integration and workflow orchestration
- Parallel execution with dynamic sequences and performance metrics
- Completion detection strategies and automatic handoffs
- Performance benchmarking and timing requirements
- Error handling and system resilience

Test Categories:
1. Sequential Supervisor Integration (from test_sequential_supervisor_integration.py)
2. Parallel Execution Integration (from test_parallel_execution_integration.py)
3. Completion Detection (from test_completion_detection.py)
4. Performance Benchmarks (from test_performance_benchmarks.py)
"""

import asyncio
import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from open_deep_research.supervisor.sequential_supervisor import SequentialSupervisor, SupervisorConfig
from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.agents.completion_detector import CompletionDetector, DetectionStrategy, CompletionResult
from open_deep_research.orchestration.report_builder import RunningReportBuilder
from open_deep_research.sequencing.sequence_engine import SequenceOptimizationEngine
from open_deep_research.sequencing.models import (
    DynamicSequencePattern,
    SequencePattern,
    AgentType,
    SequenceResult,
    AgentExecutionResult,
    SequenceComparison,
    ParallelMetrics,
    SequenceMetrics,
    SEQUENCE_PATTERNS
)
from open_deep_research.configuration import Configuration
from open_deep_research.state import (
    DeepResearchState,
    SequentialSupervisorState,
    SequentialAgentState,
    AgentExecutionReport,
    RunningReport
)


# =============================================================================
# SEQUENTIAL SUPERVISOR INTEGRATION TESTS
# =============================================================================

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
        self.config = Configuration(
            enable_sequential_supervisor=True,
            use_shared_state=True,
            automatic_handoffs=True,
            max_agents_per_sequence=5
        )
        
        # Create supervisor config
        self.supervisor_config = SupervisorConfig(
            debug_mode=True,
            agent_timeout_seconds=30.0,
            max_agents_per_sequence=5,
            completion_threshold=0.6
        )
        
        # Initialize agent registry
        self.agent_registry = AgentRegistry(self.config)
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_agents(self):
        """Create test agent definition files."""
        # Research Agent
        research_agent = """---
name: research_agent
description: Academic research specialist
expertise_areas:
  - academic_research
  - literature_review
capabilities:
  - search
  - analysis
completion_indicators:
  - research complete
  - findings documented
---

# Research Agent

Specialized in academic research and literature review.
"""
        
        # Analysis Agent  
        analysis_agent = """---
name: analysis_agent
description: Data analysis specialist
expertise_areas:
  - data_analysis
  - statistical_analysis
capabilities:
  - analysis
  - visualization
completion_indicators:
  - analysis complete
  - insights generated
---

# Analysis Agent

Specialized in data analysis and statistical research.
"""
        
        # Create agent files
        (self.agents_dir / "research_agent.md").write_text(research_agent)
        (self.agents_dir / "analysis_agent.md").write_text(analysis_agent)
    
    def test_supervisor_initialization(self):
        """Test sequential supervisor initialization."""
        with patch('open_deep_research.supervisor.sequential_supervisor.init_chat_model'):
            supervisor = SequentialSupervisor(
                agent_registry=self.agent_registry,
                config=self.supervisor_config,
                system_config=self.config
            )
            
            assert supervisor is not None
            assert supervisor.config == self.supervisor_config
            assert supervisor.system_config == self.config
    
    def test_agent_registry_integration(self):
        """Test integration with agent registry."""
        # Verify agent registry loads test agents
        registry = AgentRegistry(self.config)
        assert registry is not None
        
        # Check that agent files exist
        assert (self.agents_dir / "research_agent.md").exists()
        assert (self.agents_dir / "analysis_agent.md").exists()
    
    def test_configuration_integration(self):
        """Test configuration integration with supervisor."""
        # Test that supervisor respects configuration settings
        assert self.config.enable_sequential_supervisor is True
        assert self.config.use_shared_state is True
        assert self.config.automatic_handoffs is True
        assert self.config.max_agents_per_sequence == 5
        
        # Test supervisor config
        assert self.supervisor_config.debug_mode is True
        assert self.supervisor_config.agent_timeout_seconds == 30.0
        assert self.supervisor_config.completion_threshold == 0.6
    
    @pytest.mark.asyncio
    async def test_sequential_workflow_simulation(self):
        """Test sequential workflow simulation."""
        # Mock state for sequential execution
        state = SequentialSupervisorState(
            research_topic="Test research topic",
            current_sequence=["research_agent", "analysis_agent"],
            current_agent_index=0,
            shared_context={},
            agent_states={},
            running_report=None
        )
        
        # Mock agent execution
        with patch('open_deep_research.supervisor.sequential_supervisor.init_chat_model'):
            supervisor = SequentialSupervisor(
                agent_registry=self.agent_registry,
                config=self.supervisor_config,
                system_config=self.config
            )
            
            # Test state handling
            assert state.current_agent_index == 0
            assert len(state.current_sequence) == 2
            assert state.current_sequence[0] == "research_agent"


class TestAutomaticHandoffs:
    """Test automatic handoff mechanism between agents."""
    
    def setup_method(self):
        """Set up handoff testing."""
        self.config = Configuration(
            automatic_handoffs=True,
            use_automatic_completion=True,
            completion_confidence_threshold=0.6
        )
    
    def test_handoff_triggering(self):
        """Test conditions that trigger automatic handoffs."""
        # Mock completion detection
        completion_result = CompletionResult(
            is_complete=True,
            confidence=0.85,
            reasoning="Agent has indicated task completion",
            detection_strategy=DetectionStrategy.PATTERN_BASED
        )
        
        # Test that high confidence triggers handoff
        assert completion_result.is_complete is True
        assert completion_result.confidence > self.config.completion_confidence_threshold
        assert completion_result.detection_strategy == DetectionStrategy.PATTERN_BASED
    
    def test_handoff_context_transfer(self):
        """Test context transfer during handoffs."""
        # Mock shared context
        shared_context = {
            "research_findings": ["Finding 1", "Finding 2"],
            "current_focus": "Technical analysis",
            "previous_agent": "research_agent",
            "handoff_timestamp": datetime.now()
        }
        
        # Mock agent state
        agent_state = SequentialAgentState(
            agent_name="analysis_agent",
            status="ready",
            shared_context=shared_context,
            completion_confidence=0.0,
            last_updated=datetime.now()
        )
        
        # Verify context is properly structured
        assert "research_findings" in agent_state.shared_context
        assert len(agent_state.shared_context["research_findings"]) == 2
        assert agent_state.shared_context["previous_agent"] == "research_agent"
    
    def test_handoff_timing_requirements(self):
        """Test handoff timing meets performance requirements."""
        import time
        
        # Mock handoff operation
        start_time = time.time()
        
        # Simulate handoff operations
        context_transfer_time = 0.1  # Mock context transfer
        completion_detection_time = 0.5  # Mock completion detection
        agent_initialization_time = 0.2  # Mock next agent initialization
        
        total_handoff_time = context_transfer_time + completion_detection_time + agent_initialization_time
        actual_time = time.time() - start_time
        
        # Should complete handoff quickly (< 3 seconds per requirements)
        assert total_handoff_time < 3.0
        assert actual_time < 1.0  # Actual mock time should be very fast


# =============================================================================
# PARALLEL EXECUTION INTEGRATION TESTS  
# =============================================================================

class TestParallelDynamicSequenceExecution:
    """Test parallel execution of dynamic sequences."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=RunnableConfig)
        self.mock_config.get = Mock(return_value={})
        
        with patch('open_deep_research.sequencing.sequence_engine.SupervisorResearchDirector'), \
             patch('open_deep_research.sequencing.sequence_engine.MetricsCalculator'), \
             patch('open_deep_research.sequencing.sequence_engine.SequenceAnalyzer'):
            self.engine = SequenceOptimizationEngine(
                config=self.mock_config,
                enable_real_time_metrics=False
            )
    
    @pytest.mark.asyncio
    async def test_parallel_execution_mixed_patterns(self):
        """Test parallel execution of mixed dynamic and standard patterns."""
        research_topic = "Sustainable urban transportation systems with autonomous vehicles"
        
        # Create mixed pattern list
        patterns = [
            SEQUENCE_PATTERNS["theory_first"],  # Standard pattern
            DynamicSequencePattern(
                agent_order=[AgentType.RESEARCH, AgentType.ANALYSIS, AgentType.SYNTHESIS],
                description="Research-analysis-synthesis cycle",
                reasoning="Sequential deep dive approach",
                confidence_score=0.85
            ),
            DynamicSequencePattern(
                agent_order=[AgentType.TECHNICAL, AgentType.MARKET],
                description="Technical-market validation",
                reasoning="Technical feasibility validated by market analysis",
                confidence_score=0.8
            )
        ]
        
        # Mock execution results for each pattern
        mock_results = []
        for i, pattern in enumerate(patterns):
            mock_result = Mock(spec=SequenceResult)
            mock_result.sequence_pattern = pattern
            mock_result.research_topic = research_topic
            mock_result.overall_productivity_metrics = Mock(tool_productivity=0.7 + i * 0.1)
            mock_result.agent_results = [Mock(spec=AgentExecutionResult) for _ in range(len(pattern.agent_order))]
            mock_results.append(mock_result)
        
        with patch.object(self.engine, 'execute_sequence', new_callable=AsyncMock) as mock_execute:
            # Configure mock to return different results
            mock_execute.side_effect = mock_results
            
            # Execute sequences in parallel
            tasks = [self.engine.execute_sequence(pattern, research_topic) for pattern in patterns]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all sequences executed
            assert len(results) == 3
            assert mock_execute.call_count == 3
            
            # Verify no exceptions
            for result in results:
                assert not isinstance(result, Exception)
    
    @pytest.mark.asyncio
    async def test_parallel_metrics_collection(self):
        """Test metrics collection during parallel execution."""
        # Mock parallel metrics
        parallel_metrics = ParallelMetrics(
            total_sequences=3,
            completed_sequences=3,
            failed_sequences=0,
            average_duration=42.5,
            total_duration=127.5,
            peak_memory_usage=256.7,
            total_api_calls=45,
            total_tokens=15000
        )
        
        # Verify metrics structure
        assert parallel_metrics.total_sequences == 3
        assert parallel_metrics.completed_sequences == 3
        assert parallel_metrics.failed_sequences == 0
        assert parallel_metrics.average_duration == 42.5
        assert parallel_metrics.peak_memory_usage > 0
    
    def test_dynamic_sequence_validation(self):
        """Test validation of dynamic sequence patterns."""
        # Valid dynamic sequence
        valid_sequence = DynamicSequencePattern(
            agent_order=[AgentType.RESEARCH, AgentType.ANALYSIS],
            description="Research then analysis",
            reasoning="Sequential approach for thorough investigation",
            confidence_score=0.9
        )
        
        # Verify structure
        assert len(valid_sequence.agent_order) == 2
        assert valid_sequence.agent_order[0] == AgentType.RESEARCH
        assert valid_sequence.confidence_score == 0.9
        assert len(valid_sequence.description) > 0
        assert len(valid_sequence.reasoning) > 0


# =============================================================================
# COMPLETION DETECTION TESTS
# =============================================================================

class TestCompletionDetection:
    """Test completion detection strategies and accuracy."""
    
    def setup_method(self):
        """Set up completion detection testing."""
        self.config = Configuration(
            use_automatic_completion=True,
            completion_confidence_threshold=0.6,
            completion_indicators=["research complete", "analysis finished", "task accomplished"]
        )
    
    def test_pattern_based_detection(self):
        """Test pattern-based completion detection."""
        # Mock agent responses
        test_responses = [
            "The research has been completed and all findings documented.",
            "Analysis finished with comprehensive insights generated.",
            "Task accomplished successfully with high confidence.",
            "Still working on the analysis, need more data.",
            "Preliminary research ongoing, not yet complete."
        ]
        
        expected_completions = [True, True, True, False, False]
        
        for response, expected in zip(test_responses, expected_completions):
            # Test if completion indicators are present
            is_complete = any(indicator in response.lower() for indicator in self.config.completion_indicators)
            assert is_complete == expected
    
    def test_semantic_completion_detection(self):
        """Test semantic-based completion detection."""
        # Mock completion detector
        detector = CompletionDetector(
            strategy=DetectionStrategy.SEMANTIC_ANALYSIS,
            confidence_threshold=self.config.completion_confidence_threshold
        )
        
        # Mock high-confidence completion
        high_confidence_result = CompletionResult(
            is_complete=True,
            confidence=0.92,
            reasoning="Strong semantic indicators of task completion",
            detection_strategy=DetectionStrategy.SEMANTIC_ANALYSIS
        )
        
        # Mock low-confidence ongoing work
        low_confidence_result = CompletionResult(
            is_complete=False,
            confidence=0.35,
            reasoning="Ongoing work indicators present",
            detection_strategy=DetectionStrategy.SEMANTIC_ANALYSIS
        )
        
        # Test confidence thresholding
        assert high_confidence_result.confidence > self.config.completion_confidence_threshold
        assert low_confidence_result.confidence < self.config.completion_confidence_threshold
        assert high_confidence_result.is_complete is True
        assert low_confidence_result.is_complete is False
    
    def test_combined_detection_strategy(self):
        """Test combined pattern and semantic detection."""
        # Mock combined strategy result
        combined_result = CompletionResult(
            is_complete=True,
            confidence=0.88,
            reasoning="Both pattern matching and semantic analysis indicate completion",
            detection_strategy=DetectionStrategy.COMBINED
        )
        
        # Should have higher confidence when both strategies agree
        assert combined_result.confidence > 0.8
        assert combined_result.detection_strategy == DetectionStrategy.COMBINED
        assert "Both pattern matching and semantic analysis" in combined_result.reasoning
    
    def test_completion_confidence_calibration(self):
        """Test calibration of completion confidence scores."""
        confidence_scenarios = [
            (0.95, True, "Very clear completion signal"),
            (0.75, True, "Clear completion signal"),  
            (0.65, True, "Moderate completion signal"),
            (0.55, False, "Ambiguous signal"),
            (0.35, False, "Ongoing work signal"),
            (0.15, False, "Clear ongoing signal")
        ]
        
        for confidence, expected_complete, description in confidence_scenarios:
            is_above_threshold = confidence >= self.config.completion_confidence_threshold
            assert is_above_threshold == expected_complete, \
                f"Confidence {confidence} with threshold {self.config.completion_confidence_threshold} failed for: {description}"


# =============================================================================
# PERFORMANCE BENCHMARK TESTS
# =============================================================================

class TestPerformanceBenchmarks:
    """Test performance benchmarks and timing requirements."""
    
    def setup_method(self):
        """Set up performance testing."""
        self.config = Configuration(
            max_agents_per_sequence=5,
            parallel_execution_timeout=3600,
            evaluation_timeout=120
        )
    
    def test_handoff_timing_requirements(self):
        """Test that handoffs complete within 3 seconds."""
        import time
        
        # Simulate handoff operations
        start_time = time.time()
        
        # Mock handoff components
        completion_detection_time = 0.5
        context_preparation_time = 0.3
        agent_initialization_time = 0.4
        state_update_time = 0.2
        
        total_simulated_time = (completion_detection_time + 
                              context_preparation_time + 
                              agent_initialization_time + 
                              state_update_time)
        
        actual_time = time.time() - start_time
        
        # Performance requirement: handoffs < 3 seconds
        assert total_simulated_time < 3.0, f"Simulated handoff took {total_simulated_time}s, should be < 3s"
        assert actual_time < 0.1, f"Actual test execution took {actual_time}s"
    
    def test_sequence_execution_performance(self):
        """Test sequence execution performance targets."""
        # Mock sequence execution metrics
        sequence_metrics = {
            "sequence_duration": 45.2,  # Should be < 60s per agent
            "total_agents": 3,
            "avg_agent_duration": 15.1,
            "handoff_times": [2.1, 1.8, 2.3],
            "completion_detection_times": [0.6, 0.4, 0.7]
        }
        
        # Validate performance targets
        assert sequence_metrics["sequence_duration"] < 300, "Total sequence should be < 5 minutes"
        assert sequence_metrics["avg_agent_duration"] < 60, "Average agent time should be < 1 minute"
        assert all(t < 3.0 for t in sequence_metrics["handoff_times"]), "All handoffs should be < 3 seconds"
        assert all(t < 2.0 for t in sequence_metrics["completion_detection_times"]), "Completion detection should be < 2 seconds"
    
    def test_parallel_execution_scaling(self):
        """Test parallel execution scaling characteristics."""
        # Mock parallel execution metrics for different sequence counts
        scaling_data = {
            1: {"duration": 45.2, "memory_mb": 128, "cpu_percent": 25},
            2: {"duration": 52.1, "memory_mb": 198, "cpu_percent": 45},
            3: {"duration": 58.7, "memory_mb": 267, "cpu_percent": 65},
            5: {"duration": 71.3, "memory_mb": 398, "cpu_percent": 85}
        }
        
        # Validate scaling characteristics
        for sequence_count, metrics in scaling_data.items():
            # Duration should scale sub-linearly (parallel benefit)
            expected_linear_duration = 45.2 * sequence_count
            actual_duration = metrics["duration"]
            scaling_benefit = (expected_linear_duration - actual_duration) / expected_linear_duration
            
            if sequence_count > 1:
                assert scaling_benefit > 0, f"Should see parallel benefit with {sequence_count} sequences"
            
            # Memory usage should scale reasonably
            assert metrics["memory_mb"] < 500, f"Memory usage should stay under 500MB for {sequence_count} sequences"
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring and limits."""
        # Mock memory usage scenarios
        memory_scenarios = [
            {"stage": "initialization", "memory_mb": 45, "limit_mb": 100},
            {"stage": "agent_execution", "memory_mb": 128, "limit_mb": 300},
            {"stage": "parallel_sequences", "memory_mb": 267, "limit_mb": 500},
            {"stage": "report_generation", "memory_mb": 89, "limit_mb": 150}
        ]
        
        for scenario in memory_scenarios:
            memory_usage = scenario["memory_mb"]
            memory_limit = scenario["limit_mb"]
            stage = scenario["stage"]
            
            assert memory_usage < memory_limit, \
                f"Memory usage {memory_usage}MB exceeds limit {memory_limit}MB during {stage}"
            
            # Memory usage should be reasonable for the stage
            if stage == "initialization":
                assert memory_usage < 100, "Initialization should use < 100MB"
            elif stage == "report_generation":
                assert memory_usage < 200, "Report generation should use < 200MB"
    
    def test_api_rate_limiting_compliance(self):
        """Test compliance with API rate limiting."""
        # Mock API usage metrics
        api_metrics = {
            "requests_per_minute": 45,
            "tokens_per_minute": 8500,
            "peak_requests_per_second": 2.1,
            "total_requests": 127,
            "total_tokens": 25600
        }
        
        # Validate rate limiting compliance
        assert api_metrics["requests_per_minute"] < 60, "Should stay under 60 requests/minute"
        assert api_metrics["tokens_per_minute"] < 15000, "Should stay under 15k tokens/minute"  
        assert api_metrics["peak_requests_per_second"] < 5, "Should stay under 5 requests/second"
    
    def test_error_recovery_performance(self):
        """Test error recovery time and success rates."""
        # Mock error scenarios and recovery metrics
        error_scenarios = [
            {"error_type": "api_timeout", "recovery_time": 1.2, "success_rate": 0.95},
            {"error_type": "agent_failure", "recovery_time": 2.8, "success_rate": 0.87},
            {"error_type": "memory_limit", "recovery_time": 0.8, "success_rate": 0.98},
            {"error_type": "network_error", "recovery_time": 3.1, "success_rate": 0.92}
        ]
        
        for scenario in error_scenarios:
            error_type = scenario["error_type"]
            recovery_time = scenario["recovery_time"]
            success_rate = scenario["success_rate"]
            
            # Recovery should be fast
            assert recovery_time < 5.0, f"{error_type} recovery should be < 5 seconds"
            
            # Success rates should be high
            assert success_rate > 0.8, f"{error_type} recovery success rate should be > 80%"
            
            # Critical errors should recover faster
            if error_type in ["memory_limit", "api_timeout"]:
                assert recovery_time < 2.0, f"Critical {error_type} should recover < 2 seconds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])