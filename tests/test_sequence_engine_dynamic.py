"""Tests for sequence_engine.py dynamic pattern execution capabilities.

This test module validates that the SequenceOptimizationEngine properly handles
DynamicSequencePattern execution, metrics collection, and integration with
the existing sequence execution infrastructure.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import List

from langchain_core.runnables import RunnableConfig

from open_deep_research.sequencing.models import (
    DynamicSequencePattern,
    SequencePattern,
    AgentType,
    AgentExecutionResult,
    SequenceResult,
    ToolProductivityMetrics,
    SEQUENCE_PATTERNS
)
from open_deep_research.sequencing.sequence_engine import SequenceOptimizationEngine


class TestSequenceEngineDynamicExecution:
    """Test dynamic sequence pattern execution in SequenceOptimizationEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock configuration
        self.mock_config = Mock(spec=RunnableConfig)
        self.mock_config.get = Mock(return_value={})
        
        # Create engine instance with mocked dependencies
        with patch('open_deep_research.sequencing.sequence_engine.SupervisorResearchDirector'), \
             patch('open_deep_research.sequencing.sequence_engine.MetricsCalculator'), \
             patch('open_deep_research.sequencing.sequence_engine.SequenceAnalyzer'):
            self.engine = SequenceOptimizationEngine(
                config=self.mock_config,
                enable_real_time_metrics=False  # Disable for testing
            )
    
    def test_dynamic_sequence_pattern_handling(self):
        """Test that engine properly handles DynamicSequencePattern vs SequencePattern."""
        # Create dynamic pattern
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY],
            description="Future-back dynamic analysis",
            reasoning="Technical trends first to identify emerging opportunities",
            confidence_score=0.85,
            expected_advantages=["Future-oriented perspective", "Innovation identification"]
        )
        
        # Create standard pattern
        standard_pattern = SEQUENCE_PATTERNS["theory_first"]
        
        # Test pattern type detection
        assert isinstance(dynamic_pattern, DynamicSequencePattern)
        assert isinstance(standard_pattern, SequencePattern)
        assert not isinstance(standard_pattern, DynamicSequencePattern)
    
    @pytest.mark.asyncio
    async def test_execute_dynamic_sequence_basic(self):
        """Test basic execution of a dynamic sequence pattern."""
        # Create dynamic pattern
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY],
            description="Academic-industry validation sequence",
            reasoning="Academic foundation followed by market validation",
            confidence_score=0.8,
            expected_advantages=["Strong theoretical base", "Market applicability"]
        )
        
        research_topic = "Machine learning applications in renewable energy optimization"
        
        # Mock agent instances and their results
        mock_academic_agent = AsyncMock()
        mock_industry_agent = AsyncMock()
        
        # Create mock agent execution results
        academic_result = AgentExecutionResult(
            agent_type=AgentType.ACADEMIC,
            execution_order=1,
            received_questions=["What does research show about ML in energy?"],
            previous_insights=[],
            research_topic=research_topic,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            execution_duration=45.0,
            tool_calls_made=3,
            search_queries_executed=2,
            think_tool_usage_count=1,
            key_insights=["ML algorithms can optimize energy grids", "Academic research shows 15% efficiency gains"],
            research_findings="Academic research demonstrates significant potential...",
            refined_insights=["Energy grid optimization through ML is well-established in literature"],
            insight_quality_scores=[0.8, 0.9],
            research_depth_score=0.85,
            novelty_score=0.7,
            cognitive_offloading_detected=False,
            independent_reasoning_score=0.9
        )
        
        industry_result = AgentExecutionResult(
            agent_type=AgentType.INDUSTRY,
            execution_order=2,
            received_questions=["What market opportunities exist for ML energy solutions?"],
            previous_insights=["Energy grid optimization through ML is well-established in literature"],
            research_topic=research_topic,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            execution_duration=38.0,
            tool_calls_made=4,
            search_queries_executed=3,
            think_tool_usage_count=2,
            key_insights=["$50B market opportunity", "Major utility companies investing heavily"],
            research_findings="Market analysis shows strong commercial demand...",
            refined_insights=["Commercial viability confirmed with strong market demand"],
            insight_quality_scores=[0.9, 0.8],
            research_depth_score=0.8,
            novelty_score=0.75,
            cognitive_offloading_detected=False,
            independent_reasoning_score=0.85
        )
        
        mock_academic_agent.execute_research.return_value = academic_result
        mock_industry_agent.execute_research.return_value = industry_result
        
        # Mock agent creation
        with patch.object(self.engine, '_get_agent') as mock_get_agent:
            mock_get_agent.side_effect = [mock_academic_agent, mock_industry_agent]
            
            # Mock research director
            self.engine.research_director.direct_next_investigation = AsyncMock(
                return_value=Mock(questions=["What market opportunities exist for ML energy solutions?"])
            )
            self.engine.research_director.track_insight_productivity = AsyncMock(
                return_value=Mock(
                    insight_types=["TECHNICAL_FEASIBILITY"],
                    transition_quality=0.8
                )
            )
            
            # Mock metrics calculator
            mock_metrics = ToolProductivityMetrics(
                tool_productivity=0.75,
                research_quality_score=0.85,
                total_agent_calls=7,
                agent_efficiency=0.8,
                context_efficiency=0.9,
                time_to_value=30.0,
                insight_novelty=0.75,
                insight_relevance=0.9,
                insight_actionability=0.8,
                research_completeness=0.85,
                useful_insights_count=4,
                redundant_research_count=0,
                cognitive_offloading_incidents=0,
                relevant_context_used=0.9,
                total_context_available=1.0
            )
            self.engine.metrics_calculator.calculate_sequence_productivity = Mock(return_value=mock_metrics)
            
            # Execute dynamic sequence
            result = await self.engine.execute_sequence(dynamic_pattern, research_topic)
            
            # Verify result
            assert isinstance(result, SequenceResult)
            assert result.sequence_pattern == dynamic_pattern
            assert result.research_topic == research_topic
            assert len(result.agent_results) == 2
            assert result.agent_results[0].agent_type == AgentType.ACADEMIC
            assert result.agent_results[1].agent_type == AgentType.INDUSTRY
            assert result.overall_productivity_metrics == mock_metrics
            
            # Verify agents were called correctly
            assert mock_academic_agent.execute_research.called
            assert mock_industry_agent.execute_research.called
    
    @pytest.mark.asyncio
    async def test_execute_dynamic_sequence_method(self):
        """Test the execute_dynamic_sequence convenience method."""
        research_topic = "Blockchain applications in supply chain transparency"
        agent_order = [AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC]
        
        # Mock the execute_sequence method
        mock_result = Mock(spec=SequenceResult)
        mock_result.sequence_pattern = Mock()
        mock_result.research_topic = research_topic
        
        with patch.object(self.engine, 'execute_sequence', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result
            
            result = await self.engine.execute_dynamic_sequence(
                agent_order=agent_order,
                research_topic=research_topic,
                description="Custom business-technical-academic sequence",
                reasoning="Market needs drive technical solutions validated by academic research",
                confidence_score=0.9,
                expected_advantages=["Market-driven approach", "Technical feasibility", "Academic validation"]
            )
            
            # Verify execute_sequence was called with proper DynamicSequencePattern
            assert mock_execute.called
            call_args = mock_execute.call_args[0]
            pattern_arg = call_args[0]
            topic_arg = call_args[1]
            
            assert isinstance(pattern_arg, DynamicSequencePattern)
            assert pattern_arg.agent_order == agent_order
            assert pattern_arg.description == "Custom business-technical-academic sequence"
            assert pattern_arg.confidence_score == 0.9
            assert topic_arg == research_topic
            assert result == mock_result
    
    @pytest.mark.asyncio
    async def test_dynamic_sequence_with_variable_length(self):
        """Test execution of dynamic sequences with different lengths."""
        # Test single agent sequence
        single_agent_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Academic-only deep research",
            reasoning="Focus exclusively on academic literature and theoretical foundations",
            confidence_score=0.7
        )
        
        # Test four-agent sequence with repetition
        four_agent_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC],
            description="Academic-industry-technical-academic validation cycle",
            reasoning="Theory-practice-technology-validation approach",
            confidence_score=0.8
        )
        
        research_topic = "Sustainable urban transportation systems"
        
        # Mock agent execution for single agent
        mock_agent = AsyncMock()
        mock_result = AgentExecutionResult(
            agent_type=AgentType.ACADEMIC,
            execution_order=1,
            received_questions=["Academic research questions"],
            previous_insights=[],
            research_topic=research_topic,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            execution_duration=30.0,
            tool_calls_made=2,
            search_queries_executed=1,
            think_tool_usage_count=1,
            key_insights=["Academic insight"],
            research_findings="Research content",
            refined_insights=["Refined insight"],
            insight_quality_scores=[0.8],
            research_depth_score=0.8,
            novelty_score=0.7,
            cognitive_offloading_detected=False,
            independent_reasoning_score=0.9
        )
        mock_agent.execute_research.return_value = mock_result
        
        with patch.object(self.engine, '_get_agent', return_value=mock_agent), \
             patch.object(self.engine.metrics_calculator, 'calculate_sequence_productivity') as mock_calc:
            
            mock_calc.return_value = Mock(spec=ToolProductivityMetrics)
            
            # Test single agent execution
            result_single = await self.engine.execute_sequence(single_agent_pattern, research_topic)
            assert len(result_single.agent_results) == 1
            
            # Test four agent execution
            result_four = await self.engine.execute_sequence(four_agent_pattern, research_topic)
            assert len(result_four.agent_results) == 4
            
            # Verify agent was called 4 times for four-agent sequence
            assert mock_agent.execute_research.call_count >= 4
    
    @pytest.mark.asyncio
    async def test_dynamic_sequence_metrics_integration(self):
        """Test that dynamic sequences integrate properly with metrics collection."""
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.TECHNICAL_TRENDS, AgentType.INDUSTRY],
            description="Tech-market validation",
            reasoning="Technology first, then market assessment",
            confidence_score=0.85
        )
        
        research_topic = "Edge computing infrastructure deployment"
        execution_id = "test_execution_123"
        
        # Mock metrics aggregator
        mock_aggregator = Mock()
        self.engine.metrics_aggregator = mock_aggregator
        self.engine.enable_real_time_metrics = True
        
        # Mock agent results
        mock_agents = [AsyncMock(), AsyncMock()]
        mock_results = [
            Mock(spec=AgentExecutionResult, agent_type=AgentType.TECHNICAL_TRENDS, execution_duration=25.0),
            Mock(spec=AgentExecutionResult, agent_type=AgentType.INDUSTRY, execution_duration=30.0)
        ]
        
        for agent, result in zip(mock_agents, mock_results):
            agent.execute_research.return_value = result
        
        with patch.object(self.engine, '_get_agent') as mock_get_agent, \
             patch.object(self.engine.metrics_calculator, 'calculate_sequence_productivity') as mock_calc, \
             patch.object(self.engine.metrics_calculator, 'calculate_real_time_productivity', new_callable=AsyncMock) as mock_realtime:
            
            mock_get_agent.side_effect = mock_agents
            mock_calc.return_value = Mock(spec=ToolProductivityMetrics)
            
            # Mock other dependencies
            self.engine.research_director.direct_next_investigation = AsyncMock(
                return_value=Mock(questions=["Test questions"])
            )
            self.engine.research_director.track_insight_productivity = AsyncMock(
                return_value=Mock(insight_types=["TECHNICAL_FEASIBILITY"], transition_quality=0.8)
            )
            
            # Execute with metrics
            result = await self.engine.execute_sequence(dynamic_pattern, research_topic, execution_id)
            
            # Verify metrics aggregator was called
            assert mock_aggregator.collect_sequence_metrics.called
            
            # Verify real-time metrics were calculated
            assert mock_realtime.called
    
    def test_dynamic_pattern_synthesis_formatting(self):
        """Test synthesis formatting for dynamic patterns."""
        dynamic_pattern = DynamicSequencePattern(
            sequence_id="test_dynamic_123",
            agent_order=[AgentType.INDUSTRY, AgentType.ACADEMIC],
            description="Market-academic validation sequence",
            reasoning="Business needs validated by academic research",
            confidence_score=0.8
        )
        
        # Mock agent results
        mock_results = [
            Mock(
                spec=AgentExecutionResult,
                agent_type=AgentType.INDUSTRY,
                execution_duration=35.0,
                tool_calls_made=3,
                key_insights=["Market insight 1", "Market insight 2"],
                research_findings="Industry analysis shows strong market demand..."
            ),
            Mock(
                spec=AgentExecutionResult,
                agent_type=AgentType.ACADEMIC,
                execution_duration=40.0,
                tool_calls_made=4,
                key_insights=["Academic insight 1"],
                research_findings="Academic research validates commercial applications..."
            )
        ]
        
        research_topic = "AI-powered financial advisory services"
        
        # Test synthesis generation
        synthesis = asyncio.run(self.engine._synthesize_research_findings(
            mock_results, research_topic, dynamic_pattern
        ))
        
        # Verify synthesis contains dynamic pattern information
        assert "Dynamic Pattern" in synthesis
        assert "test_dynamic_123" in synthesis or dynamic_pattern.sequence_id[:8] in synthesis
        assert "Market-academic validation sequence" in synthesis
        assert "Industry → Academic" in synthesis
        assert research_topic in synthesis
    
    @pytest.mark.asyncio
    async def test_dynamic_sequence_error_handling(self):
        """Test error handling during dynamic sequence execution."""
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Test pattern for error handling",
            reasoning="Test reasoning",
            confidence_score=0.7
        )
        
        research_topic = "Test topic"
        execution_id = "test_error_execution"
        
        # Mock agent that raises an exception
        mock_agent = AsyncMock()
        mock_agent.execute_research.side_effect = Exception("Agent execution failed")
        
        # Mock metrics aggregator for error reporting
        mock_aggregator = Mock()
        self.engine.metrics_aggregator = mock_aggregator
        
        with patch.object(self.engine, '_get_agent', return_value=mock_agent):
            # Should propagate the exception and report to metrics
            with pytest.raises(Exception, match="Agent execution failed"):
                await self.engine.execute_sequence(dynamic_pattern, research_topic, execution_id)
            
            # Verify error was reported to metrics aggregator
            error_calls = [
                call for call in mock_aggregator.collect_sequence_metrics.call_args_list
                if len(call[1]) > 0 and call[1].get('status_update') == 'failed'
            ]
            assert len(error_calls) > 0


class TestDynamicSequenceCompatibility:
    """Test compatibility between dynamic and standard sequence patterns."""
    
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
    
    def test_mixed_pattern_type_comparison(self):
        """Test that dynamic and standard patterns can coexist in comparisons."""
        # Create both pattern types
        standard_pattern = SEQUENCE_PATTERNS["theory_first"]
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
            description="Dynamic theory-first approach",
            reasoning="LLM-optimized academic foundation",
            confidence_score=0.9,
            strategy="theory_first"  # Associate with standard strategy
        )
        
        # Both should have compatible agent_order
        assert standard_pattern.agent_order == dynamic_pattern.agent_order
        
        # Both should work with engine methods
        assert hasattr(standard_pattern, 'description')
        assert hasattr(dynamic_pattern, 'description')
        
        # Dynamic pattern has additional properties
        assert hasattr(dynamic_pattern, 'reasoning')
        assert hasattr(dynamic_pattern, 'confidence_score')
        assert hasattr(dynamic_pattern, 'sequence_length')
    
    def test_performance_summary_with_dynamic_patterns(self):
        """Test that performance summary handles mixed pattern types."""
        # Mock execution history with both pattern types
        standard_result = Mock(spec=SequenceResult)
        standard_result.sequence_pattern = SEQUENCE_PATTERNS["market_first"]
        standard_result.overall_productivity_metrics = Mock(tool_productivity=0.8)
        
        dynamic_result = Mock(spec=SequenceResult)
        dynamic_result.sequence_pattern = DynamicSequencePattern(
            sequence_id="dynamic_abc123",
            agent_order=[AgentType.INDUSTRY, AgentType.ACADEMIC],
            description="Dynamic market-first",
            reasoning="Dynamic reasoning",
            confidence_score=0.85
        )
        dynamic_result.overall_productivity_metrics = Mock(tool_productivity=0.9)
        
        # Add to execution history
        self.engine.execution_history = [standard_result, dynamic_result]
        
        # Get performance summary
        summary = self.engine.get_performance_summary()
        
        # Should handle both pattern types
        assert summary["total_executions"] == 2
        assert len(summary["strategies_tested"]) == 2
        assert "market_first" in summary["average_productivity_by_strategy"]
        assert any("dynamic_" in strategy for strategy in summary["strategies_tested"])


if __name__ == "__main__":
    pytest.main([__file__])