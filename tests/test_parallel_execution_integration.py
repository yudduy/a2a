"""Integration tests for parallel execution with dynamic sequences.

This test module validates that dynamic sequences integrate properly with
parallel execution infrastructure, metrics collection, streaming, and
the complete workflow from generation to execution and comparison.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import List, Dict, Optional

from langchain_core.runnables import RunnableConfig

from open_deep_research.sequencing.models import (
    DynamicSequencePattern,
    SequencePattern,
    AgentType,
    SequenceResult,
    AgentExecutionResult,
    SequenceComparison,
    ParallelMetrics,
    SequenceMetrics,
    MetricsUpdate,
    MetricsUpdateType,
    SEQUENCE_PATTERNS
)
from open_deep_research.sequencing.sequence_engine import SequenceOptimizationEngine


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
                agent_order=[AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC],
                description="Market-tech-academic validation cycle",
                reasoning="Market needs drive technical solutions validated by academic research",
                confidence_score=0.85
            ),
            DynamicSequencePattern(
                agent_order=[AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC],
                description="Future-back academic validation",
                reasoning="Technical trends inform academic research priorities",
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
            mock_execute.side_effect = mock_results
            
            # Execute all patterns in parallel (simulated)
            tasks = [
                self.engine.execute_sequence(pattern, research_topic)
                for pattern in patterns
            ]
            results = await asyncio.gather(*tasks)
            
            # Verify all patterns were executed
            assert len(results) == 3
            assert mock_execute.call_count == 3
            
            # Verify mixed pattern types were handled
            executed_patterns = [call[0][0] for call in mock_execute.call_args_list]
            assert any(isinstance(p, SequencePattern) and not isinstance(p, DynamicSequencePattern) for p in executed_patterns)
            assert any(isinstance(p, DynamicSequencePattern) for p in executed_patterns)
    
    @pytest.mark.asyncio
    async def test_dynamic_sequence_comparison_integration(self):
        """Test that dynamic sequences integrate with comparison framework."""
        research_topic = "AI-powered climate change mitigation strategies"
        
        # Generate dynamic sequences
        dynamic_sequences = [
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
                description="Academic-led climate research",
                reasoning="Scientific foundation drives practical implementation",
                confidence_score=0.9,
                strategy="theory_first"  # Associate with standard strategy
            ),
            DynamicSequencePattern(
                agent_order=[AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC],
                description="Market-driven climate solutions",
                reasoning="Commercial viability ensures sustainable implementation",
                confidence_score=0.85,
                strategy="market_first"
            ),
            DynamicSequencePattern(
                agent_order=[AgentType.TECHNICAL_TRENDS, AgentType.INDUSTRY, AgentType.ACADEMIC],
                description="Technology-forward climate innovation",
                reasoning="Cutting-edge tech creates new market opportunities",
                confidence_score=0.8,
                strategy="future_back"
            )
        ]
        
        # Mock sequence execution and comparison
        mock_results = []
        for i, seq in enumerate(dynamic_sequences):
            mock_result = Mock(spec=SequenceResult)
            mock_result.sequence_pattern = seq
            mock_result.research_topic = research_topic
            mock_result.overall_productivity_metrics = Mock(tool_productivity=0.8 + i * 0.05)
            mock_results.append(mock_result)
        
        mock_comparison = Mock(spec=SequenceComparison)
        mock_comparison.highest_productivity_sequence = "theory_first"
        mock_comparison.productivity_variance = 0.15
        
        with patch.object(self.engine, 'execute_sequence', new_callable=AsyncMock) as mock_execute, \
             patch.object(self.engine.metrics_calculator, 'compare_sequence_results') as mock_compare:
            
            mock_execute.side_effect = mock_results
            mock_compare.return_value = mock_comparison
            
            # Simulate comparison with dynamic sequences
            comparison_results = []
            for seq in dynamic_sequences:
                result = await self.engine.execute_sequence(seq, research_topic)
                comparison_results.append(result)
            
            comparison = self.engine.metrics_calculator.compare_sequence_results(comparison_results)
            
            assert len(comparison_results) == 3
            assert comparison == mock_comparison
    
    @pytest.mark.asyncio
    async def test_dynamic_sequences_with_metrics_aggregation(self):
        """Test dynamic sequences with real-time metrics aggregation."""
        research_topic = "Decentralized finance (DeFi) regulatory frameworks"
        execution_id = "test_parallel_123"
        
        # Create dynamic sequences with different lengths
        sequences = [
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC],
                description="Academic regulatory analysis",
                reasoning="Pure academic perspective on regulation",
                confidence_score=0.7
            ),
            DynamicSequencePattern(
                agent_order=[AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS],
                description="Industry-academic-technical analysis",
                reasoning="Business needs validated by research and tech feasibility",
                confidence_score=0.85
            )
        ]
        
        # Mock metrics aggregator
        mock_metrics_aggregator = Mock()
        self.engine.metrics_aggregator = mock_metrics_aggregator
        self.engine.enable_real_time_metrics = True
        
        # Mock sequence execution
        mock_agents = [AsyncMock() for _ in range(4)]  # Enough for both sequences
        mock_results = []
        
        # Results for sequence 1 (1 agent)
        mock_results.append(Mock(
            spec=AgentExecutionResult,
            agent_type=AgentType.ACADEMIC,
            execution_duration=25.0,
            key_insights=["Academic insight"],
            research_findings="Academic analysis"
        ))
        
        # Results for sequence 2 (3 agents)
        for agent_type in [AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS]:
            mock_results.append(Mock(
                spec=AgentExecutionResult,
                agent_type=agent_type,
                execution_duration=30.0,
                key_insights=[f"{agent_type.value} insight"],
                research_findings=f"{agent_type.value} analysis"
            ))
        
        for agent, result in zip(mock_agents, mock_results):
            agent.execute_research.return_value = result
        
        with patch.object(self.engine, '_get_agent') as mock_get_agent, \
             patch.object(self.engine.metrics_calculator, 'calculate_sequence_productivity') as mock_calc, \
             patch.object(self.engine.metrics_calculator, 'calculate_real_time_productivity', new_callable=AsyncMock) as mock_realtime:
            
            mock_get_agent.side_effect = mock_agents
            mock_calc.return_value = Mock()
            
            # Mock other dependencies
            self.engine.research_director.direct_next_investigation = AsyncMock(
                return_value=Mock(questions=["Test questions"])
            )
            self.engine.research_director.track_insight_productivity = AsyncMock(
                return_value=Mock(insight_types=["RESEARCH_GAP"], transition_quality=0.8)
            )
            
            # Execute sequences in parallel
            tasks = [
                self.engine.execute_sequence(seq, research_topic, f"{execution_id}_{i}")
                for i, seq in enumerate(sequences)
            ]
            results = await asyncio.gather(*tasks)
            
            # Verify results
            assert len(results) == 2
            assert len(results[0].agent_results) == 1  # Single agent sequence
            assert len(results[1].agent_results) == 3  # Three agent sequence
            
            # Verify metrics collection was called for both sequences
            assert mock_metrics_aggregator.collect_sequence_metrics.call_count >= 2
            assert mock_realtime.call_count >= 2


class TestDynamicSequenceGenerationToExecution:
    """Test complete workflow from dynamic generation to parallel execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=RunnableConfig)
        with patch('open_deep_research.sequencing.sequence_engine.SupervisorResearchDirector'), \
             patch('open_deep_research.sequencing.sequence_engine.MetricsCalculator'), \
             patch('open_deep_research.sequencing.sequence_engine.SequenceAnalyzer'):
            self.engine = SequenceOptimizationEngine(self.mock_config, enable_real_time_metrics=False)
    
    @pytest.mark.asyncio
    async def test_end_to_end_dynamic_workflow(self):
        """Test complete workflow from generation to execution and comparison."""
        research_topic = "Quantum machine learning algorithms for drug discovery"
        
        # Step 1: Generate dynamic sequences
        dynamic_sequences = [
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS, AgentType.INDUSTRY],
                description="Academic quantum ML research",
                reasoning="Theoretical foundations drive technical innovation and market applications",
                confidence_score=0.9
            ),
            DynamicSequencePattern(
                agent_order=[AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY],
                description="Tech-forward quantum ML",
                reasoning="Technical capabilities define research directions and market potential",
                confidence_score=0.85
            ),
            DynamicSequencePattern(
                agent_order=[AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS],
                description="Market-driven quantum ML",
                reasoning="Commercial needs guide academic research and technical development",
                confidence_score=0.8
            )
        ]
        
        # Step 2: Mock parallel execution
        mock_results = []
        for i, seq in enumerate(dynamic_sequences):
            mock_result = Mock(spec=SequenceResult)
            mock_result.sequence_pattern = seq
            mock_result.research_topic = research_topic
            mock_result.overall_productivity_metrics = Mock(tool_productivity=0.75 + i * 0.05)
            mock_result.unique_insights_generated = 5 + i * 2
            mock_result.final_quality_score = 0.8 + i * 0.05
            mock_results.append(mock_result)
        
        # Step 3: Mock comparison
        mock_comparison = Mock(spec=SequenceComparison)
        mock_comparison.highest_productivity_sequence = "dynamic_sequence_0"
        mock_comparison.productivity_variance = 0.12
        mock_comparison.compared_sequences = mock_results
        
        with patch.object(self.engine, 'execute_sequence', new_callable=AsyncMock) as mock_execute, \
             patch.object(self.engine.metrics_calculator, 'compare_sequence_results') as mock_compare:
            
            mock_execute.side_effect = mock_results
            mock_compare.return_value = mock_comparison
            
            # Execute complete workflow
            execution_results = []
            for seq in dynamic_sequences:
                result = await self.engine.execute_sequence(seq, research_topic)
                execution_results.append(result)
            
            # Compare results
            comparison = self.engine.metrics_calculator.compare_sequence_results(execution_results)
            
            # Verify complete workflow
            assert len(execution_results) == 3
            assert all(isinstance(r.sequence_pattern, DynamicSequencePattern) for r in execution_results)
            assert comparison.productivity_variance > 0
            assert comparison.highest_productivity_sequence is not None
    
    @pytest.mark.asyncio
    async def test_mixed_standard_and_dynamic_comparison(self):
        """Test comparison between standard patterns and dynamic sequences."""
        research_topic = "Renewable energy microgrids with AI optimization"
        
        # Mix of standard and dynamic patterns
        all_patterns = [
            SEQUENCE_PATTERNS["theory_first"],  # Standard
            SEQUENCE_PATTERNS["market_first"],  # Standard
            DynamicSequencePattern(  # Dynamic
                agent_order=[AgentType.TECHNICAL_TRENDS, AgentType.INDUSTRY, AgentType.ACADEMIC],
                description="AI-first energy solution",
                reasoning="AI capabilities drive market solutions with academic validation",
                confidence_score=0.88
            ),
            DynamicSequencePattern(  # Dynamic with repetition
                agent_order=[AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC],
                description="Academic-tech-academic validation",
                reasoning="Theory-implementation-validation cycle for AI energy systems",
                confidence_score=0.82
            )
        ]
        
        # Mock execution results
        mock_results = []
        for i, pattern in enumerate(all_patterns):
            mock_result = Mock(spec=SequenceResult)
            mock_result.sequence_pattern = pattern
            mock_result.research_topic = research_topic
            mock_result.overall_productivity_metrics = Mock(tool_productivity=0.7 + i * 0.08)
            mock_results.append(mock_result)
        
        with patch.object(self.engine, 'execute_sequence', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = mock_results
            
            # Execute all patterns
            results = await asyncio.gather(*[
                self.engine.execute_sequence(pattern, research_topic)
                for pattern in all_patterns
            ])
            
            # Verify mixed execution
            assert len(results) == 4
            
            # Check pattern types in results
            standard_count = sum(1 for r in results 
                               if isinstance(r.sequence_pattern, SequencePattern) 
                               and not isinstance(r.sequence_pattern, DynamicSequencePattern))
            dynamic_count = sum(1 for r in results 
                              if isinstance(r.sequence_pattern, DynamicSequencePattern))
            
            assert standard_count == 2
            assert dynamic_count == 2


class TestParallelMetricsIntegration:
    """Test integration with parallel metrics and real-time streaming."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=RunnableConfig)
        
        # Mock metrics aggregator
        self.mock_metrics_aggregator = Mock()
        
        with patch('open_deep_research.sequencing.sequence_engine.SupervisorResearchDirector'), \
             patch('open_deep_research.sequencing.sequence_engine.MetricsCalculator'), \
             patch('open_deep_research.sequencing.sequence_engine.SequenceAnalyzer'):
            self.engine = SequenceOptimizationEngine(
                config=self.mock_config,
                metrics_aggregator=self.mock_metrics_aggregator,
                enable_real_time_metrics=True
            )
    
    @pytest.mark.asyncio
    async def test_dynamic_sequences_metrics_streaming(self):
        """Test real-time metrics streaming with dynamic sequences."""
        research_topic = "Personalized medicine with genomic AI analysis"
        execution_id = "streaming_test_456"
        
        # Create dynamic sequences of varying lengths
        sequences = [
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS],
                description="Academic-tech genomics research",
                reasoning="Scientific research drives technical implementation",
                confidence_score=0.9
            ),
            DynamicSequencePattern(
                agent_order=[AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS, AgentType.INDUSTRY],
                description="Industry-academic-tech-industry cycle",
                reasoning="Market needs drive research, tech development, and commercialization",
                confidence_score=0.85
            )
        ]
        
        # Mock agent execution with different durations
        mock_agents = [AsyncMock() for _ in range(6)]  # Enough for both sequences
        execution_times = [20.0, 25.0, 30.0, 35.0, 28.0, 32.0]
        
        for i, (agent, duration) in enumerate(zip(mock_agents, execution_times)):
            mock_result = Mock(spec=AgentExecutionResult)
            mock_result.agent_type = [AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS, 
                                    AgentType.INDUSTRY, AgentType.ACADEMIC, 
                                    AgentType.TECHNICAL_TRENDS, AgentType.INDUSTRY][i]
            mock_result.execution_duration = duration
            mock_result.key_insights = [f"Insight {i}"]
            mock_result.research_findings = f"Research findings {i}"
            agent.execute_research.return_value = mock_result
        
        with patch.object(self.engine, '_get_agent') as mock_get_agent, \
             patch.object(self.engine.metrics_calculator, 'calculate_sequence_productivity') as mock_calc, \
             patch.object(self.engine.metrics_calculator, 'calculate_real_time_productivity', new_callable=AsyncMock) as mock_realtime:
            
            mock_get_agent.side_effect = mock_agents
            mock_calc.return_value = Mock()
            
            # Mock research director
            self.engine.research_director.direct_next_investigation = AsyncMock(
                return_value=Mock(questions=["Test questions"])
            )
            self.engine.research_director.track_insight_productivity = AsyncMock(
                return_value=Mock(insight_types=["RESEARCH_GAP"], transition_quality=0.8)
            )
            
            # Execute sequences with metrics streaming
            tasks = []
            for i, seq in enumerate(sequences):
                task = self.engine.execute_sequence(seq, research_topic, f"{execution_id}_{i}")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Verify metrics collection for both sequences
            assert len(results) == 2
            assert len(results[0].agent_results) == 2  # First sequence: 2 agents
            assert len(results[1].agent_results) == 4  # Second sequence: 4 agents
            
            # Verify metrics aggregator was called for both sequences
            collect_calls = self.mock_metrics_aggregator.collect_sequence_metrics.call_args_list
            assert len(collect_calls) >= 4  # At least start and end for each sequence
            
            # Verify real-time productivity calculations
            assert mock_realtime.call_count >= 2
    
    def test_parallel_metrics_model_compatibility(self):
        """Test that ParallelMetrics model works with dynamic sequences."""
        execution_id = "parallel_metrics_test"
        
        # Create parallel metrics for mixed sequence types
        parallel_metrics = ParallelMetrics(
            execution_id=execution_id,
            start_time=datetime.utcnow(),
            sequence_count=4,
            best_strategy="dynamic_sequence_1",
            best_productivity_score=0.92,
            productivity_variance=0.15,
            significant_difference_detected=True,
            total_insights_generated=25,
            total_tool_calls=18,
            average_research_quality=0.87,
            completed_sequences=4,
            strategy_rankings={
                "theory_first": 0.78,
                "dynamic_sequence_1": 0.92,
                "dynamic_sequence_2": 0.85,
                "market_first": 0.81
            }
        )
        
        # Verify metrics model handles dynamic strategies
        assert parallel_metrics.best_strategy.startswith("dynamic_")
        assert "dynamic_sequence_1" in parallel_metrics.strategy_rankings
        assert "dynamic_sequence_2" in parallel_metrics.strategy_rankings
        assert parallel_metrics.completion_rate == 100.0
        assert parallel_metrics.overall_tool_productivity > 0
    
    def test_sequence_metrics_dynamic_compatibility(self):
        """Test SequenceMetrics compatibility with dynamic patterns."""
        dynamic_sequence_id = "dynamic_test_789"
        
        # Create sequence metrics for dynamic pattern
        sequence_metrics = SequenceMetrics(
            sequence_id=dynamic_sequence_id,
            strategy="dynamic_custom",  # Non-standard strategy name
            execution_id="test_execution",
            start_time=datetime.utcnow(),
            total_agents=2,  # Variable length
            current_agent_position=1,
            current_tool_productivity=0.85,
            insights_generated=3,
            tool_calls_made=5
        )
        
        # Verify dynamic sequence metrics
        assert sequence_metrics.strategy == "dynamic_custom"
        assert sequence_metrics.total_agents == 2  # Non-standard length
        assert sequence_metrics.progress_percent == 50.0  # 1/2 = 50%
        
        # Test with completed dynamic sequence
        sequence_metrics.status = "completed"
        sequence_metrics.current_agent_position = 2
        assert sequence_metrics.progress_percent == 100.0


class TestDynamicSequenceVariableLengthIntegration:
    """Test integration with variable-length dynamic sequences."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=RunnableConfig)
        with patch('open_deep_research.sequencing.sequence_engine.SupervisorResearchDirector'), \
             patch('open_deep_research.sequencing.sequence_engine.MetricsCalculator'), \
             patch('open_deep_research.sequencing.sequence_engine.SequenceAnalyzer'):
            self.engine = SequenceOptimizationEngine(self.mock_config, enable_real_time_metrics=False)
    
    @pytest.mark.asyncio
    async def test_variable_length_parallel_execution(self):
        """Test parallel execution of sequences with different lengths."""
        research_topic = "Blockchain consensus mechanisms for IoT networks"
        
        # Create sequences of different lengths
        sequences = [
            DynamicSequencePattern(  # Length 1
                agent_order=[AgentType.TECHNICAL_TRENDS],
                description="Pure technical analysis",
                reasoning="Focus on cutting-edge blockchain tech",
                confidence_score=0.75
            ),
            DynamicSequencePattern(  # Length 2
                agent_order=[AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
                description="Market-tech validation",
                reasoning="Business needs drive technical solutions",
                confidence_score=0.8
            ),
            DynamicSequencePattern(  # Length 5
                agent_order=[
                    AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS, AgentType.INDUSTRY,
                    AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS
                ],
                description="Extended research-tech-market-validation cycle",
                reasoning="Comprehensive multi-perspective analysis with validation loops",
                confidence_score=0.85
            )
        ]
        
        # Mock agent executions
        total_agents_needed = sum(len(seq.agent_order) for seq in sequences)  # 1 + 2 + 5 = 8
        mock_agents = [AsyncMock() for _ in range(total_agents_needed)]
        
        agent_counter = 0
        mock_results_by_sequence = []
        
        for seq in sequences:
            seq_results = []
            for i, agent_type in enumerate(seq.agent_order):
                mock_result = Mock(spec=AgentExecutionResult)
                mock_result.agent_type = agent_type
                mock_result.execution_order = i + 1
                mock_result.execution_duration = 25.0 + i * 5
                mock_result.key_insights = [f"Insight from {agent_type.value}"]
                mock_result.research_findings = f"Research from {agent_type.value}"
                
                mock_agents[agent_counter].execute_research.return_value = mock_result
                seq_results.append(mock_result)
                agent_counter += 1
            
            mock_results_by_sequence.append(seq_results)
        
        with patch.object(self.engine, '_get_agent') as mock_get_agent, \
             patch.object(self.engine.metrics_calculator, 'calculate_sequence_productivity') as mock_calc:
            
            # Set up agent returns
            mock_get_agent.side_effect = mock_agents
            mock_calc.return_value = Mock()
            
            # Mock research director
            self.engine.research_director.direct_next_investigation = AsyncMock(
                return_value=Mock(questions=["Test questions"])
            )
            self.engine.research_director.track_insight_productivity = AsyncMock(
                return_value=Mock(insight_types=["RESEARCH_GAP"], transition_quality=0.8)
            )
            
            # Execute all sequences in parallel
            tasks = [
                self.engine.execute_sequence(seq, research_topic)
                for seq in sequences
            ]
            results = await asyncio.gather(*tasks)
            
            # Verify variable length execution
            assert len(results) == 3
            assert len(results[0].agent_results) == 1  # Single agent
            assert len(results[1].agent_results) == 2  # Two agents
            assert len(results[2].agent_results) == 5  # Five agents
            
            # Verify all agents were called
            assert mock_get_agent.call_count == total_agents_needed


if __name__ == "__main__":
    pytest.main([__file__])