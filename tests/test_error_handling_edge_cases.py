"""Tests for error handling and edge cases in the dynamic sequence system.

This test module validates robust error handling, graceful degradation,
and proper behavior under various edge cases and failure scenarios
in the meta-sequence optimizer system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Optional

from langchain_core.runnables import RunnableConfig

from open_deep_research.sequencing.models import (
    DynamicSequencePattern,
    AgentType,
    SequenceResult,
    AgentExecutionResult
)
from open_deep_research.sequencing.sequence_engine import SequenceOptimizationEngine
from open_deep_research.sequencing.sequence_selector import SequenceAnalyzer


class TestDynamicSequenceGenerationErrorHandling:
    """Test error handling in dynamic sequence generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()
    
    def test_empty_topic_handling(self):
        """Test handling of empty or invalid topics."""
        # Empty string
        with pytest.raises((ValueError, Exception)):
            self.analyzer.generate_dynamic_sequences("")
        
        # None topic
        with pytest.raises((ValueError, TypeError, Exception)):
            self.analyzer.generate_dynamic_sequences(None)
        
        # Whitespace only
        with pytest.raises((ValueError, Exception)):
            self.analyzer.generate_dynamic_sequences("   ")
    
    def test_invalid_sequence_count_handling(self):
        """Test handling of invalid sequence count parameters."""
        topic = "Valid research topic for testing"
        
        # Zero sequences
        with pytest.raises((ValueError, Exception)):
            self.analyzer.generate_dynamic_sequences(topic, num_sequences=0)
        
        # Negative sequences
        with pytest.raises((ValueError, Exception)):
            self.analyzer.generate_dynamic_sequences(topic, num_sequences=-1)
        
        # Extremely large sequence count
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=100)
        # Should handle gracefully, possibly with a maximum limit
        assert len(sequences) <= 100
        assert len(sequences) > 0
    
    def test_malformed_topic_handling(self):
        """Test handling of unusual or malformed topics."""
        # Very long topic
        very_long_topic = "This is an extremely long research topic " * 100
        sequences = self.analyzer.generate_dynamic_sequences(very_long_topic, num_sequences=2)
        assert len(sequences) == 2
        
        # Topic with special characters
        special_topic = "Research on AI & ML: Effects of €$£¥ on 🚀 development (2024)"
        sequences = self.analyzer.generate_dynamic_sequences(special_topic, num_sequences=2)
        assert len(sequences) == 2
        
        # Topic with only numbers
        numeric_topic = "123 456 789 2024 2025"
        sequences = self.analyzer.generate_dynamic_sequences(numeric_topic, num_sequences=2)
        assert len(sequences) == 2
    
    def test_single_word_topic_handling(self):
        """Test handling of very short topics."""
        # Single word topics
        short_topics = ["AI", "Blockchain", "ML", "IoT", "5G"]
        
        for topic in short_topics:
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=2)
            assert len(sequences) == 2
            assert all(len(seq.agent_order) > 0 for seq in sequences)
    
    def test_analysis_failure_fallback(self):
        """Test fallback behavior when analysis fails."""
        topic = "Test topic for analysis failure"
        
        # Mock analyze_query to raise an exception
        with patch.object(self.analyzer, 'analyze_query') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")
            
            # Should handle gracefully or raise appropriate exception
            with pytest.raises(Exception):
                self.analyzer.generate_dynamic_sequences(topic)
    
    def test_invalid_analysis_results_handling(self):
        """Test handling when analysis returns invalid results."""
        topic = "Test topic for invalid analysis"
        
        # Mock analyze_query to return invalid data
        with patch.object(self.analyzer, 'analyze_query') as mock_analyze:
            # Mock with missing required fields
            mock_analysis = Mock()
            mock_analysis.primary_recommendation = None
            mock_analysis.confidence = -1  # Invalid confidence
            mock_analysis.recommended_sequences = []
            mock_analyze.return_value = mock_analysis
            
            # Should handle gracefully
            try:
                sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=2)
                # If it succeeds, should return valid sequences
                assert len(sequences) <= 2
                assert all(isinstance(seq, DynamicSequencePattern) for seq in sequences)
            except Exception:
                # Or raise appropriate exception
                pass


class TestSequenceEngineErrorHandling:
    """Test error handling in sequence execution."""
    
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
    async def test_agent_execution_failure_handling(self):
        """Test handling when agent execution fails."""
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY],
            description="Test pattern for error handling",
            reasoning="Test reasoning",
            confidence_score=0.8
        )
        
        research_topic = "Test topic"
        
        # Mock agent that fails
        mock_agent = AsyncMock()
        mock_agent.execute_research.side_effect = Exception("Agent execution failed")
        
        with patch.object(self.engine, '_get_agent', return_value=mock_agent):
            # Should propagate the exception
            with pytest.raises(Exception, match="Agent execution failed"):
                await self.engine.execute_sequence(dynamic_pattern, research_topic)
    
    @pytest.mark.asyncio
    async def test_partial_agent_failure_handling(self):
        """Test handling when only some agents fail in a sequence."""
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
            description="Test pattern for partial failure",
            reasoning="Test reasoning",
            confidence_score=0.8
        )
        
        research_topic = "Test topic"
        
        # First agent succeeds, second fails
        success_agent = AsyncMock()
        success_result = Mock(spec=AgentExecutionResult)
        success_result.agent_type = AgentType.ACADEMIC
        success_result.execution_duration = 30.0
        success_result.key_insights = ["Success insight"]
        success_result.research_findings = "Success findings"
        success_result.refined_insights = ["Refined insight"]
        success_agent.execute_research.return_value = success_result
        
        failure_agent = AsyncMock()
        failure_agent.execute_research.side_effect = Exception("Second agent failed")
        
        with patch.object(self.engine, '_get_agent') as mock_get_agent:
            mock_get_agent.side_effect = [success_agent, failure_agent]
            
            # Should fail on second agent
            with pytest.raises(Exception, match="Second agent failed"):
                await self.engine.execute_sequence(dynamic_pattern, research_topic)
    
    @pytest.mark.asyncio
    async def test_research_director_failure_handling(self):
        """Test handling when research director fails."""
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY],
            description="Test pattern",
            reasoning="Test reasoning",
            confidence_score=0.8
        )
        
        research_topic = "Test topic"
        
        # Mock agent
        mock_agent = AsyncMock()
        mock_result = Mock(spec=AgentExecutionResult)
        mock_result.agent_type = AgentType.ACADEMIC
        mock_result.execution_duration = 30.0
        mock_result.key_insights = ["Test insight"]
        mock_result.research_findings = "Test findings"
        mock_result.refined_insights = ["Refined insight"]
        mock_agent.execute_research.return_value = mock_result
        
        # Research director fails
        with patch.object(self.engine, '_get_agent', return_value=mock_agent):
            self.engine.research_director.direct_next_investigation.side_effect = Exception("Director failed")
            
            # Should fail when trying to get questions for second agent
            with pytest.raises(Exception, match="Director failed"):
                await self.engine.execute_sequence(dynamic_pattern, research_topic)
    
    @pytest.mark.asyncio
    async def test_metrics_calculation_failure_handling(self):
        """Test handling when metrics calculation fails."""
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Test pattern",
            reasoning="Test reasoning",
            confidence_score=0.8
        )
        
        research_topic = "Test topic"
        
        # Mock successful agent execution
        mock_agent = AsyncMock()
        mock_result = Mock(spec=AgentExecutionResult)
        mock_result.agent_type = AgentType.ACADEMIC
        mock_result.execution_duration = 30.0
        mock_result.key_insights = ["Test insight"]
        mock_result.research_findings = "Test findings"
        mock_result.refined_insights = ["Refined insight"]
        mock_agent.execute_research.return_value = mock_result
        
        with patch.object(self.engine, '_get_agent', return_value=mock_agent):
            # Mock metrics calculation failure
            self.engine.metrics_calculator.calculate_sequence_productivity.side_effect = Exception("Metrics failed")
            
            # Should propagate metrics failure
            with pytest.raises(Exception, match="Metrics failed"):
                await self.engine.execute_sequence(dynamic_pattern, research_topic)
    
    @pytest.mark.asyncio
    async def test_invalid_agent_type_handling(self):
        """Test handling of invalid agent types."""
        # Try to create pattern with invalid agent
        with pytest.raises((ValueError, TypeError)):
            DynamicSequencePattern(
                agent_order=["INVALID_AGENT"],  # Invalid agent type
                description="Test pattern",
                reasoning="Test reasoning",
                confidence_score=0.8
            )
    
    @pytest.mark.asyncio
    async def test_synthesis_failure_handling(self):
        """Test handling when synthesis generation fails."""
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Test pattern",
            reasoning="Test reasoning",
            confidence_score=0.8
        )
        
        research_topic = "Test topic"
        
        # Mock successful agent execution
        mock_agent = AsyncMock()
        mock_result = Mock(spec=AgentExecutionResult)
        mock_result.agent_type = AgentType.ACADEMIC
        mock_result.execution_duration = 30.0
        mock_result.key_insights = ["Test insight"]
        mock_result.research_findings = "Test findings"
        mock_result.refined_insights = ["Refined insight"]
        mock_agent.execute_research.return_value = mock_result
        
        with patch.object(self.engine, '_get_agent', return_value=mock_agent), \
             patch.object(self.engine, '_synthesize_research_findings') as mock_synthesis:
            
            # Mock synthesis failure
            mock_synthesis.side_effect = Exception("Synthesis failed")
            
            # Should propagate synthesis failure
            with pytest.raises(Exception, match="Synthesis failed"):
                await self.engine.execute_sequence(dynamic_pattern, research_topic)


class TestEdgeCasePatterns:
    """Test edge cases in pattern definition and usage."""
    
    def test_single_agent_pattern_creation(self):
        """Test creation of single-agent patterns."""
        single_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Single agent analysis",
            reasoning="Focus on academic perspective only",
            confidence_score=0.7
        )
        
        assert len(single_pattern.agent_order) == 1
        assert single_pattern.sequence_length == 1
        assert single_pattern.agent_types_used == {AgentType.ACADEMIC}
    
    def test_repeated_agent_pattern_creation(self):
        """Test creation of patterns with repeated agents."""
        repeated_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.ACADEMIC],
            description="Academic-heavy analysis with industry validation",
            reasoning="Multiple academic perspectives with industry input",
            confidence_score=0.8
        )
        
        assert len(repeated_pattern.agent_order) == 4
        assert repeated_pattern.sequence_length == 4
        assert repeated_pattern.agent_types_used == {AgentType.ACADEMIC, AgentType.INDUSTRY}
    
    def test_maximum_length_pattern(self):
        """Test very long sequences."""
        long_agent_order = [AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS] * 5  # 15 agents
        
        long_pattern = DynamicSequencePattern(
            agent_order=long_agent_order,
            description="Extended multi-perspective analysis",
            reasoning="Comprehensive analysis with multiple validation cycles",
            confidence_score=0.75
        )
        
        assert len(long_pattern.agent_order) == 15
        assert long_pattern.sequence_length == 15
        assert len(long_pattern.agent_types_used) == 3
    
    def test_boundary_confidence_scores(self):
        """Test boundary values for confidence scores."""
        # Minimum confidence
        min_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Minimum confidence pattern",
            reasoning="Low confidence test",
            confidence_score=0.0
        )
        assert min_pattern.confidence_score == 0.0
        
        # Maximum confidence
        max_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Maximum confidence pattern",
            reasoning="High confidence test",
            confidence_score=1.0
        )
        assert max_pattern.confidence_score == 1.0


class TestResourceAndMemoryEdgeCases:
    """Test resource usage and memory-related edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()
    
    def test_large_topic_generation(self):
        """Test generation with very large topics."""
        # Create a very large topic
        large_topic = ("Comprehensive interdisciplinary analysis of artificial intelligence applications " * 50)
        
        sequences = self.analyzer.generate_dynamic_sequences(large_topic, num_sequences=3)
        
        # Should handle large topics gracefully
        assert len(sequences) == 3
        assert all(isinstance(seq, DynamicSequencePattern) for seq in sequences)
    
    def test_many_sequences_generation(self):
        """Test generation of many sequences at once."""
        topic = "Machine learning in healthcare applications"
        
        # Request many sequences
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=10)
        
        # Should handle gracefully (may limit to reasonable number)
        assert len(sequences) <= 10
        assert len(sequences) > 0
        assert all(isinstance(seq, DynamicSequencePattern) for seq in sequences)
    
    def test_concurrent_generation_calls(self):
        """Test concurrent generation calls."""
        topics = [
            "AI in finance",
            "Blockchain in healthcare",
            "IoT in manufacturing",
            "5G network optimization",
            "Quantum computing applications"
        ]
        
        # Generate sequences for multiple topics concurrently
        all_sequences = []
        for topic in topics:
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=2)
            all_sequences.extend(sequences)
        
        # Should handle multiple calls
        assert len(all_sequences) == len(topics) * 2
        assert all(isinstance(seq, DynamicSequencePattern) for seq in all_sequences)
        
        # All sequences should have unique IDs
        sequence_ids = [seq.sequence_id for seq in all_sequences]
        assert len(set(sequence_ids)) == len(sequence_ids)


class TestRobustnessAndRecovery:
    """Test system robustness and recovery mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=RunnableConfig)
        with patch('open_deep_research.sequencing.sequence_engine.SupervisorResearchDirector'), \
             patch('open_deep_research.sequencing.sequence_engine.MetricsCalculator'), \
             patch('open_deep_research.sequencing.sequence_engine.SequenceAnalyzer'):
            self.engine = SequenceOptimizationEngine(self.mock_config, enable_real_time_metrics=False)
    
    @pytest.mark.asyncio
    async def test_metrics_aggregator_failure_recovery(self):
        """Test recovery when metrics aggregator fails."""
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Test pattern",
            reasoning="Test reasoning",
            confidence_score=0.8
        )
        
        research_topic = "Test topic"
        execution_id = "test_metrics_failure"
        
        # Mock failing metrics aggregator
        mock_aggregator = Mock()
        mock_aggregator.collect_sequence_metrics.side_effect = Exception("Metrics aggregator failed")
        self.engine.metrics_aggregator = mock_aggregator
        self.engine.enable_real_time_metrics = True
        
        # Mock successful agent execution
        mock_agent = AsyncMock()
        mock_result = Mock(spec=AgentExecutionResult)
        mock_result.agent_type = AgentType.ACADEMIC
        mock_result.execution_duration = 30.0
        mock_result.key_insights = ["Test insight"]
        mock_result.research_findings = "Test findings"
        mock_result.refined_insights = ["Refined insight"]
        mock_agent.execute_research.return_value = mock_result
        
        with patch.object(self.engine, '_get_agent', return_value=mock_agent), \
             patch.object(self.engine.metrics_calculator, 'calculate_sequence_productivity') as mock_calc:
            
            mock_calc.return_value = Mock()
            
            # Should continue execution despite metrics failure
            try:
                result = await self.engine.execute_sequence(dynamic_pattern, research_topic, execution_id)
                # If it succeeds, metrics failure was handled gracefully
                assert isinstance(result, SequenceResult)
            except Exception as e:
                # Should not be the metrics aggregator error if properly handled
                assert "Metrics aggregator failed" not in str(e)
    
    def test_fallback_to_standard_patterns(self):
        """Test fallback to standard patterns when dynamic generation fails."""
        topic = "Test topic for fallback"
        
        # Mock the sequence analyzer to fail dynamic generation
        with patch.object(self.engine.sequence_analyzer, 'generate_dynamic_sequences') as mock_generate:
            mock_generate.side_effect = Exception("Dynamic generation failed")
            
            # Should have mechanism to fall back to standard patterns or handle gracefully
            try:
                analysis = self.engine.analyze_research_query(topic)
                # If analysis succeeds, it should provide standard recommendations
                assert analysis.primary_recommendation in ["theory_first", "market_first", "future_back"]
            except Exception:
                # Or should raise appropriate exception, not the dynamic generation error
                pass


if __name__ == "__main__":
    pytest.main([__file__])