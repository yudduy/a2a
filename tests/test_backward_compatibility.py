"""Tests for backward compatibility with existing sequence patterns.

This test module ensures that the dynamic sequence system maintains full
backward compatibility with existing SequencePattern usage, SEQUENCE_PATTERNS
dictionary, and all existing APIs and workflows.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List

from langchain_core.runnables import RunnableConfig

from open_deep_research.sequencing.models import (
    SequencePattern,
    DynamicSequencePattern,
    AgentType,
    SequenceResult,
    AgentExecutionResult,
    SEQUENCE_PATTERNS,
    THEORY_FIRST_PATTERN,
    MARKET_FIRST_PATTERN,
    FUTURE_BACK_PATTERN
)
from open_deep_research.sequencing.sequence_engine import SequenceOptimizationEngine
from open_deep_research.sequencing.sequence_selector import SequenceAnalyzer


class TestSequencePatternsBackwardCompatibility:
    """Test backward compatibility of SEQUENCE_PATTERNS dictionary and standard patterns."""
    
    def test_sequence_patterns_dictionary_intact(self):
        """Test that SEQUENCE_PATTERNS dictionary remains intact and functional."""
        # Verify dictionary structure
        assert isinstance(SEQUENCE_PATTERNS, dict)
        assert len(SEQUENCE_PATTERNS) == 3
        
        # Verify expected keys exist
        expected_keys = {"theory_first", "market_first", "future_back"}
        assert set(SEQUENCE_PATTERNS.keys()) == expected_keys
        
        # Verify all values are SequencePattern instances
        for key, pattern in SEQUENCE_PATTERNS.items():
            assert isinstance(pattern, SequencePattern)
            assert not isinstance(pattern, DynamicSequencePattern)  # Should be base class
    
    def test_standard_pattern_constants_available(self):
        """Test that standard pattern constants are available and properly configured."""
        # Test individual pattern constants
        assert THEORY_FIRST_PATTERN is not None
        assert MARKET_FIRST_PATTERN is not None
        assert FUTURE_BACK_PATTERN is not None
        
        # Verify patterns have required fields
        for pattern in [THEORY_FIRST_PATTERN, MARKET_FIRST_PATTERN, FUTURE_BACK_PATTERN]:
            assert hasattr(pattern, 'agent_order')
            assert hasattr(pattern, 'description')
            assert hasattr(pattern, 'expected_advantages')
            assert hasattr(pattern, 'confidence_score')
            assert hasattr(pattern, 'strategy')
            assert len(pattern.agent_order) == 3  # Standard patterns are 3 agents
    
    def test_theory_first_pattern_structure(self):
        """Test theory_first pattern maintains expected structure."""
        pattern = SEQUENCE_PATTERNS["theory_first"]
        
        assert pattern.agent_order == [AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS]
        assert pattern.strategy == "theory_first"
        assert pattern.confidence_score == 1.0
        assert "Theory First" in pattern.description
        assert len(pattern.expected_advantages) > 0
        assert "theoretical foundation" in pattern.expected_advantages[0].lower()
    
    def test_market_first_pattern_structure(self):
        """Test market_first pattern maintains expected structure."""
        pattern = SEQUENCE_PATTERNS["market_first"]
        
        assert pattern.agent_order == [AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS]
        assert pattern.strategy == "market_first"
        assert pattern.confidence_score == 1.0
        assert "Market First" in pattern.description
        assert len(pattern.expected_advantages) > 0
        assert "market" in pattern.expected_advantages[0].lower()
    
    def test_future_back_pattern_structure(self):
        """Test future_back pattern maintains expected structure."""
        pattern = SEQUENCE_PATTERNS["future_back"]
        
        assert pattern.agent_order == [AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY]
        assert pattern.strategy == "future_back"
        assert pattern.confidence_score == 1.0
        assert "Future Back" in pattern.description
        assert len(pattern.expected_advantages) > 0
        assert "future" in pattern.expected_advantages[0].lower()
    
    def test_pattern_dictionary_string_keys(self):
        """Test that SEQUENCE_PATTERNS uses string keys as expected."""
        for key in SEQUENCE_PATTERNS.keys():
            assert isinstance(key, str)
            assert key in ["theory_first", "market_first", "future_back"]
        
        # Test direct access by string keys
        theory_pattern = SEQUENCE_PATTERNS["theory_first"]
        market_pattern = SEQUENCE_PATTERNS["market_first"]
        future_pattern = SEQUENCE_PATTERNS["future_back"]
        
        assert theory_pattern.strategy == "theory_first"
        assert market_pattern.strategy == "market_first"
        assert future_pattern.strategy == "future_back"


class TestEngineBackwardCompatibility:
    """Test SequenceOptimizationEngine backward compatibility."""
    
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
    async def test_execute_sequence_with_standard_patterns(self):
        """Test that execute_sequence works with standard SequencePattern objects."""
        research_topic = "Impact of renewable energy on economic systems"
        
        # Mock agent execution
        mock_agent = AsyncMock()
        mock_result = Mock(spec=AgentExecutionResult)
        mock_result.agent_type = AgentType.ACADEMIC
        mock_result.execution_duration = 30.0
        mock_result.key_insights = ["Test insight"]
        mock_result.research_findings = "Test findings"
        mock_result.refined_insights = ["Refined insight"]
        mock_agent.execute_research.return_value = mock_result
        
        with patch.object(self.engine, '_get_agent', return_value=mock_agent), \
             patch.object(self.engine.metrics_calculator, 'calculate_sequence_productivity') as mock_calc, \
             patch.object(self.engine.research_director, 'direct_next_investigation', new_callable=AsyncMock) as mock_director, \
             patch.object(self.engine.research_director, 'track_insight_productivity', new_callable=AsyncMock) as mock_track:
            
            mock_calc.return_value = Mock()
            mock_director.return_value = Mock(questions=["Test question"])
            mock_track.return_value = Mock(insight_types=["RESEARCH_GAP"], transition_quality=0.8)
            
            # Test with each standard pattern
            for strategy_name, pattern in SEQUENCE_PATTERNS.items():
                result = await self.engine.execute_sequence(pattern, research_topic)
                
                assert isinstance(result, SequenceResult)
                assert result.sequence_pattern == pattern
                assert result.research_topic == research_topic
                assert len(result.agent_results) == 3  # Standard patterns have 3 agents
    
    @pytest.mark.asyncio
    async def test_compare_sequences_with_standard_patterns(self):
        """Test that compare_sequences works with standard patterns."""
        research_topic = "Blockchain applications in healthcare data management"
        
        # Mock the execution of individual sequences
        mock_results = []
        for i, strategy in enumerate(["theory_first", "market_first", "future_back"]):
            mock_result = Mock(spec=SequenceResult)
            mock_result.sequence_pattern = SEQUENCE_PATTERNS[strategy]
            mock_result.research_topic = research_topic
            mock_result.overall_productivity_metrics = Mock(tool_productivity=0.7 + i * 0.1)
            mock_results.append(mock_result)
        
        with patch.object(self.engine, 'execute_sequence', new_callable=AsyncMock) as mock_execute, \
             patch.object(self.engine.metrics_calculator, 'compare_sequence_results') as mock_compare:
            
            mock_execute.side_effect = mock_results
            mock_comparison = Mock()
            mock_compare.return_value = mock_comparison
            
            # Execute comparison
            result = await self.engine.compare_sequences(research_topic)
            
            # Verify standard patterns were used
            assert mock_execute.call_count == 3
            for i, call in enumerate(mock_execute.call_args_list):
                pattern_arg = call[0][0]
                assert pattern_arg in SEQUENCE_PATTERNS.values()
            
            assert result == mock_comparison
    
    @pytest.mark.asyncio
    async def test_execute_recommended_sequence_standard_flow(self):
        """Test execute_recommended_sequence follows standard flow."""
        research_topic = "Artificial intelligence ethics in autonomous vehicles"
        
        # Mock sequence analysis
        mock_analysis = Mock()
        mock_analysis.primary_recommendation = "theory_first"
        
        # Mock sequence execution
        mock_result = Mock(spec=SequenceResult)
        
        with patch.object(self.engine, 'analyze_research_query', return_value=mock_analysis), \
             patch.object(self.engine, 'execute_sequence', new_callable=AsyncMock, return_value=mock_result):
            
            result, analysis = await self.engine.execute_recommended_sequence(research_topic)
            
            # Verify standard pattern was used
            self.engine.execute_sequence.assert_called_once()
            pattern_arg = self.engine.execute_sequence.call_args[0][0]
            assert pattern_arg == SEQUENCE_PATTERNS["theory_first"]
            
            assert result == mock_result
            assert analysis == mock_analysis
    
    def test_get_performance_summary_with_standard_patterns(self):
        """Test performance summary works with standard patterns."""
        # Mock execution history with standard patterns
        mock_results = []
        for strategy_name, pattern in SEQUENCE_PATTERNS.items():
            mock_result = Mock(spec=SequenceResult)
            mock_result.sequence_pattern = pattern
            mock_result.overall_productivity_metrics = Mock(tool_productivity=0.8)
            mock_results.append(mock_result)
        
        self.engine.execution_history = mock_results
        
        summary = self.engine.get_performance_summary()
        
        # Verify all standard strategies are tracked
        assert summary["total_executions"] == 3
        assert set(summary["strategies_tested"]) == {"theory_first", "market_first", "future_back"}
        assert all(score == 0.8 for score in summary["average_productivity_by_strategy"].values())


class TestAnalyzerBackwardCompatibility:
    """Test SequenceAnalyzer backward compatibility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()
    
    def test_analyze_query_returns_standard_recommendations(self):
        """Test that analyze_query returns standard strategy names."""
        test_topics = [
            "Academic research on machine learning theory",
            "Market analysis of fintech startup opportunities",
            "Future trends in quantum computing technology"
        ]
        
        for topic in test_topics:
            analysis = self.analyzer.analyze_query(topic)
            
            # Should return standard strategy recommendations
            strategy_names = [strategy for strategy, confidence in analysis.recommended_sequences]
            assert all(name in ["theory_first", "market_first", "future_back"] for name in strategy_names)
            
            # Primary recommendation should be a standard strategy
            assert analysis.primary_recommendation in ["theory_first", "market_first", "future_back"]
    
    def test_explain_all_sequences_standard_strategies(self):
        """Test that explain_all_sequences works with standard strategies."""
        topic = "Sustainable energy grid modernization strategies"
        analysis = self.analyzer.analyze_query(topic)
        
        explanations = self.analyzer.explain_all_sequences(analysis)
        
        # Should have explanations for all standard strategies
        assert "theory_first" in explanations
        assert "market_first" in explanations
        assert "future_back" in explanations
        
        # Each explanation should be substantial
        for strategy, explanation in explanations.items():
            assert len(explanation) > 100
            assert strategy.replace("_", " ") in explanation.lower()


class TestDynamicAndStandardPatternInteroperability:
    """Test that dynamic and standard patterns can work together."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=RunnableConfig)
        with patch('open_deep_research.sequencing.sequence_engine.SupervisorResearchDirector'), \
             patch('open_deep_research.sequencing.sequence_engine.MetricsCalculator'), \
             patch('open_deep_research.sequencing.sequence_engine.SequenceAnalyzer'):
            self.engine = SequenceOptimizationEngine(self.mock_config, enable_real_time_metrics=False)
    
    def test_mixed_pattern_execution_history(self):
        """Test that execution history can contain both pattern types."""
        # Create mixed execution history
        standard_result = Mock(spec=SequenceResult)
        standard_result.sequence_pattern = SEQUENCE_PATTERNS["theory_first"]
        standard_result.overall_productivity_metrics = Mock(tool_productivity=0.8)
        
        dynamic_result = Mock(spec=SequenceResult)
        dynamic_result.sequence_pattern = DynamicSequencePattern(
            agent_order=[AgentType.INDUSTRY, AgentType.ACADEMIC],
            description="Dynamic pattern",
            reasoning="Dynamic reasoning",
            confidence_score=0.9
        )
        dynamic_result.overall_productivity_metrics = Mock(tool_productivity=0.85)
        
        self.engine.execution_history = [standard_result, dynamic_result]
        
        # Performance summary should handle both
        summary = self.engine.get_performance_summary()
        assert summary["total_executions"] == 2
        assert len(summary["strategies_tested"]) == 2
        assert "theory_first" in summary["strategies_tested"]
    
    def test_pattern_type_detection_in_synthesis(self):
        """Test that synthesis properly handles both pattern types."""
        # Standard pattern synthesis
        standard_pattern = SEQUENCE_PATTERNS["market_first"]
        mock_results = [Mock(spec=AgentExecutionResult) for _ in range(3)]
        for i, result in enumerate(mock_results):
            result.agent_type = standard_pattern.agent_order[i]
            result.execution_duration = 30.0
            result.tool_calls_made = 3
            result.key_insights = [f"Insight {i}"]
            result.research_findings = f"Research findings {i}"
        
        topic = "Test topic"
        
        # Test synthesis with standard pattern
        standard_synthesis = asyncio.run(
            self.engine._synthesize_research_findings(mock_results, topic, standard_pattern)
        )
        assert "Sequence Strategy: market_first" in standard_synthesis
        
        # Test synthesis with dynamic pattern
        dynamic_pattern = DynamicSequencePattern(
            sequence_id="test_123",
            agent_order=[AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS],
            description="Test dynamic pattern",
            reasoning="Test reasoning",
            confidence_score=0.8
        )
        
        dynamic_synthesis = asyncio.run(
            self.engine._synthesize_research_findings(mock_results, topic, dynamic_pattern)
        )
        assert "Dynamic Pattern" in dynamic_synthesis
        assert "test_123" in dynamic_synthesis or "test_123"[:8] in dynamic_synthesis


class TestLegacyAPICompatibility:
    """Test that legacy API methods continue to work."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=RunnableConfig)
        with patch('open_deep_research.sequencing.sequence_engine.SupervisorResearchDirector'), \
             patch('open_deep_research.sequencing.sequence_engine.MetricsCalculator'), \
             patch('open_deep_research.sequencing.sequence_engine.SequenceAnalyzer'):
            self.engine = SequenceOptimizationEngine(self.mock_config, enable_real_time_metrics=False)
    
    def test_legacy_strategy_parameter_support(self):
        """Test that legacy strategy parameters are still supported."""
        # Mock compare_sequences with specific strategies
        with patch.object(self.engine, 'execute_sequence', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = Mock(spec=SequenceResult)
            
            # Should accept legacy strategy list
            asyncio.run(self.engine.compare_sequences(
                "Test topic",
                strategies=["theory_first", "market_first"]
            ))
            
            # Should have executed only specified strategies
            assert mock_execute.call_count == 2
            executed_patterns = [call[0][0] for call in mock_execute.call_args_list]
            assert SEQUENCE_PATTERNS["theory_first"] in executed_patterns
            assert SEQUENCE_PATTERNS["market_first"] in executed_patterns
    
    def test_legacy_batch_analysis_support(self):
        """Test that batch_sequence_analysis continues to work."""
        topics = ["Topic 1", "Topic 2"]
        
        with patch.object(self.engine, 'compare_sequences', new_callable=AsyncMock) as mock_compare:
            mock_compare.return_value = Mock()
            
            results = asyncio.run(self.engine.batch_sequence_analysis(topics))
            
            assert len(results) == 2
            assert mock_compare.call_count == 2
    
    def test_sequence_analysis_enum_compatibility(self):
        """Test that sequence analysis maintains enum compatibility."""
        topic = "Test research topic"
        
        # Mock the analyzer
        with patch.object(self.engine.sequence_analyzer, 'analyze_query') as mock_analyze:
            mock_analyze.return_value = Mock(
                analysis_id="test_id",
                query_type="academic_research",
                research_domain="academic",
                complexity_score=0.7,
                scope_breadth="medium",
                recommended_sequences=[("theory_first", 0.9)],
                primary_recommendation="theory_first",
                confidence=0.9,
                explanation="Test explanation",
                reasoning={"test": "reasoning"},
                query_characteristics={},
                analysis_timestamp=Mock()
            )
            
            analysis = self.engine.analyze_research_query(topic)
            
            # Should convert string enums to proper enums
            from open_deep_research.sequencing.models import QueryType, ResearchDomain, ScopeBreadth
            assert isinstance(analysis.query_type, QueryType)
            assert isinstance(analysis.research_domain, ResearchDomain)
            assert isinstance(analysis.scope_breadth, ScopeBreadth)


if __name__ == "__main__":
    pytest.main([__file__])