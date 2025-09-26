"""Unit tests for the enhanced quality evaluation system."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
from typing import Tuple

try:
    from ..orchestration.orchestration_optimizer import (
        QualityEvaluator,
        EnhancedQualityMetrics,
        CostEfficiencyMetrics,
        ContentMetrics,
        EvaluationCriteria
    )
except ImportError:
    # For running tests directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from orchestration.orchestration_optimizer import (
        QualityEvaluator,
        EnhancedQualityMetrics,
        CostEfficiencyMetrics,
        ContentMetrics,
        EvaluationCriteria
    )


class TestQualityEvaluator:
    """Test suite for QualityEvaluator functionality."""

    @pytest.fixture
    def sample_synthesis(self) -> str:
        """Sample research synthesis for testing."""
        return """## Comprehensive Research Synthesis

This research examines the market dynamics of electric vehicles, focusing on key trends,
competitive landscape, and future opportunities. The analysis reveals significant growth
potential driven by technological innovation and regulatory support.

### Key Findings:
- Electric vehicle adoption is accelerating rapidly
- Battery technology improvements are driving cost reductions
- Government incentives play a crucial role in market expansion
- Competition is intensifying among major manufacturers

### Market Analysis:
The EV market demonstrates robust growth with compound annual growth rates exceeding 25%.
However, challenges remain in infrastructure development and supply chain optimization.

### Strategic Recommendations:
1. Invest in battery technology research and development
2. Expand charging infrastructure networks
3. Focus on software and autonomous driving capabilities
4. Develop strategic partnerships across the ecosystem"""

    @pytest.fixture
    def sample_query(self) -> str:
        """Sample research query for testing."""
        return "market analysis of electric vehicles"

    @pytest.fixture
    def mock_llm_judge(self):
        """Mock LLM Judge for testing."""
        mock_judge = AsyncMock()
        mock_judge.evaluate_single_report.return_value = Mock(
            completeness=Mock(score=8.5, reasoning="Comprehensive coverage"),
            depth=Mock(score=7.8, reasoning="Detailed analysis"),
            coherence=Mock(score=8.2, reasoning="Well-structured"),
            innovation=Mock(score=7.0, reasoning="Some novel insights"),
            actionability=Mock(score=8.0, reasoning="Practical recommendations"),
            overall_score=85.0,
            confidence_level=0.9,
            executive_summary="High-quality research report"
        )
        return mock_judge

    def test_quality_evaluator_initialization(self):
        """Test QualityEvaluator initialization."""
        evaluator = QualityEvaluator("claude-3-5-sonnet")
        assert evaluator.evaluation_model == "claude-3-5-sonnet"
        assert evaluator._llm_judge is None

    def test_quality_evaluator_initialization_with_fallback(self):
        """Test QualityEvaluator with fallback model."""
        evaluator = QualityEvaluator("fallback-model")
        assert evaluator.evaluation_model == "fallback-model"

    @pytest.mark.asyncio
    async def test_evaluate_with_fallback(self, sample_synthesis, sample_query):
        """Test evaluation with fallback method."""
        evaluator = QualityEvaluator()
        evaluator._llm_judge = None  # Force fallback

        enhanced, cost_eff, content = await evaluator.evaluate_research_quality(
            synthesis=sample_synthesis,
            query=sample_query,
            total_tokens=1000,
            total_cost=0.05
        )

        # Test enhanced quality metrics
        assert isinstance(enhanced, EnhancedQualityMetrics)
        assert enhanced.overall_score > 0
        assert enhanced.confidence_level == 0.5  # Fallback confidence
        assert enhanced.evaluation_model == "fallback"

        # Test cost efficiency metrics
        assert isinstance(cost_eff, CostEfficiencyMetrics)
        assert cost_eff.total_tokens == 1000
        assert cost_eff.total_cost == 0.05

        # Test content metrics
        assert isinstance(content, ContentMetrics)
        assert content.synthesis_length == len(sample_synthesis)
        assert content.coherence_score >= 0.0
        assert content.readability_score >= 0.0

    @pytest.mark.asyncio
    async def test_evaluate_with_llm_judge_success(self, sample_synthesis, sample_query, mock_llm_judge):
        """Test evaluation with successful LLM Judge."""
        evaluator = QualityEvaluator()
        evaluator._llm_judge = mock_llm_judge

        enhanced, cost_eff, content = await evaluator.evaluate_research_quality(
            synthesis=sample_synthesis,
            query=sample_query,
            total_tokens=1000,
            total_cost=0.05
        )

        # Verify LLM Judge was called
        mock_llm_judge.evaluate_single_report.assert_called_once()

        # Test enhanced quality metrics from LLM Judge
        assert isinstance(enhanced, EnhancedQualityMetrics)
        assert enhanced.completeness == 8.5
        assert enhanced.depth == 7.8
        assert enhanced.overall_score == 85.0
        assert enhanced.confidence_level == 0.9
        assert "Full LLM Judge evaluation" in enhanced.evaluation_notes

    @pytest.mark.asyncio
    async def test_evaluate_with_llm_judge_failure(self, sample_synthesis, sample_query, mock_llm_judge):
        """Test evaluation when LLM Judge fails but fallback works."""
        evaluator = QualityEvaluator()
        mock_llm_judge.evaluate_single_report.side_effect = Exception("API Error")
        evaluator._llm_judge = mock_llm_judge

        enhanced, cost_eff, content = await evaluator.evaluate_research_quality(
            synthesis=sample_synthesis,
            query=sample_query,
            total_tokens=1000,
            total_cost=0.05
        )

        # Should fall back to basic evaluation
        assert isinstance(enhanced, EnhancedQualityMetrics)
        assert enhanced.confidence_level == 0.5  # Fallback confidence
        assert enhanced.evaluation_model == "fallback"

    def test_content_metrics_calculation(self, sample_synthesis):
        """Test content metrics calculation."""
        evaluator = QualityEvaluator()

        content_metrics = evaluator._calculate_content_metrics(sample_synthesis)

        assert isinstance(content_metrics, ContentMetrics)
        assert content_metrics.synthesis_length == len(sample_synthesis)
        assert content_metrics.coherence_score >= 0.0
        assert content_metrics.readability_score >= 0.0
        assert content_metrics.insight_density >= 0.0
        assert content_metrics.structure_score >= 0.0

    def test_coherence_calculation(self):
        """Test coherence score calculation."""
        evaluator = QualityEvaluator()

        # Test with consistent sentence lengths
        consistent_text = "Short. Sentence. Here. Another. One."
        coherence = evaluator._calculate_coherence(consistent_text)
        assert coherence >= 0.0

        # Test with very short text
        short_text = "Hi"
        coherence = evaluator._calculate_coherence(short_text)
        assert coherence == 0.5

    def test_readability_calculation(self):
        """Test readability score calculation."""
        evaluator = QualityEvaluator()

        # Test with simple text
        simple_text = "This is a simple sentence. Another simple one."
        readability = evaluator._calculate_readability(simple_text)
        assert readability >= 0.0

        # Test with empty text
        empty_text = ""
        readability = evaluator._calculate_readability(empty_text)
        assert readability == 0.5

    def test_insight_density_calculation(self, sample_synthesis):
        """Test insight density calculation."""
        evaluator = QualityEvaluator()

        density = evaluator._calculate_insight_density(sample_synthesis)

        assert density >= 0.0
        assert density <= 1.0  # Should be capped at 1.0

        # Test with text containing insight markers
        insight_text = "However, this is important. Therefore, we conclude. Significantly, the results show."
        density = evaluator._calculate_insight_density(insight_text)
        assert density > 0.0

    def test_structure_calculation(self, sample_synthesis):
        """Test structure quality calculation."""
        evaluator = QualityEvaluator()

        structure = evaluator._calculate_structure(sample_synthesis)
        assert structure >= 0.0

        # Test with well-structured text (has sections)
        structured_text = "## Introduction\nSome content\n### Subsection\nMore content"
        structure = evaluator._calculate_structure(structured_text)
        assert structure == 0.8  # Should be well-structured

        # Test with basic text (no sections)
        basic_text = "Just plain text without sections or paragraphs"
        structure = evaluator._calculate_structure(basic_text)
        assert structure == 0.5  # Should be basic structure

    def test_cost_efficiency_calculation(self, sample_synthesis, sample_query):
        """Test cost efficiency metrics calculation."""
        evaluator = QualityEvaluator()

        # Test with normal values
        cost_eff = evaluator._calculate_cost_efficiency_metrics(
            synthesis=sample_synthesis,
            query=sample_query,
            total_tokens=1000,
            total_cost=0.05
        )

        assert isinstance(cost_eff, CostEfficiencyMetrics)
        assert cost_eff.total_tokens == 1000
        assert cost_eff.total_cost == 0.05
        assert cost_eff.quality_per_dollar >= 0.0
        assert cost_eff.quality_per_token >= 0.0

        # Test with zero cost
        cost_eff_zero = evaluator._calculate_cost_efficiency_metrics(
            synthesis=sample_synthesis,
            query=sample_query,
            total_tokens=1000,
            total_cost=0.0
        )

        assert cost_eff_zero.quality_per_dollar == 0.0
        assert cost_eff_zero.cost_per_quality_unit == 0.0

    def test_enhanced_quality_metrics_creation(self):
        """Test EnhancedQualityMetrics model creation and validation."""
        metrics = EnhancedQualityMetrics(
            completeness=8.5,
            depth=7.8,
            coherence=8.2,
            innovation=7.0,
            actionability=8.0,
            overall_score=85.0,
            confidence_level=0.9,
            evaluation_notes="Test evaluation"
        )

        assert metrics.completeness == 8.5
        assert metrics.depth == 7.8
        assert metrics.coherence == 8.2
        assert metrics.innovation == 7.0
        assert metrics.actionability == 8.0
        assert metrics.overall_score == 85.0
        assert metrics.confidence_level == 0.9
        assert metrics.evaluation_notes == "Test evaluation"
        assert isinstance(metrics.evaluation_timestamp, datetime)

    def test_cost_efficiency_metrics_creation(self):
        """Test CostEfficiencyMetrics model creation."""
        cost_metrics = CostEfficiencyMetrics(
            quality_per_dollar=17.0,
            quality_per_token=0.085,
            tokens_per_quality_unit=11.76,
            cost_per_quality_unit=0.059,
            total_tokens=1000,
            total_cost=0.05
        )

        assert cost_metrics.quality_per_dollar == 17.0
        assert cost_metrics.quality_per_token == 0.085
        assert cost_metrics.tokens_per_quality_unit == 11.76
        assert cost_metrics.cost_per_quality_unit == 0.059
        assert cost_metrics.total_tokens == 1000
        assert cost_metrics.total_cost == 0.05

    def test_content_metrics_creation(self, sample_synthesis):
        """Test ContentMetrics model creation."""
        evaluator = QualityEvaluator()
        content_metrics = evaluator._calculate_content_metrics(sample_synthesis)

        assert isinstance(content_metrics, ContentMetrics)
        assert content_metrics.synthesis_length == len(sample_synthesis)
        assert 0.0 <= content_metrics.coherence_score <= 1.0
        assert 0.0 <= content_metrics.readability_score <= 1.0
        assert 0.0 <= content_metrics.completeness_score <= 1.0
        assert 0.0 <= content_metrics.insight_density <= 1.0
        assert 0.0 <= content_metrics.structure_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_research_quality_with_empty_synthesis(self):
        """Test evaluation with empty synthesis."""
        evaluator = QualityEvaluator()

        enhanced, cost_eff, content = await evaluator.evaluate_research_quality(
            synthesis="",
            query="test query",
            total_tokens=0,
            total_cost=0.0
        )

        assert isinstance(enhanced, EnhancedQualityMetrics)
        assert enhanced.overall_score >= 0.0
        assert isinstance(cost_eff, CostEfficiencyMetrics)
        assert isinstance(content, ContentMetrics)
        assert content.synthesis_length == 0

    @pytest.mark.asyncio
    async def test_evaluate_research_quality_with_none_synthesis(self):
        """Test evaluation with None synthesis."""
        evaluator = QualityEvaluator()

        enhanced, cost_eff, content = await evaluator.evaluate_research_quality(
            synthesis=None,
            query="test query",
            total_tokens=0,
            total_cost=0.0
        )

        assert isinstance(enhanced, EnhancedQualityMetrics)
        assert enhanced.overall_score >= 0.0
        assert isinstance(cost_eff, CostEfficiencyMetrics)
        assert isinstance(content, ContentMetrics)
        assert content.synthesis_length == 0


class TestEnhancedMetricsModels:
    """Test suite for enhanced metrics models."""

    def test_enhanced_quality_metrics_weighted_calculation(self):
        """Test that overall_score is correctly calculated from individual criteria."""
        metrics = EnhancedQualityMetrics(
            completeness=8.0,  # 25% weight
            depth=9.0,        # 25% weight
            coherence=7.0,    # 20% weight
            innovation=6.0,   # 15% weight
            actionability=8.0 # 15% weight
        )

        # Manual calculation: (8*0.25) + (9*0.25) + (7*0.20) + (6*0.15) + (8*0.15)
        expected_score = (8.0 * 0.25) + (9.0 * 0.25) + (7.0 * 0.20) + (6.0 * 0.15) + (8.0 * 0.15)
        assert metrics.overall_score == expected_score

    def test_cost_efficiency_metrics_calculations(self):
        """Test cost efficiency calculations."""
        # Simulate: 1000 tokens, $0.05 cost, quality score 0.85
        quality_score = 0.85
        total_tokens = 1000
        total_cost = 0.05

        metrics = CostEfficiencyMetrics(
            quality_per_dollar=quality_score / total_cost if total_cost > 0 else 0,
            quality_per_token=quality_score / total_tokens if total_tokens > 0 else 0,
            tokens_per_quality_unit=total_tokens / quality_score if quality_score > 0 else 0,
            cost_per_quality_unit=total_cost / quality_score if quality_score > 0 else 0,
            total_tokens=total_tokens,
            total_cost=total_cost
        )

        assert metrics.quality_per_dollar == 17.0  # 0.85 / 0.05
        assert metrics.quality_per_token == 0.00085  # 0.85 / 1000
        assert abs(metrics.tokens_per_quality_unit - 1176.47) < 0.01  # 1000 / 0.85 (allow small floating point error)
        assert abs(metrics.cost_per_quality_unit - 0.0588) < 0.0001  # 0.05 / 0.85 (allow small floating point error)

    def test_content_metrics_bounds(self):
        """Test that content metrics stay within expected bounds."""
        # Test with extreme values
        metrics = ContentMetrics(
            synthesis_length=50000,  # Very long
            coherence_score=1.5,     # Above normal range
            readability_score=-0.5,  # Below normal range
            completeness_score=2.0,  # Above normal range
            insight_density=5.0,     # Way above normal range
            structure_score=1.2      # Above normal range
        )

        # Values should be clamped appropriately by the model
        assert metrics.synthesis_length == 50000
        assert metrics.coherence_score >= 0.0
        assert metrics.readability_score >= 0.0
        assert metrics.completeness_score >= 0.0
        assert metrics.insight_density >= 0.0
        assert metrics.structure_score >= 0.0


# Utility functions for running tests
def run_test_suite():
    """Run the complete test suite."""
    pytest.main([__file__, "-v", "--tb=short"])


def run_integration_tests():
    """Run only integration tests (requires API keys)."""
    pytest.main([__file__ + "::TestQualityEvaluator::test_evaluate_with_llm_judge_success", "-v"])


if __name__ == "__main__":
    run_test_suite()
