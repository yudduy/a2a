"""Integration tests for the LLM Judge evaluation system."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from open_deep_research.evaluation import LLMJudge
from open_deep_research.evaluation.models import (
    EvaluationCriteria, 
    ReportEvaluation,
    ComparativeAnalysis,
    EvaluationResult
)
from open_deep_research.state import RunningReport, AgentExecutionReport


class TestLLMJudge:
    """Test suite for LLM Judge functionality."""

    @pytest.fixture
    def sample_config(self) -> RunnableConfig:
        """Create sample configuration for testing."""
        return RunnableConfig(
            configurable={
                "evaluation_model": "anthropic:claude-3-5-sonnet",
                "evaluation_model_max_tokens": 4096,
                "apiKeys": {
                    "ANTHROPIC_API_KEY": "test_key"
                }
            }
        )

    @pytest.fixture
    def sample_reports(self) -> Dict[str, str]:
        """Create sample reports for testing."""
        return {
            "sequence_a": "This is a comprehensive research report about AI safety...",
            "sequence_b": "This report analyzes market trends in AI safety solutions..."
        }

    @pytest.fixture
    def sample_evaluation_criteria(self) -> EvaluationCriteria:
        """Create sample evaluation criteria."""
        return EvaluationCriteria(
            name="completeness",
            score=8.5,
            max_score=10.0,
            reasoning="The report covers most key aspects comprehensively",
            strengths=["Thorough analysis", "Good coverage"],
            weaknesses=["Missing some edge cases"],
            evidence_examples=["Section 3 provides detailed analysis"]
        )

    @pytest.fixture  
    def sample_report_evaluation(self, sample_evaluation_criteria) -> ReportEvaluation:
        """Create sample report evaluation."""
        return ReportEvaluation(
            report_id="test_report_1",
            sequence_name="test_sequence",
            research_topic="AI Safety Testing",
            evaluation_timestamp=datetime.now(),
            completeness=sample_evaluation_criteria,
            depth=sample_evaluation_criteria,
            coherence=sample_evaluation_criteria,
            innovation=sample_evaluation_criteria,
            actionability=sample_evaluation_criteria,
            overall_score=85.0,
            weighted_criteria_scores={"completeness": 21.25, "depth": 21.25, "coherence": 17.0, "innovation": 12.75, "actionability": 12.75},
            executive_summary="This is a high-quality research report.",
            key_strengths=["Comprehensive analysis", "Clear structure"],
            key_weaknesses=["Could use more examples"],
            recommendation_quality="Good practical value",
            confidence_level=0.9
        )

    def test_llm_judge_initialization(self, sample_config):
        """Test LLM Judge initialization with various configurations."""
        # Test basic initialization
        judge = LLMJudge(config=sample_config)
        assert judge.evaluation_model_name == "anthropic:claude-3-5-sonnet"
        assert judge.max_retries == 3
        
        # Test with custom model override
        judge_custom = LLMJudge(config=sample_config, evaluation_model="openai:gpt-4o")
        assert judge_custom.evaluation_model_name == "openai:gpt-4o"
        
        # Test with custom retries
        judge_retries = LLMJudge(config=sample_config, max_retries=5)
        assert judge_retries.max_retries == 5

    def test_llm_judge_initialization_fallbacks(self):
        """Test fallback behavior for model configuration."""
        # Test with empty config
        judge = LLMJudge()
        assert "claude-3-5-sonnet" in judge.evaluation_model_name
        
        # Test with partial config
        partial_config = RunnableConfig(configurable={})
        judge_partial = LLMJudge(config=partial_config)
        assert judge_partial.evaluation_model_name is not None

    @pytest.mark.asyncio
    async def test_model_initialization_success(self, sample_config):
        """Test successful model initialization."""
        judge = LLMJudge(config=sample_config)
        
        with patch('open_deep_research.evaluation.llm_judge.init_chat_model') as mock_init:
            mock_model = Mock()
            mock_init.return_value = mock_model
            
            model = await judge._get_evaluation_model()
            
            assert model is mock_model
            assert mock_init.called
            assert judge._model is mock_model

    @pytest.mark.asyncio
    async def test_model_initialization_failure(self, sample_config):
        """Test model initialization failure handling."""
        judge = LLMJudge(config=sample_config)
        
        with patch('open_deep_research.evaluation.llm_judge.init_chat_model') as mock_init:
            mock_init.side_effect = Exception("API key invalid")
            
            with pytest.raises(RuntimeError, match="Could not initialize evaluation model"):
                await judge._get_evaluation_model()

    @pytest.mark.asyncio
    async def test_single_report_evaluation_success(self, sample_config, sample_report_evaluation):
        """Test successful single report evaluation."""
        judge = LLMJudge(config=sample_config)
        
        with patch.object(judge, '_get_evaluation_model') as mock_get_model, \
             patch.object(judge, '_execute_with_retry') as mock_execute:
            
            mock_model = AsyncMock()
            mock_get_model.return_value = mock_model
            mock_execute.return_value = sample_report_evaluation
            
            result = await judge.evaluate_single_report(
                report="Test report content",
                research_topic="AI Safety",
                sequence_name="test_sequence"
            )
            
            assert result is not None
            assert result.sequence_name == "test_sequence"
            assert result.research_topic == "AI Safety"
            assert mock_execute.called

    @pytest.mark.asyncio
    async def test_single_report_evaluation_failure(self, sample_config):
        """Test single report evaluation failure handling."""
        judge = LLMJudge(config=sample_config)
        
        with patch.object(judge, '_get_evaluation_model') as mock_get_model, \
             patch.object(judge, '_execute_with_retry') as mock_execute:
            
            mock_get_model.return_value = AsyncMock()
            mock_execute.return_value = None  # Simulate failure
            
            result = await judge.evaluate_single_report(
                report="Test report",
                research_topic="AI Safety",
                sequence_name="test_sequence"
            )
            
            assert result is None

    @pytest.mark.asyncio
    async def test_running_report_content_extraction(self, sample_config):
        """Test extraction of content from RunningReport objects."""
        judge = LLMJudge(config=sample_config)
        
        # Create a sample RunningReport
        agent_report = AgentExecutionReport(
            agent_name="test_agent",
            agent_type="test_type",
            execution_start=datetime.now(),
            execution_end=datetime.now(),
            execution_duration=60.0,
            insights=["Key insight 1", "Key insight 2"],
            research_content="Detailed research findings...",
            questions_addressed=["Question 1", "Question 2"],
            completion_confidence=0.9,
            insight_quality_score=0.85,
            research_depth_score=0.88,
            handoff_context={},
            suggested_next_questions=[]
        )
        
        running_report = RunningReport(
            research_topic="Test Topic",
            sequence_name="test_sequence",
            start_time=datetime.now(),
            agent_reports=[agent_report],
            all_insights=["Insight 1", "Insight 2"],
            insight_connections=[],
            executive_summary="Test executive summary",
            detailed_findings=["Finding 1", "Finding 2"],
            recommendations=["Recommendation 1", "Recommendation 2"],
            total_agents_executed=1,
            total_execution_time=60.0,
            completion_status="completed"
        )
        
        content = judge._extract_report_content_from_running_report(running_report)
        
        assert "Test executive summary" in content
        assert "Finding 1" in content
        assert "Recommendation 1" in content
        assert "Insight 1" in content

    @pytest.mark.asyncio 
    async def test_execute_with_retry_success(self, sample_config):
        """Test retry mechanism with successful execution."""
        judge = LLMJudge(config=sample_config)
        
        mock_func = AsyncMock()
        mock_func.return_value = "success"
        
        result = await judge._execute_with_retry(mock_func, "arg1", "arg2", context="test")
        
        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_failure_then_success(self, sample_config):
        """Test retry mechanism with initial failures then success."""
        judge = LLMJudge(config=sample_config, max_retries=2)
        
        mock_func = AsyncMock()
        mock_func.side_effect = [Exception("First attempt"), Exception("Second attempt"), "success"]
        
        with patch('asyncio.sleep'):  # Mock sleep to speed up test
            result = await judge._execute_with_retry(mock_func, context="test")
        
        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_all_failures(self, sample_config):
        """Test retry mechanism with all attempts failing."""
        judge = LLMJudge(config=sample_config, max_retries=1)
        
        mock_func = AsyncMock()
        mock_func.side_effect = Exception("All attempts fail")
        
        with patch('asyncio.sleep'):  # Mock sleep to speed up test
            result = await judge._execute_with_retry(mock_func, context="test")
        
        assert result is None
        assert mock_func.call_count == 2  # Initial + 1 retry

    @pytest.mark.asyncio
    async def test_evaluate_reports_empty_input(self, sample_config):
        """Test evaluation with empty reports input."""
        judge = LLMJudge(config=sample_config)
        
        with pytest.raises(ValueError, match="Reports dictionary cannot be empty"):
            await judge.evaluate_reports({}, "Test Topic")

    @pytest.mark.asyncio
    async def test_evaluate_reports_integration(self, sample_config, sample_reports, sample_report_evaluation):
        """Test full report evaluation integration."""
        judge = LLMJudge(config=sample_config)
        
        # Create mock comparative analysis
        mock_comparative_analysis = ComparativeAnalysis(
            research_topic="Test Topic",
            num_reports_compared=2,
            analysis_timestamp=datetime.now(),
            overall_ranking=[{"sequence": "sequence_a", "score": 85.0}],
            best_sequence="sequence_a",
            best_sequence_reasoning="Superior analysis depth",
            criteria_leaders={"completeness": "sequence_a"},
            criteria_analysis={"completeness": "Sequence A provides more comprehensive coverage"},
            pairwise_comparisons=[],
            sequence_strengths_patterns={"sequence_a": ["thorough analysis"]},
            sequence_weakness_patterns={"sequence_a": ["could be more concise"]},
            sequence_selection_guide=[{"context": "comprehensive research", "recommendation": "sequence_a"}],
            improvement_recommendations={"sequence_a": ["improve conciseness"]},
            evaluation_confidence=0.9,
            methodology_notes="Standard LLM evaluation approach"
        )
        
        with patch.object(judge, '_evaluate_individual_reports') as mock_individual, \
             patch.object(judge, '_perform_comparative_analysis') as mock_comparative:
            
            mock_individual.return_value = [sample_report_evaluation]
            mock_comparative.return_value = mock_comparative_analysis
            
            result = await judge.evaluate_reports(sample_reports, "Test Topic")
            
            assert isinstance(result, EvaluationResult)
            assert result.winning_sequence == "test_sequence"  # From sample_report_evaluation
            assert result.processing_time > 0
            assert len(result.individual_evaluations) == 1

    def test_compare_reports_functionality(self, sample_config, sample_report_evaluation):
        """Test report comparison functionality."""
        judge = LLMJudge(config=sample_config)
        
        # Create multiple evaluations
        eval1 = sample_report_evaluation
        eval1.sequence_name = "sequence_a"
        eval1.overall_score = 85.0
        
        eval2 = sample_report_evaluation.model_copy()
        eval2.sequence_name = "sequence_b" 
        eval2.overall_score = 78.0
        
        comparison = judge.compare_reports([eval1, eval2])
        
        assert comparison["winner"] == "sequence_a"
        assert comparison["winner_score"] == 85.0
        assert comparison["performance_spread"] == 7.0
        assert len(comparison["ranking"]) == 2
        assert comparison["ranking"][0]["sequence_name"] == "sequence_a"

    def test_compare_reports_empty_input(self, sample_config):
        """Test compare_reports with empty input."""
        judge = LLMJudge(config=sample_config)
        
        result = judge.compare_reports([])
        
        assert "error" in result
        assert "No evaluations provided" in result["error"]

    def test_determine_winner_functionality(self, sample_config, sample_report_evaluation):
        """Test winner determination functionality."""
        judge = LLMJudge(config=sample_config)
        
        # Create evaluations with different scores
        eval1 = sample_report_evaluation
        eval1.sequence_name = "winner_sequence"
        eval1.overall_score = 92.0
        
        eval2 = sample_report_evaluation.model_copy()
        eval2.sequence_name = "runner_up"
        eval2.overall_score = 84.0
        
        winner_info = judge.determine_winner([eval1, eval2])
        
        assert winner_info["winner"] == "winner_sequence"
        assert winner_info["winning_score"] == 92.0
        assert winner_info["margin_over_second"] == 8.0
        assert winner_info["confidence"] == 0.9

    def test_determine_winner_empty_input(self, sample_config):
        """Test determine_winner with empty input."""
        judge = LLMJudge(config=sample_config)
        
        result = judge.determine_winner([])
        
        assert "error" in result
        assert "No evaluations provided" in result["error"]

    def test_evaluation_result_helper_methods(self, sample_report_evaluation):
        """Test EvaluationResult helper methods."""
        # Create a sample EvaluationResult
        comparative_analysis = ComparativeAnalysis(
            research_topic="Test Topic",
            num_reports_compared=1,
            analysis_timestamp=datetime.now(),
            overall_ranking=[],
            best_sequence="test_sequence",
            best_sequence_reasoning="Test reasoning",
            criteria_leaders={},
            criteria_analysis={},
            pairwise_comparisons=[],
            sequence_strengths_patterns={},
            sequence_weakness_patterns={},
            sequence_selection_guide=[],
            improvement_recommendations={},
            evaluation_confidence=0.9,
            methodology_notes="Test methodology"
        )
        
        result = EvaluationResult(
            individual_evaluations=[sample_report_evaluation],
            comparative_analysis=comparative_analysis,
            score_statistics={"mean": 85.0},
            performance_gaps={},
            winning_sequence="test_sequence",
            winning_sequence_score=85.0,
            key_differentiators=["thorough analysis"],
            sequence_recommendations={"test_sequence": "best for comprehensive research"},
            evaluation_model="test_model"
        )
        
        # Test get_winner_details
        winner_details = result.get_winner_details()
        assert winner_details["sequence_name"] == "test_sequence"
        assert winner_details["overall_score"] == 85.0
        assert "completeness" in winner_details["criteria_scores"]
        
        # Test get_criteria_rankings
        rankings = result.get_criteria_rankings()
        assert "completeness" in rankings
        assert len(rankings["completeness"]) == 1
        assert rankings["completeness"][0]["sequence_name"] == "test_sequence"

    @pytest.mark.asyncio
    async def test_token_limit_handling(self, sample_config):
        """Test handling of token limit exceeded errors."""
        judge = LLMJudge(config=sample_config)
        
        # Mock a token limit error
        token_error = Exception("Token limit exceeded")
        
        with patch('open_deep_research.evaluation.llm_judge.is_token_limit_exceeded') as mock_token_check:
            mock_token_check.return_value = True
            
            mock_func = AsyncMock()
            mock_func.side_effect = token_error
            
            with patch('asyncio.sleep'):  # Speed up test
                result = await judge._execute_with_retry(mock_func, context="token_test")
            
            assert result is None
            assert mock_token_check.called

    def test_model_configuration_patterns(self, sample_config):
        """Test integration with existing model configuration patterns."""
        judge = LLMJudge(config=sample_config)
        
        # Verify configuration integration
        config = judge.configuration
        assert hasattr(config, 'max_structured_output_retries')
        
        # Test model name handling
        assert judge.evaluation_model_name == "anthropic:claude-3-5-sonnet"
        
        # Test fallback behavior
        judge_no_config = LLMJudge()
        assert judge_no_config.evaluation_model_name is not None


# Utility functions for running tests
def run_test_suite():
    """Run the complete test suite."""
    pytest.main([__file__, "-v", "--tb=short"])


def run_integration_tests():
    """Run only integration tests (requires API keys)."""
    pytest.main([__file__ + "::TestLLMJudge::test_evaluate_reports_integration", "-v"])


if __name__ == "__main__":
    print("Running LLM Judge Test Suite...")
    run_test_suite()