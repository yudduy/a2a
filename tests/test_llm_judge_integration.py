"""Test LLM Judge integration into the main deep research workflow."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from open_deep_research.state import AgentState, RunningReport
from open_deep_research.deep_researcher import (
    extract_sequence_reports_for_evaluation,
    create_enhanced_final_report_prompt,
    create_orchestration_insights
)

class TestLLMJudgeIntegration:
    """Test suite for LLM Judge integration."""
    
    @pytest.mark.asyncio
    async def test_extract_sequence_reports_from_parallel_results(self):
        """Test extracting sequence reports from parallel execution results."""
        # Mock parallel execution results
        state = {
            "parallel_sequence_results": {
                "sequence_results": {
                    "seq_0_comprehensive": {
                        "comprehensive_findings": [
                            "Key finding 1 from comprehensive sequence",
                            "Key finding 2 from comprehensive sequence"
                        ],
                        "agent_results": [
                            {
                                "agent_type": "research_analyst",
                                "key_insights": ["Research insight 1", "Research insight 2"]
                            },
                            {
                                "agent_type": "domain_expert", 
                                "key_insights": ["Domain insight 1"]
                            }
                        ],
                        "total_duration": 45.2,
                        "productivity_score": 0.85
                    },
                    "seq_1_focused": {
                        "comprehensive_findings": [
                            "Key finding from focused sequence"
                        ],
                        "agent_results": [
                            {
                                "agent_type": "specialist",
                                "key_insights": ["Specialist insight"]
                            }
                        ],
                        "total_duration": 32.1,
                        "productivity_score": 0.92
                    }
                }
            }
        }
        
        # Extract sequence reports
        sequence_reports = await extract_sequence_reports_for_evaluation(state)
        
        # Verify extraction
        assert len(sequence_reports) == 2
        assert "seq_0_comprehensive" in sequence_reports
        assert "seq_1_focused" in sequence_reports
        
        # Check content structure
        comp_report = sequence_reports["seq_0_comprehensive"]
        assert "Key finding 1 from comprehensive sequence" in comp_report
        assert "research_analyst" in comp_report
        assert "Duration: 45.2 seconds" in comp_report
        assert "Productivity Score: 0.85" in comp_report
        
        focused_report = sequence_reports["seq_1_focused"]
        assert "Key finding from focused sequence" in focused_report
        assert "specialist" in focused_report
    
    @pytest.mark.asyncio 
    async def test_extract_sequence_reports_from_running_report(self):
        """Test extracting sequence reports from sequential execution running report."""
        # Create mock running report
        running_report = RunningReport(
            research_topic="AI Safety Research",
            sequence_name="sequential_analysis",
            start_time=datetime.now(),
            agent_reports=[],
            all_insights=["Insight 1", "Insight 2"],
            insight_connections=[],
            executive_summary="This is the executive summary of findings.",
            detailed_findings=["Finding 1", "Finding 2", "Finding 3"],
            recommendations=["Recommendation 1", "Recommendation 2"],
            total_agents_executed=3,
            total_execution_time=120.5,
            completion_status="completed"
        )
        
        state = {
            "running_report": running_report,
            "notes": ["Additional note 1", "Additional note 2"]
        }
        
        # Extract sequence reports
        sequence_reports = await extract_sequence_reports_for_evaluation(state)
        
        # Verify extraction
        assert len(sequence_reports) == 1
        assert "sequential_analysis" in sequence_reports
        
        report_content = sequence_reports["sequential_analysis"]
        assert "This is the executive summary of findings." in report_content
        assert "Finding 1" in report_content
        assert "Finding 2" in report_content
        assert "Recommendation 1" in report_content
    
    @pytest.mark.asyncio
    async def test_extract_sequence_reports_from_strategic_sequences(self):
        """Test extracting reports from strategic sequences with notes."""
        from open_deep_research.supervisor.sequence_models import AgentSequence
        
        strategic_sequences = [
            AgentSequence(
                sequence_name="Comprehensive Research Sequence",
                agent_names=["general_researcher", "academic_analyst"],
                rationale="Sequential comprehensive research approach",
                approach_description="Linear progression through specialized research agents",
                expected_outcomes=["Foundational research", "Specialized insights"],
                confidence_score=0.8,
                research_focus="Comprehensive coverage"
            ),
            AgentSequence(
                sequence_name="Focused Strategic Sequence",
                agent_names=["domain_expert", "industry_analyst"],
                rationale="Targeted research approach",
                approach_description="Strategic focus on key research agents",
                expected_outcomes=["Deep domain expertise", "Strategic insights"],
                confidence_score=0.7,
                research_focus="Strategic focus"
            )
        ]
        
        state = {
            "strategic_sequences": strategic_sequences,
            "notes": [
                "Research finding 1",
                "Research finding 2", 
                "Research finding 3",
                "Research finding 4"
            ]
        }
        
        # Extract sequence reports
        sequence_reports = await extract_sequence_reports_for_evaluation(state)
        
        # Verify extraction
        assert len(sequence_reports) == 2
        assert "Comprehensive Research Sequence" in sequence_reports
        assert "Focused Strategic Sequence" in sequence_reports
        
        comp_report = sequence_reports["Comprehensive Research Sequence"]
        assert "Comprehensive coverage" in comp_report
        assert "Linear progression through specialized research agents" in comp_report
        # Check note distribution (every 2nd note starting from index 0)
        assert "Research finding 1" in comp_report
        assert "Research finding 3" in comp_report
    
    def test_create_enhanced_final_report_prompt_with_evaluation(self):
        """Test creating enhanced final report prompt with evaluation results."""
        # Mock evaluation result
        evaluation_result = MagicMock()
        evaluation_result.individual_evaluations = [MagicMock(), MagicMock()]
        evaluation_result.winning_sequence = "Best Sequence"
        evaluation_result.winning_sequence_score = 87.5
        evaluation_result.key_differentiators = [
            "Superior research depth",
            "Excellent coherence",
            "Strong actionability"
        ]
        evaluation_result.performance_gaps = {
            "Alternative Sequence": 12.3,
            "Third Sequence": 8.7
        }
        evaluation_result.comparative_analysis = MagicMock()
        evaluation_result.comparative_analysis.best_sequence_reasoning = "This approach excelled due to systematic methodology"
        
        # Mock state
        state = {
            "research_brief": "AI safety research",
            "messages": []
        }
        
        with patch('open_deep_research.deep_researcher.final_report_generation_prompt') as mock_prompt, \
             patch('open_deep_research.deep_researcher.get_buffer_string') as mock_buffer, \
             patch('open_deep_research.deep_researcher.get_today_str') as mock_date:
            
            mock_prompt.format.return_value = "Base report prompt"
            mock_buffer.return_value = ""
            mock_date.return_value = "2024-01-01"
            
            # Create enhanced prompt
            enhanced_prompt = create_enhanced_final_report_prompt(
                state=state,
                findings="Research findings here",
                evaluation_result=evaluation_result
            )
            
            # Verify enhancement
            assert "Base report prompt" in enhanced_prompt
            assert "ORCHESTRATION EVALUATION INSIGHTS" in enhanced_prompt
            assert "**Best Performing Approach:** Best Sequence" in enhanced_prompt
            assert "**Score:** 87.5/100" in enhanced_prompt
            assert "Superior research depth" in enhanced_prompt
            assert "This approach excelled due to systematic methodology" in enhanced_prompt
            assert "Alternative Sequence: 12.3 points behind best approach" in enhanced_prompt
    
    def test_create_enhanced_final_report_prompt_without_evaluation(self):
        """Test creating final report prompt without evaluation results."""
        state = {
            "research_brief": "AI safety research",
            "messages": []
        }
        
        with patch('open_deep_research.deep_researcher.final_report_generation_prompt') as mock_prompt, \
             patch('open_deep_research.deep_researcher.get_buffer_string') as mock_buffer, \
             patch('open_deep_research.deep_researcher.get_today_str') as mock_date:
            
            mock_prompt.format.return_value = "Base report prompt"
            mock_buffer.return_value = ""
            mock_date.return_value = "2024-01-01"
            
            # Create prompt without evaluation
            enhanced_prompt = create_enhanced_final_report_prompt(
                state=state,
                findings="Research findings here",
                evaluation_result=None
            )
            
            # Verify no enhancement
            assert enhanced_prompt == "Base report prompt"
            assert "ORCHESTRATION EVALUATION INSIGHTS" not in enhanced_prompt
    
    def test_create_orchestration_insights(self):
        """Test creating structured orchestration insights."""
        # Mock evaluation result with detailed structure
        evaluation_result = MagicMock()
        evaluation_result.winning_sequence = "Best Sequence"
        evaluation_result.winning_sequence_score = 89.2
        
        # Mock individual evaluations
        winning_eval = MagicMock()
        winning_eval.sequence_name = "Best Sequence"
        winning_eval.key_strengths = ["Comprehensive analysis", "Clear methodology", "Strong conclusions"]
        winning_eval.completeness = MagicMock(score=9.2)
        winning_eval.depth = MagicMock(score=8.8)
        winning_eval.coherence = MagicMock(score=9.0)
        winning_eval.innovation = MagicMock(score=7.5)
        winning_eval.actionability = MagicMock(score=8.5)
        
        other_eval = MagicMock()
        other_eval.sequence_name = "Alternative Sequence"
        other_eval.key_strengths = ["Speed of execution"]
        
        evaluation_result.individual_evaluations = [winning_eval, other_eval]
        
        # Mock comparative analysis
        evaluation_result.comparative_analysis = MagicMock()
        evaluation_result.comparative_analysis.criteria_leaders = {
            "completeness": "Best Sequence",
            "depth": "Best Sequence",
            "coherence": "Alternative Sequence"
        }
        
        # Create insights
        insights = create_orchestration_insights(evaluation_result)
        
        # Verify structure and content
        assert insights["summary"] == "Evaluated 2 research approaches"
        assert insights["best_approach"]["name"] == "Best Sequence"
        assert insights["best_approach"]["score"] == 89.2
        assert insights["best_approach"]["advantages"] == ["Comprehensive analysis", "Clear methodology", "Strong conclusions"]
        
        # Check key learnings (criteria with score >= 8.0)
        expected_learnings = [
            "Superior completeness: 9.2/10",
            "Superior depth: 8.8/10", 
            "Superior coherence: 9.0/10",
            "Superior actionability: 8.5/10"
        ]
        for learning in expected_learnings:
            assert learning in insights["key_learnings"]
        
        # Check recommendations
        assert insights["recommendations"]["Best Sequence"] == "Best for: Comprehensive analysis"
        assert insights["recommendations"]["Alternative Sequence"] == "Best for: Speed of execution"
        
        # Check methodology effectiveness
        assert insights["methodology_effectiveness"]["completeness"] == "Best Sequence"
        assert insights["methodology_effectiveness"]["coherence"] == "Alternative Sequence"


if __name__ == "__main__":
    # Run a simple test to verify basic functionality
    import sys
    import os
    
    # Add the src directory to the path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    # Run the test
    async def main():
        test_instance = TestLLMJudgeIntegration()
        
        print("Testing extract_sequence_reports_from_parallel_results...")
        await test_instance.test_extract_sequence_reports_from_parallel_results()
        print("✓ Passed")
        
        print("\nTesting extract_sequence_reports_from_running_report...")
        await test_instance.test_extract_sequence_reports_from_running_report()
        print("✓ Passed")
        
        print("\nTesting extract_sequence_reports_from_strategic_sequences...")
        await test_instance.test_extract_sequence_reports_from_strategic_sequences()
        print("✓ Passed")
        
        print("\nTesting create_enhanced_final_report_prompt_with_evaluation...")
        test_instance.test_create_enhanced_final_report_prompt_with_evaluation()
        print("✓ Passed")
        
        print("\nTesting create_enhanced_final_report_prompt_without_evaluation...")
        test_instance.test_create_enhanced_final_report_prompt_without_evaluation()
        print("✓ Passed")
        
        print("\nTesting create_orchestration_insights...")
        test_instance.test_create_orchestration_insights()
        print("✓ Passed")
        
        print("\n🎉 All LLM Judge integration tests passed!")
    
    asyncio.run(main())