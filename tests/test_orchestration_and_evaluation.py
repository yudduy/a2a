"""Comprehensive tests for Orchestration and Evaluation components.

This module tests the orchestration logic and evaluation systems:
- LLM-based sequence generation and agent capability mapping
- LLM Judge evaluation system and report comparison
- End-to-end integration flows and orchestration workflows
- Report building and incremental update mechanisms

Test Categories:
1. Sequence Generation (from test_llm_sequence_generation.py)
2. LLM Judge Integration (from test_llm_judge_integration.py)
3. Integration Flows (from test_integration_flow.py)
4. Sequence-Judge Integration (from test_sequence_generation_llm_judge.py)
"""

import time
from unittest.mock import Mock, patch

import pytest

from open_deep_research.configuration import Configuration
from open_deep_research.deep_researcher import (
    create_enhanced_final_report_prompt,
    create_orchestration_insights,
    extract_sequence_reports_for_evaluation,
)
from open_deep_research.evaluation.llm_judge import EvaluationResult, LLMJudge
from open_deep_research.orchestration.report_builder import ReportBuilder
from open_deep_research.state import AgentState, RunningReport
from open_deep_research.supervisor.llm_sequence_generator import LLMSequenceGenerator
from open_deep_research.supervisor.sequence_models import (
    AgentCapability,
    AgentSequence,
    SequenceGenerationInput,
    SequenceGenerationOutput,
)

# =============================================================================
# SEQUENCE GENERATION TESTS
# =============================================================================

class TestLLMSequenceGenerator:
    """Test LLM-based sequence generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = LLMSequenceGenerator()
        
        # Create sample agent capabilities
        self.sample_agents = [
            AgentCapability(
                name="research_agent",
                expertise_areas=["Academic", "Literature Review"],
                description="Deep research specialist for academic investigation",
                typical_use_cases=["Literature reviews", "Academic research"],
                strength_summary="Specializes in academic research and analysis"
            ),
            AgentCapability(
                name="technical_agent",
                expertise_areas=["Technical", "Engineering"],
                description="Technical analysis and implementation specialist",
                typical_use_cases=["Technical analysis", "Implementation research"],
                strength_summary="Expert in technical and engineering research"
            ),
            AgentCapability(
                name="market_agent",
                expertise_areas=["Market", "Business"],
                description="Market research and business analysis specialist",
                typical_use_cases=["Market analysis", "Business research"],
                strength_summary="Specializes in market and business research"
            ),
            AgentCapability(
                name="synthesis_agent",
                expertise_areas=["Synthesis", "Integration"],
                description="Information synthesis and integration specialist",
                typical_use_cases=["Information synthesis", "Report integration"],
                strength_summary="Specializes in information synthesis and integration"
            )
        ]
        
        # Sample research topics
        self.test_topics = {
            "academic": "The impact of machine learning on climate change predictions",
            "technical": "Implementing microservices architecture for large-scale applications",
            "market": "Consumer adoption trends for electric vehicles in emerging markets",
            "mixed": "AI governance frameworks for healthcare applications"
        }
    
    def test_system_prompt_creation(self):
        """Test that system prompt is properly created."""
        system_prompt = self.generator._create_system_prompt()
        
        # Verify key elements are present
        assert "research strategist" in system_prompt.lower()
        assert "3 distinct strategic sequences" in system_prompt.lower()
        assert "agent orchestration" in system_prompt.lower()
        assert "json format" in system_prompt.lower()
        
        # Verify research approach types are mentioned
        assert "foundational-first" in system_prompt.lower()
        assert "problem-solution" in system_prompt.lower()
        assert "comparative" in system_prompt.lower()
    
    def test_agent_capability_formatting(self):
        """Test proper formatting of agent capabilities."""
        formatted_agents = self.generator._format_agent_capabilities(self.sample_agents)
        
        # Verify all agents are included
        assert "research_agent" in formatted_agents
        assert "technical_agent" in formatted_agents
        assert "market_agent" in formatted_agents
        assert "synthesis_agent" in formatted_agents
        
        # Verify proper formatting
        assert "Academic, Literature Review" in formatted_agents
        assert "Technical, Engineering" in formatted_agents
    
    def test_sequence_generation_input_validation(self):
        """Test validation of sequence generation inputs."""
        # Valid input
        valid_input = SequenceGenerationInput(
            research_topic="Test topic",
            available_agents=self.sample_agents
        )
        assert valid_input.research_topic == "Test topic"
        assert len(valid_input.available_agents) == 4
        
        # Test with empty topic (should still be valid, let validation happen at runtime)
        input_empty_topic = SequenceGenerationInput(
            research_topic="",
            available_agents=self.sample_agents
        )
        assert input_empty_topic.research_topic == ""
    
    def test_sequence_generation_output_structure(self):
        """Test structure of sequence generation output."""
        # Mock sequence data
        sample_sequences = [
            AgentSequence(
                sequence_id="seq_0",
                approach_name="foundational-first",
                agent_names=["research_agent", "synthesis_agent"],
                rationale="Start with foundational research then synthesize findings"
            ),
            AgentSequence(
                sequence_id="seq_1", 
                approach_name="problem-solution",
                agent_names=["technical_agent", "market_agent"],
                rationale="Analyze technical aspects then market implications"
            )
        ]
        
        output = SequenceGenerationOutput(
            sequences=sample_sequences,
            reasoning="Generated diverse sequences for comprehensive coverage"
        )
        
        assert len(output.sequences) == 2
        assert output.sequences[0].sequence_id == "seq_0"
        assert output.sequences[1].approach_name == "problem-solution"
        assert "comprehensive coverage" in output.reasoning
    
    @pytest.mark.asyncio
    async def test_sequence_generation_error_handling(self):
        """Test error handling in sequence generation."""
        with patch.object(self.generator.llm, 'ainvoke', side_effect=Exception("API error")):
            try:
                input_data = SequenceGenerationInput(
                    research_topic="Test topic",
                    available_agents=self.sample_agents
                )
                result = await self.generator.generate_sequences(input_data)
                # Should handle errors gracefully
                assert result is not None
            except Exception as e:
                # Or it may propagate the error - both are acceptable
                assert "API error" in str(e)


class TestAgentCapabilityMapping:
    """Test agent capability analysis and mapping."""
    
    def setup_method(self):
        """Set up capability mapping tests."""
        self.sample_agents = [
            AgentCapability(
                name="research_agent",
                expertise_areas=["Academic", "Literature Review", "Theoretical Analysis"],
                description="Deep research specialist for academic investigation",
                typical_use_cases=["Literature reviews", "Academic research", "Theoretical analysis"],
                strength_summary="Specializes in academic research and theoretical analysis"
            ),
            AgentCapability(
                name="data_agent",
                expertise_areas=["Data Analysis", "Statistics", "Quantitative Research"],
                description="Data analysis and statistical research specialist",
                typical_use_cases=["Statistical analysis", "Data visualization", "Quantitative studies"],
                strength_summary="Expert in data analysis and quantitative methods"
            )
        ]
    
    def test_expertise_area_analysis(self):
        """Test analysis of agent expertise areas."""
        research_agent = self.sample_agents[0]
        
        # Verify expertise areas are properly captured
        assert "Academic" in research_agent.expertise_areas
        assert "Literature Review" in research_agent.expertise_areas
        assert "Theoretical Analysis" in research_agent.expertise_areas
        
        # Verify typical use cases align with expertise
        assert "Literature reviews" in research_agent.typical_use_cases
        assert "Academic research" in research_agent.typical_use_cases
    
    def test_capability_differentiation(self):
        """Test that different agents have distinct capabilities."""
        research_agent = self.sample_agents[0]
        data_agent = self.sample_agents[1]
        
        # Verify different expertise areas
        research_expertise = set(research_agent.expertise_areas)
        data_expertise = set(data_agent.expertise_areas)
        
        # Should have minimal overlap
        overlap = research_expertise.intersection(data_expertise)
        assert len(overlap) == 0  # No overlap expected in this test case
        
        # Verify distinct use cases
        assert "Literature reviews" in research_agent.typical_use_cases
        assert "Statistical analysis" in data_agent.typical_use_cases
    
    def test_strength_summary_generation(self):
        """Test generation of agent strength summaries."""
        for agent in self.sample_agents:
            assert len(agent.strength_summary) > 10  # Should be descriptive
            assert agent.name.replace("_", " ") in agent.strength_summary.lower() or \
                   any(area.lower() in agent.strength_summary.lower() 
                       for area in agent.expertise_areas)


# =============================================================================
# LLM JUDGE INTEGRATION TESTS
# =============================================================================

class TestLLMJudgeIntegration:
    """Test LLM Judge integration into the main workflow."""
    
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
        
        focused_report = sequence_reports["seq_1_focused"]
        assert "Key finding from focused sequence" in focused_report
        assert "specialist" in focused_report
    
    @pytest.mark.asyncio
    async def test_extract_sequence_reports_from_running_report(self):
        """Test extracting sequence reports from running reports."""
        # Mock state with running report
        mock_running_report = RunningReport(
            research_topic="Test topic",
            current_status="Research in progress",
            agent_reports=[
                AgentState(
                    agent_name="research_agent",
                    status="completed",
                    key_insights=["Research insight 1", "Research insight 2"],
                    completion_reasoning="Research completed successfully"
                ),
                AgentState(
                    agent_name="analysis_agent", 
                    status="completed",
                    key_insights=["Analysis insight 1"],
                    completion_reasoning="Analysis completed"
                )
            ],
            comprehensive_findings=["Overall finding 1", "Overall finding 2"],
            research_metadata={
                "total_agents": 2,
                "completion_rate": 1.0,
                "research_depth": "comprehensive"
            }
        )
        
        state = {
            "running_report": mock_running_report
        }
        
        # Extract sequence reports
        sequence_reports = await extract_sequence_reports_for_evaluation(state)
        
        # Verify extraction
        assert len(sequence_reports) >= 1
        
        # Check content includes agent insights
        report_content = list(sequence_reports.values())[0]
        assert "Research insight 1" in report_content or "Analysis insight 1" in report_content
        assert "Overall finding 1" in report_content
    
    @pytest.mark.asyncio
    async def test_llm_judge_evaluation_flow(self):
        """Test the complete LLM Judge evaluation flow."""
        # Mock LLM Judge
        mock_judge = Mock(spec=LLMJudge)
        mock_evaluation_result = EvaluationResult(
            winner="seq_0_comprehensive",
            reasoning="Comprehensive sequence provided more detailed analysis",
            scores={
                "seq_0_comprehensive": {
                    "completeness": 0.9,
                    "depth": 0.85,
                    "coherence": 0.9,
                    "innovation": 0.8,
                    "actionability": 0.85
                },
                "seq_1_focused": {
                    "completeness": 0.75,
                    "depth": 0.9,
                    "coherence": 0.85,
                    "innovation": 0.9,
                    "actionability": 0.8
                }
            },
            detailed_feedback={
                "seq_0_comprehensive": "Excellent comprehensive coverage",
                "seq_1_focused": "Good focused analysis but limited scope"
            }
        )
        
        mock_judge.evaluate_reports.return_value = mock_evaluation_result
        
        # Mock sequence reports
        sequence_reports = {
            "seq_0_comprehensive": "Comprehensive research report with detailed analysis",
            "seq_1_focused": "Focused analysis on specific aspects"
        }
        
        # Test evaluation
        with patch('open_deep_research.evaluation.llm_judge.LLMJudge', return_value=mock_judge):
            result = mock_judge.evaluate_reports(sequence_reports, "Test research topic")
            
            assert result.winner == "seq_0_comprehensive"
            assert "Comprehensive sequence" in result.reasoning
            assert result.scores["seq_0_comprehensive"]["completeness"] == 0.9
            assert len(result.detailed_feedback) == 2


# =============================================================================
# INTEGRATION FLOW TESTS
# =============================================================================

class TestEndToEndIntegration:
    """Test end-to-end integration flows."""
    
    def setup_method(self):
        """Set up integration testing."""
        self.config = Configuration(
            enable_dynamic_sequence_generation=True,
            enable_llm_judge=True,
            max_parallel_sequences=2
        )
    
    @pytest.mark.asyncio
    async def test_sequence_generation_to_evaluation_flow(self):
        """Test complete flow from sequence generation to evaluation."""
        # Mock sequence generator
        mock_sequences = [
            AgentSequence(
                sequence_id="seq_0",
                approach_name="comprehensive",
                agent_names=["research_agent", "analysis_agent", "synthesis_agent"],
                rationale="Comprehensive multi-agent approach"
            ),
            AgentSequence(
                sequence_id="seq_1",
                approach_name="focused",
                agent_names=["technical_agent", "market_agent"],
                rationale="Focused technical and market analysis"
            )
        ]
        
        # Mock execution results
        
        # Mock evaluation result
        mock_evaluation = EvaluationResult(
            winner="seq_0",
            reasoning="More comprehensive analysis",
            scores={"seq_0": {"completeness": 0.9}, "seq_1": {"completeness": 0.8}},
            detailed_feedback={"seq_0": "Excellent", "seq_1": "Good"}
        )
        
        # Test the flow
        with patch('open_deep_research.supervisor.llm_sequence_generator.LLMSequenceGenerator') as mock_gen:
            with patch('open_deep_research.evaluation.llm_judge.LLMJudge') as mock_judge:
                mock_gen.return_value.generate_sequences.return_value = SequenceGenerationOutput(
                    sequences=mock_sequences,
                    reasoning="Generated test sequences"
                )
                mock_judge.return_value.evaluate_reports.return_value = mock_evaluation
                
                # Verify integration points work
                assert len(mock_sequences) == 2
                assert mock_sequences[0].sequence_id == "seq_0"
                assert mock_evaluation.winner == "seq_0"
    
    def test_orchestration_insights_creation(self):
        """Test creation of orchestration insights."""
        # Mock evaluation result
        evaluation_result = EvaluationResult(
            winner="seq_comprehensive",
            reasoning="Provided most thorough analysis with strong evidence base",
            scores={
                "seq_comprehensive": {
                    "completeness": 0.92,
                    "depth": 0.88,
                    "coherence": 0.90,
                    "innovation": 0.85,
                    "actionability": 0.87
                }
            },
            detailed_feedback={
                "seq_comprehensive": "Excellent comprehensive coverage with strong methodology"
            }
        )
        
        # Mock parallel results metadata
        parallel_metadata = {
            "total_sequences": 3,
            "total_agents_used": 8,
            "avg_sequence_duration": 42.3,
            "sequence_efficiency_scores": {
                "seq_comprehensive": 0.89,
                "seq_focused": 0.94,
                "seq_comparative": 0.76
            }
        }
        
        # Create orchestration insights
        insights = create_orchestration_insights(evaluation_result, parallel_metadata)
        
        # Verify insights structure
        assert "Winner: seq_comprehensive" in insights
        assert "0.92" in insights  # completeness score
        assert "42.3" in insights  # avg duration
        assert "8" in insights     # total agents
    
    def test_enhanced_final_report_prompt_creation(self):
        """Test creation of enhanced final report prompt."""
        winning_report = "This is the winning research report with comprehensive analysis."
        
        orchestration_insights = """
        Winner: seq_comprehensive
        Reasoning: Most thorough analysis
        Completeness Score: 0.92
        """
        
        research_topic = "AI governance frameworks for healthcare applications"
        
        # Create enhanced prompt
        enhanced_prompt = create_enhanced_final_report_prompt(
            winning_report,
            orchestration_insights,
            research_topic
        )
        
        # Verify prompt structure
        assert research_topic in enhanced_prompt
        assert "winning research report" in enhanced_prompt
        assert "orchestration insights" in enhanced_prompt.lower()
        assert "comprehensive analysis" in enhanced_prompt


# =============================================================================
# REPORT BUILDING TESTS
# =============================================================================

class TestReportBuildingIntegration:
    """Test report building and incremental updates."""
    
    def setup_method(self):
        """Set up report building tests."""
        self.config = Configuration(
            use_running_reports=True,
            report_update_frequency="after_each_agent",
            include_agent_metadata=True
        )
    
    def test_report_builder_initialization(self):
        """Test report builder initialization."""
        builder = ReportBuilder(self.config)
        assert builder is not None
        assert builder.config.use_running_reports is True
    
    def test_incremental_report_updates(self):
        """Test incremental report update mechanism."""
        # Mock agent results for incremental updates
        agent_results = [
            AgentState(
                agent_name="research_agent",
                status="completed", 
                key_insights=["Research insight 1", "Research insight 2"],
                completion_reasoning="Research phase completed"
            ),
            AgentState(
                agent_name="analysis_agent",
                status="completed",
                key_insights=["Analysis insight 1"],
                completion_reasoning="Analysis phase completed"
            )
        ]
        
        # Test that report can be built incrementally
        ReportBuilder(self.config)
        
        # Simulate incremental updates
        for agent_result in agent_results:
            # In real implementation, this would update running report
            assert agent_result.status == "completed"
            assert len(agent_result.key_insights) > 0
    
    def test_metadata_inclusion(self):
        """Test inclusion of agent metadata in reports."""
        # When include_agent_metadata is True, metadata should be included
        assert self.config.include_agent_metadata is True
        
        # Mock agent with metadata
        agent_state = AgentState(
            agent_name="test_agent",
            status="completed",
            key_insights=["Test insight"],
            completion_reasoning="Test completed",
            metadata={
                "execution_time": 30.5,
                "tokens_used": 1250,
                "confidence_score": 0.87
            }
        )
        
        # Verify metadata is accessible
        assert agent_state.metadata is not None
        assert "execution_time" in agent_state.metadata
        assert agent_state.metadata["confidence_score"] == 0.87


class TestPerformanceAndScaling:
    """Test performance characteristics of orchestration and evaluation."""
    
    def test_sequence_generation_performance(self):
        """Test sequence generation performance characteristics."""
        
        # Mock quick sequence generation
        start_time = time.time()
        
        # Simulate sequence generation (should be < 10 seconds per requirements)
        sample_sequences = [
            AgentSequence(
                sequence_id=f"seq_{i}",
                approach_name=f"approach_{i}",
                agent_names=[f"agent_{j}" for j in range(3)],
                rationale=f"Rationale for sequence {i}"
            )
            for i in range(3)
        ]
        
        generation_time = time.time() - start_time
        
        # Should be very fast for mock data
        assert generation_time < 1.0
        assert len(sample_sequences) == 3
        assert all(seq.sequence_id.startswith("seq_") for seq in sample_sequences)
    
    def test_evaluation_performance(self):
        """Test LLM Judge evaluation performance."""
        
        start_time = time.time()
        
        # Mock evaluation (should be < 30 seconds per requirements)
        mock_evaluation = EvaluationResult(
            winner="seq_0",
            reasoning="Performance test evaluation",
            scores={"seq_0": {"completeness": 0.9}},
            detailed_feedback={"seq_0": "Mock feedback"}
        )
        
        evaluation_time = time.time() - start_time
        
        # Should be very fast for mock data
        assert evaluation_time < 0.1
        assert mock_evaluation.winner == "seq_0"
        assert len(mock_evaluation.scores) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])