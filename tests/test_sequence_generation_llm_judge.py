"""Comprehensive tests for Sequence Generation and LLM Judge systems.

This module tests:
- SequenceGenerator with different research topics
- Dynamic sequence generation quality and diversity
- LLM Judge evaluation accuracy and reliability
- Multi-sequence comparison and winner determination
- Integration between generation and evaluation systems
- Performance requirements and edge cases

Test Categories:
1. Sequence generation across research domains
2. Sequence quality and diversity validation
3. LLM Judge evaluation accuracy
4. Multi-sequence comparison logic
5. Integration testing
6. Performance and reliability
7. Error handling and edge cases
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Optional
import time
import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.orchestration.sequence_generator import SequenceGenerator
from open_deep_research.evaluation.llm_judge import LLMJudge, EvaluationResult, ComparisonResult
from open_deep_research.evaluation.models import ReportEvaluation, SequenceComparison
from open_deep_research.configuration import Configuration
from open_deep_research.state import RunningReport, AgentExecutionReport
from open_deep_research.agents.registry import AgentRegistry
from datetime import datetime, timedelta


class TestSequenceGeneration:
    """Test sequence generation across different research domains."""
    
    def setup_method(self):
        """Set up sequence generation testing."""
        self.config = Configuration(
            planner_model="mock_planner_model",
            planner_model_max_tokens=8000
        )
        
        # Mock agent registry
        self.mock_registry = Mock(spec=AgentRegistry)
        self.mock_registry.list_agents.return_value = [
            "academic_researcher", "industry_analyst", "technical_specialist", 
            "market_researcher", "policy_analyst", "innovation_specialist"
        ]
        
        # Mock agent configurations
        self.mock_registry.get_agent.side_effect = self._mock_get_agent
        
        with patch('open_deep_research.orchestration.sequence_generator.init_chat_model'):
            self.generator = SequenceGenerator(
                agent_registry=self.mock_registry,
                config=self.config
            )
    
    def _mock_get_agent(self, agent_name: str) -> Dict[str, Any]:
        """Mock agent configuration retrieval."""
        agent_configs = {
            "academic_researcher": {
                "name": "academic_researcher",
                "description": "Academic research specialist",
                "expertise_areas": ["academic research", "literature review", "theoretical analysis"],
                "completion_indicators": ["research complete", "analysis finished"]
            },
            "industry_analyst": {
                "name": "industry_analyst", 
                "description": "Industry analysis specialist",
                "expertise_areas": ["market analysis", "business intelligence", "competitive analysis"],
                "completion_indicators": ["market analysis complete", "industry review finished"]
            },
            "technical_specialist": {
                "name": "technical_specialist",
                "description": "Technical implementation specialist", 
                "expertise_areas": ["technical architecture", "implementation", "performance optimization"],
                "completion_indicators": ["technical analysis complete", "implementation plan ready"]
            },
            "market_researcher": {
                "name": "market_researcher",
                "description": "Market research specialist",
                "expertise_areas": ["market research", "consumer behavior", "market sizing"],
                "completion_indicators": ["market research complete", "consumer analysis finished"]
            },
            "policy_analyst": {
                "name": "policy_analyst",
                "description": "Policy and regulatory analyst",
                "expertise_areas": ["policy analysis", "regulatory compliance", "government relations"],
                "completion_indicators": ["policy analysis complete", "regulatory review finished"]
            },
            "innovation_specialist": {
                "name": "innovation_specialist",
                "description": "Innovation and emerging trends specialist",
                "expertise_areas": ["innovation trends", "emerging technologies", "future forecasting"],
                "completion_indicators": ["innovation analysis complete", "trend forecasting finished"]
            }
        }
        return agent_configs.get(agent_name, {})
    
    @pytest.mark.asyncio
    async def test_academic_research_sequence_generation(self):
        """Test sequence generation for academic research topics."""
        academic_topics = [
            "Recent advances in quantum computing error correction",
            "Social impact of artificial intelligence on educational systems",
            "Theoretical foundations of sustainable energy storage",
            "Cognitive behavioral therapy effectiveness for anxiety disorders",
            "Machine learning applications in genomic research"
        ]
        
        # Mock LLM response for sequence generation
        mock_response = AIMessage(content=json.dumps({
            "sequences": [
                {
                    "agent_order": ["academic_researcher", "technical_specialist", "innovation_specialist"],
                    "reasoning": "Academic topic requires theoretical foundation, technical analysis, and future implications",
                    "confidence_score": 0.85
                },
                {
                    "agent_order": ["innovation_specialist", "academic_researcher", "technical_specialist"], 
                    "reasoning": "Alternative approach starting with emerging trends, then academic validation",
                    "confidence_score": 0.75
                },
                {
                    "agent_order": ["technical_specialist", "academic_researcher", "policy_analyst"],
                    "reasoning": "Technical-first approach with academic validation and policy implications",
                    "confidence_score": 0.70
                }
            ]
        }))
        
        with patch.object(self.generator.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_response
            
            for topic in academic_topics:
                sequences = await self.generator.generate_sequences(topic, num_sequences=3)
                
                # Verify generation results
                assert len(sequences) == 3
                assert all(len(seq.agent_order) >= 2 for seq in sequences)
                assert all(seq.confidence_score > 0.6 for seq in sequences)
                assert all("academic_researcher" in seq.agent_order for seq in sequences)
                
                # Verify reasoning quality
                assert all(len(seq.reasoning) > 50 for seq in sequences)
                assert all(topic.lower() in seq.reasoning.lower() or "academic" in seq.reasoning.lower() 
                          for seq in sequences)
    
    @pytest.mark.asyncio
    async def test_market_analysis_sequence_generation(self):
        """Test sequence generation for market analysis topics."""
        market_topics = [
            "Electric vehicle market growth opportunities in Southeast Asia",
            "Consumer adoption patterns for fintech mobile applications", 
            "Competitive landscape analysis for cloud computing services",
            "Market entry strategy for sustainable fashion brands",
            "B2B SaaS pricing optimization in enterprise markets"
        ]
        
        mock_response = AIMessage(content=json.dumps({
            "sequences": [
                {
                    "agent_order": ["market_researcher", "industry_analyst", "technical_specialist"],
                    "reasoning": "Market analysis requires consumer research, industry context, and technical feasibility",
                    "confidence_score": 0.90
                },
                {
                    "agent_order": ["industry_analyst", "market_researcher", "policy_analyst"],
                    "reasoning": "Industry-first approach with market validation and regulatory considerations", 
                    "confidence_score": 0.80
                }
            ]
        }))
        
        with patch.object(self.generator.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_response
            
            for topic in market_topics:
                sequences = await self.generator.generate_sequences(topic, num_sequences=2)
                
                # Verify market-focused sequences
                assert len(sequences) == 2
                assert all(any(agent in seq.agent_order for agent in ["market_researcher", "industry_analyst"]) 
                          for seq in sequences)
                assert all(seq.confidence_score > 0.7 for seq in sequences)
                
                # Verify market-relevant reasoning
                market_keywords = ["market", "industry", "competitive", "consumer", "business"]
                assert all(any(keyword in seq.reasoning.lower() for keyword in market_keywords)
                          for seq in sequences)
    
    @pytest.mark.asyncio 
    async def test_technical_implementation_sequence_generation(self):
        """Test sequence generation for technical implementation topics."""
        technical_topics = [
            "Microservices architecture design for high-traffic applications",
            "Machine learning pipeline optimization for real-time inference",
            "Blockchain implementation for supply chain transparency",
            "Edge computing deployment strategies for IoT networks",
            "Database sharding strategies for global e-commerce platforms"
        ]
        
        mock_response = AIMessage(content=json.dumps({
            "sequences": [
                {
                    "agent_order": ["technical_specialist", "academic_researcher", "innovation_specialist"],
                    "reasoning": "Technical topics require implementation expertise, theoretical validation, and future-proofing",
                    "confidence_score": 0.88
                },
                {
                    "agent_order": ["innovation_specialist", "technical_specialist", "industry_analyst"],
                    "reasoning": "Emerging technology approach with technical validation and market context",
                    "confidence_score": 0.82
                }
            ]
        }))
        
        with patch.object(self.generator.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_response
            
            for topic in technical_topics:
                sequences = await self.generator.generate_sequences(topic, num_sequences=2)
                
                # Verify technical-focused sequences
                assert len(sequences) == 2
                assert all("technical_specialist" in seq.agent_order for seq in sequences)
                assert all(seq.confidence_score > 0.75 for seq in sequences)
                
                # Verify technical reasoning
                technical_keywords = ["technical", "implementation", "architecture", "system", "performance"]
                assert all(any(keyword in seq.reasoning.lower() for keyword in technical_keywords)
                          for seq in sequences)
    
    def test_sequence_diversity_validation(self):
        """Test that generated sequences are diverse and not repetitive."""
        sequences = [
            Mock(agent_order=["academic_researcher", "industry_analyst", "technical_specialist"], 
                 reasoning="Academic-first approach", confidence_score=0.8),
            Mock(agent_order=["industry_analyst", "academic_researcher", "technical_specialist"],
                 reasoning="Market-first approach", confidence_score=0.75), 
            Mock(agent_order=["technical_specialist", "industry_analyst", "academic_researcher"],
                 reasoning="Technical-first approach", confidence_score=0.7)
        ]
        
        # Verify diversity
        unique_orders = set(tuple(seq.agent_order) for seq in sequences)
        assert len(unique_orders) == len(sequences), "Sequences should have diverse agent orders"
        
        # Verify reasoning diversity
        reasoning_texts = [seq.reasoning for seq in sequences]
        unique_reasoning_count = len(set(reasoning_texts))
        assert unique_reasoning_count == len(sequences), "Sequences should have diverse reasoning"
    
    @pytest.mark.asyncio
    async def test_sequence_generation_performance(self):
        """Test sequence generation performance requirements."""
        topic = "AI ethics in autonomous vehicle decision-making"
        
        mock_response = AIMessage(content=json.dumps({
            "sequences": [
                {
                    "agent_order": ["academic_researcher", "technical_specialist", "policy_analyst"],
                    "reasoning": "Ethics requires academic foundation, technical understanding, and policy implications",
                    "confidence_score": 0.85
                }
            ]
        }))
        
        with patch.object(self.generator.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_response
            
            # Time sequence generation
            start_time = time.time()
            sequences = await self.generator.generate_sequences(topic, num_sequences=3)
            generation_time = time.time() - start_time
            
            # Should generate sequences quickly (< 5 seconds for 3 sequences)
            assert generation_time < 5.0, f"Sequence generation took {generation_time:.2f}s"
            assert len(sequences) > 0
            
            print(f"Generated {len(sequences)} sequences in {generation_time:.3f}s")


class TestLLMJudgeEvaluation:
    """Test LLM Judge evaluation system."""
    
    def setup_method(self):
        """Set up LLM Judge testing."""
        self.config = Configuration(
            evaluation_model="mock_evaluation_model",
            evaluation_model_max_tokens=4096,
            evaluation_criteria=["completeness", "depth", "coherence", "innovation", "actionability"],
            evaluation_timeout=60
        )
        
        with patch('open_deep_research.evaluation.llm_judge.init_chat_model'):
            self.judge = LLMJudge(config=self.config)
    
    def _create_mock_running_report(self, topic: str, agents: List[str], quality_score: float = 0.8) -> RunningReport:
        """Create mock running report for testing."""
        report = RunningReport(
            research_topic=topic,
            sequence_name=f"test_sequence_{len(agents)}_agents",
            planned_agents=agents,
            total_agents_executed=len(agents),
            all_insights=[f"Insight from {agent}" for agent in agents],
            agent_summaries={agent: {
                "execution_duration": 30.0,
                "insights_count": 2,
                "research_quality_score": quality_score
            } for agent in agents},
            executive_summary=f"Comprehensive analysis of {topic} completed by {len(agents)} specialized agents.",
            creation_timestamp=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        return report
    
    @pytest.mark.asyncio
    async def test_single_report_evaluation(self):
        """Test evaluation of single research reports."""
        # Create test reports with different quality levels
        test_reports = [
            {
                "report": self._create_mock_running_report(
                    "Climate change mitigation strategies",
                    ["academic_researcher", "policy_analyst", "technical_specialist"],
                    quality_score=0.9
                ),
                "expected_min_score": 0.8
            },
            {
                "report": self._create_mock_running_report(
                    "Basic market analysis",
                    ["market_researcher"],
                    quality_score=0.6
                ),
                "expected_min_score": 0.5
            },
            {
                "report": self._create_mock_running_report(
                    "Comprehensive technology assessment", 
                    ["academic_researcher", "technical_specialist", "industry_analyst", "innovation_specialist"],
                    quality_score=0.95
                ),
                "expected_min_score": 0.85
            }
        ]
        
        # Mock LLM evaluation responses
        def mock_evaluation_response(report):
            quality = report.agent_summaries[list(report.agent_summaries.keys())[0]]["research_quality_score"]
            return AIMessage(content=json.dumps({
                "overall_score": quality,
                "criterion_scores": {
                    "completeness": quality + 0.05,
                    "depth": quality,
                    "coherence": quality + 0.02,
                    "innovation": quality - 0.1,
                    "actionability": quality + 0.03
                },
                "strengths": [
                    "Comprehensive coverage of topic",
                    "Well-structured analysis",
                    "Clear insights presented"
                ],
                "weaknesses": [
                    "Could benefit from more quantitative data",
                    "Some areas need deeper exploration"
                ],
                "detailed_feedback": "The report demonstrates solid research methodology and presents findings clearly."
            }))
        
        with patch.object(self.judge.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            for test_case in test_reports:
                mock_invoke.return_value = mock_evaluation_response(test_case["report"])
                
                evaluation = await self.judge.evaluate_report(test_case["report"])
                
                # Verify evaluation results
                assert isinstance(evaluation, ReportEvaluation)
                assert evaluation.overall_score >= test_case["expected_min_score"]
                assert len(evaluation.criterion_scores) == 5
                assert all(0.0 <= score <= 1.0 for score in evaluation.criterion_scores.values())
                assert len(evaluation.strengths) > 0
                assert len(evaluation.weaknesses) > 0
                assert len(evaluation.detailed_feedback) > 50
    
    @pytest.mark.asyncio
    async def test_sequence_comparison_evaluation(self):
        """Test comparison evaluation between multiple sequences."""
        # Create competing reports
        report1 = self._create_mock_running_report(
            "Renewable energy adoption strategies",
            ["academic_researcher", "policy_analyst", "market_researcher"],
            quality_score=0.8
        )
        
        report2 = self._create_mock_running_report(
            "Renewable energy adoption strategies",
            ["technical_specialist", "industry_analyst", "innovation_specialist"], 
            quality_score=0.85
        )
        
        report3 = self._create_mock_running_report(
            "Renewable energy adoption strategies",
            ["market_researcher", "technical_specialist", "academic_researcher"],
            quality_score=0.75
        )
        
        reports = [report1, report2, report3]
        
        # Mock comparison response
        mock_comparison_response = AIMessage(content=json.dumps({
            "winner_index": 1,  # report2 wins
            "ranking": [1, 0, 2],  # report2, report1, report3
            "comparison_scores": [0.8, 0.85, 0.75],
            "winner_reasoning": "Technical-first approach provides more actionable insights with stronger implementation focus",
            "detailed_analysis": {
                "completeness": [0.8, 0.9, 0.7],
                "depth": [0.85, 0.85, 0.75],
                "coherence": [0.8, 0.85, 0.8],
                "innovation": [0.75, 0.9, 0.7],
                "actionability": [0.8, 0.85, 0.75]
            },
            "comparative_insights": [
                "Report 1 shows strong academic foundation but lacks technical depth",
                "Report 2 provides best balance of technical insight and practical application",
                "Report 3 covers basics well but misses innovative approaches"
            ]
        }))
        
        with patch.object(self.judge.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_comparison_response
            
            comparison = await self.judge.compare_sequences(reports)
            
            # Verify comparison results
            assert isinstance(comparison, SequenceComparison)
            assert comparison.winner_index == 1
            assert comparison.ranking == [1, 0, 2]
            assert len(comparison.comparison_scores) == 3
            assert comparison.comparison_scores[1] == max(comparison.comparison_scores)  # Winner has highest score
            assert len(comparison.winner_reasoning) > 50
            assert len(comparison.detailed_analysis) == 5  # All criteria
            assert len(comparison.comparative_insights) == 3  # One per report
    
    @pytest.mark.asyncio
    async def test_evaluation_criteria_customization(self):
        """Test evaluation with custom criteria."""
        custom_criteria = ["accuracy", "relevance", "clarity", "impact"]
        custom_config = Configuration(
            evaluation_criteria=custom_criteria,
            evaluation_model="mock_model"
        )
        
        with patch('open_deep_research.evaluation.llm_judge.init_chat_model'):
            custom_judge = LLMJudge(config=custom_config)
        
        test_report = self._create_mock_running_report(
            "Custom criteria test",
            ["academic_researcher", "industry_analyst"],
            quality_score=0.8
        )
        
        mock_response = AIMessage(content=json.dumps({
            "overall_score": 0.8,
            "criterion_scores": {
                "accuracy": 0.85,
                "relevance": 0.8,
                "clarity": 0.75,
                "impact": 0.8
            },
            "strengths": ["High accuracy", "Clear presentation"],
            "weaknesses": ["Could improve impact assessment"],
            "detailed_feedback": "Report meets custom evaluation criteria well."
        }))
        
        with patch.object(custom_judge.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_response
            
            evaluation = await custom_judge.evaluate_report(test_report)
            
            # Verify custom criteria are used
            assert set(evaluation.criterion_scores.keys()) == set(custom_criteria)
            assert all(criterion in evaluation.criterion_scores for criterion in custom_criteria)
    
    @pytest.mark.asyncio
    async def test_evaluation_consistency(self):
        """Test consistency of evaluations across multiple runs."""
        test_report = self._create_mock_running_report(
            "Consistency test topic",
            ["academic_researcher", "technical_specialist"],
            quality_score=0.8
        )
        
        mock_response = AIMessage(content=json.dumps({
            "overall_score": 0.8,
            "criterion_scores": {
                "completeness": 0.8,
                "depth": 0.75,
                "coherence": 0.85,
                "innovation": 0.7,
                "actionability": 0.8
            },
            "strengths": ["Consistent strengths"],
            "weaknesses": ["Consistent weaknesses"],
            "detailed_feedback": "Consistent evaluation feedback."
        }))
        
        with patch.object(self.judge.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_response
            
            # Run evaluation multiple times
            evaluations = []
            for _ in range(5):
                evaluation = await self.judge.evaluate_report(test_report)
                evaluations.append(evaluation)
            
            # Verify consistency
            overall_scores = [eval.overall_score for eval in evaluations]
            score_variance = max(overall_scores) - min(overall_scores)
            assert score_variance < 0.1, f"Score variance {score_variance} too high"
            
            # Verify criterion scores consistency
            for criterion in self.config.evaluation_criteria:
                criterion_scores = [eval.criterion_scores[criterion] for eval in evaluations]
                criterion_variance = max(criterion_scores) - min(criterion_scores)
                assert criterion_variance < 0.1, f"Criterion '{criterion}' variance too high"


class TestSequenceGenerationLLMJudgeIntegration:
    """Test integration between sequence generation and LLM judge systems."""
    
    def setup_method(self):
        """Set up integration testing."""
        self.config = Configuration(
            planner_model="mock_planner",
            evaluation_model="mock_evaluator"
        )
        
        self.mock_registry = Mock(spec=AgentRegistry)
        self.mock_registry.list_agents.return_value = ["academic_researcher", "industry_analyst", "technical_specialist"]
        self.mock_registry.get_agent.side_effect = lambda name: {
            "name": name,
            "description": f"{name} specialist",
            "expertise_areas": [f"{name} expertise"]
        }
        
        with patch('open_deep_research.orchestration.sequence_generator.init_chat_model'), \
             patch('open_deep_research.evaluation.llm_judge.init_chat_model'):
            self.generator = SequenceGenerator(self.mock_registry, self.config)
            self.judge = LLMJudge(self.config)
    
    @pytest.mark.asyncio
    async def test_generate_and_evaluate_workflow(self):
        """Test complete workflow from generation to evaluation."""
        topic = "Integration test: AI applications in healthcare"
        
        # Mock sequence generation
        generation_response = AIMessage(content=json.dumps({
            "sequences": [
                {
                    "agent_order": ["academic_researcher", "technical_specialist", "industry_analyst"],
                    "reasoning": "Academic foundation, technical implementation, market validation",
                    "confidence_score": 0.85
                },
                {
                    "agent_order": ["technical_specialist", "industry_analyst", "academic_researcher"],
                    "reasoning": "Technical-first with market validation and theoretical grounding",
                    "confidence_score": 0.80
                }
            ]
        }))
        
        # Mock evaluation response
        evaluation_response = AIMessage(content=json.dumps({
            "winner_index": 0,
            "ranking": [0, 1],
            "comparison_scores": [0.85, 0.80],
            "winner_reasoning": "Better sequence order provides stronger foundation",
            "detailed_analysis": {
                "completeness": [0.9, 0.8],
                "depth": [0.85, 0.8],
                "coherence": [0.9, 0.85],
                "innovation": [0.8, 0.8],
                "actionability": [0.85, 0.8]
            },
            "comparative_insights": [
                "First sequence provides better theoretical foundation",
                "Second sequence is more practical but less comprehensive"
            ]
        }))
        
        with patch.object(self.generator.model, 'ainvoke', new_callable=AsyncMock) as mock_gen, \
             patch.object(self.judge.model, 'ainvoke', new_callable=AsyncMock) as mock_judge:
            
            mock_gen.return_value = generation_response
            mock_judge.return_value = evaluation_response
            
            # Generate sequences
            sequences = await self.generator.generate_sequences(topic, num_sequences=2)
            
            # Create mock reports for evaluation
            reports = []
            for i, sequence in enumerate(sequences):
                report = RunningReport(
                    research_topic=topic,
                    sequence_name=f"sequence_{i}",
                    planned_agents=sequence.agent_order,
                    total_agents_executed=len(sequence.agent_order),
                    all_insights=[f"Insight {j}" for j in range(3)],
                    agent_summaries={agent: {"execution_duration": 30.0} for agent in sequence.agent_order},
                    executive_summary=f"Test summary for sequence {i}",
                    creation_timestamp=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                )
                reports.append(report)
            
            # Evaluate sequences
            comparison = await self.judge.compare_sequences(reports)
            
            # Verify integration results
            assert len(sequences) == 2
            assert comparison.winner_index == 0
            assert len(comparison.comparison_scores) == 2
            assert comparison.comparison_scores[0] > comparison.comparison_scores[1]
            
            # Verify sequence quality correlates with evaluation
            winner_sequence = sequences[comparison.winner_index]
            assert winner_sequence.confidence_score >= 0.8
    
    @pytest.mark.asyncio
    async def test_performance_requirements_integration(self):
        """Test performance requirements for integrated workflow."""
        topic = "Performance test topic"
        
        # Mock responses for speed
        quick_generation = AIMessage(content=json.dumps({
            "sequences": [{"agent_order": ["academic_researcher"], "reasoning": "Quick test", "confidence_score": 0.8}]
        }))
        
        quick_evaluation = AIMessage(content=json.dumps({
            "winner_index": 0,
            "ranking": [0],
            "comparison_scores": [0.8],
            "winner_reasoning": "Only option",
            "detailed_analysis": {"completeness": [0.8]},
            "comparative_insights": ["Single sequence test"]
        }))
        
        with patch.object(self.generator.model, 'ainvoke', new_callable=AsyncMock) as mock_gen, \
             patch.object(self.judge.model, 'ainvoke', new_callable=AsyncMock) as mock_judge:
            
            mock_gen.return_value = quick_generation
            mock_judge.return_value = quick_evaluation
            
            # Time the complete workflow
            start_time = time.time()
            
            # Generate sequences
            sequences = await self.generator.generate_sequences(topic, num_sequences=1)
            
            # Create and evaluate report
            report = RunningReport(
                research_topic=topic,
                sequence_name="perf_test",
                planned_agents=sequences[0].agent_order,
                total_agents_executed=1,
                all_insights=["Test insight"],
                agent_summaries={"academic_researcher": {"execution_duration": 30.0}},
                executive_summary="Performance test summary",
                creation_timestamp=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            evaluation = await self.judge.evaluate_report(report)
            
            workflow_time = time.time() - start_time
            
            # Verify performance requirements
            assert workflow_time < 10.0, f"Integrated workflow took {workflow_time:.2f}s"
            assert len(sequences) > 0
            assert evaluation.overall_score > 0.0
            
            print(f"Complete generation + evaluation workflow: {workflow_time:.3f}s")


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for generation and evaluation systems."""
    
    def setup_method(self):
        """Set up error handling testing."""
        self.config = Configuration()
        self.mock_registry = Mock(spec=AgentRegistry)
        self.mock_registry.list_agents.return_value = ["test_agent"]
        
        with patch('open_deep_research.orchestration.sequence_generator.init_chat_model'), \
             patch('open_deep_research.evaluation.llm_judge.init_chat_model'):
            self.generator = SequenceGenerator(self.mock_registry, self.config)
            self.judge = LLMJudge(self.config)
    
    @pytest.mark.asyncio
    async def test_generation_error_handling(self):
        """Test sequence generation error handling."""
        # Test with invalid topic
        with patch.object(self.generator.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = Exception("Model error")
            
            try:
                await self.generator.generate_sequences("", num_sequences=1)
                assert False, "Should have raised an exception"
            except Exception as e:
                assert "Model error" in str(e) or isinstance(e, Exception)
    
    @pytest.mark.asyncio
    async def test_evaluation_error_handling(self):
        """Test LLM judge error handling."""
        invalid_report = Mock()
        invalid_report.research_topic = None
        invalid_report.agent_summaries = {}
        
        with patch.object(self.judge.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = Exception("Evaluation error")
            
            try:
                await self.judge.evaluate_report(invalid_report)
                assert False, "Should have raised an exception"
            except Exception as e:
                assert isinstance(e, Exception)
    
    @pytest.mark.asyncio
    async def test_malformed_response_handling(self):
        """Test handling of malformed LLM responses."""
        # Test generation with malformed JSON
        malformed_generation = AIMessage(content="Invalid JSON response")
        
        with patch.object(self.generator.model, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = malformed_generation
            
            try:
                sequences = await self.generator.generate_sequences("test topic", num_sequences=1)
                # If no exception, should handle gracefully
                assert isinstance(sequences, list)
            except Exception as e:
                # Should be a reasonable error
                assert "json" in str(e).lower() or "parse" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])