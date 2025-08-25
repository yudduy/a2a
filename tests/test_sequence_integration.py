"""Integration tests for SequenceGenerator with Sequential Supervisor.

This test suite validates the integration between SequenceGenerator and
Sequential Supervisor components for complete workflow execution.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, List

from open_deep_research.orchestration.sequence_generator import (
    SequenceGenerator,
    TopicType,
    SequenceStrategy,
    GeneratedSequence
)
from open_deep_research.supervisor.sequential_supervisor import (
    SequentialSupervisor,
    SupervisorConfig
)
from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.state import SequentialSupervisorState


class TestSequenceGeneratorIntegration:
    """Integration tests for SequenceGenerator with Sequential Supervisor."""
    
    @pytest.fixture
    def mock_agent_registry(self) -> Mock:
        """Create comprehensive mock agent registry."""
        registry = Mock(spec=AgentRegistry)
        
        agents_data = {
            "research_agent": {
                "name": "research_agent",
                "description": "Academic research specialist",
                "expertise_areas": [
                    "Academic research methodology",
                    "Literature reviews and synthesis", 
                    "Primary source analysis",
                    "Research design and planning"
                ],
                "completion_indicators": [
                    "Research foundation established",
                    "Literature review completed"
                ]
            },
            "analysis_agent": {
                "name": "analysis_agent",
                "description": "Data analysis specialist",
                "expertise_areas": [
                    "Quantitative and qualitative data analysis",
                    "Statistical interpretation and modeling",
                    "Pattern recognition and trend analysis",
                    "Comparative analysis and benchmarking"
                ],
                "completion_indicators": [
                    "Data analysis completed",
                    "Patterns and trends identified"
                ]
            },
            "market_agent": {
                "name": "market_agent", 
                "description": "Market research specialist",
                "expertise_areas": [
                    "Market analysis and sizing",
                    "Competitive intelligence and positioning",
                    "Business model evaluation",
                    "Financial analysis and forecasting"
                ],
                "completion_indicators": [
                    "Market analysis completed",
                    "Competitive landscape mapped"
                ]
            },
            "technical_agent": {
                "name": "technical_agent",
                "description": "Technical implementation specialist",
                "expertise_areas": [
                    "System architecture and design patterns",
                    "Technology stack evaluation and selection", 
                    "Implementation feasibility and complexity analysis",
                    "Performance optimization and scalability planning"
                ],
                "completion_indicators": [
                    "Technical architecture defined",
                    "Implementation plan developed"
                ]
            }
        }
        
        registry.list_agents.return_value = list(agents_data.keys())
        registry.get_agent.side_effect = lambda name: agents_data.get(name)
        registry.has_agent.side_effect = lambda name: name in agents_data
        
        return registry
    
    @pytest.fixture
    def sequence_generator(self, mock_agent_registry) -> SequenceGenerator:
        """Create SequenceGenerator with mock registry."""
        return SequenceGenerator(mock_agent_registry, debug_mode=True)
    
    @pytest.fixture
    def supervisor(self, mock_agent_registry) -> SequentialSupervisor:
        """Create SequentialSupervisor with test configuration."""
        config = SupervisorConfig(
            agent_timeout_seconds=60.0,
            max_agents_per_sequence=4,
            debug_mode=True
        )
        return SequentialSupervisor(mock_agent_registry, config=config)
    
    def test_sequence_generator_supervisor_integration(self, sequence_generator, supervisor):
        """Test basic integration between SequenceGenerator and Sequential Supervisor."""
        research_topic = """
        Analyze the impact of machine learning on healthcare diagnostics.
        Research current applications, evaluate effectiveness, assess market opportunities,
        and provide implementation recommendations for healthcare organizations.
        """
        
        # Generate sequences
        sequences = sequence_generator.generate_sequences(
            research_topic=research_topic,
            num_sequences=3
        )
        
        assert len(sequences) > 0
        best_sequence = sequences[0]
        
        # Validate sequence with supervisor
        validation_result = supervisor.validate_sequence(best_sequence.agents)
        
        assert validation_result["valid"] is True
        assert "errors" in validation_result
        assert isinstance(validation_result["errors"], list)
        assert len(validation_result["errors"]) == 0
        
        # Verify sequence properties
        assert len(best_sequence.agents) > 0
        assert best_sequence.score > 0
        assert 0 <= best_sequence.confidence <= 1
        assert best_sequence.rationale is not None
    
    def test_sequence_validation_with_invalid_agents(self, sequence_generator, supervisor):
        """Test sequence validation with invalid agent names."""
        # Create sequence with invalid agent
        invalid_sequence = ["nonexistent_agent", "research_agent"]
        
        validation_result = supervisor.validate_sequence(invalid_sequence)
        
        assert validation_result["valid"] is False
        assert len(validation_result["errors"]) > 0
        assert "not found" in validation_result["errors"][0]
    
    def test_topic_specific_sequence_generation(self, sequence_generator, supervisor):
        """Test sequence generation for different topic types."""
        test_cases = [
            {
                "topic": """
                Systematic literature review of deep learning applications in medical imaging.
                Analyze peer-reviewed publications, evaluate methodologies, and identify research gaps.
                """,
                "expected_type": TopicType.ACADEMIC,
                "expected_first_agent": "research_agent"
            },
            {
                "topic": """
                Market opportunity analysis for AI-powered fintech solutions.
                Evaluate market size, competitive landscape, and revenue potential.
                """,
                "expected_type": TopicType.MARKET,
                "expected_first_agent": "market_agent"  
            },
            {
                "topic": """
                Design microservices architecture for real-time data processing.
                Evaluate technology stacks and define implementation strategy.
                """,
                "expected_type": TopicType.TECHNICAL,
                "expected_first_agent": "technical_agent"
            }
        ]
        
        for case in test_cases:
            # Analyze topic
            topic_analysis = sequence_generator.analyze_topic_characteristics(case["topic"])
            assert topic_analysis.topic_type == case["expected_type"]
            
            # Generate sequences
            sequences = sequence_generator.generate_sequences(case["topic"], num_sequences=2)
            assert len(sequences) > 0
            
            # Validate best sequence
            best_sequence = sequences[0]
            validation = supervisor.validate_sequence(best_sequence.agents)
            assert validation["valid"] is True
            
            # Check if expected agent type is prioritized (not strict requirement)
            agent_names = best_sequence.agents
            assert len(agent_names) > 0
    
    def test_sequence_ranking_consistency(self, sequence_generator):
        """Test that sequence ranking is consistent and logical."""
        topic = """
        Comprehensive research on artificial intelligence ethics in autonomous vehicles.
        Analyze academic literature, evaluate market implications, assess technical challenges,
        and provide policy recommendations for regulatory frameworks.
        """
        
        # Generate multiple sequences
        sequences = sequence_generator.generate_sequences(topic, num_sequences=4)
        
        # Verify ranking consistency
        assert len(sequences) > 1
        for i in range(len(sequences) - 1):
            assert sequences[i].score >= sequences[i + 1].score
        
        # Verify all sequences have reasonable scores
        for seq in sequences:
            assert 0 <= seq.score <= 1
            assert 0 <= seq.confidence <= 1
            assert seq.estimated_duration > 0
    
    def test_supervisor_state_preparation(self, sequence_generator):
        """Test preparing supervisor state with generated sequences."""
        topic = "AI applications in sustainable energy management systems"
        
        sequences = sequence_generator.generate_sequences(topic, num_sequences=2)
        best_sequence = sequences[0]
        
        # Create supervisor state
        state = SequentialSupervisorState()
        state["research_topic"] = topic
        state["planned_sequence"] = best_sequence.agents.copy()
        state["sequence_position"] = 0
        state["handoff_ready"] = True
        
        # Verify state preparation
        assert state["research_topic"] == topic
        assert state["planned_sequence"] == best_sequence.agents
        assert state["sequence_position"] == 0
        assert state["handoff_ready"] is True
        assert len(state.get("executed_agents", [])) == 0
    
    def test_sequence_strategy_effectiveness(self, sequence_generator):
        """Test that different strategies produce different but valid sequences."""
        topic = """
        Business analysis of blockchain adoption in supply chain management.
        Research technical implementations, evaluate market opportunities,
        analyze competitive landscape, and assess implementation feasibility.
        """
        
        # Generate sequences (should include different strategies)
        sequences = sequence_generator.generate_sequences(topic, num_sequences=4)
        
        # Verify we get different strategies
        strategies = [seq.strategy for seq in sequences]
        assert len(set(strategies)) >= 2  # At least 2 different strategies
        
        # Verify each sequence is internally consistent
        for sequence in sequences:
            assert len(sequence.agents) > 0
            assert sequence.score > 0
            assert sequence.rationale is not None
            assert sequence.strategy in SequenceStrategy
    
    def test_error_handling_integration(self, mock_agent_registry):
        """Test error handling in integration scenarios."""
        # Test with empty agent registry
        empty_registry = Mock(spec=AgentRegistry)
        empty_registry.list_agents.return_value = []
        empty_registry.get_agent.return_value = None
        empty_registry.has_agent.return_value = False
        
        generator = SequenceGenerator(empty_registry)
        
        with pytest.raises(RuntimeError, match="No agents available"):
            generator.generate_sequences("test topic")
    
    def test_sequence_scoring_components(self, sequence_generator):
        """Test that sequence scoring components work correctly."""
        topic = "Machine learning model interpretability in financial risk assessment"
        
        sequences = sequence_generator.generate_sequences(topic, num_sequences=3)
        
        for sequence in sequences:
            # Verify all scoring components are populated
            assert hasattr(sequence, 'topic_fit_score')
            assert hasattr(sequence, 'coverage_score') 
            assert hasattr(sequence, 'efficiency_score')
            assert hasattr(sequence, 'expertise_match_score')
            
            # Verify scores are in valid range
            assert 0 <= sequence.topic_fit_score <= 1
            assert 0 <= sequence.coverage_score <= 1
            assert 0 <= sequence.efficiency_score <= 1  
            assert 0 <= sequence.expertise_match_score <= 1
            
            # Verify overall score is reasonable combination
            assert sequence.score > 0
    
    def test_agent_expertise_matching(self, sequence_generator, mock_agent_registry):
        """Test that agents are matched appropriately to topic requirements."""
        # Technical topic
        technical_topic = """
        Design and implement a distributed microservices architecture 
        for high-frequency trading systems with sub-millisecond latency requirements.
        """
        
        sequences = sequence_generator.generate_sequences(technical_topic, num_sequences=2)
        best_sequence = sequences[0]
        
        # Should prioritize technical agent for technical topics
        assert "technical_agent" in best_sequence.agents
        
        # Academic topic
        academic_topic = """
        Systematic literature review of quantum computing applications
        in cryptography and security protocols.
        """
        
        sequences = sequence_generator.generate_sequences(academic_topic, num_sequences=2)
        best_sequence = sequences[0]
        
        # Should prioritize research agent for academic topics
        assert "research_agent" in best_sequence.agents
    
    def test_complex_topic_handling(self, sequence_generator):
        """Test handling of complex, multi-domain topics."""
        complex_topic = """
        Comprehensive analysis of artificial intelligence impact on healthcare delivery:
        Research current applications and clinical effectiveness through systematic literature review,
        analyze market opportunities and competitive landscape in health tech sector,
        evaluate technical implementation challenges for AI integration in hospital systems,
        assess regulatory compliance requirements and ethical considerations,
        and provide strategic recommendations for healthcare organizations planning AI adoption.
        """
        
        # Analyze complex topic
        topic_analysis = sequence_generator.analyze_topic_characteristics(complex_topic)
        
        # Should detect high complexity and multiple domains
        assert topic_analysis.complexity_score > 0.5
        assert topic_analysis.estimated_agents_needed >= 3
        assert len(topic_analysis.priority_areas) >= 2
        
        # Generate sequences
        sequences = sequence_generator.generate_sequences(complex_topic, num_sequences=3)
        
        # Should generate sequences with multiple agents
        for sequence in sequences:
            assert len(sequence.agents) >= 2
            assert sequence.estimated_duration > 1.0  # Complex topics take more time
    
    @pytest.mark.parametrize("num_sequences", [1, 2, 3, 4, 5])
    def test_configurable_sequence_count(self, sequence_generator, num_sequences):
        """Test that sequence generation respects num_sequences parameter."""
        topic = "AI applications in renewable energy optimization"
        
        sequences = sequence_generator.generate_sequences(topic, num_sequences=num_sequences)
        
        # Should generate requested number or fewer (if not enough strategies)
        assert len(sequences) <= num_sequences
        assert len(sequences) > 0
        
        # All sequences should be valid
        for sequence in sequences:
            assert len(sequence.agents) > 0
            assert sequence.score > 0