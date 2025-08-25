"""Comprehensive tests for the SequenceGenerator component.

This test suite validates the sequence generation logic, topic analysis,
and ranking algorithms of the SequenceGenerator.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

from open_deep_research.orchestration.sequence_generator import (
    SequenceGenerator,
    TopicType,
    SequenceStrategy,
    TopicAnalysis,
    GeneratedSequence
)
from open_deep_research.agents.registry import AgentRegistry


class TestSequenceGenerator:
    """Test suite for SequenceGenerator functionality."""
    
    @pytest.fixture
    def mock_agent_registry(self) -> Mock:
        """Create mock agent registry with sample agents."""
        registry = Mock(spec=AgentRegistry)
        
        # Sample agent configurations
        agents_data = {
            "research_agent": {
                "name": "research_agent",
                "description": "Research specialist",
                "expertise_areas": [
                    "Academic research methodology",
                    "Literature reviews and synthesis",
                    "Primary source analysis"
                ]
            },
            "analysis_agent": {
                "name": "analysis_agent", 
                "description": "Analysis specialist",
                "expertise_areas": [
                    "Quantitative and qualitative data analysis",
                    "Statistical interpretation and modeling",
                    "Pattern recognition and trend analysis"
                ]
            },
            "market_agent": {
                "name": "market_agent",
                "description": "Market research specialist", 
                "expertise_areas": [
                    "Market analysis and sizing",
                    "Competitive intelligence and positioning",
                    "Business model evaluation"
                ]
            },
            "technical_agent": {
                "name": "technical_agent",
                "description": "Technical specialist",
                "expertise_areas": [
                    "System architecture and design patterns",
                    "Technology stack evaluation and selection",
                    "Implementation feasibility and complexity analysis"
                ]
            },
            "synthesis_agent": {
                "name": "synthesis_agent",
                "description": "Synthesis specialist",
                "expertise_areas": [
                    "Information synthesis and integration",
                    "Report writing and documentation",
                    "Strategic recommendations"
                ]
            }
        }
        
        registry.list_agents.return_value = list(agents_data.keys())
        registry.get_agent.side_effect = lambda name: agents_data.get(name)
        
        return registry
    
    @pytest.fixture
    def sequence_generator(self, mock_agent_registry) -> SequenceGenerator:
        """Create SequenceGenerator instance with mock registry."""
        return SequenceGenerator(mock_agent_registry, debug_mode=True)
    
    def test_initialization(self, sequence_generator):
        """Test SequenceGenerator initialization."""
        assert sequence_generator is not None
        assert sequence_generator.debug_mode is True
        assert sequence_generator.agent_registry is not None
        assert len(sequence_generator._domain_patterns) > 0
        assert len(sequence_generator._agent_type_mappings) > 0
    
    def test_analyze_topic_characteristics_academic(self, sequence_generator):
        """Test topic analysis for academic research topic."""
        topic = """
        Conduct a comprehensive literature review on machine learning interpretability methods.
        This research should analyze peer-reviewed publications from the last 5 years,
        focusing on explainable AI techniques and their validation through empirical studies.
        The methodology should include systematic database searches and citation analysis.
        """
        
        analysis = sequence_generator.analyze_topic_characteristics(topic)
        
        assert isinstance(analysis, TopicAnalysis)
        assert analysis.topic_type == TopicType.ACADEMIC
        assert analysis.complexity_score > 0.5
        assert len(analysis.keywords) > 0
        assert "academic" in analysis.domain_indicators
        assert analysis.domain_indicators["academic"] > analysis.domain_indicators.get("market", 0)
        assert analysis.estimated_agents_needed >= 2
    
    def test_analyze_topic_characteristics_market(self, sequence_generator):
        """Test topic analysis for market-focused research topic."""
        topic = """
        Market opportunity analysis for AI-powered customer service solutions.
        Evaluate the competitive landscape, pricing strategies, and revenue potential.
        Assess market size, growth projections, and key customer segments.
        Analyze competitor positioning and identify business model opportunities.
        """
        
        analysis = sequence_generator.analyze_topic_characteristics(topic)
        
        assert analysis.topic_type == TopicType.MARKET
        assert analysis.market_relevance > 0.5
        assert "market" in analysis.domain_indicators
        assert analysis.domain_indicators["market"] > 0.3
    
    def test_analyze_topic_characteristics_technical(self, sequence_generator):
        """Test topic analysis for technical research topic."""
        topic = """
        Technical architecture evaluation for microservices-based e-commerce platform.
        Analyze system design patterns, database architecture, and API integration strategies.
        Assess performance scalability, security implementation, and deployment approaches.
        Evaluate technology stack options and infrastructure requirements.
        """
        
        analysis = sequence_generator.analyze_topic_characteristics(topic)
        
        assert analysis.topic_type == TopicType.TECHNICAL
        assert analysis.technical_complexity > 0.5
        assert "technical" in analysis.domain_indicators
        assert analysis.domain_indicators["technical"] > 0.3
    
    def test_analyze_topic_characteristics_mixed(self, sequence_generator):
        """Test topic analysis for mixed/complex research topic."""
        topic = """
        Comprehensive analysis of blockchain technology adoption in financial services.
        Research academic literature on distributed ledger technologies, analyze market
        trends and business applications, evaluate technical implementation challenges,
        and assess regulatory implications across different geographic markets.
        """
        
        analysis = sequence_generator.analyze_topic_characteristics(topic)
        
        # Should detect mixed topic or have high scores across multiple domains
        assert len([score for score in analysis.domain_indicators.values() if score > 0.2]) >= 2
        assert analysis.estimated_agents_needed >= 3
    
    def test_generate_sequences_basic(self, sequence_generator, mock_agent_registry):
        """Test basic sequence generation functionality."""
        topic = "Research machine learning applications in healthcare diagnosis"
        available_agents = ["research_agent", "analysis_agent", "technical_agent"]
        
        sequences = sequence_generator.generate_sequences(
            research_topic=topic,
            available_agents=available_agents,
            num_sequences=2
        )
        
        assert len(sequences) <= 2
        assert all(isinstance(seq, GeneratedSequence) for seq in sequences)
        assert all(len(seq.agents) > 0 for seq in sequences)
        assert all(seq.score >= 0 for seq in sequences)
        
        # Verify sequences are ranked (highest score first)
        if len(sequences) > 1:
            assert sequences[0].score >= sequences[1].score
    
    def test_generate_sequences_validation(self, sequence_generator):
        """Test input validation for sequence generation."""
        # Empty topic
        with pytest.raises(ValueError, match="Research topic cannot be empty"):
            sequence_generator.generate_sequences("", num_sequences=1)
        
        # Invalid num_sequences
        with pytest.raises(ValueError, match="num_sequences must be between 1 and 5"):
            sequence_generator.generate_sequences("test topic", num_sequences=0)
        
        with pytest.raises(ValueError, match="num_sequences must be between 1 and 5"):
            sequence_generator.generate_sequences("test topic", num_sequences=6)
        
        # Invalid agents
        with pytest.raises(ValueError, match="Agents not found in registry"):
            sequence_generator.generate_sequences(
                "test topic", 
                available_agents=["nonexistent_agent"],
                num_sequences=1
            )
    
    def test_create_theory_first_sequence(self, sequence_generator, mock_agent_registry):
        """Test theory-first sequence creation."""
        available_agents = ["research_agent", "analysis_agent", "market_agent", "technical_agent"]
        topic_analysis = TopicAnalysis(
            topic_type=TopicType.ACADEMIC,
            complexity_score=0.8,
            scope_breadth=0.6,
            keywords=["research", "analysis"],
            domain_indicators={"academic": 0.9, "analysis": 0.7, "market": 0.3},
            estimated_agents_needed=3,
            priority_areas=["academic", "analysis"],
            market_relevance=0.4,
            technical_complexity=0.5
        )
        
        sequence = sequence_generator.create_theory_first_sequence(available_agents, topic_analysis)
        
        assert len(sequence) > 0
        # Should start with research agent if available
        if "research_agent" in available_agents:
            assert "research_agent" in sequence
        # Should include analysis agent
        if "analysis_agent" in available_agents:
            assert "analysis_agent" in sequence
    
    def test_create_market_first_sequence(self, sequence_generator, mock_agent_registry):
        """Test market-first sequence creation."""
        available_agents = ["market_agent", "technical_agent", "research_agent", "analysis_agent"]
        topic_analysis = TopicAnalysis(
            topic_type=TopicType.MARKET,
            complexity_score=0.6,
            scope_breadth=0.7,
            keywords=["market", "business"],
            domain_indicators={"market": 0.9, "technical": 0.4},
            estimated_agents_needed=3,
            priority_areas=["market", "technical"],
            market_relevance=0.9,
            technical_complexity=0.4
        )
        
        sequence = sequence_generator.create_market_first_sequence(available_agents, topic_analysis)
        
        assert len(sequence) > 0
        # Should start with market agent if available
        if "market_agent" in available_agents:
            assert sequence[0] == "market_agent"
    
    def test_create_technical_first_sequence(self, sequence_generator, mock_agent_registry):
        """Test technical-first sequence creation."""
        available_agents = ["technical_agent", "research_agent", "analysis_agent", "market_agent"]
        topic_analysis = TopicAnalysis(
            topic_type=TopicType.TECHNICAL,
            complexity_score=0.7,
            scope_breadth=0.5,
            keywords=["technical", "architecture"],
            domain_indicators={"technical": 0.9, "research": 0.3},
            estimated_agents_needed=3,
            priority_areas=["technical"],
            technical_complexity=0.9
        )
        
        sequence = sequence_generator.create_technical_first_sequence(available_agents, topic_analysis)
        
        assert len(sequence) > 0
        # Should start with technical agent if available
        if "technical_agent" in available_agents:
            assert sequence[0] == "technical_agent"
    
    def test_rank_sequences_by_topic(self, sequence_generator, mock_agent_registry):
        """Test sequence ranking by topic fitness."""
        topic_analysis = TopicAnalysis(
            topic_type=TopicType.ACADEMIC,
            complexity_score=0.7,
            scope_breadth=0.6,
            keywords=["research", "academic", "study"],
            domain_indicators={"academic": 0.8, "analysis": 0.5},
            estimated_agents_needed=3,
            priority_areas=["academic", "analysis"]
        )
        
        # Create test sequences
        sequences = [
            GeneratedSequence(
                sequence_name="theory_first",
                strategy=SequenceStrategy.THEORY_FIRST,
                agents=["research_agent", "analysis_agent"],
                score=0.0,
                rationale="Academic approach",
                estimated_duration=2.0,
                confidence=0.0
            ),
            GeneratedSequence(
                sequence_name="market_first", 
                strategy=SequenceStrategy.MARKET_FIRST,
                agents=["market_agent", "research_agent"],
                score=0.0,
                rationale="Market approach",
                estimated_duration=2.0,
                confidence=0.0
            )
        ]
        
        ranked = sequence_generator.rank_sequences_by_topic(sequences, topic_analysis)
        
        assert len(ranked) == 2
        assert all(seq.score > 0 for seq in ranked)
        assert ranked[0].score >= ranked[1].score
        
        # Theory-first should score higher for academic topics
        theory_first_seq = next((s for s in ranked if s.strategy == SequenceStrategy.THEORY_FIRST), None)
        market_first_seq = next((s for s in ranked if s.strategy == SequenceStrategy.MARKET_FIRST), None)
        
        if theory_first_seq and market_first_seq:
            assert theory_first_seq.score >= market_first_seq.score
    
    def test_keyword_extraction(self, sequence_generator):
        """Test keyword extraction from research topics."""
        text = "Analyze machine learning algorithms for natural language processing applications"
        keywords = sequence_generator._extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "machine" in keywords
        assert "learning" in keywords
        assert "algorithms" in keywords
        assert "natural" in keywords
        assert "language" in keywords
        assert "processing" in keywords
        assert "applications" in keywords
        
        # Should not include stop words
        assert "for" not in keywords
        assert "the" not in keywords
    
    def test_domain_indicators_calculation(self, sequence_generator):
        """Test domain indicator calculation."""
        academic_text = "conduct research study analysis literature review methodology findings"
        indicators = sequence_generator._calculate_domain_indicators(academic_text)
        
        assert "academic" in indicators
        assert "market" in indicators  
        assert "technical" in indicators
        assert indicators["academic"] > indicators["market"]
        assert indicators["academic"] > indicators["technical"]
    
    def test_complexity_score_calculation(self, sequence_generator):
        """Test complexity score calculation."""
        simple_topic = "Study cats."
        complex_topic = """
        Conduct a comprehensive, in-depth analysis of machine learning interpretability 
        methods across multiple domains including computer vision, natural language processing,
        and time series analysis. The research should evaluate various explainable AI
        techniques, assess their effectiveness through empirical validation, and provide
        detailed comparative analysis of performance metrics.
        """
        
        simple_keywords = sequence_generator._extract_keywords(simple_topic)
        complex_keywords = sequence_generator._extract_keywords(complex_topic)
        
        simple_score = sequence_generator._calculate_complexity_score(simple_topic, simple_keywords)
        complex_score = sequence_generator._calculate_complexity_score(complex_topic, complex_keywords)
        
        assert complex_score > simple_score
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1
    
    def test_agent_type_classification(self, sequence_generator, mock_agent_registry):
        """Test agent type classification."""
        available_agents = ["research_agent", "analysis_agent", "market_agent", "technical_agent"]
        
        research_agents = sequence_generator._get_agents_by_type("research", available_agents)
        assert "research_agent" in research_agents
        
        analysis_agents = sequence_generator._get_agents_by_type("analysis", available_agents)
        assert "analysis_agent" in analysis_agents
        
        market_agents = sequence_generator._get_agents_by_type("market", available_agents)
        assert "market_agent" in market_agents
        
        technical_agents = sequence_generator._get_agents_by_type("technical", available_agents)
        assert "technical_agent" in technical_agents
    
    def test_best_agent_selection(self, sequence_generator, mock_agent_registry):
        """Test best agent selection logic."""
        candidates = ["research_agent", "analysis_agent"]
        topic_analysis = TopicAnalysis(
            topic_type=TopicType.ACADEMIC,
            complexity_score=0.7,
            scope_breadth=0.6,
            keywords=["research", "academic", "literature"],
            domain_indicators={"academic": 0.8},
            estimated_agents_needed=2,
            priority_areas=["academic"]
        )
        
        best = sequence_generator._select_best_agent(candidates, "research", topic_analysis)
        assert best is not None
        assert best in candidates
    
    def test_empty_agent_list_handling(self, sequence_generator):
        """Test handling of empty agent lists."""
        topic = "Test research topic"
        
        with pytest.raises(RuntimeError, match="No agents available"):
            sequence_generator.generate_sequences(topic, available_agents=[])
    
    def test_single_agent_sequence(self, sequence_generator, mock_agent_registry):
        """Test sequence generation with only one agent available."""
        topic = "Simple research task"
        available_agents = ["research_agent"]
        
        sequences = sequence_generator.generate_sequences(
            research_topic=topic,
            available_agents=available_agents,
            num_sequences=1
        )
        
        assert len(sequences) == 1
        assert len(sequences[0].agents) == 1
        assert sequences[0].agents[0] == "research_agent"
    
    def test_strategy_selection_for_topic_types(self, sequence_generator, mock_agent_registry):
        """Test strategy selection based on different topic types."""
        academic_analysis = TopicAnalysis(
            topic_type=TopicType.ACADEMIC,
            complexity_score=0.7,
            scope_breadth=0.6,
            keywords=["research"],
            domain_indicators={"academic": 0.9},
            estimated_agents_needed=3,
            priority_areas=["academic"]
        )
        
        strategies = sequence_generator._select_strategies_for_topic(academic_analysis, 3)
        assert SequenceStrategy.THEORY_FIRST in strategies
        
        market_analysis = TopicAnalysis(
            topic_type=TopicType.MARKET,
            complexity_score=0.6,
            scope_breadth=0.7,
            keywords=["market"],
            domain_indicators={"market": 0.9},
            estimated_agents_needed=3,
            priority_areas=["market"],
            market_relevance=0.9
        )
        
        strategies = sequence_generator._select_strategies_for_topic(market_analysis, 3)
        assert SequenceStrategy.MARKET_FIRST in strategies
    
    def test_scoring_components(self, sequence_generator, mock_agent_registry):
        """Test individual scoring components."""
        topic_analysis = TopicAnalysis(
            topic_type=TopicType.ACADEMIC,
            complexity_score=0.7,
            scope_breadth=0.6,
            keywords=["research", "academic"],
            domain_indicators={"academic": 0.8},
            estimated_agents_needed=2,
            priority_areas=["academic"]
        )
        
        sequence = GeneratedSequence(
            sequence_name="test_sequence",
            strategy=SequenceStrategy.THEORY_FIRST,
            agents=["research_agent", "analysis_agent"],
            score=0.0,
            rationale="Test sequence",
            estimated_duration=2.0,
            confidence=0.0
        )
        
        # Test individual scoring methods
        topic_fit = sequence_generator._calculate_topic_fit_score(sequence, topic_analysis)
        assert 0 <= topic_fit <= 1
        
        coverage = sequence_generator._calculate_coverage_score(sequence, topic_analysis)
        assert 0 <= coverage <= 1
        
        efficiency = sequence_generator._calculate_efficiency_score(sequence, topic_analysis)
        assert 0 <= efficiency <= 1
        
        expertise = sequence_generator._calculate_expertise_match_score(sequence, topic_analysis)
        assert 0 <= expertise <= 1
    
    def test_end_to_end_sequence_generation(self, sequence_generator):
        """Test complete end-to-end sequence generation workflow."""
        sample_topics = {
            "academic": "Systematic literature review of deep learning applications in medical image analysis",
            "market": "Market opportunity assessment for AI-powered fintech solutions in emerging markets", 
            "technical": "Scalable microservices architecture design for real-time data processing systems",
            "mixed": "Comprehensive evaluation of blockchain technology adoption in supply chain management"
        }
        
        for topic_type, topic in sample_topics.items():
            sequences = sequence_generator.generate_sequences(
                research_topic=topic,
                num_sequences=3
            )
            
            # Verify we get valid sequences
            assert len(sequences) <= 3
            assert all(isinstance(seq, GeneratedSequence) for seq in sequences)
            assert all(seq.score > 0 for seq in sequences)
            assert all(len(seq.agents) > 0 for seq in sequences)
            
            # Verify sequences are properly ranked
            if len(sequences) > 1:
                for i in range(len(sequences) - 1):
                    assert sequences[i].score >= sequences[i + 1].score


class TestTopicAnalysis:
    """Test suite for TopicAnalysis data structure."""
    
    def test_topic_analysis_creation(self):
        """Test TopicAnalysis object creation."""
        analysis = TopicAnalysis(
            topic_type=TopicType.ACADEMIC,
            complexity_score=0.7,
            scope_breadth=0.6,
            keywords=["test", "keywords"],
            domain_indicators={"academic": 0.8},
            estimated_agents_needed=3,
            priority_areas=["academic"]
        )
        
        assert analysis.topic_type == TopicType.ACADEMIC
        assert analysis.complexity_score == 0.7
        assert analysis.scope_breadth == 0.6
        assert len(analysis.keywords) == 2
        assert analysis.domain_indicators["academic"] == 0.8
        assert analysis.estimated_agents_needed == 3
        assert "academic" in analysis.priority_areas


class TestGeneratedSequence:
    """Test suite for GeneratedSequence data structure."""
    
    def test_generated_sequence_creation(self):
        """Test GeneratedSequence object creation."""
        sequence = GeneratedSequence(
            sequence_name="test_sequence",
            strategy=SequenceStrategy.THEORY_FIRST,
            agents=["agent1", "agent2"],
            score=0.8,
            rationale="Test rationale",
            estimated_duration=2.5,
            confidence=0.9
        )
        
        assert sequence.sequence_name == "test_sequence"
        assert sequence.strategy == SequenceStrategy.THEORY_FIRST
        assert len(sequence.agents) == 2
        assert sequence.score == 0.8
        assert sequence.rationale == "Test rationale"
        assert sequence.estimated_duration == 2.5
        assert sequence.confidence == 0.9


