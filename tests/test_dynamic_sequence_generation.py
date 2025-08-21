"""Tests for SequenceAnalyzer.generate_dynamic_sequences() functionality.

This test module validates the dynamic sequence generation capabilities,
including topic analysis, confidence scoring, reasoning generation, and
proper sequence pattern creation for various research domains and query types.
"""

import pytest
from typing import List, Set
from unittest.mock import Mock, patch

from open_deep_research.sequencing.models import (
    DynamicSequencePattern,
    AgentType,
    QueryType,
    ResearchDomain,
    ScopeBreadth
)
from open_deep_research.sequencing.sequence_selector import SequenceAnalyzer


class TestDynamicSequenceGeneration:
    """Test cases for dynamic sequence generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()
    
    def test_generate_dynamic_sequences_basic(self):
        """Test basic dynamic sequence generation with default parameters."""
        topic = "Impact of artificial intelligence on modern healthcare systems"
        
        sequences = self.analyzer.generate_dynamic_sequences(topic)
        
        # Should return default number of sequences (3)
        assert len(sequences) == 3
        
        # All should be DynamicSequencePattern instances
        for seq in sequences:
            assert isinstance(seq, DynamicSequencePattern)
        
        # Should be sorted by confidence score (highest first)
        confidence_scores = [seq.confidence_score for seq in sequences]
        assert confidence_scores == sorted(confidence_scores, reverse=True)
        
        # All sequences should have required fields
        for seq in sequences:
            assert seq.sequence_id is not None
            assert len(seq.agent_order) > 0
            assert seq.description != ""
            assert seq.reasoning != ""
            assert 0.0 <= seq.confidence_score <= 1.0
            assert 0.0 <= seq.topic_alignment_score <= 1.0
            assert isinstance(seq.expected_advantages, list)
    
    def test_generate_dynamic_sequences_custom_count(self):
        """Test dynamic sequence generation with custom sequence count."""
        topic = "Quantum computing applications in financial modeling"
        
        # Test with different counts
        for count in [1, 2, 5]:
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=count)
            assert len(sequences) == count
            
            # Verify all are valid patterns
            for seq in sequences:
                assert isinstance(seq, DynamicSequencePattern)
                assert len(seq.agent_order) > 0
    
    def test_academic_research_topic_sequences(self):
        """Test dynamic sequences for academic research topics."""
        academic_topics = [
            "Recent advances in machine learning theory and computational complexity",
            "Systematic review of climate change mitigation strategies in peer-reviewed literature",
            "Theoretical frameworks for understanding cognitive behavioral therapy effectiveness"
        ]
        
        for topic in academic_topics:
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
            
            # Should generate valid sequences
            assert len(sequences) == 3
            
            # Primary sequence should likely favor academic-first approaches
            primary_seq = sequences[0]
            assert primary_seq.confidence_score > 0.5
            
            # At least one sequence should start with Academic agent for academic topics
            academic_first_sequences = [
                seq for seq in sequences 
                if seq.agent_order[0] == AgentType.ACADEMIC
            ]
            assert len(academic_first_sequences) >= 1
    
    def test_market_analysis_topic_sequences(self):
        """Test dynamic sequences for market analysis topics."""
        market_topics = [
            "Market opportunities for sustainable fashion brands in emerging economies",
            "Competitive landscape analysis for electric vehicle charging infrastructure",
            "Business model innovation in the subscription economy marketplace"
        ]
        
        for topic in market_topics:
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
            
            # Should generate valid sequences
            assert len(sequences) == 3
            
            # At least one sequence should start with Industry agent for market topics
            industry_first_sequences = [
                seq for seq in sequences 
                if seq.agent_order[0] == AgentType.INDUSTRY
            ]
            assert len(industry_first_sequences) >= 1
    
    def test_technical_innovation_topic_sequences(self):
        """Test dynamic sequences for technical/innovation topics."""
        technical_topics = [
            "Emerging trends in edge computing architectures for IoT applications",
            "Future developments in quantum computing hardware and software integration",
            "Next-generation blockchain consensus mechanisms and scalability solutions"
        ]
        
        for topic in technical_topics:
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
            
            # Should generate valid sequences
            assert len(sequences) == 3
            
            # At least one sequence should start with Technical Trends for innovation topics
            tech_first_sequences = [
                seq for seq in sequences 
                if seq.agent_order[0] == AgentType.TECHNICAL_TRENDS
            ]
            assert len(tech_first_sequences) >= 1
    
    def test_hybrid_multi_domain_topic_sequences(self):
        """Test dynamic sequences for complex multi-domain topics."""
        hybrid_topics = [
            "Comprehensive analysis of AI ethics: technical implementation, academic research, and market implications",
            "Interdisciplinary approach to sustainable urban development: technology, policy, and business models",
            "Digital transformation in healthcare: clinical research, market dynamics, and technical infrastructure"
        ]
        
        for topic in hybrid_topics:
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=4)
            
            # Should generate requested number of sequences
            assert len(sequences) == 4
            
            # Should have diverse starting agents for hybrid topics
            starting_agents = {seq.agent_order[0] for seq in sequences}
            assert len(starting_agents) >= 2  # At least 2 different starting agents
    
    def test_sequence_confidence_and_reasoning_quality(self):
        """Test quality of confidence scores and reasoning generation."""
        topic = "Artificial intelligence applications in personalized medicine"
        
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
        
        for seq in sequences:
            # Confidence scores should be reasonable (not all identical)
            assert 0.3 <= seq.confidence_score <= 1.0
            
            # Reasoning should be substantial and informative
            assert len(seq.reasoning) > 100  # Should be detailed
            assert topic.lower() in seq.reasoning.lower() or any(
                keyword in seq.reasoning.lower() 
                for keyword in ["artificial intelligence", "personalized medicine", "ai", "medical"]
            )
            
            # Expected advantages should be relevant
            assert len(seq.expected_advantages) >= 2
            for advantage in seq.expected_advantages:
                assert len(advantage) > 10  # Should be descriptive
    
    def test_topic_alignment_scores(self):
        """Test topic alignment score calculation."""
        topic = "Blockchain technology adoption in supply chain management"
        
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
        
        for seq in sequences:
            # Topic alignment should be reasonable
            assert 0.0 <= seq.topic_alignment_score <= 1.0
            
            # Primary sequence should have high alignment
            if seq == sequences[0]:  # Primary sequence
                assert seq.topic_alignment_score >= 0.6
    
    def test_agent_order_diversity(self):
        """Test that generated sequences provide diverse agent orderings."""
        topic = "Impact of renewable energy technologies on global economics"
        
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=5)
        
        # Collect all unique agent orderings
        unique_orderings = set()
        for seq in sequences:
            ordering_tuple = tuple(seq.agent_order)
            unique_orderings.add(ordering_tuple)
        
        # Should have diverse orderings (at least 2 different ones)
        assert len(unique_orderings) >= 2
        
        # Should use all agent types across sequences
        all_agents_used = set()
        for seq in sequences:
            all_agents_used.update(seq.agent_types_used)
        
        assert AgentType.ACADEMIC in all_agents_used
        assert AgentType.INDUSTRY in all_agents_used
        assert AgentType.TECHNICAL_TRENDS in all_agents_used
    
    def test_sequence_length_variations(self):
        """Test that sequences can have different lengths when appropriate."""
        topic = "Comprehensive analysis of quantum computing across all domains"
        
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=4)
        
        # Most sequences should be length 3 (standard), but allow flexibility
        sequence_lengths = [seq.sequence_length for seq in sequences]
        
        # Should have at least some 3-agent sequences
        assert 3 in sequence_lengths
        
        # All sequences should be reasonable length (1-5 agents)
        for length in sequence_lengths:
            assert 1 <= length <= 5
    
    def test_strategy_association_consistency(self):
        """Test that strategy associations are consistent when present."""
        topic = "Theoretical foundations of machine learning in academic literature"
        
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
        
        for seq in sequences:
            if seq.strategy is not None:
                # If strategy is associated, agent order should match expected patterns
                if seq.strategy == "theory_first":
                    assert seq.agent_order[0] == AgentType.ACADEMIC
                elif seq.strategy == "market_first":
                    assert seq.agent_order[0] == AgentType.INDUSTRY
                elif seq.strategy == "future_back":
                    assert seq.agent_order[0] == AgentType.TECHNICAL_TRENDS
    
    def test_empty_or_invalid_topic_handling(self):
        """Test handling of edge cases with topics."""
        # Empty topic
        with pytest.raises(Exception):  # Should handle gracefully or raise appropriate error
            self.analyzer.generate_dynamic_sequences("")
        
        # Very short topic
        short_sequences = self.analyzer.generate_dynamic_sequences("AI", num_sequences=2)
        assert len(short_sequences) == 2
        
        # Very long topic
        long_topic = "This is a very long research topic that contains many detailed aspects and considerations about multiple domains including academic research theoretical frameworks market analysis business opportunities technical implementation challenges future trends emerging technologies and comprehensive interdisciplinary approaches that require extensive investigation across all possible dimensions and perspectives"
        long_sequences = self.analyzer.generate_dynamic_sequences(long_topic, num_sequences=3)
        assert len(long_sequences) == 3
    
    def test_sequence_uniqueness_and_ids(self):
        """Test that generated sequences have unique IDs and distinct characteristics."""
        topic = "Smart city infrastructure development and urban planning"
        
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=4)
        
        # All sequence IDs should be unique
        sequence_ids = [seq.sequence_id for seq in sequences]
        assert len(set(sequence_ids)) == len(sequence_ids)
        
        # Sequences should have different characteristics (not identical)
        descriptions = [seq.description for seq in sequences]
        reasonings = [seq.reasoning for seq in sequences]
        
        # Should have variety in descriptions and reasoning
        assert len(set(descriptions)) >= 2
        assert len(set(reasonings)) >= 2


class TestDynamicSequenceAnalysisIntegration:
    """Test integration between sequence analysis and dynamic generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()
    
    def test_analysis_drives_dynamic_generation(self):
        """Test that query analysis properly drives dynamic sequence generation."""
        topic = "Academic research methods in computational linguistics"
        
        # First, analyze the query
        analysis = self.analyzer.analyze_query(topic)
        
        # Then generate dynamic sequences
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
        
        # Primary dynamic sequence should align with analysis recommendation
        primary_sequence = sequences[0]
        
        # Confidence should be consistent with analysis
        assert primary_sequence.confidence_score >= 0.4  # Should have reasonable confidence
        
        # If analysis is academic-focused, primary sequence should reflect this
        if analysis.query_type in [QueryType.ACADEMIC_RESEARCH]:
            assert primary_sequence.strategy == "theory_first" or primary_sequence.agent_order[0] == AgentType.ACADEMIC
    
    def test_complex_query_characteristics_influence(self):
        """Test that complex query characteristics influence sequence generation."""
        # Innovation-focused query
        innovation_topic = "Emerging trends and future breakthroughs in artificial general intelligence"
        innovation_sequences = self.analyzer.generate_dynamic_sequences(innovation_topic)
        
        # Should have sequences that leverage future-oriented approaches
        future_oriented_sequences = [
            seq for seq in innovation_sequences 
            if seq.agent_order[0] == AgentType.TECHNICAL_TRENDS or "future" in seq.reasoning.lower()
        ]
        assert len(future_oriented_sequences) >= 1
        
        # Business-focused query
        business_topic = "Market opportunities and revenue models for B2B SaaS platforms"
        business_sequences = self.analyzer.generate_dynamic_sequences(business_topic)
        
        # Should have sequences that prioritize market analysis
        market_focused_sequences = [
            seq for seq in business_sequences 
            if seq.agent_order[0] == AgentType.INDUSTRY or "market" in seq.reasoning.lower()
        ]
        assert len(market_focused_sequences) >= 1


class TestDynamicSequencePerformance:
    """Test performance and resource characteristics of dynamic sequence generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()
    
    def test_generation_speed_reasonable(self):
        """Test that dynamic sequence generation completes in reasonable time."""
        import time
        
        topic = "Sustainable energy systems and grid modernization strategies"
        
        start_time = time.time()
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should complete quickly (analysis should be efficient)
        assert generation_time < 5.0  # Should be under 5 seconds for 3 sequences
        assert len(sequences) == 3
    
    def test_multiple_topic_batch_performance(self):
        """Test performance when generating sequences for multiple topics."""
        topics = [
            "Machine learning applications in drug discovery",
            "Blockchain technology in supply chain transparency",
            "Renewable energy grid integration challenges",
            "Digital transformation in education systems",
            "Artificial intelligence ethics and governance"
        ]
        
        all_sequences = []
        for topic in topics:
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=2)
            all_sequences.extend(sequences)
        
        # Should generate expected total number
        assert len(all_sequences) == len(topics) * 2
        
        # All should be valid
        for seq in all_sequences:
            assert isinstance(seq, DynamicSequencePattern)
            assert len(seq.agent_order) > 0


if __name__ == "__main__":
    pytest.main([__file__])