"""Tests for DynamicSequencePattern model validation and properties.

This test module validates the DynamicSequencePattern model structure,
properties, and field validation to ensure the dynamic sequence system
maintains data integrity and proper constraints.
"""

import pytest
from uuid import UUID
from typing import Set

from open_deep_research.sequencing.models import (
    DynamicSequencePattern,
    AgentType,
    SequencePattern,
    SEQUENCE_PATTERNS
)


class TestDynamicSequencePattern:
    """Test cases for DynamicSequencePattern model validation."""
    
    def test_dynamic_sequence_pattern_creation(self):
        """Test basic DynamicSequencePattern creation and validation."""
        # Valid pattern creation
        pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
            description="Test dynamic sequence for academic research",
            reasoning="Academic foundation builds strong theoretical base for subsequent analysis",
            confidence_score=0.85,
            expected_advantages=["Strong theoretical foundation", "Evidence-based analysis"],
            topic_alignment_score=0.9
        )
        
        # Verify fields are set correctly
        assert len(pattern.agent_order) == 3
        assert pattern.agent_order[0] == AgentType.ACADEMIC
        assert pattern.description == "Test dynamic sequence for academic research"
        assert pattern.confidence_score == 0.85
        assert pattern.topic_alignment_score == 0.9
        assert len(pattern.expected_advantages) == 2
        
        # Verify auto-generated fields
        assert pattern.sequence_id is not None
        assert UUID(pattern.sequence_id)  # Should be valid UUID
        assert pattern.strategy is None  # Default for dynamic patterns
    
    def test_dynamic_sequence_pattern_properties(self):
        """Test computed properties of DynamicSequencePattern."""
        # Test with 2-agent sequence
        pattern = DynamicSequencePattern(
            agent_order=[AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
            description="Market-technical sequence",
            reasoning="Quick market validation followed by technical feasibility",
            confidence_score=0.7
        )
        
        assert pattern.sequence_length == 2
        assert pattern.agent_types_used == {AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS}
        
        # Test with single agent
        single_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Academic-only analysis",
            reasoning="Pure academic research approach",
            confidence_score=0.6
        )
        
        assert single_pattern.sequence_length == 1
        assert single_pattern.agent_types_used == {AgentType.ACADEMIC}
        
        # Test with repeated agents
        repeated_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.ACADEMIC],
            description="Academic-industry-academic cycle",
            reasoning="Theory-practice-theory validation cycle",
            confidence_score=0.75
        )
        
        assert repeated_pattern.sequence_length == 3
        assert repeated_pattern.agent_types_used == {AgentType.ACADEMIC, AgentType.INDUSTRY}
    
    def test_confidence_score_validation(self):
        """Test confidence score field validation constraints."""
        # Valid confidence scores
        for score in [0.0, 0.5, 1.0, 0.85, 0.123]:
            pattern = DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC],
                description="Test pattern",
                reasoning="Test reasoning",
                confidence_score=score
            )
            assert pattern.confidence_score == score
        
        # Invalid confidence scores should raise validation error
        with pytest.raises(ValueError):
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC],
                description="Test pattern",
                reasoning="Test reasoning",
                confidence_score=-0.1  # Below minimum
            )
        
        with pytest.raises(ValueError):
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC],
                description="Test pattern",
                reasoning="Test reasoning",
                confidence_score=1.1  # Above maximum
            )
    
    def test_topic_alignment_score_validation(self):
        """Test topic alignment score field validation constraints."""
        # Valid alignment scores
        for score in [0.0, 0.5, 1.0, 0.42]:
            pattern = DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC],
                description="Test pattern",
                reasoning="Test reasoning",
                confidence_score=0.8,
                topic_alignment_score=score
            )
            assert pattern.topic_alignment_score == score
        
        # Invalid alignment scores
        with pytest.raises(ValueError):
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC],
                description="Test pattern",
                reasoning="Test reasoning",
                confidence_score=0.8,
                topic_alignment_score=-0.1
            )
        
        with pytest.raises(ValueError):
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC],
                description="Test pattern",
                reasoning="Test reasoning",
                confidence_score=0.8,
                topic_alignment_score=1.2
            )
    
    def test_agent_order_validation(self):
        """Test agent order field requirements and validation."""
        # Empty agent order should raise validation error
        with pytest.raises(ValueError):
            DynamicSequencePattern(
                agent_order=[],
                description="Empty sequence",
                reasoning="No agents",
                confidence_score=0.5
            )
        
        # Valid single agent
        single_agent_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Single agent sequence",
            reasoning="Academic-only analysis",
            confidence_score=0.7
        )
        assert len(single_agent_pattern.agent_order) == 1
        
        # Valid multiple agents with repetition
        multi_agent_pattern = DynamicSequencePattern(
            agent_order=[
                AgentType.ACADEMIC, 
                AgentType.INDUSTRY, 
                AgentType.TECHNICAL_TRENDS,
                AgentType.ACADEMIC  # Repetition allowed
            ],
            description="Multi-agent with repetition",
            reasoning="Complex analysis with academic validation",
            confidence_score=0.8
        )
        assert len(multi_agent_pattern.agent_order) == 4
    
    def test_default_values(self):
        """Test default field values for DynamicSequencePattern."""
        pattern = DynamicSequencePattern(
            agent_order=[AgentType.INDUSTRY],
            description="Minimal pattern",
            reasoning="Basic reasoning"
        )
        
        # Check defaults
        assert pattern.confidence_score == 0.0  # Should default to 0.0
        assert pattern.topic_alignment_score == 0.0  # Should default to 0.0
        assert pattern.expected_advantages == []  # Should default to empty list
        assert pattern.strategy is None  # Should default to None
        assert pattern.sequence_id is not None  # Should be auto-generated
    
    def test_strategy_field_compatibility(self):
        """Test strategy field for backward compatibility."""
        # Dynamic pattern without strategy (default)
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY],
            description="Dynamic pattern",
            reasoning="Dynamic reasoning",
            confidence_score=0.7
        )
        assert dynamic_pattern.strategy is None
        
        # Dynamic pattern with strategy association
        associated_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
            description="Theory-first dynamic pattern",
            reasoning="Follows theory-first approach",
            confidence_score=0.85,
            strategy="theory_first"
        )
        assert associated_pattern.strategy == "theory_first"
    
    def test_compatibility_with_sequence_pattern(self):
        """Test that DynamicSequencePattern can coexist with SequencePattern."""
        # Create both types
        standard_pattern = SEQUENCE_PATTERNS["theory_first"]
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
            description="Dynamic theory-first pattern",
            reasoning="LLM-generated theory-first approach",
            confidence_score=0.9
        )
        
        # Both should have agent_order
        assert hasattr(standard_pattern, 'agent_order')
        assert hasattr(dynamic_pattern, 'agent_order')
        assert standard_pattern.agent_order == dynamic_pattern.agent_order
        
        # Both should have description
        assert hasattr(standard_pattern, 'description')
        assert hasattr(dynamic_pattern, 'description')
        
        # DynamicSequencePattern has additional fields
        assert hasattr(dynamic_pattern, 'reasoning')
        assert hasattr(dynamic_pattern, 'topic_alignment_score')
        assert hasattr(dynamic_pattern, 'sequence_length')
        assert hasattr(dynamic_pattern, 'agent_types_used')
    
    def test_expected_advantages_handling(self):
        """Test expected advantages list handling."""
        # Empty advantages
        pattern1 = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC],
            description="Pattern without advantages",
            reasoning="Simple pattern",
            confidence_score=0.5,
            expected_advantages=[]
        )
        assert pattern1.expected_advantages == []
        
        # Multiple advantages
        advantages = [
            "Strong theoretical foundation",
            "Evidence-based analysis",
            "Academic rigor applied to practical problems"
        ]
        pattern2 = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY],
            description="Pattern with advantages",
            reasoning="Comprehensive approach",
            confidence_score=0.8,
            expected_advantages=advantages
        )
        assert pattern2.expected_advantages == advantages
        assert len(pattern2.expected_advantages) == 3


class TestDynamicSequencePatternSerialization:
    """Test serialization and deserialization of DynamicSequencePattern."""
    
    def test_model_dump_and_load(self):
        """Test that DynamicSequencePattern can be serialized and deserialized."""
        original_pattern = DynamicSequencePattern(
            agent_order=[AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY],
            description="Future-back dynamic sequence",
            reasoning="Technical trends drive academic validation and market assessment",
            confidence_score=0.85,
            expected_advantages=["Future-oriented perspective", "Innovation identification"],
            topic_alignment_score=0.9,
            strategy="future_back"
        )
        
        # Serialize to dict
        pattern_dict = original_pattern.model_dump()
        
        # Verify key fields are present
        assert pattern_dict['agent_order'] == ['technical_trends', 'academic', 'industry']
        assert pattern_dict['description'] == "Future-back dynamic sequence"
        assert pattern_dict['confidence_score'] == 0.85
        assert pattern_dict['topic_alignment_score'] == 0.9
        assert pattern_dict['strategy'] == "future_back"
        
        # Deserialize back to model
        reconstructed_pattern = DynamicSequencePattern(**{
            **pattern_dict,
            'agent_order': [AgentType(agent) for agent in pattern_dict['agent_order']]
        })
        
        # Verify reconstruction
        assert reconstructed_pattern.agent_order == original_pattern.agent_order
        assert reconstructed_pattern.description == original_pattern.description
        assert reconstructed_pattern.confidence_score == original_pattern.confidence_score
        assert reconstructed_pattern.topic_alignment_score == original_pattern.topic_alignment_score
        assert reconstructed_pattern.expected_advantages == original_pattern.expected_advantages
    
    def test_json_serialization(self):
        """Test JSON serialization compatibility."""
        import json
        
        pattern = DynamicSequencePattern(
            agent_order=[AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
            description="Market-tech validation sequence",
            reasoning="Market needs drive technical implementation",
            confidence_score=0.75
        )
        
        # Should be JSON serializable
        pattern_dict = pattern.model_dump()
        json_str = json.dumps(pattern_dict)
        
        # Should be JSON deserializable
        loaded_dict = json.loads(json_str)
        assert loaded_dict['description'] == "Market-tech validation sequence"
        assert loaded_dict['confidence_score'] == 0.75


if __name__ == "__main__":
    pytest.main([__file__])