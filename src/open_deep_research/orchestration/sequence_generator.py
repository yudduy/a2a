"""DEPRECATED: This module has been consolidated into core.sequence_generator.

This module is kept for backward compatibility. All functionality has been moved
to open_deep_research.core.sequence_generator.UnifiedSequenceGenerator.

Please update your imports to use the new unified sequence generator:
  from open_deep_research.core.sequence_generator import UnifiedSequenceGenerator
"""

import warnings
from typing import Dict, List, Optional, Any, Tuple, Set

from open_deep_research.core.sequence_generator import (
    UnifiedSequenceGenerator,
    TopicType,
    SequenceStrategy, 
    TopicAnalysis,
    GeneratedSequence,
    SequenceGenerationInput,
    AgentCapability
)
from open_deep_research.agents.registry import AgentRegistry

# Issue deprecation warning
warnings.warn(
    "orchestration.sequence_generator is deprecated. "
    "Use core.sequence_generator.UnifiedSequenceGenerator instead.",
    DeprecationWarning,
    stacklevel=2
)


class SequenceGenerator:
    """DEPRECATED: Backward compatibility wrapper for UnifiedSequenceGenerator.
    
    This class provides backward compatibility for existing code using the old
    SequenceGenerator. All functionality is delegated to UnifiedSequenceGenerator
    with rule-based generation mode.
    """
    
    def __init__(self, agent_registry: AgentRegistry, debug_mode: bool = False):
        """Initialize with backward compatibility wrapper.
        
        Args:
            agent_registry: Registry containing available agents  
            debug_mode: Enable detailed logging for debugging
        """
        warnings.warn(
            "SequenceGenerator is deprecated. Use UnifiedSequenceGenerator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self._unified_generator = UnifiedSequenceGenerator(
            agent_registry=agent_registry,
            debug_mode=debug_mode
        )
        self.agent_registry = agent_registry
        self.debug_mode = debug_mode
    
    def generate_sequences(
        self,
        research_topic: str,
        available_agents: Optional[List[str]] = None,
        num_sequences: int = 3,
        strategies: Optional[List[SequenceStrategy]] = None
    ) -> List[GeneratedSequence]:
        """Generate sequences using unified generator with rule-based mode.
        
        Args:
            research_topic: The research topic to analyze
            available_agents: List of available agent names
            num_sequences: Number of sequences to generate
            strategies: Specific strategies to use
            
        Returns:
            List of generated sequences ranked by fitness score
        """
        # Convert to new format
        if available_agents is None:
            available_agents = self.agent_registry.list_agents()
        
        # Create AgentCapability objects from agent names
        agent_capabilities = []
        for agent_name in available_agents:
            agent_config = self.agent_registry.get_agent(agent_name)
            if agent_config:
                capability = AgentCapability(
                    name=agent_name,
                    expertise_areas=agent_config.get("expertise_areas", []),
                    description=agent_config.get("description", f"Agent: {agent_name}"),
                    typical_use_cases=agent_config.get("typical_use_cases", []),
                    strength_summary=agent_config.get("strength_summary", f"Specialized {agent_name}")
                )
                agent_capabilities.append(capability)
        
        # Create input for unified generator
        input_data = SequenceGenerationInput(
            research_topic=research_topic,
            available_agents=agent_capabilities,
            generation_mode="rule_based",  # Force rule-based for backward compatibility
            num_sequences=num_sequences,
            strategies=strategies
        )
        
        # Generate using unified generator (synchronous)
        result = self._unified_generator.generate_sequences_sync(input_data)
        
        # Convert AgentSequence results back to GeneratedSequence format
        generated_sequences = []
        for i, agent_seq in enumerate(result.output.sequences):
            gen_seq = GeneratedSequence(
                sequence_name=agent_seq.sequence_name,
                strategy=strategies[i] if strategies and i < len(strategies) else SequenceStrategy.BALANCED,
                agents=agent_seq.agent_names,
                score=agent_seq.confidence_score,
                rationale=agent_seq.rationale,
                estimated_duration=len(agent_seq.agent_names) * 0.5,  # Simple estimate
                confidence=agent_seq.confidence_score
            )
            generated_sequences.append(gen_seq)
        
        return generated_sequences
    
    def analyze_topic_characteristics(self, research_topic: str) -> TopicAnalysis:
        """Analyze topic characteristics using unified generator."""
        return self._unified_generator.analyze_topic_characteristics(research_topic)
    
    def create_theory_first_sequence(self, agents: List[str], topic_analysis: TopicAnalysis) -> List[str]:
        """Create theory-first sequence using unified generator."""
        return self._unified_generator._create_theory_first_sequence(agents, topic_analysis)
    
    def create_market_first_sequence(self, agents: List[str], topic_analysis: TopicAnalysis) -> List[str]:
        """Create market-first sequence using unified generator."""
        return self._unified_generator._create_market_first_sequence(agents, topic_analysis)
    
    def create_technical_first_sequence(self, agents: List[str], topic_analysis: TopicAnalysis) -> List[str]:
        """Create technical-first sequence using unified generator."""
        return self._unified_generator._create_technical_first_sequence(agents, topic_analysis)
    
    def rank_sequences_by_topic(
        self,
        sequences: List[GeneratedSequence],
        topic_analysis: TopicAnalysis
    ) -> List[GeneratedSequence]:
        """Rank sequences by topic fitness using unified generator."""
        # This is a simplified ranking for backward compatibility
        # The real ranking happens inside the unified generator
        return sorted(sequences, key=lambda s: s.score, reverse=True)


# Re-export all types for backward compatibility
__all__ = [
    "SequenceGenerator",
    "TopicType",
    "SequenceStrategy", 
    "TopicAnalysis",
    "GeneratedSequence",
    "UnifiedSequenceGenerator"
]