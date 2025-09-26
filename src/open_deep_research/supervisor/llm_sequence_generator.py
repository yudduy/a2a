"""DEPRECATED: This module has been consolidated into core.sequence_generator.

This module is kept for backward compatibility. All functionality has been moved
to open_deep_research.core.sequence_generator.UnifiedSequenceGenerator.

Please update your imports to use the new unified sequence generator:
  from open_deep_research.core.sequence_generator import UnifiedSequenceGenerator
"""

import warnings
from typing import Optional

from langchain_core.runnables import RunnableConfig

from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.core.sequence_generator import (
    SequenceGenerationInput,
    UnifiedSequenceGenerator,
)

from .sequence_models import AgentSequence as OriginalAgentSequence
from .sequence_models import SequenceGenerationInput as OriginalInput
from .sequence_models import SequenceGenerationMetadata as OriginalMetadata
from .sequence_models import SequenceGenerationOutput as OriginalOutput
from .sequence_models import SequenceGenerationResult as OriginalResult

# Issue deprecation warning
warnings.warn(
    "supervisor.llm_sequence_generator is deprecated. "
    "Use core.sequence_generator.UnifiedSequenceGenerator instead.",
    DeprecationWarning,
    stacklevel=2
)


class LLMSequenceGenerator:
    """DEPRECATED: Backward compatibility wrapper for UnifiedSequenceGenerator.
    
    This class provides backward compatibility for existing code using the old
    LLMSequenceGenerator. All functionality is delegated to UnifiedSequenceGenerator
    with LLM-based generation mode.
    """
    
    def __init__(self, model_config: Optional[RunnableConfig] = None):
        """Initialize with backward compatibility wrapper.
        
        Args:
            model_config: Optional configuration for the LLM model
        """
        warnings.warn(
            "LLMSequenceGenerator is deprecated. Use UnifiedSequenceGenerator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create a basic agent registry for the unified generator
        # In practice, this should be injected properly
        self.agent_registry = AgentRegistry()
        
        self._unified_generator = UnifiedSequenceGenerator(
            agent_registry=self.agent_registry,
            model_config=model_config
        )
        self.model_config = model_config
    
    async def generate_sequences(
        self, 
        input_data: OriginalInput
    ) -> OriginalResult:
        """Generate strategic agent sequences using LLM reasoning.
        
        Args:
            input_data: Input containing research topic and available agents
            
        Returns:
            SequenceGenerationResult with sequences and metadata
        """
        # Convert to new input format
        new_input = SequenceGenerationInput(
            research_topic=input_data.research_topic,
            research_brief=input_data.research_brief,
            available_agents=input_data.available_agents,
            research_type=input_data.research_type,
            constraints=input_data.constraints or {},
            generation_mode="llm_based",  # Force LLM-based for backward compatibility
            num_sequences=3
        )
        
        # Generate using unified generator
        result = await self._unified_generator.generate_sequences(new_input)
        
        # Convert back to original format
        original_output = OriginalOutput(
            research_analysis=result.output.research_analysis,
            sequences=[
                OriginalAgentSequence(
                    sequence_name=seq.sequence_name,
                    agent_names=seq.agent_names,
                    rationale=seq.rationale,
                    approach_description=seq.approach_description,
                    expected_outcomes=seq.expected_outcomes,
                    confidence_score=seq.confidence_score,
                    research_focus=seq.research_focus
                )
                for seq in result.output.sequences
            ],
            reasoning_summary=result.output.reasoning_summary,
            recommended_sequence=result.output.recommended_sequence,
            alternative_considerations=result.output.alternative_considerations
        )
        
        original_metadata = OriginalMetadata(
            generation_timestamp=result.metadata.generation_timestamp,
            model_used=result.metadata.model_used,
            input_token_count=result.metadata.input_token_count,
            output_token_count=result.metadata.output_token_count,
            generation_time_seconds=result.metadata.generation_time_seconds,
            fallback_used=result.metadata.fallback_used,
            error_details=result.metadata.error_details
        )
        
        return OriginalResult(
            output=original_output,
            metadata=original_metadata,
            success=result.success
        )
    
    def generate_sequences_sync(
        self, 
        input_data: OriginalInput
    ) -> OriginalResult:
        """Synchronous version of sequence generation.
        
        Args:
            input_data: Input containing research topic and available agents
            
        Returns:
            SequenceGenerationResult with sequences and metadata
        """
        # Convert to new input format
        new_input = SequenceGenerationInput(
            research_topic=input_data.research_topic,
            research_brief=input_data.research_brief,
            available_agents=input_data.available_agents,
            research_type=input_data.research_type,
            constraints=input_data.constraints or {},
            generation_mode="llm_based",  # Force LLM-based for backward compatibility
            num_sequences=3
        )
        
        # Generate using unified generator
        result = self._unified_generator.generate_sequences_sync(new_input)
        
        # Convert back to original format
        original_output = OriginalOutput(
            research_analysis=result.output.research_analysis,
            sequences=[
                OriginalAgentSequence(
                    sequence_name=seq.sequence_name,
                    agent_names=seq.agent_names,
                    rationale=seq.rationale,
                    approach_description=seq.approach_description,
                    expected_outcomes=seq.expected_outcomes,
                    confidence_score=seq.confidence_score,
                    research_focus=seq.research_focus
                )
                for seq in result.output.sequences
            ],
            reasoning_summary=result.output.reasoning_summary,
            recommended_sequence=result.output.recommended_sequence,
            alternative_considerations=result.output.alternative_considerations
        )
        
        original_metadata = OriginalMetadata(
            generation_timestamp=result.metadata.generation_timestamp,
            model_used=result.metadata.model_used,
            input_token_count=result.metadata.input_token_count,
            output_token_count=result.metadata.output_token_count,
            generation_time_seconds=result.metadata.generation_time_seconds,
            fallback_used=result.metadata.fallback_used,
            error_details=result.metadata.error_details
        )
        
        return OriginalResult(
            output=original_output,
            metadata=original_metadata,
            success=result.success
        )


# Re-export all types for backward compatibility
__all__ = [
    "LLMSequenceGenerator",
    "UnifiedSequenceGenerator"
]