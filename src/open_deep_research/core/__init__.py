"""Core modules for the Open Deep Research system.

This package contains the fundamental components that power the research system,
including unified sequence generation, shared data models, and core utilities.
"""

from .sequence_generator import (
    UnifiedSequenceGenerator,
    SequenceStrategy,
    TopicType,
    TopicAnalysis,
    GeneratedSequence,
    SequenceGenerationInput,
    SequenceGenerationOutput,
    SequenceGenerationResult
)

__all__ = [
    "UnifiedSequenceGenerator",
    "SequenceStrategy", 
    "TopicType",
    "TopicAnalysis",
    "GeneratedSequence",
    "SequenceGenerationInput",
    "SequenceGenerationOutput",
    "SequenceGenerationResult"
]