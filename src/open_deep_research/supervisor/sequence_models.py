"""Pydantic models for LLM-based sequence generation.

This module defines structured output models for the LLM sequence generator,
including agent capability descriptions, sequence reasoning, and metadata.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class SequenceStrategy(Enum):
    """Sequence generation strategies."""
    THEORY_FIRST = "theory_first"
    MARKET_FIRST = "market_first"  
    TECHNICAL_FIRST = "technical_first"
    ANALYSIS_FIRST = "analysis_first"
    BALANCED = "balanced"
    CUSTOM = "custom"
    LLM_GENERATED = "llm_generated"


class AgentCapability(BaseModel):
    """Represents an agent's capabilities for LLM reasoning."""
    
    name: str = Field(description="Agent name/identifier")
    expertise_areas: List[str] = Field(description="Areas of expertise/specialization")
    description: str = Field(description="Brief description of agent capabilities")
    typical_use_cases: List[str] = Field(description="Common scenarios where this agent is useful")
    strength_summary: str = Field(description="One-line summary of agent's main strength")
    core_responsibilities: List[str] = Field(
        default_factory=list,
        description="Core responsibilities extracted from agent system prompt"
    )
    completion_indicators: List[str] = Field(
        default_factory=list,
        description="Indicators that signal when the agent has completed its work"
    )


class AgentSequence(BaseModel):
    """Represents a single strategic sequence of agents."""
    
    sequence_name: str = Field(description="Descriptive name for this sequence strategy")
    agent_names: List[str] = Field(description="Ordered list of agent names to execute")
    rationale: str = Field(description="Detailed reasoning for why this sequence is effective")
    approach_description: str = Field(description="High-level description of the research approach")
    expected_outcomes: List[str] = Field(description="Expected outcomes from this sequence")
    confidence_score: float = Field(
        ge=0.0, le=1.0, 
        description="Confidence in sequence effectiveness (0-1)"
    )
    research_focus: str = Field(description="Primary research focus of this sequence")


class SequenceGenerationInput(BaseModel):
    """Input data for LLM sequence generation."""
    
    research_topic: str = Field(description="The research topic/question to address")
    research_brief: Optional[str] = Field(default=None, description="Additional research context or brief")
    available_agents: List[AgentCapability] = Field(description="Available agents with capabilities")
    research_type: Optional[str] = Field(default=None, description="Type of research (academic, technical, market, etc.)")
    constraints: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Any constraints on sequence generation"
    )
    generation_mode: str = Field(
        default="hybrid",
        description="Generation mode: 'rule_based', 'llm_based', or 'hybrid'"
    )
    num_sequences: int = Field(default=3, ge=1, le=5, description="Number of sequences to generate")
    strategies: Optional[List[SequenceStrategy]] = Field(default=None, description="Specific strategies to use")


class SequenceGenerationOutput(BaseModel):
    """Complete output from LLM sequence generation."""
    
    research_analysis: str = Field(description="Analysis of the research requirements")
    sequences: List[AgentSequence] = Field(
        min_items=1, max_items=5,
        description="Generated strategic agent sequences"
    )
    reasoning_summary: str = Field(description="Summary of overall reasoning approach")
    recommended_sequence: int = Field(
        ge=0,
        description="Index of recommended sequence"
    )
    alternative_considerations: List[str] = Field(
        description="Alternative approaches or considerations"
    )
    generation_mode: str = Field(description="Mode used for generation")
    topic_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Topic analysis results")


class SequenceGenerationMetadata(BaseModel):
    """Metadata about the sequence generation process."""
    
    generation_timestamp: str = Field(description="When sequences were generated")
    model_used: str = Field(description="Method/model used for generation")
    input_token_count: Optional[int] = Field(description="Tokens in input")
    output_token_count: Optional[int] = Field(description="Tokens in output")
    generation_time_seconds: Optional[float] = Field(description="Time taken for generation")
    fallback_used: bool = Field(default=False, description="Whether fallback logic was used")
    error_details: Optional[str] = Field(default=None, description="Any error details if fallback used")
    generation_mode: str = Field(description="Mode used for generation")


class SequenceGenerationResult(BaseModel):
    """Complete result including output and metadata."""
    
    output: SequenceGenerationOutput
    metadata: SequenceGenerationMetadata
    success: bool = Field(description="Whether generation was successful")