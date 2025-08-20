"""Base models for the sequential agent ordering optimization framework.

This module defines the core data structures for tracking sequence patterns,
results, insight transitions, and tool productivity metrics across different
agent orderings in the research process.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class SequenceStrategy(Enum):
    """Strategic sequence patterns for specialized agent ordering."""
    
    THEORY_FIRST = "theory_first"      # Academic → Industry → Technical
    MARKET_FIRST = "market_first"      # Industry → Academic → Technical  
    FUTURE_BACK = "future_back"        # Technical → Academic → Industry


class AgentType(Enum):
    """Types of specialized research agents."""
    
    ACADEMIC = "academic"              # Theory & research focus
    INDUSTRY = "industry"              # Market & business focus
    TECHNICAL_TRENDS = "technical_trends"  # Implementation + trends focus


class InsightType(Enum):
    """Categories of insights that can be transferred between agents."""
    
    THEORETICAL_FOUNDATION = "theoretical_foundation"
    MARKET_OPPORTUNITY = "market_opportunity"
    TECHNICAL_FEASIBILITY = "technical_feasibility"
    IMPLEMENTATION_BARRIER = "implementation_barrier"
    RESEARCH_GAP = "research_gap"
    COMPETITIVE_ADVANTAGE = "competitive_advantage"
    FUTURE_TREND = "future_trend"
    VALIDATION_CRITERIA = "validation_criteria"


class SequencePattern(BaseModel):
    """Defines a specific sequence of specialized agents for research execution."""
    
    sequence_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy: SequenceStrategy
    agent_order: List[AgentType]
    description: str
    expected_advantages: List[str] = Field(default_factory=list)
    
    def __post_init__(self):
        """Validate sequence pattern consistency."""
        if len(self.agent_order) != 3:
            raise ValueError("Sequence must contain exactly 3 agents")
        
        expected_orders = {
            SequenceStrategy.THEORY_FIRST: [
                AgentType.ACADEMIC, 
                AgentType.INDUSTRY, 
                AgentType.TECHNICAL_TRENDS
            ],
            SequenceStrategy.MARKET_FIRST: [
                AgentType.INDUSTRY, 
                AgentType.ACADEMIC, 
                AgentType.TECHNICAL_TRENDS
            ],
            SequenceStrategy.FUTURE_BACK: [
                AgentType.TECHNICAL_TRENDS, 
                AgentType.ACADEMIC, 
                AgentType.INDUSTRY
            ]
        }
        
        if self.agent_order != expected_orders[self.strategy]:
            raise ValueError(f"Agent order {self.agent_order} doesn't match strategy {self.strategy}")


class InsightTransition(BaseModel):
    """Tracks insight flow and productivity between sequential agents."""
    
    transition_id: str = Field(default_factory=lambda: str(uuid4()))
    from_agent: AgentType
    to_agent: AgentType
    source_insight: str
    insight_type: InsightType
    generated_questions: List[str]
    
    # Productivity metrics
    question_quality_score: float = Field(ge=0.0, le=1.0)
    research_depth_achieved: float = Field(ge=0.0, le=1.0)
    novel_findings_discovered: int = Field(ge=0)
    time_to_productive_research: float = Field(ge=0.0)  # seconds
    
    # Adaptive learning data
    productive_transition: bool = False
    productivity_score: float = Field(ge=0.0, le=1.0, default=0.0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ToolProductivityMetrics(BaseModel):
    """Core metrics tracking tool productivity and agent efficiency."""
    
    # Tool Productivity (Primary Metric): Quality / Agent_Calls
    tool_productivity: float = Field(ge=0.0)
    research_quality_score: float = Field(ge=0.0, le=1.0)
    total_agent_calls: int = Field(ge=1)
    
    # Efficiency Metrics
    agent_efficiency: float = Field(ge=0.0)  # Useful_Insights / Total_Agent_Calls
    context_efficiency: float = Field(ge=0.0)  # Relevant_Context_Used / Total_Context_Available
    time_to_value: float = Field(ge=0.0)  # Time to first significant insight (seconds)
    
    # Quality Breakdown
    insight_novelty: float = Field(ge=0.0, le=1.0)
    insight_relevance: float = Field(ge=0.0, le=1.0)
    insight_actionability: float = Field(ge=0.0, le=1.0)
    research_completeness: float = Field(ge=0.0, le=1.0)
    
    # Agent Performance
    useful_insights_count: int = Field(ge=0)
    redundant_research_count: int = Field(ge=0)
    cognitive_offloading_incidents: int = Field(ge=0)
    
    # Context Usage
    relevant_context_used: float = Field(ge=0.0, le=1.0)
    total_context_available: float = Field(ge=0.0, le=1.0)


class AgentExecutionResult(BaseModel):
    """Results from a single agent's execution in a sequence."""
    
    agent_type: AgentType
    execution_order: int  # 1, 2, or 3
    
    # Input Context
    received_questions: List[str]
    previous_insights: List[str]
    research_topic: str
    
    # Execution Data
    start_time: datetime
    end_time: datetime
    execution_duration: float  # seconds
    
    # Tool Usage
    tool_calls_made: int
    search_queries_executed: int
    think_tool_usage_count: int
    
    # Output
    key_insights: List[str]
    research_findings: str
    refined_insights: List[str]  # Insights specifically prepared for next agent
    
    # Quality Metrics
    insight_quality_scores: List[float]
    research_depth_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    
    # Cognitive Load
    cognitive_offloading_detected: bool = False
    independent_reasoning_score: float = Field(ge=0.0, le=1.0)


class SequenceResult(BaseModel):
    """Complete results from executing a sequence pattern."""
    
    result_id: str = Field(default_factory=lambda: str(uuid4()))
    sequence_pattern: SequencePattern
    research_topic: str
    
    # Execution Timeline
    start_time: datetime
    end_time: datetime
    total_duration: float  # seconds
    
    # Agent Results
    agent_results: List[AgentExecutionResult]
    insight_transitions: List[InsightTransition]
    
    # Final Output
    final_research_synthesis: str
    comprehensive_findings: List[str]
    
    # Aggregate Metrics
    overall_productivity_metrics: ToolProductivityMetrics
    sequence_efficiency_score: float = Field(ge=0.0, le=1.0)
    
    # Comparative Analysis
    unique_insights_generated: int = Field(ge=0)
    research_breadth_score: float = Field(ge=0.0, le=1.0)
    research_depth_score: float = Field(ge=0.0, le=1.0)
    
    # Quality Assessment
    final_quality_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    actionability_score: float = Field(ge=0.0, le=1.0)


class SequenceComparison(BaseModel):
    """Comparative analysis between different sequence patterns."""
    
    comparison_id: str = Field(default_factory=lambda: str(uuid4()))
    research_topic: str
    compared_sequences: List[SequenceResult]
    
    # Productivity Variance Analysis
    productivity_variance: float = Field(ge=0.0)  # Standard deviation of tool productivity
    significant_difference_detected: bool = False  # True if >20% variance found
    
    # Best Performing Sequence
    highest_productivity_sequence: SequenceStrategy
    productivity_advantage: float = Field(ge=0.0)  # Percentage advantage
    
    # Detailed Comparisons
    productivity_rankings: Dict[SequenceStrategy, float]
    quality_rankings: Dict[SequenceStrategy, float]
    efficiency_rankings: Dict[SequenceStrategy, float]
    
    # Insight Analysis
    unique_insights_by_sequence: Dict[SequenceStrategy, List[str]]
    shared_insights_across_sequences: List[str]
    sequence_specific_advantages: Dict[SequenceStrategy, List[str]]
    
    # Statistical Validation
    statistical_significance: float = Field(ge=0.0, le=1.0)
    confidence_level: float = Field(ge=0.0, le=1.0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AdaptiveLearningState(BaseModel):
    """State for tracking and adapting sequence optimization over time."""
    
    # Historical Performance
    sequence_performance_history: Dict[SequenceStrategy, List[float]] = Field(default_factory=dict)
    insight_transition_patterns: Dict[str, List[InsightTransition]] = Field(default_factory=dict)
    
    # Learning Parameters
    productive_transition_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    adaptation_learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Optimization State
    optimal_sequence_for_topic_types: Dict[str, SequenceStrategy] = Field(default_factory=dict)
    transition_success_rates: Dict[str, float] = Field(default_factory=dict)
    
    # Performance Tracking
    total_sequences_executed: int = Field(default=0, ge=0)
    significant_improvements_detected: int = Field(default=0, ge=0)
    
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# Predefined sequence patterns for the three strategic approaches
THEORY_FIRST_PATTERN = SequencePattern(
    strategy=SequenceStrategy.THEORY_FIRST,
    agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
    description="Start with theoretical foundation, then explore market applications, finish with technical implementation",
    expected_advantages=[
        "Strong theoretical grounding",
        "Evidence-based market analysis", 
        "Technically informed implementation strategy"
    ]
)

MARKET_FIRST_PATTERN = SequencePattern(
    strategy=SequenceStrategy.MARKET_FIRST,
    agent_order=[AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS],
    description="Begin with market opportunities, validate with academic research, conclude with technical feasibility",
    expected_advantages=[
        "Market-driven research focus",
        "Commercial viability emphasis",
        "Practical implementation priority"
    ]
)

FUTURE_BACK_PATTERN = SequencePattern(
    strategy=SequenceStrategy.FUTURE_BACK,
    agent_order=[AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY],
    description="Start with future technical trends, ground in academic theory, assess market readiness",
    expected_advantages=[
        "Future-oriented perspective",
        "Innovation-focused research",
        "Technology-push market analysis"
    ]
)

# Registry of all available sequence patterns
SEQUENCE_PATTERNS = {
    SequenceStrategy.THEORY_FIRST: THEORY_FIRST_PATTERN,
    SequenceStrategy.MARKET_FIRST: MARKET_FIRST_PATTERN,
    SequenceStrategy.FUTURE_BACK: FUTURE_BACK_PATTERN
}