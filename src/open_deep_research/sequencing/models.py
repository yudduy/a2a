"""Base models for the sequential agent ordering optimization framework.

This module defines the core data structures for tracking sequence patterns,
results, insight transitions, and tool productivity metrics across different
agent orderings in the research process.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


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


class QueryType(Enum):
    """Classification of research query types for sequence selection."""
    
    ACADEMIC_RESEARCH = "academic_research"
    MARKET_ANALYSIS = "market_analysis"
    TECHNICAL_FEASIBILITY = "technical_feasibility"
    INNOVATION_EXPLORATION = "innovation_exploration"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    TREND_ANALYSIS = "trend_analysis"
    HYBRID_MULTI_DOMAIN = "hybrid_multi_domain"


class ResearchDomain(Enum):
    """Primary research domains identified in queries."""
    
    ACADEMIC = "academic"
    MARKET = "market"
    TECHNICAL = "technical"
    HYBRID = "hybrid"


class ScopeBreadth(Enum):
    """Scope breadth classifications for research queries."""
    
    NARROW = "narrow"
    MEDIUM = "medium"
    BROAD = "broad"


class SequenceAnalysis(BaseModel):
    """Analysis results for query and sequence recommendations."""
    
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    query_type: QueryType
    research_domain: ResearchDomain
    complexity_score: float = Field(ge=0.0, le=1.0)
    scope_breadth: ScopeBreadth
    
    # Sequence recommendations with confidence scores
    recommended_sequences: List[Tuple[str, float]] = Field(default_factory=list)
    primary_recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Analysis results
    explanation: str
    reasoning: Dict[str, str] = Field(default_factory=dict)
    query_characteristics: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    original_query: str
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


class SequencePattern(BaseModel):
    """Defines a flexible sequence of specialized agents for research execution."""
    
    sequence_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_order: List[AgentType]
    description: str
    expected_advantages: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=1.0)
    reasoning: str = Field(default="")
    strategy: Optional[str] = None  # For backward compatibility


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
    highest_productivity_sequence: str
    productivity_advantage: float = Field(ge=0.0)  # Percentage advantage
    
    # Detailed Comparisons
    productivity_rankings: Dict[str, float]
    quality_rankings: Dict[str, float]
    efficiency_rankings: Dict[str, float]
    
    # Insight Analysis
    unique_insights_by_sequence: Dict[str, List[str]]
    shared_insights_across_sequences: List[str]
    sequence_specific_advantages: Dict[str, List[str]]
    
    # Statistical Validation
    statistical_significance: float = Field(ge=0.0, le=1.0)
    confidence_level: float = Field(ge=0.0, le=1.0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AdaptiveLearningState(BaseModel):
    """State for tracking and adapting sequence optimization over time."""
    
    # Historical Performance
    sequence_performance_history: Dict[str, List[float]] = Field(default_factory=dict)
    insight_transition_patterns: Dict[str, List[InsightTransition]] = Field(default_factory=dict)
    
    # Learning Parameters
    productive_transition_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    adaptation_learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Optimization State
    optimal_sequence_for_topic_types: Dict[str, str] = Field(default_factory=dict)
    transition_success_rates: Dict[str, float] = Field(default_factory=dict)
    
    # Performance Tracking
    total_sequences_executed: int = Field(default=0, ge=0)
    significant_improvements_detected: int = Field(default=0, ge=0)
    
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class DynamicSequencePattern(BaseModel):
    """Dynamic sequence pattern generated based on research topic analysis."""
    
    sequence_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_order: List[AgentType]
    description: str
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    expected_advantages: List[str] = Field(default_factory=list)
    topic_alignment_score: float = Field(ge=0.0, le=1.0, default=0.0)
    strategy: Optional[str] = None  # Optional strategy association
    
    @property
    def sequence_length(self) -> int:
        """Get the length of this sequence."""
        return len(self.agent_order)
    
    @property
    def agent_types_used(self) -> Set[AgentType]:
        """Get the unique agent types in this sequence."""
        return set(self.agent_order)


# Real-time metrics models for production-ready streaming

class MetricType(Enum):
    """Types of metrics being tracked."""
    
    TOOL_PRODUCTIVITY = "tool_productivity"
    RESEARCH_QUALITY = "research_quality"
    AGENT_EFFICIENCY = "agent_efficiency"
    TIME_TO_VALUE = "time_to_value"
    CONTEXT_EFFICIENCY = "context_efficiency"
    INSIGHT_QUALITY = "insight_quality"
    EXECUTION_PROGRESS = "execution_progress"
    RESOURCE_USAGE = "resource_usage"


class MetricsUpdateType(Enum):
    """Types of metrics updates for real-time streaming."""
    
    REAL_TIME = "real_time"
    SNAPSHOT = "snapshot"
    COMPARISON = "comparison"
    WINNER_DETECTED = "winner_detected"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    AGENT_COMPLETED = "agent_completed"
    ERROR = "error"


class SequenceMetrics(BaseModel):
    """Real-time metrics for a single sequence execution."""
    
    sequence_id: str
    strategy: str
    execution_id: str
    start_time: datetime
    
    # Current state
    status: str = "pending"  # pending, running, completed, failed
    current_agent_position: int = 0
    total_agents: int = 3
    
    # Real-time productivity metrics
    current_tool_productivity: float = 0.0
    current_research_quality: float = 0.0
    current_agent_efficiency: float = 0.0
    current_context_efficiency: float = 0.0
    
    # Performance tracking
    time_to_first_insight: Optional[float] = None
    insights_generated: int = 0
    tool_calls_made: int = 0
    execution_duration: float = 0.0
    
    # Quality metrics
    avg_insight_quality: float = 0.0
    research_depth_score: float = 0.0
    novelty_score: float = 0.0
    
    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    api_calls_count: int = 0
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Agent progression
    agent_completion_times: List[float] = Field(default_factory=list)
    agent_insights_count: List[int] = Field(default_factory=list)
    agent_tool_calls: List[int] = Field(default_factory=list)
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.status == "completed":
            return 100.0
        elif self.status == "failed":
            return 0.0
        else:
            return (self.current_agent_position / self.total_agents) * 100.0
    
    @property
    def estimated_completion_time(self) -> Optional[datetime]:
        """Estimate completion time based on current progress."""
        if self.current_agent_position == 0 or self.status == "completed":
            return None
        
        if not self.agent_completion_times:
            return None
        
        avg_time_per_agent = sum(self.agent_completion_times) / len(self.agent_completion_times)
        remaining_agents = self.total_agents - self.current_agent_position
        estimated_remaining_seconds = avg_time_per_agent * remaining_agents
        
        from datetime import timedelta
        return datetime.utcnow() + timedelta(seconds=estimated_remaining_seconds)


class ParallelMetrics(BaseModel):
    """Aggregated metrics across all parallel sequences."""
    
    execution_id: str
    start_time: datetime
    sequence_count: int
    
    # Comparative metrics
    best_strategy: Optional[str] = None
    best_productivity_score: float = 0.0
    productivity_variance: float = 0.0
    significant_difference_detected: bool = False
    
    # Aggregate performance
    total_insights_generated: int = 0
    total_tool_calls: int = 0
    average_research_quality: float = 0.0
    average_agent_efficiency: float = 0.0
    
    # Resource metrics
    peak_memory_usage: float = 0.0
    average_cpu_usage: float = 0.0
    total_api_calls: int = 0
    
    # Completion tracking
    completed_sequences: int = 0
    failed_sequences: int = 0
    active_sequences: int = 0
    
    # Performance rankings
    strategy_rankings: Dict[str, float] = Field(default_factory=dict)
    
    # Timeline metrics
    first_completion_time: Optional[datetime] = None
    last_completion_time: Optional[datetime] = None
    
    @property
    def completion_rate(self) -> float:
        """Calculate completion rate percentage."""
        total = self.completed_sequences + self.failed_sequences + self.active_sequences
        return (self.completed_sequences / total * 100) if total > 0 else 0.0
    
    @property
    def overall_tool_productivity(self) -> float:
        """Calculate overall tool productivity."""
        if self.total_tool_calls == 0:
            return 0.0
        return self.average_research_quality / self.total_tool_calls * self.total_insights_generated
    
    @property
    def execution_duration(self) -> float:
        """Get current execution duration in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()


class WinnerAnalysis(BaseModel):
    """Analysis of the winning sequence strategy."""
    
    winning_strategy: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    productivity_advantage: float = Field(ge=0.0)  # Percentage advantage
    
    # Comparison data
    all_strategy_scores: Dict[str, float]
    variance_threshold_exceeded: bool
    statistical_significance: float = Field(ge=0.0, le=1.0)
    
    # Performance characteristics
    winner_strengths: List[str] = Field(default_factory=list)
    comparative_advantages: Dict[str, float] = Field(default_factory=dict)
    
    # Quality analysis
    unique_insights_count: int = Field(ge=0)
    quality_superiority: float = Field(ge=0.0)
    efficiency_advantage: float = Field(ge=0.0)
    
    # Detection metadata
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    detection_trigger: str = "variance_threshold"  # variance_threshold, completion, manual


class MetricsUpdate(BaseModel):
    """Real-time metrics update message for streaming."""
    
    update_id: str = Field(default_factory=lambda: str(uuid4()))
    update_type: MetricsUpdateType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_id: str
    
    # Update data
    sequence_metrics: Optional[Dict[str, SequenceMetrics]] = None
    parallel_metrics: Optional[ParallelMetrics] = None
    winner_analysis: Optional[WinnerAnalysis] = None
    
    # Specific update data
    updated_strategy: Optional[str] = None
    metric_deltas: Dict[str, float] = Field(default_factory=dict)
    
    # Context
    message: Optional[str] = None
    alert_level: str = "info"  # info, warning, error, success
    
    def to_websocket_message(self) -> Dict[str, Any]:
        """Convert to WebSocket message format."""
        return {
            "update_id": self.update_id,
            "type": self.update_type.value,
            "timestamp": self.timestamp.isoformat(),
            "execution_id": self.execution_id,
            "sequence_metrics": {
                str(k): v.model_dump() if hasattr(v, 'model_dump') else v.__dict__
                for k, v in (self.sequence_metrics or {}).items()
            },
            "parallel_metrics": self.parallel_metrics.model_dump() if self.parallel_metrics else None,
            "winner_analysis": self.winner_analysis.model_dump() if self.winner_analysis else None,
            "updated_strategy": self.updated_strategy if self.updated_strategy else None,
            "metric_deltas": self.metric_deltas,
            "message": self.message,
            "alert_level": self.alert_level
        }


# Standard sequence patterns for backward compatibility
THEORY_FIRST_PATTERN = SequencePattern(
    agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
    description="Theory First: Academic → Industry → Technical (strong theoretical foundation)",
    expected_advantages=[
        "Strong theoretical foundation",
        "Academic rigor applied to market analysis",
        "Technical implementation builds on solid theory"
    ],
    confidence_score=1.0,
    reasoning="Academic insights provide theoretical foundation, then industry validates market relevance, finally technical trends assess implementation feasibility"
)

MARKET_FIRST_PATTERN = SequencePattern(
    agent_order=[AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS],
    description="Market First: Industry → Academic → Technical (market-driven approach)",
    expected_advantages=[
        "Market-driven insights",
        "Industry needs guide academic research",
        "Technical implementation focuses on market requirements"
    ],
    confidence_score=1.0,
    reasoning="Industry insights identify market opportunities, academic research validates approaches, technical trends assess implementation paths"
)

FUTURE_BACK_PATTERN = SequencePattern(
    agent_order=[AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY],
    description="Future Back: Technical → Academic → Industry (future-oriented perspective)",
    expected_advantages=[
        "Future-oriented perspective",
        "Technical trends drive academic research direction",
        "Industry analysis focuses on emerging opportunities"
    ],
    confidence_score=1.0,
    reasoning="Technical trends identify future directions, academic research provides theoretical backing, industry analysis evaluates market readiness"
)

# Add strategy attribute to patterns for compatibility
THEORY_FIRST_PATTERN.strategy = "theory_first"
MARKET_FIRST_PATTERN.strategy = "market_first"
FUTURE_BACK_PATTERN.strategy = "future_back"

# Mapping of strategies to patterns
SEQUENCE_PATTERNS = {
    "theory_first": THEORY_FIRST_PATTERN,
    "market_first": MARKET_FIRST_PATTERN,
    "future_back": FUTURE_BACK_PATTERN
}