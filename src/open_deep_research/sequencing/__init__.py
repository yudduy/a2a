"""Sequential Agent Ordering Optimization Framework for OpenDeepResearch.

This module implements a comprehensive framework for proving that different sequential
orderings of specialized agents produce measurably different productivity outcomes
in research tasks.

Key Components:
- SequenceOptimizationEngine: Core engine for executing and comparing sequences
- SupervisorResearchDirector: Dynamic question generation and insight tracking
- Specialized Agents: Academic, Industry, and Technical Trends agents
- MetricsCalculator: Tool productivity metrics and comparative analysis
- Models: Data structures for sequence patterns and results

Core Metrics:
- Tool Productivity (TP) = Research Quality / Agent Calls
- Agent Efficiency = Useful Insights / Total Agent Calls
- Context Efficiency = Relevant Context Used / Total Context Available
- Time to Value = Time to first significant insight

The framework executes three strategic sequence patterns:
1. Theory First: Academic → Industry → Technical (strong theoretical foundation)
2. Market First: Industry → Academic → Technical (market-driven approach)
3. Future Back: Technical → Academic → Industry (future-oriented perspective)

Usage:
    from open_deep_research.sequencing import SequenceOptimizationEngine
    
    engine = SequenceOptimizationEngine(config)
    comparison = await engine.compare_sequences("AI safety research")
    print(f"Productivity variance: {comparison.productivity_variance:.3f}")
"""

from .models import (
    AgentType,
    AgentExecutionResult,
    QueryType,
    ResearchDomain,
    ScopeBreadth,
    SequenceAnalysis,
    SequencePattern,
    SequenceResult,
    SequenceStrategy,
    ToolProductivityMetrics,
    InsightTransition,
    SEQUENCE_PATTERNS,
    THEORY_FIRST_PATTERN,
    MARKET_FIRST_PATTERN,
    FUTURE_BACK_PATTERN
)
from .research_director import SupervisorResearchDirector
from .sequence_engine import SequenceOptimizationEngine
from .sequence_selector import SequenceAnalyzer
from .metrics import MetricsCalculator
from .specialized_agents import (
    AcademicAgent,
    IndustryAgent,
    TechnicalTrendsAgent,
    SpecializedAgent,
    ResearchContext
)
from .parallel_executor import (
    ParallelSequenceExecutor,
    ParallelExecutionResult,
    ParallelExecutionProgress,
    StreamMessage,
    ResourceMonitor,
    parallel_executor_context
)
from .stream_multiplexer import (
    StreamMultiplexer,
    StreamSubscription,
    WebSocketConnection,
    ConnectionState,
    DeliveryGuarantee,
    MessageBuffer,
    create_stream_multiplexer
)
from .langgraph_wrapper import (
    DelegationEvent,
    DelegationState,
    ParallelDelegationState,
    parallel_execution_graph,
    get_stream_stats,
    cleanup_streams
)

__all__ = [
    # Core Engine
    "SequenceOptimizationEngine",
    
    # Analysis and Selection
    "SequenceAnalyzer",
    
    # Models and Data Structures
    "AgentType",
    "AgentExecutionResult",
    "QueryType",
    "ResearchDomain",
    "ScopeBreadth",
    "SequenceAnalysis",
    "SequencePattern", 
    "SequenceResult",
    "SequenceStrategy",
    "ToolProductivityMetrics",
    "InsightTransition",
    "SEQUENCE_PATTERNS",
    "THEORY_FIRST_PATTERN",
    "MARKET_FIRST_PATTERN", 
    "FUTURE_BACK_PATTERN",
    
    # Director and Metrics
    "SupervisorResearchDirector",
    "MetricsCalculator",
    
    # Specialized Agents
    "AcademicAgent",
    "IndustryAgent",
    "TechnicalTrendsAgent",
    "SpecializedAgent",
    "ResearchContext",
    
    # Parallel Execution
    "ParallelSequenceExecutor",
    "ParallelExecutionResult",
    "ParallelExecutionProgress",
    "StreamMessage",
    "ResourceMonitor",
    "parallel_executor_context",
    
    # Stream Multiplexing
    "StreamMultiplexer",
    "StreamSubscription", 
    "WebSocketConnection",
    "ConnectionState",
    "DeliveryGuarantee",
    "MessageBuffer",
    "create_stream_multiplexer",
    
    # LangGraph Integration
    "DelegationEvent",
    "DelegationState",
    "ParallelDelegationState",
    "parallel_execution_graph",
    "get_stream_stats",
    "cleanup_streams"
]

# Version information
__version__ = "1.0.0"
__author__ = "OpenDeepResearch Team"
__description__ = "Sequential Agent Ordering Optimization Framework"