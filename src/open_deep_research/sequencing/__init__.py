"""Dynamic Sequential Agent Ordering Optimization Framework for OpenDeepResearch.

This module implements a comprehensive framework for dynamically generating and executing 
optimal sequences of specialized agents based on research topic analysis, proving that 
different orderings produce measurably different productivity outcomes.

Key Components:
- SequenceOptimizationEngine: Core engine for executing dynamic and static sequences
- SequenceAnalyzer: LLM-powered dynamic sequence generation based on topic analysis
- SupervisorResearchDirector: Dynamic question generation and insight tracking
- Specialized Agents: Academic, Industry, and Technical Trends agents
- MetricsCalculator: Tool productivity metrics and comparative analysis
- DynamicSequencePattern: Flexible sequence representation for any agent ordering

Core Metrics:
- Tool Productivity (TP) = Research Quality / Agent Calls
- Agent Efficiency = Useful Insights / Total Agent Calls
- Context Efficiency = Relevant Context Used / Total Context Available
- Time to Value = Time to first significant insight

Dynamic Sequence Generation:
The framework now intelligently generates optimal sequences based on:
- Research topic domain analysis (academic, market, technical)
- Query complexity and scope assessment
- Historical performance patterns
- LLM-powered reasoning for sequence optimization

Usage:
    from open_deep_research.sequencing import SequenceAnalyzer, SequenceOptimizationEngine
    
    analyzer = SequenceAnalyzer()
    sequences = analyzer.generate_dynamic_sequences("AI safety research")
    
    engine = SequenceOptimizationEngine(config)
    results = await engine.execute_sequences_parallel(sequences, "AI safety research")
"""

from .models import (
    AgentType,
    AgentExecutionResult,
    QueryType,
    ResearchDomain,
    ScopeBreadth,
    SequenceAnalysis,
    SequencePattern,
    DynamicSequencePattern,
    SequenceResult,
    ToolProductivityMetrics,
    InsightTransition,
    SEQUENCE_PATTERNS
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
# Legacy langgraph wrapper removed - now using LLM-generated sequences

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
    "DynamicSequencePattern",
    "SequenceResult",
    "ToolProductivityMetrics",
    "InsightTransition",
    "SEQUENCE_PATTERNS",
    
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
    
    # LangGraph Integration - Legacy removed
]

# Version information
__version__ = "1.0.0"
__author__ = "OpenDeepResearch Team"
__description__ = "Sequential Agent Ordering Optimization Framework"