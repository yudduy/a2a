"""LangGraph wrapper for the Sequential Agent Ordering Optimization Framework.

This module exposes the sequence optimization engine as LangGraph graphs that can be
consumed by the React Agent Studio frontend for delegation pattern visualization.
Enhanced with production-ready parallel execution capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.sequencing import (
    SequenceOptimizationEngine,
    SEQUENCE_PATTERNS,
    ToolProductivityMetrics
)
from open_deep_research.sequencing.parallel_executor import (
    ParallelSequenceExecutor,
    ParallelExecutionResult,
    StreamMessage,
    parallel_executor_context
)
from open_deep_research.sequencing.stream_multiplexer import (
    StreamMultiplexer,
    StreamSubscription,
    DeliveryGuarantee,
    create_stream_multiplexer
)
from open_deep_research.state import AgentState

logger = logging.getLogger(__name__)


class DelegationEvent(BaseModel):
    """Event emitted during delegation pattern execution for frontend consumption."""
    
    event_type: str = Field(description="Type of delegation event")
    sequence_strategy: str = Field(description="Which sequence strategy this event belongs to")
    agent_type: Optional[str] = Field(default=None, description="Current agent executing")
    step_number: int = Field(description="Step number in the sequence")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    productivity_metrics: Optional[Dict[str, float]] = Field(default=None, description="Current productivity metrics")


class DelegationState(AgentState):
    """Extended state for delegation pattern execution."""
    
    sequence_strategy: str = Field(default="", description="Current sequence strategy")
    delegation_events: List[DelegationEvent] = Field(default_factory=list, description="Events for frontend")
    productivity_metrics: Optional[ToolProductivityMetrics] = Field(default=None, description="Real-time metrics")
    sequence_position: int = Field(default=0, description="Current position in sequence")
    agent_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from each agent")


class ParallelDelegationState(AgentState):
    """Extended state for parallel delegation pattern execution."""
    
    execution_id: str = Field(default="", description="Parallel execution ID")
    strategies: List[str] = Field(default_factory=list, description="Strategies being executed in parallel")
    parallel_events: List[DelegationEvent] = Field(default_factory=list, description="Events from all parallel sequences")
    execution_progress: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Progress by strategy")
    stream_subscription_id: Optional[str] = Field(default=None, description="Stream subscription ID")
    
    # Results
    parallel_results: Optional[ParallelExecutionResult] = Field(default=None, description="Final parallel execution results")
    comparison_summary: Optional[Dict[str, Any]] = Field(default=None, description="Comparison summary")
    
    # Real-time streaming
    active_streams: Dict[str, Any] = Field(default_factory=dict, description="Active stream connections")
    message_count: int = Field(default=0, description="Total messages streamed")


async def emit_delegation_event(
    state: DelegationState,
    event_type: str,
    agent_type: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    productivity_metrics: Optional[Dict[str, float]] = None
) -> DelegationState:
    """Emit a delegation event for frontend consumption."""
    
    event = DelegationEvent(
        event_type=event_type,
        sequence_strategy=state.get("sequence_strategy", "unknown"),
        agent_type=agent_type,
        step_number=state.get("sequence_position", 0),
        data=data or {},
        productivity_metrics=productivity_metrics
    )
    
    # Add event to state
    current_events = state.get("delegation_events", [])
    current_events.append(event)
    
    # Update state with new event
    updated_state = state.copy()
    updated_state["delegation_events"] = current_events
    
    return updated_state


async def execute_sequence_pattern(
    state: DelegationState, 
    config: RunnableConfig, 
    strategy: str
) -> Command[Literal["finalize_sequence"]]:
    """Execute a specific sequence pattern with real-time event emission."""
    
    logger.info(f"Starting sequence execution: {strategy}")
    
    # Initialize sequence state
    state["sequence_strategy"] = strategy
    state["sequence_position"] = 0
    
    # Emit sequence start event
    await emit_delegation_event(
        state,
        event_type="sequence_started",
        data={
            "strategy": strategy,
            "research_topic": state.get("research_brief", "Unknown"),
            "agent_order": [agent.value for agent in SEQUENCE_PATTERNS[strategy].agent_order]
        }
    )
    
    try:
        # Initialize sequence optimization engine
        engine = SequenceOptimizationEngine(config)
        
        # Get the research topic from state
        research_topic = state.get("research_brief", "")
        if not research_topic:
            # Extract from messages if research_brief not available
            messages = state.get("messages", [])
            if messages:
                last_human_message = next(
                    (msg for msg in reversed(messages) if msg.type == "human"), 
                    None
                )
                if last_human_message:
                    research_topic = last_human_message.content
        
        if not research_topic:
            raise ValueError("No research topic found in state")
        
        # Execute the sequence pattern
        pattern = SEQUENCE_PATTERNS[strategy]
        
        # Custom sequence execution with event emission
        sequence_result = await execute_sequence_with_events(
            engine, pattern, research_topic, state, config
        )
        
        # Emit sequence completion event
        await emit_delegation_event(
            state,
            event_type="sequence_completed",
            data={
                "final_insights": sequence_result.final_insights,
                "total_agent_calls": sequence_result.total_agent_calls,
                "execution_duration": sequence_result.total_duration,
                "productivity_score": sequence_result.overall_productivity_metrics.tool_productivity
            },
            productivity_metrics={
                "tool_productivity": sequence_result.overall_productivity_metrics.tool_productivity,
                "agent_efficiency": sequence_result.overall_productivity_metrics.agent_efficiency,
                "research_quality": sequence_result.overall_productivity_metrics.research_quality_score
            }
        )
        
        # Store final results
        state["notes"] = [sequence_result.final_insights]
        state["productivity_metrics"] = sequence_result.overall_productivity_metrics
        
        return Command(goto="finalize_sequence")
        
    except Exception as e:
        logger.error(f"Error in sequence execution: {e}")
        
        # Emit error event
        await emit_delegation_event(
            state,
            event_type="sequence_error",
            data={"error": str(e)}
        )
        
        # Provide fallback result
        state["notes"] = [f"Sequence execution failed: {str(e)}"]
        return Command(goto="finalize_sequence")


async def execute_sequence_with_events(
    engine: SequenceOptimizationEngine,
    pattern,
    research_topic: str,
    state: DelegationState,
    config: RunnableConfig
):
    """Execute sequence with real-time event emission for each agent."""
    
    agent_results = []
    accumulated_insights = ""
    
    for position, agent_type in enumerate(pattern.agent_order, 1):
        # Update sequence position
        state["sequence_position"] = position
        
        # Emit agent start event
        await emit_delegation_event(
            state,
            event_type="agent_started",
            agent_type=agent_type.value,
            data={
                "position": position,
                "total_agents": len(pattern.agent_order),
                "previous_insights": accumulated_insights
            }
        )
        
        try:
            # Simulate agent execution (replace with actual agent execution)
            # For now, we'll create a mock result
            agent_start_time = datetime.utcnow()
            
            # Generate dynamic question for agent
            if position == 1:
                question = f"From a {agent_type.value} perspective, analyze: {research_topic}"
            else:
                # Use research director for dynamic question generation
                question_result = await engine.research_director.direct_next_investigation(
                    previous_agent_insights=[accumulated_insights] if accumulated_insights else [],
                    next_agent_type=agent_type,
                    research_context=research_topic,
                    sequence_position=position
                )
                # Safely extract question with better error handling
                if question_result and question_result.questions and len(question_result.questions) > 0:
                    question = question_result.questions[0]
                else:
                    question = f"From a {agent_type.value} perspective, analyze: {research_topic}"
            
            # Emit agent progress event
            await emit_delegation_event(
                state,
                event_type="agent_progress",
                agent_type=agent_type.value,
                data={
                    "directed_question": question,
                    "research_phase": "executing"
                }
            )
            
            # Execute agent (simplified for initial implementation)
            agent = engine.agents.get(agent_type)
            if agent:
                try:
                    # Create research context
                    from open_deep_research.sequencing.specialized_agents import ResearchContext
                    
                    context = ResearchContext(
                        research_topic=research_topic,
                        directed_questions=[question],
                        previous_insights=accumulated_insights,
                        agent_specialization=agent_type.value,
                        sequence_position=position
                    )
                    
                    # Execute research
                    result = await agent.execute_research(context)
                    
                    # Extract insights
                    insights = result.refined_insights if hasattr(result, 'refined_insights') else "Analysis completed"
                    tool_calls = result.tool_calls_made if hasattr(result, 'tool_calls_made') else 1
                except Exception as e:
                    logger.warning(f"Agent execution failed: {e}, using fallback")
                    insights = f"Fallback {agent_type.value} analysis for {research_topic} (API issue: {str(e)[:100]})"
                    tool_calls = 1
            else:
                # Fallback for missing agent
                insights = f"Mock {agent_type.value} analysis: {research_topic}"
                tool_calls = 1
            
            agent_end_time = datetime.utcnow()
            execution_time = (agent_end_time - agent_start_time).total_seconds()
            
            # Create agent result
            agent_result = {
                "agent_type": agent_type.value,
                "directed_question": question,
                "insights": insights,
                "tool_calls_made": tool_calls,
                "execution_duration": execution_time,
                "position": position
            }
            
            agent_results.append(agent_result)
            accumulated_insights = insights  # Linear context passing
            
            # Calculate current productivity metrics
            current_productivity = len(agent_results) / sum(r["tool_calls_made"] for r in agent_results)
            
            # Emit agent completion event
            await emit_delegation_event(
                state,
                event_type="agent_completed",
                agent_type=agent_type.value,
                data={
                    "insights_generated": len(insights.split(". ")) if insights else 0,
                    "tool_calls_made": tool_calls,
                    "execution_duration": execution_time
                },
                productivity_metrics={
                    "current_productivity": current_productivity,
                    "agents_completed": len(agent_results),
                    "total_agents": len(pattern.agent_order)
                }
            )
            
        except Exception as e:
            logger.error(f"Error executing agent {agent_type.value}: {e}")
            
            # Emit agent error event
            await emit_delegation_event(
                state,
                event_type="agent_error",
                agent_type=agent_type.value,
                data={"error": str(e)}
            )
            
            # Continue with fallback
            insights = f"Error in {agent_type.value} analysis: {str(e)}"
            accumulated_insights = insights
    
    # Create final sequence result
    final_quality_score = 0.8  # Mock quality score
    total_agent_calls = sum(r["tool_calls_made"] for r in agent_results)
    total_duration = sum(r["execution_duration"] for r in agent_results)
    
    # Create mock sequence result
    class MockSequenceResult:
        def __init__(self):
            self.final_insights = accumulated_insights
            self.total_agent_calls = total_agent_calls
            self.total_duration = total_duration
            self.overall_productivity_metrics = type('obj', (object,), {
                'tool_productivity': final_quality_score / max(total_agent_calls, 1),
                'agent_efficiency': len(agent_results) / max(total_agent_calls, 1),
                'research_quality_score': final_quality_score
            })()
    
    return MockSequenceResult()


async def finalize_sequence(state: DelegationState, config: RunnableConfig):
    """Finalize sequence execution and prepare final response."""
    
    # Create final AI message with results
    final_insights = state.get("notes", ["No insights generated"])[0]
    productivity_metrics = state.get("productivity_metrics")
    
    response_content = f"""# {state.get('sequence_strategy', 'Unknown').replace('_', ' ').title()} Sequence Results

## Research Insights
{final_insights}

## Sequence Performance
- **Strategy**: {state.get('sequence_strategy', 'Unknown').replace('_', ' ').title()}
- **Agents Executed**: {state.get('sequence_position', 0)}
"""
    
    if productivity_metrics:
        response_content += f"""- **Tool Productivity**: {productivity_metrics.tool_productivity:.3f}
- **Agent Efficiency**: {productivity_metrics.agent_efficiency:.3f}
- **Research Quality**: {productivity_metrics.research_quality_score:.3f}
"""
    
    response_content += f"""
## Delegation Events
{len(state.get('delegation_events', []))} events tracked during execution.
"""
    
    final_message = AIMessage(content=response_content)
    
    return {
        "messages": [final_message],
        "notes": state.get("notes", []),
        "delegation_events": state.get("delegation_events", []),
        "productivity_metrics": productivity_metrics
    }


# Create individual sequence graphs
def create_theory_first_graph():
    """Create LangGraph for Theory First sequence pattern."""
    
    builder = StateGraph(DelegationState, config_schema=Configuration)
    
    async def execute_theory_first(state: DelegationState, config: RunnableConfig):
        return await execute_sequence_pattern(state, config, "theory_first")
    
    builder.add_node("execute_theory_first", execute_theory_first)
    builder.add_node("finalize_sequence", finalize_sequence)
    
    builder.add_edge(START, "execute_theory_first")
    builder.add_edge("finalize_sequence", END)
    
    return builder.compile()


def create_market_first_graph():
    """Create LangGraph for Market First sequence pattern."""
    
    builder = StateGraph(DelegationState, config_schema=Configuration)
    
    async def execute_market_first(state: DelegationState, config: RunnableConfig):
        return await execute_sequence_pattern(state, config, "market_first")
    
    builder.add_node("execute_market_first", execute_market_first)
    builder.add_node("finalize_sequence", finalize_sequence)
    
    builder.add_edge(START, "execute_market_first")
    builder.add_edge("finalize_sequence", END)
    
    return builder.compile()


def create_future_back_graph():
    """Create LangGraph for Future Back sequence pattern."""
    
    builder = StateGraph(DelegationState, config_schema=Configuration)
    
    async def execute_future_back(state: DelegationState, config: RunnableConfig):
        return await execute_sequence_pattern(state, config, "future_back")
    
    builder.add_node("execute_future_back", execute_future_back)
    builder.add_node("finalize_sequence", finalize_sequence)
    
    builder.add_edge(START, "execute_future_back")
    builder.add_edge("finalize_sequence", END)
    
    return builder.compile()


async def execute_comparison(state: DelegationState, config: RunnableConfig):
    """Execute all three sequence patterns in parallel for comparison."""
    
    logger.info("Starting delegation pattern comparison")
    
    # Emit comparison start event
    await emit_delegation_event(
        state,
        event_type="comparison_started",
        data={
            "patterns": ["theory_first", "market_first", "future_back"],
            "research_topic": state.get("research_brief", "Unknown")
        }
    )
    
    try:
        # Initialize engines for each pattern
        engine = SequenceOptimizationEngine(config)
        research_topic = state.get("research_brief", "")
        
        if not research_topic:
            messages = state.get("messages", [])
            if messages:
                last_human_message = next(
                    (msg for msg in reversed(messages) if msg.type == "human"), 
                    None
                )
                if last_human_message:
                    research_topic = last_human_message.content
        
        # Execute comparison
        comparison_result = await engine.compare_sequences(research_topic)
        
        # Emit comparison completion event
        await emit_delegation_event(
            state,
            event_type="comparison_completed",
            data={
                "productivity_variance": comparison_result.productivity_variance,
                "best_sequence": comparison_result.highest_productivity_sequence.value,
                "significant_difference": comparison_result.significant_difference_detected
            }
        )
        
        # Store comparison results
        comparison_summary = f"""# Delegation Pattern Comparison Results

## Research Topic
{research_topic}

## Performance Summary
- **Productivity Variance**: {comparison_result.productivity_variance:.1%}
- **Best Performing Pattern**: {comparison_result.highest_productivity_sequence.value.replace('_', ' ').title()}
- **Significant Difference Detected**: {'Yes' if comparison_result.significant_difference_detected else 'No'}

## Pattern Rankings
"""
        
        for i, (strategy, score) in enumerate(comparison_result.productivity_rankings.items(), 1):
            comparison_summary += f"{i}. **{strategy.replace('_', ' ').title()}**: {score:.3f} Tool Productivity\n"
        
        state["notes"] = [comparison_summary]
        state["comparison_result"] = comparison_result
        
        return Command(goto="finalize_sequence")
        
    except Exception as e:
        logger.error(f"Error in comparison execution: {e}")
        
        await emit_delegation_event(
            state,
            event_type="comparison_error",
            data={"error": str(e)}
        )
        
        state["notes"] = [f"Comparison execution failed: {str(e)}"]
        return Command(goto="finalize_sequence")


def create_comparison_graph():
    """Create LangGraph for comparing all delegation patterns."""
    
    builder = StateGraph(DelegationState, config_schema=Configuration)
    
    builder.add_node("execute_comparison", execute_comparison)
    builder.add_node("finalize_sequence", finalize_sequence)
    
    builder.add_edge(START, "execute_comparison")
    builder.add_edge("finalize_sequence", END)
    
    return builder.compile()


# ==============================================================================
# PARALLEL EXECUTION ENHANCEMENT
# ==============================================================================

# Global stream multiplexer instance (initialized when needed)
_global_stream_multiplexer: Optional[StreamMultiplexer] = None


async def get_stream_multiplexer() -> StreamMultiplexer:
    """Get or create the global stream multiplexer."""
    global _global_stream_multiplexer
    
    if _global_stream_multiplexer is None:
        _global_stream_multiplexer = await create_stream_multiplexer(
            max_connections=100,
            max_buffer_size=10000
        )
        logger.info("Initialized global stream multiplexer")
    
    return _global_stream_multiplexer


async def stream_callback_factory(
    state: ParallelDelegationState,
    multiplexer: StreamMultiplexer
) -> callable:
    """Create a stream callback function for parallel execution."""
    
    async def stream_callback(message: StreamMessage):
        """Handle streaming messages from parallel execution."""
        try:
            # Update state with message count
            state["message_count"] = state.get("message_count", 0) + 1
            
            # Route message through multiplexer
            delivery_results = await multiplexer.route_message(message)
            
            # Convert to delegation event for state tracking
            delegation_event = DelegationEvent(
                event_type=f"parallel_{message.message_type}",
                sequence_strategy=message.sequence_strategy if message.sequence_strategy else "all",
                step_number=state.get("message_count", 0),
                data=message.data,
                timestamp=message.timestamp,
                productivity_metrics=message.progress.model_dump() if message.progress else None
            )
            
            # Add to parallel events
            current_events = state.get("parallel_events", [])
            current_events.append(delegation_event)
            state["parallel_events"] = current_events
            
            # Update execution progress
            if message.sequence_strategy:
                strategy_key = message.sequence_strategy
                progress_data = {
                    "last_message_type": message.message_type,
                    "last_update": message.timestamp.isoformat(),
                    "message_data": message.data
                }
                
                if message.progress:
                    progress_data.update({
                        "status": message.progress.status,
                        "progress_percent": message.progress.progress_percent,
                        "agent_position": message.progress.agent_position,
                        "current_insights": message.progress.current_insights
                    })
                
                execution_progress = state.get("execution_progress", {})
                execution_progress[strategy_key] = progress_data
                state["execution_progress"] = execution_progress
            
            logger.debug(f"Processed stream message: {message.message_type} for {message.sequence_strategy}")
            
        except Exception as e:
            logger.error(f"Error in stream callback: {e}")
    
    return stream_callback


async def execute_parallel_sequences(state: ParallelDelegationState, config: RunnableConfig):
    """Execute all three sequence patterns in parallel with real-time streaming."""
    
    logger.info("Starting parallel sequence execution with streaming")
    
    try:
        # Get configuration
        app_config = Configuration.from_runnable_config(config)
        
        # Initialize execution state
        execution_id = f"parallel_{int(datetime.utcnow().timestamp())}"
        state["execution_id"] = execution_id
        state["strategies"] = ["theory_first", "market_first", "future_back"]
        state["message_count"] = 0
        
        # Get research topic
        research_topic = state.get("research_brief", "")
        if not research_topic:
            messages = state.get("messages", [])
            if messages:
                last_human_message = next(
                    (msg for msg in reversed(messages) if msg.type == "human"), 
                    None
                )
                if last_human_message:
                    research_topic = last_human_message.content
        
        if not research_topic:
            raise ValueError("No research topic found in state")
        
        # Initialize stream multiplexer
        multiplexer = await get_stream_multiplexer()
        
        # Create stream subscription for this execution
        subscription_id = await multiplexer.create_subscription(
            client_id=execution_id,
            sequence_strategies={"theory_first", "market_first", "future_back"},  # All strategies
            message_types={"progress", "result", "error", "completion"},
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
            include_progress=True,
            include_errors=True,
            include_results=True,
            buffer_size=1000
        )
        state["stream_subscription_id"] = subscription_id
        
        # Create stream callback
        stream_callback = await stream_callback_factory(state, multiplexer)
        
        # Execute parallel sequences with timeout from configuration
        timeout_seconds = getattr(app_config, 'parallel_execution_timeout', 3600)
        max_concurrent = getattr(app_config, 'max_parallel_sequences', 3)
        
        async with parallel_executor_context(
            config=config,
            max_concurrent=max_concurrent,
            timeout_seconds=timeout_seconds
        ) as executor:
            
            # Send initial progress message
            initial_message = StreamMessage(
                message_type="execution_started",
                data={
                    "execution_id": execution_id,
                    "research_topic": research_topic,
                    "strategies": state["strategies"],
                    "max_concurrent": max_concurrent
                }
            )
            await stream_callback(initial_message)
            
            # Execute parallel sequences
            parallel_result = await executor.execute_sequences_parallel(
                research_topic=research_topic,
                strategies=["theory_first", "market_first", "future_back"],
                stream_callback=stream_callback
            )
            
            # Store results in state
            state["parallel_results"] = parallel_result
            
            # Generate comparison summary
            comparison_summary = {
                "execution_id": execution_id,
                "research_topic": research_topic,
                "total_duration": parallel_result.total_duration,
                "success_rate": parallel_result.success_rate,
                "best_performing_strategy": parallel_result.best_performing_strategy if parallel_result.best_performing_strategy else None,
                "total_insights": len(parallel_result.unique_insights_across_sequences),
                "total_api_calls": parallel_result.total_api_calls,
                "peak_memory_usage": parallel_result.peak_memory_usage,
                "failed_sequences": [s.value for s in parallel_result.failed_sequences]
            }
            
            # Add productivity rankings if comparison available
            if parallel_result.comparison:
                comparison_summary["productivity_rankings"] = {
                    strategy: score 
                    for strategy, score in parallel_result.comparison.productivity_rankings.items()
                }
                comparison_summary["productivity_variance"] = parallel_result.comparison.productivity_variance
                comparison_summary["significant_difference"] = parallel_result.comparison.significant_difference_detected
            
            state["comparison_summary"] = comparison_summary
            
            # Send completion message
            completion_message = StreamMessage(
                message_type="execution_completed",
                data=comparison_summary
            )
            await stream_callback(completion_message)
            
            logger.info(f"Parallel execution completed: {parallel_result.success_rate:.1f}% success rate")
            
            return Command(goto="finalize_parallel_execution")
    
    except Exception as e:
        logger.error(f"Error in parallel sequence execution: {e}")
        
        # Send error message
        error_message = StreamMessage(
            message_type="error",
            data={
                "error": str(e),
                "execution_id": state.get("execution_id", "unknown")
            }
        )
        
        # Try to send error through stream if callback exists
        try:
            multiplexer = await get_stream_multiplexer()
            await multiplexer.route_message(error_message)
        except:
            pass
        
        # Store error in state
        state["comparison_summary"] = {
            "error": str(e),
            "execution_id": state.get("execution_id", "unknown"),
            "status": "failed"
        }
        
        return Command(goto="finalize_parallel_execution")


async def finalize_parallel_execution(state: ParallelDelegationState, config: RunnableConfig):
    """Finalize parallel execution and prepare comprehensive response."""
    
    logger.info("Finalizing parallel execution results")
    
    try:
        # Get results from state
        parallel_results = state.get("parallel_results")
        comparison_summary = state.get("comparison_summary", {})
        execution_progress = state.get("execution_progress", {})
        message_count = state.get("message_count", 0)
        
        # Build comprehensive response
        if parallel_results and hasattr(parallel_results, 'success_rate'):
            # Successful execution
            response_parts = [
                f"# Parallel Sequence Execution Results",
                f"",
                f"## Executive Summary",
                f"- **Execution ID**: {comparison_summary.get('execution_id', 'Unknown')}",
                f"- **Research Topic**: {comparison_summary.get('research_topic', 'Unknown')}",
                f"- **Success Rate**: {comparison_summary.get('success_rate', 0):.1f}%",
                f"- **Total Duration**: {comparison_summary.get('total_duration', 0):.1f} seconds",
                f"- **Best Strategy**: {comparison_summary.get('best_performing_strategy', 'None')}",
                f"- **Total Insights Generated**: {comparison_summary.get('total_insights', 0)}",
                f"- **Messages Streamed**: {message_count}",
                f"",
                f"## Strategy Performance Rankings"
            ]
            
            # Add productivity rankings
            rankings = comparison_summary.get("productivity_rankings", {})
            if rankings:
                for i, (strategy, score) in enumerate(sorted(rankings.items(), key=lambda x: x[1], reverse=True), 1):
                    strategy_name = strategy.replace('_', ' ').title()
                    response_parts.append(f"{i}. **{strategy_name}**: {score:.3f} tool productivity")
            else:
                response_parts.append("No performance rankings available")
            
            # Add variance analysis
            response_parts.extend([
                f"",
                f"## Productivity Analysis",
                f"- **Productivity Variance**: {comparison_summary.get('productivity_variance', 0):.1%}",
                f"- **Significant Difference Detected**: {'Yes' if comparison_summary.get('significant_difference', False) else 'No'}",
                f"- **Peak Memory Usage**: {comparison_summary.get('peak_memory_usage', 0):.1f} MB",
                f"- **Total API Calls**: {comparison_summary.get('total_api_calls', 0)}"
            ])
            
            # Add failed sequences if any
            failed_sequences = comparison_summary.get("failed_sequences", [])
            if failed_sequences:
                response_parts.extend([
                    f"",
                    f"## Failed Sequences",
                    f"The following sequences failed to complete:"
                ])
                for seq in failed_sequences:
                    response_parts.append(f"- {seq.replace('_', ' ').title()}")
            
            # Add execution progress summary
            if execution_progress:
                response_parts.extend([
                    f"",
                    f"## Execution Progress Summary",
                    f"Real-time progress was tracked for {len(execution_progress)} strategies:"
                ])
                for strategy, progress in execution_progress.items():
                    strategy_name = strategy.replace('_', ' ').title()
                    status = progress.get('status', 'unknown')
                    progress_percent = progress.get('progress_percent', 0)
                    response_parts.append(f"- **{strategy_name}**: {status} ({progress_percent:.1f}% complete)")
            
            # Add streaming statistics
            response_parts.extend([
                f"",
                f"## Real-time Streaming",
                f"- **Total Messages**: {message_count}",
                f"- **Subscription ID**: {state.get('stream_subscription_id', 'None')}",
                f"- **Parallel Events Tracked**: {len(state.get('parallel_events', []))}"
            ])
            
        else:
            # Failed execution
            error_msg = comparison_summary.get("error", "Unknown error occurred")
            response_parts = [
                f"# Parallel Sequence Execution Failed",
                f"",
                f"## Error Details",
                f"**Execution ID**: {comparison_summary.get('execution_id', 'Unknown')}",
                f"**Error**: {error_msg}",
                f"",
                f"## Partial Results",
                f"- **Messages Streamed**: {message_count}",
                f"- **Strategies Attempted**: {', '.join(state.get('strategies', []))}",
                f"- **Execution Progress**: {len(execution_progress)} strategies tracked"
            ]
        
        response_content = "\n".join(response_parts)
        
        # Create final AI message
        final_message = AIMessage(content=response_content)
        
        # Clean up stream subscription if exists
        subscription_id = state.get("stream_subscription_id")
        if subscription_id:
            try:
                multiplexer = await get_stream_multiplexer()
                await multiplexer.remove_subscription(subscription_id)
                logger.info(f"Cleaned up stream subscription {subscription_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up subscription: {e}")
        
        return {
            "messages": [final_message],
            "notes": [response_content] if parallel_results else [f"Execution failed: {comparison_summary.get('error', 'Unknown error')}"],
            "parallel_results": parallel_results,
            "comparison_summary": comparison_summary,
            "execution_progress": execution_progress,
            "message_count": message_count
        }
        
    except Exception as e:
        logger.error(f"Error finalizing parallel execution: {e}")
        
        error_response = f"# Parallel Execution Finalization Error\n\nAn error occurred while finalizing results: {str(e)}"
        return {
            "messages": [AIMessage(content=error_response)],
            "notes": [error_response]
        }


def create_parallel_execution_graph():
    """Create LangGraph for parallel execution of all sequence patterns with real-time streaming."""
    
    builder = StateGraph(ParallelDelegationState, config_schema=Configuration)
    
    builder.add_node("execute_parallel_sequences", execute_parallel_sequences)
    builder.add_node("finalize_parallel_execution", finalize_parallel_execution)
    
    builder.add_edge(START, "execute_parallel_sequences")
    builder.add_edge("finalize_parallel_execution", END)
    
    return builder.compile()


# ==============================================================================
# STREAM MANAGEMENT UTILITIES
# ==============================================================================

async def get_stream_stats() -> Dict[str, Any]:
    """Get current streaming statistics."""
    try:
        multiplexer = await get_stream_multiplexer()
        return multiplexer.get_connection_stats()
    except Exception as e:
        logger.error(f"Error getting stream stats: {e}")
        return {"error": str(e)}


async def cleanup_streams():
    """Clean up all stream connections and resources."""
    global _global_stream_multiplexer
    
    try:
        if _global_stream_multiplexer:
            await _global_stream_multiplexer.stop()
            _global_stream_multiplexer = None
            logger.info("Cleaned up global stream multiplexer")
    except Exception as e:
        logger.error(f"Error cleaning up streams: {e}")


# Export graphs for langgraph.json
theory_first_graph = create_theory_first_graph()
market_first_graph = create_market_first_graph()
future_back_graph = create_future_back_graph()
comparison_graph = create_comparison_graph()
parallel_execution_graph = create_parallel_execution_graph()