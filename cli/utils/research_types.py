"""Research-specific type definitions for CLI system.

This module contains type definitions used across the research CLI system.
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


# Core research state types
class ResearchState(BaseModel):
    """Core research execution state."""

    query: str = Field(description="Research query being executed")
    status: str = Field(default="initializing", description="Current execution status")
    start_time: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Execution start time")
    end_time: Optional[str] = Field(default=None, description="Execution end time")

    # Sequence and execution data
    sequences: Optional[List[Dict[str, Any]]] = Field(default=None, description="Generated research sequences")
    sequence_count: int = Field(default=0, description="Number of sequences generated")
    execution_results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sequence execution results")
    failed_sequences: List[int] = Field(default_factory=list, description="Indices of failed sequences")
    successful_executions: int = Field(default=0, description="Number of successful executions")

    # Results and analysis
    synthesis: Optional[str] = Field(default=None, description="Research synthesis")
    total_papers: int = Field(default=0, description="Total papers found")
    total_insights: int = Field(default=0, description="Total insights generated")

    # Quality and evaluation
    quality_score: Optional[float] = Field(default=None, description="Research quality score")
    final_result: Optional[Any] = Field(default=None, description="Final research result")

    # Metadata
    query_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Query analysis results")
    trace_id: Optional[str] = Field(default=None, description="Execution trace identifier")
    completion_timestamp: Optional[str] = Field(default=None, description="Completion timestamp")

    # Streaming support
    stream_writer: Optional[Any] = Field(default=None, description="Stream writer function")


class StreamMessage(BaseModel):
    """Message for streaming execution updates."""

    message_id: str = Field(description="Unique message identifier")
    sequence_id: str = Field(description="Sequence identifier")
    message_type: str = Field(description="Message type (progress, result, error)")
    timestamp: int = Field(description="Message timestamp (milliseconds)")
    content: Any = Field(description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class RoutedMessage(BaseModel):
    """Message routed through the system."""

    message_id: str = Field(description="Unique message identifier")
    sequence_id: str = Field(description="Sequence identifier")
    agent_type: Optional[str] = Field(default=None, description="Agent type that generated message")
    current_agent: Optional[str] = Field(default=None, description="Current agent name")
    message_type: str = Field(description="Message type")
    timestamp: int = Field(description="Message timestamp")
    content: Any = Field(description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message metadata")


# Agent and sequence types
class AgentCapability(BaseModel):
    """Agent capability definition."""

    name: str = Field(description="Agent name")
    expertise_areas: List[str] = Field(default_factory=list, description="Areas of expertise")
    description: str = Field(description="Agent description")
    typical_use_cases: List[str] = Field(default_factory=list, description="Typical use cases")
    strength_summary: str = Field(default="", description="Summary of agent strengths")
    core_responsibilities: List[str] = Field(default_factory=list, description="Core responsibilities")
    completion_indicators: List[str] = Field(default_factory=list, description="Completion indicators")


class SequenceStrategy(BaseModel):
    """Research sequence strategy."""

    THEORY_FIRST = "theory_first"
    MARKET_FIRST = "market_first"
    FUTURE_BACK = "future_back"
    PARALLEL_ALL = "parallel_all"


class AgentType(BaseModel):
    """Agent type enumeration."""

    ACADEMIC = "academic"
    INDUSTRY = "industry"
    TECHNICAL_TRENDS = "technical_trends"
    MARKET_ANALYSIS = "market_analysis"
    SYNTHESIS = "synthesis"


class ConnectionState(BaseModel):
    """Connection state enumeration."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class DeliveryGuarantee(BaseModel):
    """Delivery guarantee enumeration."""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


# Metrics and monitoring types
class SequenceProgress(BaseModel):
    """Progress tracking for individual sequences."""

    sequence_id: str = Field(description="Sequence identifier")
    current_agent: Optional[str] = Field(default=None, description="Currently executing agent")
    agents_completed: int = Field(default=0, description="Number of completed agents")
    total_agents: int = Field(default=0, description="Total agents in sequence")
    completion_percentage: float = Field(default=0.0, description="Completion percentage (0-100)")
    estimated_time_remaining: Optional[int] = Field(default=None, description="Estimated remaining time (seconds)")
    last_activity: int = Field(description="Last activity timestamp")
    status: str = Field(default="initializing", description="Sequence status")


class SequenceMetrics(BaseModel):
    """Metrics for sequence execution."""

    sequence_id: str = Field(description="Sequence identifier")
    message_count: int = Field(default=0, description="Total messages processed")
    research_duration: float = Field(default=0.0, description="Research duration (seconds)")
    tokens_used: int = Field(default=0, description="Total tokens consumed")
    average_response_time: float = Field(default=0.0, description="Average response time")
    agent_calls: Optional[int] = Field(default=None, description="Number of agent calls")
    insights_generated: Optional[int] = Field(default=None, description="Insights generated")
    quality_score: Optional[float] = Field(default=None, description="Quality score")
    efficiency_score: Optional[float] = Field(default=None, description="Efficiency score")


class RealTimeMetrics(BaseModel):
    """Real-time system metrics."""

    messages_per_second: float = Field(default=0.0, description="Message processing rate")
    average_latency: float = Field(default=0.0, description="Average message latency (ms)")
    connection_health: float = Field(default=100.0, description="Connection health (0-100)")
    buffer_utilization: float = Field(default=0.0, description="Buffer utilization (0-100)")
    error_rate: float = Field(default=0.0, description="Error rate percentage")
    throughput_efficiency: float = Field(default=100.0, description="Throughput efficiency (0-100)")
    last_updated: int = Field(description="Last update timestamp")


# WebSocket and streaming types
class WebSocketFrame(BaseModel):
    """WebSocket frame structure."""

    type: str = Field(description="Frame type (message, ack, ping, pong, error)")
    payload: Optional[Any] = Field(default=None, description="Frame payload")
    timestamp: int = Field(description="Frame timestamp")
    frame_id: str = Field(description="Unique frame identifier")


class SubscriptionMessage(BaseModel):
    """Subscription message for streaming."""

    subscription_id: str = Field(description="Subscription identifier")
    sequence_strategies: List[str] = Field(description="Subscribed sequence strategies")
    message_types: List[str] = Field(description="Subscribed message types")
    delivery_guarantee: str = Field(description="Delivery guarantee")
    client_id: str = Field(description="Client identifier")


class AckMessage(BaseModel):
    """Acknowledgment message."""

    message_id: str = Field(description="Message to acknowledge")
    sequence_id: str = Field(description="Sequence identifier")
    status: str = Field(description="Acknowledgment status")
    timestamp: int = Field(description="Acknowledgment timestamp")


class ErrorMessage(BaseModel):
    """Error message structure."""

    error_id: str = Field(description="Error identifier")
    error_type: str = Field(description="Error type")
    message: str = Field(description="Error message")
    sequence_id: Optional[str] = Field(default=None, description="Related sequence ID")
    recoverable: bool = Field(default=False, description="Whether error is recoverable")
    timestamp: int = Field(description="Error timestamp")


# Performance and health monitoring
class HealthCheckResult(BaseModel):
    """Health check result."""

    connection_healthy: bool = Field(description="Connection health status")
    latency: float = Field(description="Connection latency (ms)")
    last_message_time: int = Field(description="Last message timestamp")
    buffer_utilization: float = Field(description="Buffer utilization percentage")
    error_count: int = Field(description="Error count")
    timestamp: int = Field(description="Health check timestamp")


class PerformanceHints(BaseModel):
    """Performance optimization hints."""

    enable_message_batching: bool = Field(default=False, description="Enable message batching")
    batch_size: int = Field(default=10, description="Message batch size")
    compression_threshold: int = Field(default=1000, description="Compression threshold (bytes)")
    enable_lazy_rendering: bool = Field(default=False, description="Enable lazy rendering")
    message_retention_limit: int = Field(default=1000, description="Message retention limit")


# Utility functions for type handling
def create_stream_message(message_type: str, content: Any, sequence_id: str = "main") -> StreamMessage:
    """Create stream message.

    Args:
        message_type: Type of message
        content: Message content
        sequence_id: Sequence identifier

    Returns:
        StreamMessage instance
    """
    return StreamMessage(
        message_id=f"msg_{int(datetime.utcnow().timestamp() * 1000)}",
        sequence_id=sequence_id,
        message_type=message_type,
        timestamp=int(datetime.utcnow().timestamp() * 1000),
        content=content
    )


def create_routed_message(agent_type: str, content: Any, sequence_id: str = "main") -> RoutedMessage:
    """Create routed message.

    Args:
        agent_type: Type of agent generating message
        content: Message content
        sequence_id: Sequence identifier

    Returns:
        RoutedMessage instance
    """
    return RoutedMessage(
        message_id=f"routed_{int(datetime.utcnow().timestamp() * 1000)}",
        sequence_id=sequence_id,
        agent_type=agent_type,
        message_type="agent_output",
        timestamp=int(datetime.utcnow().timestamp() * 1000),
        content=content
    )


def format_duration(start_time: str, end_time: Optional[str] = None) -> str:
    """Format duration between timestamps.

    Args:
        start_time: Start timestamp (ISO format)
        end_time: End timestamp (ISO format)

    Returns:
        Formatted duration string
    """
    start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    end = datetime.fromisoformat(end_time.replace('Z', '+00:00')) if end_time else datetime.utcnow()

    duration = end - start
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


# Async utility functions
async def gather_with_timeout(tasks: List[asyncio.Task], timeout: float = 30.0):
    """Gather tasks with timeout.

    Args:
        tasks: List of async tasks
        timeout: Timeout in seconds

    Returns:
        List of task results
    """
    try:
        return await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
    except asyncio.TimeoutError:
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        raise


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Chunk list into smaller pieces.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        import json
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default
