"""Production-ready real-time metrics aggregation for parallel sequence execution.

This module provides comprehensive metrics collection, aggregation, and streaming
capabilities for monitoring parallel sequence execution with real-time updates,
winner detection, and performance analysis.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from .models import (
 
    SequenceResult, 
    ToolProductivityMetrics, 
    AgentExecutionResult,
    InsightTransition
)

logger = logging.getLogger(__name__)


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


class AggregationLevel(Enum):
    """Aggregation levels for metrics."""
    
    SEQUENCE = "sequence"      # Per individual sequence
    STRATEGY = "strategy"      # Per strategy across executions
    PARALLEL = "parallel"      # Across all parallel sequences
    GLOBAL = "global"          # System-wide metrics


class MetricsUpdateType(Enum):
    """Types of metrics updates."""
    
    REAL_TIME = "real_time"
    SNAPSHOT = "snapshot"
    COMPARISON = "comparison"
    WINNER_DETECTED = "winner_detected"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    AGENT_COMPLETED = "agent_completed"
    ERROR = "error"


@dataclass
class SequenceMetrics:
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
    agent_completion_times: List[float] = field(default_factory=list)
    agent_insights_count: List[int] = field(default_factory=list)
    agent_tool_calls: List[int] = field(default_factory=list)
    
    # Trend data (last 10 updates)
    productivity_trend: deque = field(default_factory=lambda: deque(maxlen=10))
    quality_trend: deque = field(default_factory=lambda: deque(maxlen=10))
    efficiency_trend: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def update_metrics(self, agent_result: Optional[AgentExecutionResult] = None):
        """Update metrics with new agent result."""
        if agent_result:
            self.current_agent_position += 1
            self.agent_completion_times.append(agent_result.execution_duration)
            self.agent_insights_count.append(len(agent_result.key_insights))
            self.agent_tool_calls.append(agent_result.tool_calls_made)
            
            # Update totals
            self.insights_generated += len(agent_result.key_insights)
            self.tool_calls_made += agent_result.tool_calls_made
            
            # Update quality scores
            if agent_result.insight_quality_scores:
                self.avg_insight_quality = sum(agent_result.insight_quality_scores) / len(agent_result.insight_quality_scores)
            
            self.research_depth_score = agent_result.research_depth_score
            self.novelty_score = agent_result.novelty_score
            
            # Calculate tool productivity
            if self.tool_calls_made > 0:
                research_quality = (self.research_depth_score + self.novelty_score + self.avg_insight_quality) / 3
                self.current_tool_productivity = research_quality / self.tool_calls_made
                self.current_research_quality = research_quality
            
            # Calculate agent efficiency
            if self.tool_calls_made > 0:
                self.current_agent_efficiency = self.insights_generated / self.tool_calls_made
            
            # Track time to first insight
            if self.time_to_first_insight is None and self.insights_generated > 0:
                self.time_to_first_insight = sum(self.agent_completion_times)
        
        # Update execution duration
        self.execution_duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Update trends
        self.productivity_trend.append(self.current_tool_productivity)
        self.quality_trend.append(self.current_research_quality)
        self.efficiency_trend.append(self.current_agent_efficiency)
    
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
    def productivity_trend_direction(self) -> str:
        """Determine if productivity is trending up, down, or stable."""
        if len(self.productivity_trend) < 2:
            return "stable"
        
        recent_values = list(self.productivity_trend)[-5:]  # Last 5 values
        if len(recent_values) < 2:
            return "stable"
        
        slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
        
        if slope > 0.01:
            return "up"
        elif slope < -0.01:
            return "down"
        else:
            return "stable"
    
    @property
    def estimated_completion_time(self) -> Optional[datetime]:
        """Estimate completion time based on current progress."""
        if self.current_agent_position == 0 or self.status == "completed":
            return None
        
        avg_time_per_agent = sum(self.agent_completion_times) / len(self.agent_completion_times)
        remaining_agents = self.total_agents - self.current_agent_position
        estimated_remaining_seconds = avg_time_per_agent * remaining_agents
        
        return datetime.utcnow() + timedelta(seconds=estimated_remaining_seconds)


@dataclass
class ParallelMetrics:
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
    strategy_rankings: Dict[str, float] = field(default_factory=dict)
    
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
    """Real-time metrics update message."""
    
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
                str(k.value): v.__dict__ if hasattr(v, '__dict__') else v
                for k, v in (self.sequence_metrics or {}).items()
            },
            "parallel_metrics": self.parallel_metrics.__dict__ if self.parallel_metrics else None,
            "winner_analysis": self.winner_analysis.model_dump() if self.winner_analysis else None,
            "updated_strategy": self.updated_strategy if self.updated_strategy else None,
            "metric_deltas": self.metric_deltas,
            "message": self.message,
            "alert_level": self.alert_level
        }


class ComparisonAnalyzer:
    """Intelligent winner detection and comparative analysis."""
    
    def __init__(self, variance_threshold: float = 0.2):
        """Initialize comparison analyzer.
        
        Args:
            variance_threshold: Threshold for detecting significant differences (default 20%)
        """
        self.variance_threshold = variance_threshold
        self.confidence_weights = {
            "productivity_difference": 0.4,
            "sample_size": 0.2,
            "consistency": 0.2,
            "statistical_significance": 0.2
        }
    
    def detect_winner(
        self, 
        sequence_metrics: Dict[str, SequenceMetrics],
        min_completion_threshold: float = 0.5
    ) -> Optional[WinnerAnalysis]:
        """Detect winner based on current metrics.
        
        Args:
            sequence_metrics: Current metrics for all sequences
            min_completion_threshold: Minimum completion progress to consider (0.5 = 50%)
            
        Returns:
            WinnerAnalysis if a clear winner is detected, None otherwise
        """
        # Filter to sequences with sufficient progress
        eligible_sequences = {
            strategy: metrics for strategy, metrics in sequence_metrics.items()
            if metrics.progress_percent >= min_completion_threshold * 100
        }
        
        if len(eligible_sequences) < 2:
            return None
        
        # Extract productivity scores
        productivity_scores = {
            strategy: metrics.current_tool_productivity
            for strategy, metrics in eligible_sequences.items()
        }
        
        if not productivity_scores or all(score == 0 for score in productivity_scores.values()):
            return None
        
        # Calculate variance
        scores = list(productivity_scores.values())
        if len(scores) < 2:
            return None
        
        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        variance = std_dev / mean_score if mean_score > 0 else 0.0
        
        # Check if variance exceeds threshold
        if variance < self.variance_threshold:
            return None
        
        # Find best strategy
        best_strategy = max(productivity_scores.keys(), key=lambda k: productivity_scores[k])
        best_score = productivity_scores[best_strategy]
        worst_score = min(productivity_scores.values())
        
        # Calculate advantage percentage
        productivity_advantage = ((best_score - worst_score) / worst_score * 100) if worst_score > 0 else 0
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            productivity_scores,
            eligible_sequences,
            variance,
            best_strategy
        )
        
        # Analyze winner strengths
        best_metrics = eligible_sequences[best_strategy]
        winner_strengths = self._analyze_winner_strengths(best_metrics, eligible_sequences)
        
        # Calculate comparative advantages
        comparative_advantages = self._calculate_comparative_advantages(
            best_metrics, eligible_sequences
        )
        
        return WinnerAnalysis(
            winning_strategy=best_strategy,
            confidence_score=confidence,
            productivity_advantage=productivity_advantage,
            all_strategy_scores=productivity_scores,
            variance_threshold_exceeded=True,
            statistical_significance=min(variance * 2, 1.0),
            winner_strengths=winner_strengths,
            comparative_advantages=comparative_advantages,
            unique_insights_count=best_metrics.insights_generated,
            quality_superiority=best_metrics.current_research_quality - mean_score,
            efficiency_advantage=best_metrics.current_agent_efficiency - statistics.mean([
                m.current_agent_efficiency for m in eligible_sequences.values()
            ])
        )
    
    def _calculate_confidence(
        self,
        productivity_scores: Dict[str, float],
        sequence_metrics: Dict[str, SequenceMetrics],
        variance: float,
        best_strategy: str
    ) -> float:
        """Calculate confidence in winner detection."""
        
        scores = list(productivity_scores.values())
        best_score = productivity_scores[best_strategy]
        
        # Productivity difference component
        score_range = max(scores) - min(scores)
        max_possible = max(scores)
        productivity_diff_score = (score_range / max_possible) if max_possible > 0 else 0
        
        # Sample size component (based on progress)
        avg_progress = sum(m.progress_percent for m in sequence_metrics.values()) / len(sequence_metrics)
        sample_size_score = min(avg_progress / 100, 1.0)
        
        # Consistency component (based on trend stability)
        best_metrics = sequence_metrics[best_strategy]
        consistency_score = 1.0 if best_metrics.productivity_trend_direction == "up" else 0.7
        
        # Statistical significance (based on variance)
        stat_significance = min(variance / self.variance_threshold, 1.0)
        
        # Weighted average
        confidence = (
            productivity_diff_score * self.confidence_weights["productivity_difference"] +
            sample_size_score * self.confidence_weights["sample_size"] +
            consistency_score * self.confidence_weights["consistency"] +
            stat_significance * self.confidence_weights["statistical_significance"]
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _analyze_winner_strengths(
        self,
        winner_metrics: SequenceMetrics,
        all_metrics: Dict[str, SequenceMetrics]
    ) -> List[str]:
        """Analyze specific strengths of the winning strategy."""
        strengths = []
        
        # Compare against averages
        avg_productivity = sum(m.current_tool_productivity for m in all_metrics.values()) / len(all_metrics)
        avg_quality = sum(m.current_research_quality for m in all_metrics.values()) / len(all_metrics)
        avg_efficiency = sum(m.current_agent_efficiency for m in all_metrics.values()) / len(all_metrics)
        avg_insights = sum(m.insights_generated for m in all_metrics.values()) / len(all_metrics)
        
        if winner_metrics.current_tool_productivity > avg_productivity * 1.2:
            strengths.append("Superior tool productivity (>20% above average)")
        
        if winner_metrics.current_research_quality > avg_quality * 1.15:
            strengths.append("Higher research quality")
        
        if winner_metrics.current_agent_efficiency > avg_efficiency * 1.15:
            strengths.append("Better agent efficiency")
        
        if winner_metrics.insights_generated > avg_insights * 1.2:
            strengths.append("Generates more insights")
        
        if winner_metrics.time_to_first_insight and winner_metrics.time_to_first_insight < 120:
            strengths.append("Fast time to first insight")
        
        if winner_metrics.productivity_trend_direction == "up":
            strengths.append("Improving productivity trend")
        
        if winner_metrics.error_count == 0:
            strengths.append("Error-free execution")
        
        return strengths
    
    def _calculate_comparative_advantages(
        self,
        winner_metrics: SequenceMetrics,
        all_metrics: Dict[str, SequenceMetrics]
    ) -> Dict[str, float]:
        """Calculate specific comparative advantages."""
        
        others = [m for m in all_metrics.values() if m.strategy != winner_metrics.strategy]
        if not others:
            return {}
        
        advantages = {}
        
        # Productivity advantage
        avg_productivity = sum(m.current_tool_productivity for m in others) / len(others)
        if avg_productivity > 0:
            advantages["productivity"] = (winner_metrics.current_tool_productivity - avg_productivity) / avg_productivity * 100
        
        # Quality advantage
        avg_quality = sum(m.current_research_quality for m in others) / len(others)
        if avg_quality > 0:
            advantages["quality"] = (winner_metrics.current_research_quality - avg_quality) / avg_quality * 100
        
        # Efficiency advantage
        avg_efficiency = sum(m.current_agent_efficiency for m in others) / len(others)
        if avg_efficiency > 0:
            advantages["efficiency"] = (winner_metrics.current_agent_efficiency - avg_efficiency) / avg_efficiency * 100
        
        # Speed advantage (if time_to_first_insight available)
        if winner_metrics.time_to_first_insight:
            other_times = [m.time_to_first_insight for m in others if m.time_to_first_insight]
            if other_times:
                avg_time = sum(other_times) / len(other_times)
                if avg_time > 0:
                    advantages["speed"] = (avg_time - winner_metrics.time_to_first_insight) / avg_time * 100
        
        return {k: round(v, 1) for k, v in advantages.items()}


class MetricsAggregator:
    """Production-ready real-time metrics aggregation and streaming."""
    
    def __init__(
        self,
        update_interval: float = 1.0,
        history_size: int = 1000,
        winner_detection_enabled: bool = True,
        variance_threshold: float = 0.2
    ):
        """Initialize metrics aggregator.
        
        Args:
            update_interval: Interval between metrics updates (seconds)
            history_size: Number of historical updates to maintain
            winner_detection_enabled: Enable automatic winner detection
            variance_threshold: Threshold for detecting significant differences
        """
        self.update_interval = update_interval
        self.history_size = history_size
        self.winner_detection_enabled = winner_detection_enabled
        
        # Metrics storage
        self.sequence_metrics: Dict[str, Dict[str, SequenceMetrics]] = {}
        self.parallel_metrics: Dict[str, ParallelMetrics] = {}
        self.winner_analyses: Dict[str, WinnerAnalysis] = {}
        
        # Update streaming
        self.update_subscribers: Set[asyncio.Queue] = set()
        self.update_history: deque = deque(maxlen=history_size)
        
        # Analysis components
        self.comparison_analyzer = ComparisonAnalyzer(variance_threshold)
        
        # Background tasks
        self.aggregation_task: Optional[asyncio.Task] = None
        self.streaming_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Thread safety
        self.lock = asyncio.Lock()
        
        logger.info("MetricsAggregator initialized")
    
    async def start(self):
        """Start the metrics aggregation system."""
        if self._running:
            return
        
        self._running = True
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.streaming_task = asyncio.create_task(self._streaming_loop())
        
        logger.info("MetricsAggregator started")
    
    async def stop(self):
        """Stop the metrics aggregation system."""
        self._running = False
        
        if self.aggregation_task:
            self.aggregation_task.cancel()
            try:
                await self.aggregation_task
            except asyncio.CancelledError:
                pass
        
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MetricsAggregator stopped")
    
    def register_execution(
        self,
        execution_id: str,
        strategies: List[str],
        start_time: Optional[datetime] = None
    ):
        """Register a new parallel execution for tracking."""
        
        if start_time is None:
            start_time = datetime.utcnow()
        
        # Initialize sequence metrics
        sequence_metrics = {}
        for strategy in strategies:
            sequence_metrics[strategy] = SequenceMetrics(
                sequence_id=str(uuid4()),
                strategy=strategy,
                execution_id=execution_id,
                start_time=start_time
            )
        
        # Initialize parallel metrics
        parallel_metrics = ParallelMetrics(
            execution_id=execution_id,
            start_time=start_time,
            sequence_count=len(strategies),
            active_sequences=len(strategies)
        )
        
        self.sequence_metrics[execution_id] = sequence_metrics
        self.parallel_metrics[execution_id] = parallel_metrics
        
        logger.info(f"Registered execution {execution_id} with {len(strategies)} strategies")
    
    def collect_sequence_metrics(
        self,
        execution_id: str,
        strategy: str,
        agent_result: Optional[AgentExecutionResult] = None,
        status_update: Optional[str] = None,
        error: Optional[str] = None
    ) -> SequenceMetrics:
        """Collect and update metrics for a specific sequence."""
        
        if execution_id not in self.sequence_metrics:
            raise ValueError(f"Execution {execution_id} not registered")
        
        sequence_metrics = self.sequence_metrics[execution_id].get(strategy)
        if not sequence_metrics:
            raise ValueError(f"Strategy {strategy} not found in execution {execution_id}")
        
        # Update status
        if status_update:
            sequence_metrics.status = status_update
        
        # Handle errors
        if error:
            sequence_metrics.error_count += 1
            sequence_metrics.last_error = error
            if sequence_metrics.status != "failed":
                sequence_metrics.status = "error"
        
        # Update with agent result
        if agent_result:
            sequence_metrics.update_metrics(agent_result)
        
        logger.debug(f"Collected metrics for {strategy} in execution {execution_id}")
        return sequence_metrics
    
    def aggregate_parallel_metrics(self, execution_id: str) -> ParallelMetrics:
        """Aggregate metrics across all parallel sequences."""
        
        if execution_id not in self.parallel_metrics:
            raise ValueError(f"Execution {execution_id} not registered")
        
        parallel_metrics = self.parallel_metrics[execution_id]
        sequence_metrics = self.sequence_metrics.get(execution_id, {})
        
        if not sequence_metrics:
            return parallel_metrics
        
        # Reset counters
        parallel_metrics.completed_sequences = 0
        parallel_metrics.failed_sequences = 0
        parallel_metrics.active_sequences = 0
        parallel_metrics.total_insights_generated = 0
        parallel_metrics.total_tool_calls = 0
        
        # Aggregate from sequences
        productivity_scores = []
        quality_scores = []
        efficiency_scores = []
        
        for strategy, metrics in sequence_metrics.items():
            # Count by status
            if metrics.status == "completed":
                parallel_metrics.completed_sequences += 1
            elif metrics.status in ["failed", "error"]:
                parallel_metrics.failed_sequences += 1
            else:
                parallel_metrics.active_sequences += 1
            
            # Aggregate totals
            parallel_metrics.total_insights_generated += metrics.insights_generated
            parallel_metrics.total_tool_calls += metrics.tool_calls_made
            
            # Collect scores for averages
            if metrics.current_tool_productivity > 0:
                productivity_scores.append(metrics.current_tool_productivity)
                parallel_metrics.strategy_rankings[strategy] = metrics.current_tool_productivity
            
            if metrics.current_research_quality > 0:
                quality_scores.append(metrics.current_research_quality)
            
            if metrics.current_agent_efficiency > 0:
                efficiency_scores.append(metrics.current_agent_efficiency)
            
            # Track resource usage
            parallel_metrics.peak_memory_usage = max(
                parallel_metrics.peak_memory_usage, metrics.memory_usage_mb
            )
            parallel_metrics.total_api_calls += metrics.api_calls_count
        
        # Calculate averages
        parallel_metrics.average_research_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )
        parallel_metrics.average_agent_efficiency = (
            sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0
        )
        
        # Find best strategy
        if parallel_metrics.strategy_rankings:
            best_strategy = max(
                parallel_metrics.strategy_rankings.keys(),
                key=lambda k: parallel_metrics.strategy_rankings[k]
            )
            parallel_metrics.best_strategy = best_strategy
            parallel_metrics.best_productivity_score = parallel_metrics.strategy_rankings[best_strategy]
        
        # Calculate variance
        if len(productivity_scores) > 1:
            mean_productivity = sum(productivity_scores) / len(productivity_scores)
            variance = statistics.stdev(productivity_scores) / mean_productivity if mean_productivity > 0 else 0
            parallel_metrics.productivity_variance = variance
            parallel_metrics.significant_difference_detected = variance > 0.2
        
        # Track completion times
        completed_metrics = [m for m in sequence_metrics.values() if m.status == "completed"]
        if completed_metrics:
            completion_times = [
                m.start_time + timedelta(seconds=m.execution_duration) 
                for m in completed_metrics
            ]
            if not parallel_metrics.first_completion_time:
                parallel_metrics.first_completion_time = min(completion_times)
            parallel_metrics.last_completion_time = max(completion_times)
        
        return parallel_metrics
    
    async def stream_metrics_updates(self) -> AsyncIterator[MetricsUpdate]:
        """Stream real-time metrics updates."""
        
        # Create subscriber queue
        subscriber_queue = asyncio.Queue(maxsize=100)
        self.update_subscribers.add(subscriber_queue)
        
        try:
            while self._running:
                try:
                    # Wait for update with timeout
                    update = await asyncio.wait_for(subscriber_queue.get(), timeout=30.0)
                    yield update
                except asyncio.TimeoutError:
                    # Send heartbeat update
                    heartbeat = MetricsUpdate(
                        update_type=MetricsUpdateType.REAL_TIME,
                        execution_id="heartbeat",
                        message="Metrics stream active"
                    )
                    yield heartbeat
                
        except asyncio.CancelledError:
            pass
        finally:
            self.update_subscribers.discard(subscriber_queue)
    
    async def _aggregation_loop(self):
        """Background loop for metrics aggregation and winner detection."""
        
        while self._running:
            try:
                await asyncio.sleep(self.update_interval)
                
                async with self.lock:
                    # Process each active execution
                    for execution_id in list(self.parallel_metrics.keys()):
                        # Aggregate parallel metrics
                        parallel_metrics = self.aggregate_parallel_metrics(execution_id)
                        
                        # Check for winner detection
                        if self.winner_detection_enabled:
                            sequence_metrics = self.sequence_metrics.get(execution_id, {})
                            winner_analysis = self.comparison_analyzer.detect_winner(sequence_metrics)
                            
                            if winner_analysis and execution_id not in self.winner_analyses:
                                self.winner_analyses[execution_id] = winner_analysis
                                
                                # Create winner detection update
                                winner_update = MetricsUpdate(
                                    update_type=MetricsUpdateType.WINNER_DETECTED,
                                    execution_id=execution_id,
                                    parallel_metrics=parallel_metrics,
                                    winner_analysis=winner_analysis,
                                    message=f"Winner detected: {winner_analysis.winning_strategy}",
                                    alert_level="success"
                                )
                                
                                await self._broadcast_update(winner_update)
                        
                        # Create regular update
                        metrics_update = MetricsUpdate(
                            update_type=MetricsUpdateType.REAL_TIME,
                            execution_id=execution_id,
                            sequence_metrics=sequence_metrics,
                            parallel_metrics=parallel_metrics
                        )
                        
                        await self._broadcast_update(metrics_update)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _streaming_loop(self):
        """Background loop for managing streaming connections."""
        
        while self._running:
            try:
                await asyncio.sleep(30.0)  # Cleanup interval
                
                # Clean up disconnected subscribers
                active_subscribers = set()
                for subscriber in self.update_subscribers:
                    if not subscriber.full():
                        active_subscribers.add(subscriber)
                
                self.update_subscribers = active_subscribers
                
                logger.debug(f"Active subscribers: {len(self.update_subscribers)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _broadcast_update(self, update: MetricsUpdate):
        """Broadcast update to all subscribers."""
        
        # Add to history
        self.update_history.append(update)
        
        # Broadcast to subscribers
        disconnected_subscribers = set()
        for subscriber in self.update_subscribers:
            try:
                subscriber.put_nowait(update)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full, dropping update")
            except Exception as e:
                logger.warning(f"Error broadcasting to subscriber: {e}")
                disconnected_subscribers.add(subscriber)
        
        # Remove disconnected subscribers
        self.update_subscribers -= disconnected_subscribers
    
    def get_execution_snapshot(self, execution_id: str) -> Dict[str, Any]:
        """Get current metrics snapshot for an execution."""
        
        sequence_metrics = self.sequence_metrics.get(execution_id, {})
        parallel_metrics = self.parallel_metrics.get(execution_id)
        winner_analysis = self.winner_analyses.get(execution_id)
        
        return {
            "execution_id": execution_id,
            "sequence_metrics": {
                str(k.value): v.__dict__ for k, v in sequence_metrics.items()
            },
            "parallel_metrics": parallel_metrics.__dict__ if parallel_metrics else None,
            "winner_analysis": winner_analysis.model_dump() if winner_analysis else None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics and statistics."""
        
        total_executions = len(self.parallel_metrics)
        active_executions = sum(
            1 for pm in self.parallel_metrics.values()
            if pm.active_sequences > 0
        )
        
        total_sequences = sum(
            len(sm) for sm in self.sequence_metrics.values()
        )
        
        return {
            "total_executions": total_executions,
            "active_executions": active_executions,
            "total_sequences": total_sequences,
            "active_subscribers": len(self.update_subscribers),
            "update_history_size": len(self.update_history),
            "winner_detections": len(self.winner_analyses),
            "uptime_seconds": time.time() - (time.time() if hasattr(self, '_start_time') else time.time()),
            "update_interval": self.update_interval
        }


# Context manager for production usage
class MetricsAggregatorContext:
    """Context manager for metrics aggregator lifecycle."""
    
    def __init__(self, **kwargs):
        self.aggregator = MetricsAggregator(**kwargs)
    
    async def __aenter__(self):
        await self.aggregator.start()
        return self.aggregator
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aggregator.stop()