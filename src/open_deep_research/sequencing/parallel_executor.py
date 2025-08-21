"""Production-ready parallel execution engine for running concurrent sequences.

This module provides robust parallel execution capabilities for running 3 sequences
simultaneously with real-time streaming, thread safety, and proper error handling.
"""

import asyncio
import logging
import time
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from .models import (
    SequenceComparison,
    SequenceResult,
    SequencePattern,
    DynamicSequencePattern,
    ToolProductivityMetrics
)
from .sequence_engine import SequenceOptimizationEngine

# Import metrics aggregator for real-time integration
try:
    from .metrics_aggregator import MetricsAggregator, MetricsUpdate, MetricsUpdateType
    METRICS_AVAILABLE = True
except ImportError:
    MetricsAggregator = None
    MetricsUpdate = None
    MetricsUpdateType = None
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ParallelExecutionProgress(BaseModel):
    """Real-time progress tracking for parallel sequence execution."""
    
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    sequence_strategy: Optional[str] = None
    sequence_pattern: Optional[Union[SequencePattern, DynamicSequencePattern]] = None
    status: str = Field(default="pending")  # pending, running, completed, failed
    agent_position: int = Field(default=0)  # current agent executing
    total_agents: int = Field(default=3)  # dynamic based on sequence length
    
    # Progress metrics
    start_time: Optional[datetime] = None
    current_agent_start: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Agent results
    agent_results: List[Dict[str, Any]] = Field(default_factory=list)
    current_insights: List[str] = Field(default_factory=list)
    
    # Resource usage
    memory_usage_mb: float = Field(default=0.0)
    cpu_usage_percent: float = Field(default=0.0)
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = Field(default=0)
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage (0-100)."""
        if self.status == "completed":
            return 100.0
        elif self.status == "failed":
            return 0.0
        elif self.total_agents == 0:
            return 0.0
        else:
            return (self.agent_position / self.total_agents) * 100.0
    
    @property
    def is_active(self) -> bool:
        """Check if execution is currently active."""
        return self.status in ["pending", "running"]


class ParallelExecutionResult(BaseModel):
    """Comprehensive results from parallel sequence execution."""
    
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    research_topic: str
    
    # Execution metadata
    start_time: datetime
    end_time: datetime
    total_duration: float  # seconds
    
    # Sequence results
    sequence_results: Dict[str, SequenceResult]
    progress_snapshots: Dict[str, List[ParallelExecutionProgress]]
    
    # Comparative analysis
    comparison: SequenceComparison
    
    # Performance metrics
    max_concurrency_achieved: int
    average_memory_usage: float
    peak_memory_usage: float
    total_api_calls: int
    
    # Quality metrics
    unique_insights_across_sequences: List[str]
    insight_overlap_matrix: Dict[Tuple[str, str], float]
    
    # Failure analysis
    failed_sequences: List[str] = Field(default_factory=list)
    error_summary: Dict[str, str] = Field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate percentage of successful sequence executions."""
        total = len(self.sequence_results) + len(self.failed_sequences)
        if total == 0:
            return 0.0
        return (len(self.sequence_results) / total) * 100.0
    
    @property
    def best_performing_strategy(self) -> Optional[str]:
        """Get the best performing strategy based on tool productivity."""
        if not self.sequence_results:
            return None
        
        best_strategy = None
        best_score = -1.0
        
        for strategy, result in self.sequence_results.items():
            productivity = result.overall_productivity_metrics.tool_productivity
            if productivity > best_score:
                best_score = productivity
                best_strategy = strategy
        
        return best_strategy


class StreamMessage(BaseModel):
    """Message structure for real-time streaming."""
    
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    sequence_strategy: Optional[str] = None
    message_type: str  # progress, result, error, completion
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Message content
    data: Dict[str, Any] = Field(default_factory=dict)
    progress: Optional[ParallelExecutionProgress] = None
    
    # Delivery metadata
    retries: int = Field(default=0)
    max_retries: int = Field(default=3)
    
    def to_json(self) -> Dict[str, Any]:
        """Convert message to JSON for streaming."""
        return {
            "message_id": self.message_id,
            "sequence_strategy": self.sequence_strategy if self.sequence_strategy else None,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "progress": self.progress.model_dump() if self.progress else None
        }


class ResourceMonitor:
    """Enhanced resource monitoring with memory management and leak detection."""
    
    def __init__(self):
        self.start_time = time.time()
        self.memory_readings: List[float] = []
        self.cpu_readings: List[float] = []
        self.gc_stats: List[Dict[str, int]] = []
        self._monitoring = False
        self._process = None
        if PSUTIL_AVAILABLE:
            try:
                self._process = psutil.Process()
            except Exception:
                pass
        self._peak_memory = 0.0
        self._memory_threshold = 512.0  # MB
        self._gc_forced_count = 0
        
        # Weak references to track objects
        self._tracked_objects: Set[weakref.ref] = set()
    
    async def start_monitoring(self):
        """Start resource monitoring."""
        self._monitoring = True
        asyncio.create_task(self._monitor_loop())
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
    
    async def _monitor_loop(self):
        """Enhanced monitoring loop with memory management."""
        while self._monitoring:
            try:
                # Get memory usage
                if self._process and PSUTIL_AVAILABLE:
                    try:
                        memory_info = self._process.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)  # RSS in MB
                    except Exception:
                        # Fallback to basic method
                        import sys
                        memory_mb = sys.getsizeof(locals()) / (1024 * 1024)
                else:
                    import sys
                    memory_mb = sys.getsizeof(locals()) / (1024 * 1024)
                
                self.memory_readings.append(memory_mb)
                
                # Track peak memory
                if memory_mb > self._peak_memory:
                    self._peak_memory = memory_mb
                
                # Get CPU usage
                if self._process and PSUTIL_AVAILABLE:
                    try:
                        cpu_percent = self._process.cpu_percent()
                    except Exception:
                        cpu_percent = min(len(self.memory_readings) * 2.5, 100.0)
                else:
                    cpu_percent = min(len(self.memory_readings) * 2.5, 100.0)
                
                self.cpu_readings.append(cpu_percent)
                
                # Monitor garbage collection
                gc_stats = {
                    'generation_0': gc.get_count()[0],
                    'generation_1': gc.get_count()[1],
                    'generation_2': gc.get_count()[2],
                }
                if hasattr(gc, 'get_stats') and gc.get_stats():
                    gc_stats['collected'] = gc.get_stats()[0].get('collected', 0)
                self.gc_stats.append(gc_stats)
                
                # Force GC if memory usage is high
                if memory_mb > self._memory_threshold:
                    await self._force_gc_if_needed()
                
                # Clean up weak references
                self._cleanup_weak_refs()
                
                # Keep only recent readings (last 300 = 5 minutes at 1s intervals)
                if len(self.memory_readings) > 300:
                    self.memory_readings = self.memory_readings[-150:]
                    self.cpu_readings = self.cpu_readings[-150:]
                    self.gc_stats = self.gc_stats[-150:]
                
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    def get_current_metrics(self) -> Tuple[float, float]:
        """Get current memory and CPU usage."""
        memory = self.memory_readings[-1] if self.memory_readings else 0.0
        cpu = self.cpu_readings[-1] if self.cpu_readings else 0.0
        return memory, cpu
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage."""
        return self._peak_memory
    
    def get_average_memory(self) -> float:
        """Get average memory usage."""
        return sum(self.memory_readings) / len(self.memory_readings) if self.memory_readings else 0.0
    
    def get_memory_pressure(self) -> float:
        """Calculate memory pressure as percentage of threshold."""
        current_memory = self.memory_readings[-1] if self.memory_readings else 0.0
        return (current_memory / self._memory_threshold) * 100.0
    
    def track_object(self, obj: Any) -> None:
        """Track an object with weak reference for leak detection."""
        try:
            weak_ref = weakref.ref(obj)
            self._tracked_objects.add(weak_ref)
        except TypeError:
            # Object doesn't support weak references
            pass
    
    def get_tracked_object_count(self) -> int:
        """Get count of tracked objects still alive."""
        return len([ref for ref in self._tracked_objects if ref() is not None])
    
    async def _force_gc_if_needed(self) -> None:
        """Force garbage collection if memory pressure is high."""
        if self.get_memory_pressure() > 80.0:
            logger.info(f"High memory pressure detected, forcing garbage collection")
            
            # Run GC in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._perform_gc)
            self._gc_forced_count += 1
    
    def _perform_gc(self) -> None:
        """Perform garbage collection."""
        collected = gc.collect()
        logger.debug(f"Forced GC collected {collected} objects")
    
    def _cleanup_weak_refs(self) -> None:
        """Clean up dead weak references."""
        dead_refs = [ref for ref in self._tracked_objects if ref() is None]
        for ref in dead_refs:
            self._tracked_objects.discard(ref)
    
    def get_resource_report(self) -> Dict[str, Any]:
        """Get comprehensive resource report."""
        return {
            'memory': {
                'current_mb': self.memory_readings[-1] if self.memory_readings else 0.0,
                'peak_mb': self._peak_memory,
                'average_mb': self.get_average_memory(),
                'pressure_percent': self.get_memory_pressure(),
                'threshold_mb': self._memory_threshold
            },
            'cpu': {
                'current_percent': self.cpu_readings[-1] if self.cpu_readings else 0.0,
                'average_percent': sum(self.cpu_readings) / len(self.cpu_readings) if self.cpu_readings else 0.0
            },
            'gc': {
                'forced_collections': self._gc_forced_count,
                'tracked_objects': self.get_tracked_object_count(),
                'latest_stats': self.gc_stats[-1] if self.gc_stats else {}
            },
            'uptime_seconds': time.time() - self.start_time,
            'psutil_available': PSUTIL_AVAILABLE
        }


class ParallelSequenceExecutor:
    """Production-ready parallel execution engine for sequence strategies."""
    
    def __init__(
        self,
        config: RunnableConfig,
        max_concurrent: int = 3,
        timeout_seconds: int = 3600,  # 1 hour
        retry_attempts: int = 2,
        metrics_aggregator: Optional['MetricsAggregator'] = None,
        enable_real_time_metrics: bool = True
    ):
        """Initialize parallel executor.
        
        Args:
            config: Runtime configuration for agents and models
            max_concurrent: Maximum number of concurrent sequences
            timeout_seconds: Timeout for individual sequence execution
            retry_attempts: Number of retry attempts for failed sequences
            metrics_aggregator: Optional metrics aggregator for real-time tracking
            enable_real_time_metrics: Enable real-time metrics collection
        """
        self.config = config
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        
        # Metrics integration
        self.metrics_aggregator = metrics_aggregator
        self.enable_real_time_metrics = enable_real_time_metrics and METRICS_AVAILABLE
        
        # Execution tracking
        self.active_executions: Dict[str, ParallelExecutionProgress] = {}
        self.execution_history: List[ParallelExecutionResult] = []
        
        # Resource management
        self.resource_monitor = ResourceMonitor()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent)
        
        logger.info(f"ParallelSequenceExecutor initialized: max_concurrent={max_concurrent}, "
                   f"metrics_enabled={self.enable_real_time_metrics}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.resource_monitor.start_monitoring()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.resource_monitor.stop_monitoring()
        self.thread_pool.shutdown(wait=True)
    
    async def execute_sequences_parallel(
        self,
        research_topic: str,
        sequences: Optional[Union[List[str], List[Union[SequencePattern, DynamicSequencePattern]]]] = None,
        stream_callback: Optional[Callable[[StreamMessage], None]] = None
    ) -> ParallelExecutionResult:
        """Execute multiple sequences in parallel with real-time streaming.
        
        Args:
            research_topic: The research topic to investigate
            sequences: List of strategies/patterns to execute (default: all three strategies)
            stream_callback: Callback function for real-time progress updates
            
        Returns:
            ParallelExecutionResult with comprehensive execution data
        """
        # Handle backward compatibility and default sequences
        if sequences is None:
            sequences = ["theory_first", "market_first", "future_back"]
        
        # Convert sequences to standardized format
        sequence_patterns = []
        sequence_strategies = []
        
        for seq in sequences:
            if isinstance(seq, str):
                # Legacy strategy - need to create pattern from it
                sequence_strategies.append(seq)
                # We'll create the pattern later when we have SEQUENCE_PATTERNS or create a default
            elif isinstance(seq, (SequencePattern, DynamicSequencePattern)):
                sequence_patterns.append(seq)
                # Extract strategy if available
                if hasattr(seq, 'strategy'):
                    sequence_strategies.append(seq.strategy)
                else:
                    # For DynamicSequencePattern without strategy, use a placeholder
                    sequence_strategies.append(None)
            else:
                raise ValueError(f"Unsupported sequence type: {type(seq)}")
        
        if len(sequences) > self.max_concurrent:
            raise ValueError(f"Cannot execute {len(sequences)} sequences concurrently (max: {self.max_concurrent})")
        
        execution_id = str(uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Starting parallel execution {execution_id}: {len(sequences)} sequences for '{research_topic}'")
        
        # Register execution with metrics aggregator
        if self.enable_real_time_metrics and self.metrics_aggregator:
            # Only register strategies that are not None
            valid_strategies = [s for s in sequence_strategies if s is not None]
            if valid_strategies:
                self.metrics_aggregator.register_execution(
                    execution_id=execution_id,
                    strategies=valid_strategies,
                    start_time=start_time
                )
                logger.debug(f"Registered execution {execution_id} with metrics aggregator")
        
        # Send initial stream message
        if stream_callback:
            initial_message = StreamMessage(
                message_type="execution_started",
                data={
                    "execution_id": execution_id,
                    "research_topic": research_topic,
                    "sequences": [s if isinstance(s, str) else str(s.sequence_id) if hasattr(s, 'sequence_id') else str(i) for i, s in enumerate(sequences)],
                    "estimated_duration": len(sequences) * 300,  # 5 minutes per sequence
                    "metrics_enabled": self.enable_real_time_metrics
                }
            )
            await self._safe_stream_callback(stream_callback, initial_message)
        
        # Create progress trackers
        progress_trackers = {}
        for i, seq in enumerate(sequences):
            # Determine total agents for this sequence
            if isinstance(seq, (SequencePattern, DynamicSequencePattern)):
                total_agents = len(seq.agent_order)
                pattern = seq
                strategy = getattr(seq, 'strategy', None)
            else:  # String strategy
                total_agents = 3  # Default for backward compatibility
                pattern = None
                strategy = seq
            
            progress = ParallelExecutionProgress(
                execution_id=execution_id,
                sequence_strategy=strategy,
                sequence_pattern=pattern,
                status="pending",
                total_agents=total_agents
            )
            progress_trackers[i] = progress
            # Use index for key to handle dynamic patterns without strategy
            key = f"{execution_id}_{i}_{strategy if strategy else 'dynamic'}"
            self.active_executions[key] = progress
        
        try:
            # Execute sequences in parallel
            tasks = []
            for i, seq in enumerate(sequences):
                task = asyncio.create_task(
                    self._execute_single_sequence(
                        research_topic=research_topic,
                        sequence=seq,
                        progress_tracker=progress_trackers[i],
                        stream_callback=stream_callback
                    )
                )
                tasks.append((i, seq, task))
            
            # Wait for all tasks to complete with timeout
            sequence_results = {}
            failed_sequences = []
            error_summary = {}
            progress_snapshots = {i: [] for i in range(len(sequences))}
            
            for i, seq, task in tasks:
                seq_id = self._get_sequence_id(seq)
                try:
                    result = await asyncio.wait_for(task, timeout=self.timeout_seconds)
                    if result:
                        sequence_results[i] = result
                        logger.info(f"Sequence {seq_id} completed successfully")
                    else:
                        failed_sequences.append(i)
                        error_summary[i] = "No result returned"
                        logger.warning(f"Sequence {seq_id} returned no result")
                except asyncio.TimeoutError:
                    failed_sequences.append(i)
                    error_summary[i] = f"Timeout after {self.timeout_seconds} seconds"
                    logger.error(f"Sequence {seq_id} timed out")
                except Exception as e:
                    failed_sequences.append(i)
                    error_summary[i] = str(e)
                    logger.error(f"Sequence {seq_id} failed: {e}")
                
                # Collect progress snapshots
                if i in progress_trackers:
                    progress_snapshots[i].append(progress_trackers[i])
            
            # Generate comparison if we have results
            comparison = None
            if len(sequence_results) >= 2:
                try:
                    from .metrics import MetricsCalculator
                    metrics_calculator = MetricsCalculator()
                    comparison = metrics_calculator.compare_sequence_results(list(sequence_results.values()))
                    
                    # Update metrics aggregator with final comparison
                    if self.enable_real_time_metrics and self.metrics_aggregator:
                        parallel_metrics = self.metrics_aggregator.aggregate_parallel_metrics(execution_id)
                        logger.info(f"Final parallel metrics aggregated for execution {execution_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate comparison: {e}")
            
            # Calculate insight overlap
            insight_overlap = self._calculate_insight_overlap(sequence_results)
            unique_insights = self._extract_unique_insights(sequence_results)
            
            # Create final result
            end_time = datetime.utcnow()
            total_duration = (end_time - start_time).total_seconds()
            
            result = ParallelExecutionResult(
                execution_id=execution_id,
                research_topic=research_topic,
                start_time=start_time,
                end_time=end_time,
                total_duration=total_duration,
                sequence_results=sequence_results,
                progress_snapshots=progress_snapshots,
                comparison=comparison,
                max_concurrency_achieved=len(sequences),
                average_memory_usage=self.resource_monitor.get_average_memory(),
                peak_memory_usage=self.resource_monitor.get_peak_memory(),
                total_api_calls=sum(
                    result.overall_productivity_metrics.total_agent_calls 
                    for result in sequence_results.values()
                ),
                unique_insights_across_sequences=unique_insights,
                insight_overlap_matrix=insight_overlap,
                failed_sequences=failed_sequences,
                error_summary=error_summary
            )
            
            # Store in history
            self.execution_history.append(result)
            
            # Send completion message
            if stream_callback:
                completion_message = StreamMessage(
                    message_type="execution_completed",
                    data={
                        "execution_id": execution_id,
                        "success_rate": result.success_rate,
                        "best_strategy": result.best_performing_strategy if result.best_performing_strategy else None,
                        "total_duration": total_duration,
                        "insights_generated": len(unique_insights)
                    }
                )
                await self._safe_stream_callback(stream_callback, completion_message)
            
            logger.info(f"Parallel execution {execution_id} completed: {result.success_rate:.1f}% success rate")
            
            return result
            
        finally:
            # Clean up active executions
            for i, seq in enumerate(sequences):
                strategy = seq if isinstance(seq, str) else getattr(seq, 'strategy', None)
                key = f"{execution_id}_{i}_{strategy if strategy else 'dynamic'}"
                self.active_executions.pop(key, None)
    
    def _get_sequence_id(self, sequence: Union[str, SequencePattern, DynamicSequencePattern]) -> str:
        """Get a readable ID for a sequence."""
        if isinstance(sequence, str):
            return sequence.value
        elif hasattr(sequence, 'sequence_id'):
            return str(sequence.sequence_id)
        elif hasattr(sequence, 'strategy'):
            return sequence.strategy
        else:
            return "dynamic_sequence"
    
    async def _execute_single_sequence(
        self,
        research_topic: str,
        sequence: Union[str, SequencePattern, DynamicSequencePattern],
        progress_tracker: ParallelExecutionProgress,
        stream_callback: Optional[Callable[[StreamMessage], None]] = None
    ) -> Optional[SequenceResult]:
        """Execute a single sequence with progress tracking and error handling."""
        
        async with self.semaphore:  # Limit concurrency
            engine = None
            retry_count = 0
            sequence_id = self._get_sequence_id(sequence)
            
            while retry_count <= self.retry_attempts:
                try:
                    # Update progress
                    progress_tracker.status = "running"
                    progress_tracker.start_time = datetime.utcnow()
                    progress_tracker.retry_count = retry_count
                    
                    # Get current resource metrics
                    memory, cpu = self.resource_monitor.get_current_metrics()
                    progress_tracker.memory_usage_mb = memory
                    progress_tracker.cpu_usage_percent = cpu
                    
                    # Determine the strategy for the stream message
                    strategy = sequence if isinstance(sequence, str) else getattr(sequence, 'strategy', None)
                    
                    # Send progress update
                    if stream_callback:
                        progress_message = StreamMessage(
                            sequence_strategy=strategy,
                            message_type="progress",
                            progress=progress_tracker,
                            data={
                                "status": "sequence_started",
                                "sequence_id": sequence_id,
                                "strategy": strategy if strategy else "dynamic",
                                "retry_attempt": retry_count
                            }
                        )
                        await self._safe_stream_callback(stream_callback, progress_message)
                    
                    # Initialize engine with metrics integration
                    engine = SequenceOptimizationEngine(
                        config=self.config,
                        metrics_aggregator=self.metrics_aggregator,
                        enable_real_time_metrics=self.enable_real_time_metrics
                    )
                    
                    # Get the pattern to execute
                    if isinstance(sequence, (SequencePattern, DynamicSequencePattern)):
                        pattern = sequence
                    else:  # String strategy - need to handle this case
                        # Try to create a pattern from strategy or use a fallback
                        try:
                            # Check if we have SEQUENCE_PATTERNS available
                            from .models import SEQUENCE_PATTERNS
                            pattern = SEQUENCE_PATTERNS[sequence]
                        except (ImportError, KeyError, AttributeError):
                            # Fallback: create a basic pattern from the strategy
                            from .models import AgentType
                            if sequence == "theory_first":
                                agent_order = [AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS]
                            elif sequence == "market_first":
                                agent_order = [AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS]
                            elif sequence == "future_back":
                                agent_order = [AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY]
                            else:
                                agent_order = [AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS]
                            
                            pattern = SequencePattern(
                                agent_order=agent_order,
                                description=f"Generated pattern for {sequence.value}",
                                expected_advantages=[f"Optimized for {sequence.value} approach"]
                            )
                            # Add strategy attribute for compatibility
                            pattern.strategy = sequence
                    
                    # Execute sequence with progress updates
                    result = await self._execute_with_progress_updates(
                        engine=engine,
                        pattern=pattern,
                        research_topic=research_topic,
                        progress_tracker=progress_tracker,
                        stream_callback=stream_callback,
                        execution_id=progress_tracker.execution_id
                    )
                    
                    # Mark as completed
                    progress_tracker.status = "completed"
                    progress_tracker.agent_position = progress_tracker.total_agents
                    
                    logger.info(f"Sequence {sequence_id} completed successfully after {retry_count} retries")
                    return result
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = f"Attempt {retry_count}/{self.retry_attempts + 1} failed: {str(e)}"
                    
                    progress_tracker.error_message = error_msg
                    progress_tracker.retry_count = retry_count
                    
                    logger.warning(f"Sequence {sequence_id} failed (attempt {retry_count}): {e}")
                    
                    # Send error update
                    if stream_callback:
                        error_message = StreamMessage(
                            sequence_strategy=strategy,
                            message_type="error",
                            data={
                                "error": str(e),
                                "sequence_id": sequence_id,
                                "retry_attempt": retry_count,
                                "will_retry": retry_count <= self.retry_attempts
                            }
                        )
                        await self._safe_stream_callback(stream_callback, error_message)
                    
                    if retry_count <= self.retry_attempts:
                        # Wait before retry with exponential backoff
                        wait_time = min(2 ** retry_count, 30)  # Max 30 seconds
                        logger.info(f"Retrying sequence {sequence_id} in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        # Final failure
                        progress_tracker.status = "failed"
                        logger.error(f"Sequence {sequence_id} failed permanently after {retry_count} attempts")
                        return None
            
            return None  # Should never reach here
    
    async def _execute_with_progress_updates(
        self,
        engine: SequenceOptimizationEngine,
        pattern,
        research_topic: str,
        progress_tracker: ParallelExecutionProgress,
        stream_callback: Optional[Callable[[StreamMessage], None]] = None,
        execution_id: Optional[str] = None
    ) -> SequenceResult:
        """Execute sequence with detailed progress updates."""
        
        # Track execution start
        execution_start = datetime.utcnow()
        agent_results = []
        previous_insights = []
        
        try:
            # Execute sequence using the engine (which handles metrics integration)
            sequence_result = await engine.execute_sequence(
                sequence_pattern=pattern,
                research_topic=research_topic,
                execution_id=execution_id
            )
            
            # Update progress tracker with final results
            progress_tracker.current_insights = []
            for agent_result in sequence_result.agent_results:
                progress_tracker.current_insights.extend(agent_result.key_insights)
                progress_tracker.agent_results.append({
                    "agent_type": agent_result.agent_type.value,
                    "position": agent_result.execution_order,
                    "insights_count": len(agent_result.key_insights),
                    "execution_duration": agent_result.execution_duration,
                    "tool_calls": agent_result.tool_calls_made
                })
            
            # Send final progress update
            if stream_callback:
                strategy = getattr(pattern, 'strategy', None)
                completion_message = StreamMessage(
                    sequence_strategy=strategy,
                    message_type="progress",
                    progress=progress_tracker,
                    data={
                        "status": "sequence_completed",
                        "total_insights": len(progress_tracker.current_insights),
                        "total_duration": sequence_result.total_duration,
                        "tool_productivity": sequence_result.overall_productivity_metrics.tool_productivity,
                        "sequence_id": getattr(pattern, 'sequence_id', 'unknown')
                    }
                )
                await self._safe_stream_callback(stream_callback, completion_message)
            
            return sequence_result
            
        except Exception as e:
            strategy = getattr(pattern, 'strategy', None)
            pattern_id = getattr(pattern, 'sequence_id', 'unknown')
            logger.error(f"Error in sequence execution for {strategy if strategy else pattern_id}: {e}")
            raise
    
    async def _safe_stream_callback(
        self, 
        callback: Callable[[StreamMessage], None], 
        message: StreamMessage
    ):
        """Safely execute stream callback with error handling."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            logger.warning(f"Stream callback error: {e}")
    
    def _calculate_insight_overlap(
        self, 
        sequence_results: Dict[str, SequenceResult]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate insight overlap between sequences."""
        overlap_matrix = {}
        strategies = list(sequence_results.keys())
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies):
                if i != j:
                    insights1 = set(sequence_results[strategy1].comprehensive_findings)
                    insights2 = set(sequence_results[strategy2].comprehensive_findings)
                    
                    if insights1 and insights2:
                        intersection = len(insights1.intersection(insights2))
                        union = len(insights1.union(insights2))
                        overlap = intersection / union if union > 0 else 0.0
                    else:
                        overlap = 0.0
                    
                    overlap_matrix[(strategy1, strategy2)] = overlap
        
        return overlap_matrix
    
    def _extract_unique_insights(
        self, 
        sequence_results: Dict[str, SequenceResult]
    ) -> List[str]:
        """Extract unique insights across all sequences."""
        all_insights = set()
        
        for result in sequence_results.values():
            all_insights.update(result.comprehensive_findings)
        
        return list(all_insights)
    
    def get_active_executions(self) -> Dict[str, ParallelExecutionProgress]:
        """Get currently active executions."""
        return {k: v for k, v in self.active_executions.items() if v.is_active}
    
    def get_execution_history(self) -> List[ParallelExecutionResult]:
        """Get history of parallel executions."""
        return self.execution_history.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all executions."""
        if not self.execution_history:
            return {"message": "No parallel executions recorded"}
        
        total_executions = len(self.execution_history)
        successful_executions = [e for e in self.execution_history if e.success_rate > 50]
        
        avg_duration = sum(e.total_duration for e in self.execution_history) / total_executions
        avg_success_rate = sum(e.success_rate for e in self.execution_history) / total_executions
        
        strategy_performance = {}
        for execution in self.execution_history:
            for strategy, result in execution.sequence_results.items():
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(result.overall_productivity_metrics.tool_productivity)
        
        avg_productivity_by_strategy = {
            strategy: sum(scores) / len(scores)
            for strategy, scores in strategy_performance.items()
            if scores
        }
        
        return {
            "total_parallel_executions": total_executions,
            "successful_executions": len(successful_executions),
            "average_duration_seconds": avg_duration,
            "average_success_rate": avg_success_rate,
            "average_productivity_by_strategy": avg_productivity_by_strategy,
            "peak_memory_usage": max(e.peak_memory_usage for e in self.execution_history),
            "total_api_calls": sum(e.total_api_calls for e in self.execution_history)
        }


@asynccontextmanager
async def parallel_executor_context(
    config: RunnableConfig,
    max_concurrent: int = 3,
    timeout_seconds: int = 3600
) -> AsyncGenerator[ParallelSequenceExecutor, None]:
    """Async context manager for parallel executor with proper cleanup."""
    executor = ParallelSequenceExecutor(
        config=config,
        max_concurrent=max_concurrent,
        timeout_seconds=timeout_seconds
    )
    
    try:
        async with executor:
            yield executor
    finally:
        # Cleanup is handled by the executor's __aexit__ method
        pass