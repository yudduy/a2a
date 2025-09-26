"""Production-ready WebSocket stream multiplexer for parallel sequence execution.

This module provides robust WebSocket message routing, connection management,
and delivery guarantees for real-time streaming of multiple concurrent sequences.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from .parallel_executor import StreamMessage

# Import metrics components for integration
try:
    from .metrics_aggregator import MetricsAggregator, MetricsUpdate, MetricsUpdateType
    METRICS_AVAILABLE = True
except ImportError:
    MetricsAggregator = None
    MetricsUpdate = None
    MetricsUpdateType = None
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class DeliveryGuarantee(Enum):
    """Message delivery guarantee levels."""
    
    AT_MOST_ONCE = "at_most_once"      # No delivery confirmation needed
    AT_LEAST_ONCE = "at_least_once"    # Requires acknowledgment
    EXACTLY_ONCE = "exactly_once"      # Requires deduplication


@dataclass
class ConnectionMetrics:
    """Metrics for WebSocket connection performance."""
    
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    reconnection_count: int = 0
    total_connection_time: float = 0.0
    last_activity: float = 0.0
    latency_samples: List[float] = None
    
    def __post_init__(self):
        if self.latency_samples is None:
            self.latency_samples = []
    
    @property
    def average_latency(self) -> float:
        """Calculate average message latency."""
        return sum(self.latency_samples) / len(self.latency_samples) if self.latency_samples else 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate message delivery success rate."""
        total = self.messages_sent + self.messages_failed
        return (self.messages_sent / total * 100) if total > 0 else 100.0


class StreamSubscription(BaseModel):
    """Subscription configuration for stream filtering."""
    
    subscription_id: str = Field(default_factory=lambda: str(uuid4()))
    client_id: str
    sequence_strategies: Set[str] = Field(default_factory=set)
    message_types: Set[str] = Field(default_factory=set)
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_MOST_ONCE
    
    # Filtering options
    include_progress: bool = True
    include_errors: bool = True
    include_results: bool = True
    include_metrics: bool = True  # Include real-time metrics updates
    include_winner_detection: bool = True  # Include winner detection alerts
    max_message_rate: Optional[int] = None  # Messages per second
    
    # Metrics-specific filtering
    metrics_update_types: Set[str] = Field(default_factory=set)  # Filter by metrics update type
    min_confidence_threshold: float = 0.0  # Minimum confidence for winner detection
    
    # Buffer settings
    buffer_size: int = 1000
    buffer_overflow_strategy: str = "drop_oldest"  # drop_oldest, drop_newest, block
    
    created_at: float = Field(default_factory=time.time)
    last_activity: float = Field(default_factory=time.time)
    
    def matches_message(self, message: Union[StreamMessage, 'MetricsUpdate']) -> bool:
        """Check if message matches subscription filters."""
        # Handle MetricsUpdate messages
        if METRICS_AVAILABLE and isinstance(message, MetricsUpdate):
            return self._matches_metrics_message(message)
        
        # Handle regular StreamMessage
        # Check sequence strategy filter
        if self.sequence_strategies and message.sequence_strategy not in self.sequence_strategies:
            return False
        
        # Check message type filter
        if self.message_types and message.message_type not in self.message_types:
            return False
        
        # Check content filters
        if not self.include_progress and message.message_type == "progress":
            return False
        
        if not self.include_errors and message.message_type == "error":
            return False
        
        if not self.include_results and message.message_type == "result":
            return False
        
        return True
    
    def _matches_metrics_message(self, metrics_update: 'MetricsUpdate') -> bool:
        """Check if metrics update matches subscription filters."""
        # Check if metrics are enabled
        if not self.include_metrics:
            return False
        
        # Check metrics update type filter
        if self.metrics_update_types and metrics_update.update_type.value not in self.metrics_update_types:
            return False
        
        # Check sequence strategy filter for metrics
        if (self.sequence_strategies and 
            metrics_update.updated_strategy and 
            metrics_update.updated_strategy not in self.sequence_strategies):
            return False
        
        # Check winner detection filter
        if (metrics_update.update_type == MetricsUpdateType.WINNER_DETECTED and 
            not self.include_winner_detection):
            return False
        
        # Check confidence threshold for winner detection
        if (metrics_update.update_type == MetricsUpdateType.WINNER_DETECTED and
            metrics_update.winner_analysis and
            metrics_update.winner_analysis.confidence_score < self.min_confidence_threshold):
            return False
        
        return True


class MessageBuffer:
    """Thread-safe message buffer with overflow handling."""
    
    def __init__(self, max_size: int = 1000, overflow_strategy: str = "drop_oldest"):
        self.max_size = max_size
        self.overflow_strategy = overflow_strategy
        self.buffer = deque(maxlen=max_size if overflow_strategy == "drop_oldest" else None)
        self.lock = asyncio.Lock()
        self._dropped_count = 0
    
    async def add_message(self, message: StreamMessage) -> bool:
        """Add message to buffer, returns True if successful."""
        async with self.lock:
            if self.overflow_strategy == "drop_oldest":
                # deque with maxlen automatically drops oldest
                self.buffer.append(message)
                return True
            
            elif self.overflow_strategy == "drop_newest":
                if len(self.buffer) >= self.max_size:
                    self._dropped_count += 1
                    return False
                self.buffer.append(message)
                return True
            
            elif self.overflow_strategy == "block":
                # In a real implementation, this would block until space is available
                # For now, we'll just drop if full
                if len(self.buffer) >= self.max_size:
                    return False
                self.buffer.append(message)
                return True
            
            return False
    
    async def get_messages(self, count: Optional[int] = None) -> List[StreamMessage]:
        """Get messages from buffer."""
        async with self.lock:
            if count is None:
                messages = list(self.buffer)
                self.buffer.clear()
                return messages
            else:
                messages = []
                for _ in range(min(count, len(self.buffer))):
                    if self.buffer:
                        messages.append(self.buffer.popleft())
                return messages
    
    async def peek_messages(self, count: int = 10) -> List[StreamMessage]:
        """Peek at messages without removing them."""
        async with self.lock:
            return list(list(self.buffer)[:count])
    
    @property
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    @property
    def dropped_count(self) -> int:
        """Get number of dropped messages."""
        return self._dropped_count


class WebSocketConnection:
    """Represents a WebSocket connection with state management."""
    
    def __init__(
        self,
        connection_id: str,
        websocket,  # WebSocket object (implementation-specific)
        client_info: Optional[Dict[str, Any]] = None
    ):
        self.connection_id = connection_id
        self.websocket = websocket
        self.client_info = client_info or {}
        
        # Connection state
        self.state = ConnectionState.CONNECTING
        self.connected_at = time.time()
        self.last_ping = time.time()
        
        # Metrics
        self.metrics = ConnectionMetrics()
        
        # Message tracking
        self.pending_acks: Dict[str, float] = {}  # message_id -> timestamp
        self.acknowledged_messages: Set[str] = set()
        
        # Rate limiting
        self.message_timestamps = deque(maxlen=100)  # Track recent message times
        
        # Heartbeat
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.heartbeat_interval = 30.0  # seconds
    
    async def send_message(self, message: StreamMessage) -> bool:
        """Send message through WebSocket with error handling."""
        try:
            if self.state != ConnectionState.CONNECTED:
                logger.warning(f"Attempted to send message on disconnected connection {self.connection_id}")
                return False
            
            # Track send time for latency calculation
            send_time = time.time()
            
            # Serialize message
            message_data = message.to_json()
            message_json = json.dumps(message_data)
            
            # Send through WebSocket (implementation-specific)
            await self.websocket.send(message_json)
            
            # Update metrics
            self.metrics.messages_sent += 1
            self.metrics.last_activity = send_time
            self.message_timestamps.append(send_time)
            
            # Track for acknowledgment if required
            if hasattr(message, 'requires_ack') and message.requires_ack:
                self.pending_acks[message.message_id] = send_time
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message on connection {self.connection_id}: {e}")
            self.metrics.messages_failed += 1
            await self._handle_connection_error(e)
            return False
    
    async def handle_incoming_message(self, raw_message: str):
        """Handle incoming message from WebSocket."""
        try:
            message_data = json.loads(raw_message)
            self.metrics.messages_received += 1
            self.metrics.last_activity = time.time()
            
            # Handle acknowledgments
            if message_data.get("type") == "ack":
                message_id = message_data.get("message_id")
                if message_id and message_id in self.pending_acks:
                    send_time = self.pending_acks.pop(message_id)
                    latency = time.time() - send_time
                    self.metrics.latency_samples.append(latency)
                    self.acknowledged_messages.add(message_id)
                    
                    # Keep only recent latency samples
                    if len(self.metrics.latency_samples) > 100:
                        self.metrics.latency_samples = self.metrics.latency_samples[-50:]
            
            # Handle pong responses
            elif message_data.get("type") == "pong":
                self.last_ping = time.time()
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON received on connection {self.connection_id}: {e}")
        except Exception as e:
            logger.error(f"Error handling incoming message on connection {self.connection_id}: {e}")
    
    async def start_heartbeat(self):
        """Start heartbeat monitoring."""
        if self.heartbeat_task:
            return
        
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def stop_heartbeat(self):
        """Stop heartbeat monitoring."""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
    
    async def _heartbeat_loop(self):
        """Heartbeat monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send ping
                ping_message = {
                    "type": "ping",
                    "timestamp": time.time()
                }
                await self.websocket.send(json.dumps(ping_message))
                
                # Check for missed pongs
                if time.time() - self.last_ping > self.heartbeat_interval * 2:
                    logger.warning(f"Connection {self.connection_id} missed heartbeat")
                    await self._handle_connection_timeout()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error on connection {self.connection_id}: {e}")
                await self._handle_connection_error(e)
                break
    
    async def _handle_connection_error(self, error: Exception):
        """Handle connection errors."""
        logger.error(f"Connection {self.connection_id} error: {error}")
        self.state = ConnectionState.FAILED
        await self.stop_heartbeat()
    
    async def _handle_connection_timeout(self):
        """Handle connection timeout."""
        logger.warning(f"Connection {self.connection_id} timed out")
        self.state = ConnectionState.DISCONNECTED
        await self.stop_heartbeat()
    
    def get_message_rate(self, window_seconds: int = 60) -> float:
        """Calculate current message rate."""
        now = time.time()
        cutoff = now - window_seconds
        recent_messages = [t for t in self.message_timestamps if t > cutoff]
        return len(recent_messages) / window_seconds
    
    async def close(self):
        """Close connection gracefully."""
        self.state = ConnectionState.DISCONNECTED
        await self.stop_heartbeat()
        
        try:
            await self.websocket.close()
        except Exception as e:
            logger.warning(f"Error closing connection {self.connection_id}: {e}")


class StreamMultiplexer:
    """Production-ready WebSocket stream multiplexer for parallel sequence execution."""
    
    def __init__(
        self,
        max_connections: int = 100,
        max_buffer_size: int = 10000,
        cleanup_interval: int = 300,  # 5 minutes
        metrics_aggregator: Optional['MetricsAggregator'] = None
    ):
        """Initialize stream multiplexer.
        
        Args:
            max_connections: Maximum number of concurrent connections
            max_buffer_size: Maximum buffer size per subscription
            cleanup_interval: Interval for cleaning up inactive connections (seconds)
            metrics_aggregator: Optional metrics aggregator for real-time metrics streaming
        """
        self.max_connections = max_connections
        self.max_buffer_size = max_buffer_size
        self.cleanup_interval = cleanup_interval
        self.metrics_aggregator = metrics_aggregator
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.client_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # client_id -> subscription_ids
        
        # Message routing
        self.message_buffers: Dict[str, MessageBuffer] = {}  # subscription_id -> buffer
        self.global_message_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.total_messages_routed = 0
        self.failed_deliveries = 0
        self.start_time = time.time()
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.delivery_task: Optional[asyncio.Task] = None
        self.metrics_streaming_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self.lock = asyncio.Lock()
        
        logger.info(f"StreamMultiplexer initialized: max_connections={max_connections}, "
                   f"metrics_enabled={metrics_aggregator is not None}")
    
    async def start(self):
        """Start background tasks."""
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if not self.delivery_task:
            self.delivery_task = asyncio.create_task(self._delivery_loop())
        
        # Start metrics streaming if aggregator is available
        if self.metrics_aggregator and METRICS_AVAILABLE and not self.metrics_streaming_task:
            self.metrics_streaming_task = asyncio.create_task(self._metrics_streaming_loop())
        
        logger.info("StreamMultiplexer started")
    
    async def stop(self):
        """Stop background tasks and close all connections."""
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.delivery_task:
            self.delivery_task.cancel()
            try:
                await self.delivery_task
            except asyncio.CancelledError:
                pass
        
        if self.metrics_streaming_task:
            self.metrics_streaming_task.cancel()
            try:
                await self.metrics_streaming_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self.lock:
            for connection in list(self.connections.values()):
                await connection.close()
            self.connections.clear()
            self.subscriptions.clear()
            self.client_subscriptions.clear()
            self.message_buffers.clear()
        
        logger.info("StreamMultiplexer stopped")
    
    async def add_connection(
        self,
        websocket,
        client_id: str,
        client_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new WebSocket connection."""
        if len(self.connections) >= self.max_connections:
            raise ValueError(f"Maximum connections ({self.max_connections}) reached")
        
        connection_id = str(uuid4())
        connection = WebSocketConnection(connection_id, websocket, client_info)
        
        async with self.lock:
            self.connections[connection_id] = connection
        
        # Start connection heartbeat
        await connection.start_heartbeat()
        connection.state = ConnectionState.CONNECTED
        
        logger.info(f"Added connection {connection_id} for client {client_id}")
        return connection_id
    
    async def remove_connection(self, connection_id: str):
        """Remove a WebSocket connection."""
        async with self.lock:
            connection = self.connections.pop(connection_id, None)
            if connection:
                await connection.close()
                
                # Clean up subscriptions for this connection
                client_id = connection.client_info.get("client_id")
                if client_id and client_id in self.client_subscriptions:
                    subscription_ids = self.client_subscriptions.pop(client_id, set())
                    for sub_id in subscription_ids:
                        self.subscriptions.pop(sub_id, None)
                        self.message_buffers.pop(sub_id, None)
        
        logger.info(f"Removed connection {connection_id}")
    
    async def create_subscription(
        self,
        client_id: str,
        sequence_strategies: Optional[Set[str]] = None,
        message_types: Optional[Set[str]] = None,
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_MOST_ONCE,
        **kwargs
    ) -> str:
        """Create a new stream subscription."""
        subscription = StreamSubscription(
            client_id=client_id,
            sequence_strategies=sequence_strategies or set(),
            message_types=message_types or set(),
            delivery_guarantee=delivery_guarantee,
            **kwargs
        )
        
        # Create message buffer
        buffer = MessageBuffer(
            max_size=min(subscription.buffer_size, self.max_buffer_size),
            overflow_strategy=subscription.buffer_overflow_strategy
        )
        
        async with self.lock:
            self.subscriptions[subscription.subscription_id] = subscription
            self.message_buffers[subscription.subscription_id] = buffer
            self.client_subscriptions[client_id].add(subscription.subscription_id)
        
        logger.info(f"Created subscription {subscription.subscription_id} for client {client_id}")
        return subscription.subscription_id
    
    async def remove_subscription(self, subscription_id: str):
        """Remove a stream subscription."""
        async with self.lock:
            subscription = self.subscriptions.pop(subscription_id, None)
            self.message_buffers.pop(subscription_id, None)
            
            if subscription:
                client_subs = self.client_subscriptions.get(subscription.client_id, set())
                client_subs.discard(subscription_id)
                if not client_subs:
                    self.client_subscriptions.pop(subscription.client_id, None)
        
        logger.info(f"Removed subscription {subscription_id}")
    
    async def create_metrics_subscription(
        self,
        client_id: str,
        execution_id: Optional[str] = None,
        strategies: Optional[Set[str]] = None,
        update_types: Optional[Set[str]] = None,
        min_confidence: float = 0.0,
        include_winner_detection: bool = True,
        **kwargs
    ) -> str:
        """Create a subscription specifically for metrics updates.
        
        Args:
            client_id: Client identifier
            execution_id: Optional execution ID to filter by
            strategies: Optional sequence strategies to filter by
            update_types: Optional metrics update types to filter by
            min_confidence: Minimum confidence threshold for winner detection
            include_winner_detection: Include winner detection alerts
            **kwargs: Additional subscription parameters
            
        Returns:
            Subscription ID
        """
        # Configure metrics-specific settings
        metrics_config = {
            'include_metrics': True,
            'include_winner_detection': include_winner_detection,
            'min_confidence_threshold': min_confidence,
            'metrics_update_types': update_types or set(),
            'sequence_strategies': strategies or set(),
            'message_types': {'metrics_update', 'winner_detected', 'real_time'},
            **kwargs
        }
        
        subscription_id = await self.create_subscription(client_id, **metrics_config)
        
        logger.info(f"Created metrics subscription {subscription_id} for client {client_id} "
                   f"(execution_id={execution_id}, strategies={len(strategies or [])}, "
                   f"winner_detection={include_winner_detection})")
        
        return subscription_id
    
    async def route_message(self, message: StreamMessage) -> Dict[str, bool]:
        """Route message to matching subscriptions.
        
        Returns:
            Dictionary mapping subscription_id to delivery success status
        """
        delivery_results = {}
        
        # Add to global history
        self.global_message_history.append(message)
        self.total_messages_routed += 1
        
        async with self.lock:
            # Find matching subscriptions
            for subscription_id, subscription in self.subscriptions.items():
                if subscription.matches_message(message):
                    # Check rate limiting
                    if subscription.max_message_rate:
                        # Simplified rate limiting - could be enhanced
                        now = time.time()
                        if now - subscription.last_activity < (1.0 / subscription.max_message_rate):
                            delivery_results[subscription_id] = False
                            continue
                    
                    # Add to buffer
                    buffer = self.message_buffers.get(subscription_id)
                    if buffer:
                        success = await buffer.add_message(message)
                        delivery_results[subscription_id] = success
                        
                        if success:
                            subscription.last_activity = time.time()
                        else:
                            self.failed_deliveries += 1
                            logger.warning(f"Failed to buffer message for subscription {subscription_id}")
        
        return delivery_results
    
    async def get_client_connection(self, client_id: str) -> Optional[WebSocketConnection]:
        """Get active connection for a client."""
        async with self.lock:
            for connection in self.connections.values():
                if connection.client_info.get("client_id") == client_id:
                    if connection.state == ConnectionState.CONNECTED:
                        return connection
        return None
    
    async def get_subscription_messages(
        self,
        subscription_id: str,
        count: Optional[int] = None
    ) -> List[StreamMessage]:
        """Get messages for a specific subscription."""
        buffer = self.message_buffers.get(subscription_id)
        if buffer:
            return await buffer.get_messages(count)
        return []
    
    async def broadcast_to_strategy(
        self,
        message: StreamMessage,
        strategy: str
    ) -> Dict[str, bool]:
        """Broadcast message to all subscriptions for a specific strategy."""
        message.sequence_strategy = strategy
        return await self.route_message(message)
    
    async def route_metrics_update(self, metrics_update: 'MetricsUpdate') -> Dict[str, bool]:
        """Route metrics update to matching subscriptions.
        
        Returns:
            Dictionary mapping subscription_id to delivery success status
        """
        if not METRICS_AVAILABLE:
            return {}
        
        delivery_results = {}
        
        # Add to global history
        self.global_message_history.append(metrics_update)
        self.total_messages_routed += 1
        
        async with self.lock:
            # Find matching subscriptions
            for subscription_id, subscription in self.subscriptions.items():
                if subscription.matches_message(metrics_update):
                    # Check rate limiting
                    if subscription.max_message_rate:
                        # Simplified rate limiting - could be enhanced
                        now = time.time()
                        if now - subscription.last_activity < (1.0 / subscription.max_message_rate):
                            delivery_results[subscription_id] = False
                            continue
                    
                    # Convert metrics update to stream message for buffer compatibility
                    stream_message = StreamMessage(
                        message_type="metrics_update",
                        sequence_strategy=metrics_update.updated_strategy,
                        data=metrics_update.to_websocket_message()
                    )
                    
                    # Add to buffer
                    buffer = self.message_buffers.get(subscription_id)
                    if buffer:
                        success = await buffer.add_message(stream_message)
                        delivery_results[subscription_id] = success
                        
                        if success:
                            subscription.last_activity = time.time()
                        else:
                            self.failed_deliveries += 1
                            logger.warning(f"Failed to buffer metrics update for subscription {subscription_id}")
        
        return delivery_results
    
    async def _delivery_loop(self):
        """Background loop for message delivery."""
        while True:
            try:
                await asyncio.sleep(0.1)  # 100ms delivery interval
                
                async with self.lock:
                    # Process each subscription's buffer
                    for subscription_id, subscription in self.subscriptions.items():
                        client_connection = await self.get_client_connection(subscription.client_id)
                        
                        if not client_connection:
                            continue
                        
                        buffer = self.message_buffers.get(subscription_id)
                        if not buffer or buffer.size == 0:
                            continue
                        
                        # Get messages to deliver
                        batch_size = min(10, buffer.size)  # Deliver in small batches
                        messages = await buffer.get_messages(batch_size)
                        
                        # Deliver messages
                        for message in messages:
                            success = await client_connection.send_message(message)
                            if not success:
                                # Re-add message to buffer if delivery failed
                                await buffer.add_message(message)
                                break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in delivery loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _cleanup_loop(self):
        """Background cleanup of inactive connections and subscriptions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                now = time.time()
                cleanup_threshold = 3600  # 1 hour
                
                async with self.lock:
                    # Clean up inactive connections
                    inactive_connections = []
                    for conn_id, connection in self.connections.items():
                        if (now - connection.metrics.last_activity) > cleanup_threshold:
                            if connection.state != ConnectionState.CONNECTED:
                                inactive_connections.append(conn_id)
                    
                    for conn_id in inactive_connections:
                        await self.remove_connection(conn_id)
                        logger.info(f"Cleaned up inactive connection {conn_id}")
                    
                    # Clean up old messages from buffers
                    for buffer in self.message_buffers.values():
                        # Simple cleanup - remove old messages (enhance with timestamp checking)
                        if buffer.size > buffer.max_size * 0.9:
                            await buffer.get_messages(buffer.size // 4)
                
                logger.debug(f"Cleanup completed: {len(inactive_connections)} connections removed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_streaming_loop(self):
        """Background loop for streaming metrics updates from aggregator."""
        if not self.metrics_aggregator or not METRICS_AVAILABLE:
            return
        
        logger.info("Starting metrics streaming loop")
        
        try:
            # Subscribe to metrics updates from aggregator
            async for metrics_update in self.metrics_aggregator.stream_metrics_updates():
                try:
                    # Route metrics update to subscribers
                    delivery_results = await self.route_metrics_update(metrics_update)
                    
                    # Log delivery summary
                    successful_deliveries = sum(1 for success in delivery_results.values() if success)
                    total_subscriptions = len(delivery_results)
                    
                    if total_subscriptions > 0:
                        logger.debug(f"Metrics update delivered to {successful_deliveries}/{total_subscriptions} subscriptions")
                    
                except Exception as e:
                    logger.error(f"Error routing metrics update: {e}")
                    await asyncio.sleep(1.0)
                    
        except asyncio.CancelledError:
            logger.info("Metrics streaming loop cancelled")
        except Exception as e:
            logger.error(f"Error in metrics streaming loop: {e}")
        finally:
            logger.info("Metrics streaming loop stopped")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics."""
        active_connections = sum(
            1 for conn in self.connections.values() 
            if conn.state == ConnectionState.CONNECTED
        )
        
        total_subscriptions = len(self.subscriptions)
        total_buffer_size = sum(buffer.size for buffer in self.message_buffers.values())
        
        uptime = time.time() - self.start_time
        messages_per_second = self.total_messages_routed / uptime if uptime > 0 else 0
        
        return {
            "active_connections": active_connections,
            "total_connections": len(self.connections),
            "total_subscriptions": total_subscriptions,
            "total_messages_routed": self.total_messages_routed,
            "failed_deliveries": self.failed_deliveries,
            "delivery_success_rate": ((self.total_messages_routed - self.failed_deliveries) / 
                                    max(self.total_messages_routed, 1)) * 100,
            "messages_per_second": messages_per_second,
            "total_buffer_size": total_buffer_size,
            "uptime_seconds": uptime
        }
    
    def get_subscription_stats(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific subscription."""
        subscription = self.subscriptions.get(subscription_id)
        buffer = self.message_buffers.get(subscription_id)
        
        if not subscription or not buffer:
            return None
        
        return {
            "subscription_id": subscription_id,
            "client_id": subscription.client_id,
            "strategies": [s.value for s in subscription.sequence_strategies],
            "message_types": list(subscription.message_types),
            "delivery_guarantee": subscription.delivery_guarantee.value,
            "buffer_size": buffer.size,
            "max_buffer_size": buffer.max_size,
            "dropped_messages": buffer.dropped_count,
            "created_at": subscription.created_at,
            "last_activity": subscription.last_activity
        }


# Convenience function for creating configured multiplexer
async def create_stream_multiplexer(
    max_connections: int = 100,
    max_buffer_size: int = 10000
) -> StreamMultiplexer:
    """Create and start a configured stream multiplexer."""
    multiplexer = StreamMultiplexer(
        max_connections=max_connections,
        max_buffer_size=max_buffer_size
    )
    await multiplexer.start()
    return multiplexer