"""Production-ready WebSocket and HTTP API endpoints for real-time metrics streaming.

This module provides comprehensive API endpoints for accessing real-time metrics,
creating subscriptions, and managing metrics streaming for parallel sequence execution.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .metrics_aggregator import MetricsAggregator
from .parallel_executor import ParallelSequenceExecutor
from .stream_multiplexer import StreamMultiplexer, create_stream_multiplexer

logger = logging.getLogger(__name__)


# API Models for request/response

class MetricsSubscriptionRequest(BaseModel):
    """Request model for creating metrics subscriptions."""
    
    client_id: str
    execution_id: Optional[str] = None
    strategies: Optional[List[str]] = None  # Strategy names
    update_types: Optional[List[str]] = None  # MetricsUpdateType values
    min_confidence: float = 0.0
    include_winner_detection: bool = True
    include_progress: bool = True
    include_errors: bool = True
    max_message_rate: Optional[int] = None
    buffer_size: int = 1000


class MetricsSubscriptionResponse(BaseModel):
    """Response model for subscription creation."""
    
    subscription_id: str
    client_id: str
    created_at: datetime
    websocket_url: str
    settings: Dict[str, Any]


class MetricsSnapshotResponse(BaseModel):
    """Response model for metrics snapshots."""
    
    execution_id: str
    timestamp: datetime
    sequence_metrics: Dict[str, Any]
    parallel_metrics: Optional[Dict[str, Any]]
    winner_analysis: Optional[Dict[str, Any]]
    system_metrics: Dict[str, Any]


class ExecutionListResponse(BaseModel):
    """Response model for listing active executions."""
    
    executions: List[Dict[str, Any]]
    total_count: int
    active_count: int
    completed_count: int


class SystemStatsResponse(BaseModel):
    """Response model for system statistics."""
    
    metrics_aggregator_stats: Dict[str, Any]
    stream_multiplexer_stats: Dict[str, Any]
    active_executions: int
    total_subscriptions: int
    uptime_seconds: float


# Global instances (initialized by application startup)
metrics_aggregator: Optional[MetricsAggregator] = None
stream_multiplexer: Optional[StreamMultiplexer] = None
parallel_executor: Optional[ParallelSequenceExecutor] = None


class MetricsAPI:
    """Production-ready metrics API with WebSocket streaming and HTTP endpoints."""
    
    def __init__(
        self,
        title: str = "Open Deep Research Metrics API",
        version: str = "1.0.0",
        enable_cors: bool = True
    ):
        """Initialize the metrics API.
        
        Args:
            title: API title
            version: API version
            enable_cors: Enable CORS middleware
        """
        self.app = FastAPI(
            title=title,
            version=version,
            description="Real-time metrics streaming API for parallel sequence execution"
        )
        
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Configure appropriately for production
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        self._setup_routes()
        logger.info(f"MetricsAPI initialized: {title} v{version}")
    
    def _setup_routes(self):
        """Set up API routes."""
        # WebSocket endpoints
        self.app.websocket("/ws/metrics")(self.websocket_metrics_stream)
        self.app.websocket("/ws/metrics/{execution_id}")(self.websocket_execution_metrics)
        
        # HTTP endpoints
        self.app.post("/api/v1/subscriptions", response_model=MetricsSubscriptionResponse)(
            self.create_subscription
        )
        self.app.delete("/api/v1/subscriptions/{subscription_id}")(
            self.delete_subscription
        )
        self.app.get("/api/v1/subscriptions/{subscription_id}")(
            self.get_subscription
        )
        
        self.app.get("/api/v1/executions", response_model=ExecutionListResponse)(
            self.list_executions
        )
        self.app.get("/api/v1/executions/{execution_id}/metrics", response_model=MetricsSnapshotResponse)(
            self.get_execution_metrics
        )
        self.app.get("/api/v1/executions/{execution_id}/winner")(
            self.get_winner_analysis
        )
        
        self.app.get("/api/v1/system/stats", response_model=SystemStatsResponse)(
            self.get_system_stats
        )
        self.app.get("/api/v1/system/health")(
            self.health_check
        )
        
        # Startup and shutdown events
        self.app.on_event("startup")(self.startup)
        self.app.on_event("shutdown")(self.shutdown)
    
    async def startup(self):
        """Initialize global components on startup."""
        global metrics_aggregator, stream_multiplexer, parallel_executor
        
        try:
            # Initialize metrics aggregator
            metrics_aggregator = MetricsAggregator(
                update_interval=1.0,
                winner_detection_enabled=True
            )
            await metrics_aggregator.start()
            
            # Initialize stream multiplexer with metrics integration
            stream_multiplexer = await create_stream_multiplexer(
                max_connections=100,
                max_buffer_size=10000
            )
            stream_multiplexer.metrics_aggregator = metrics_aggregator
            
            logger.info("Metrics API startup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during startup: {e}")
            raise
    
    async def shutdown(self):
        """Clean up global components on shutdown."""
        global metrics_aggregator, stream_multiplexer, parallel_executor
        
        try:
            if stream_multiplexer:
                await stream_multiplexer.stop()
            
            if metrics_aggregator:
                await metrics_aggregator.stop()
            
            logger.info("Metrics API shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    # WebSocket endpoints
    
    async def websocket_metrics_stream(self, websocket: WebSocket):
        """WebSocket endpoint for streaming all metrics updates."""
        await websocket.accept()
        client_id = str(uuid4())
        connection_id = None
        subscription_id = None
        
        try:
            # Add WebSocket connection
            if stream_multiplexer:
                connection_id = await stream_multiplexer.add_connection(
                    websocket=websocket,
                    client_id=client_id,
                    client_info={"type": "metrics_stream", "connected_at": datetime.utcnow()}
                )
                
                # Create default metrics subscription
                subscription_id = await stream_multiplexer.create_metrics_subscription(
                    client_id=client_id,
                    include_winner_detection=True,
                    include_metrics=True
                )
                
                logger.info(f"WebSocket metrics stream connected: client={client_id}, "
                           f"connection={connection_id}, subscription={subscription_id}")
                
                # Keep connection alive and handle messages
                while True:
                    try:
                        # Wait for client messages (heartbeat, subscription updates, etc.)
                        message = await websocket.receive_text()
                        
                        # Handle client messages
                        await self._handle_websocket_message(
                            websocket, client_id, subscription_id, message
                        )
                        
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"Error in WebSocket message handling: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        }))
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket metrics stream disconnected: client={client_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket metrics stream: {e}")
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Internal server error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            except:
                pass
        finally:
            # Clean up connection and subscription
            if stream_multiplexer:
                if subscription_id:
                    await stream_multiplexer.remove_subscription(subscription_id)
                if connection_id:
                    await stream_multiplexer.remove_connection(connection_id)
    
    async def websocket_execution_metrics(self, websocket: WebSocket, execution_id: str):
        """WebSocket endpoint for streaming metrics for a specific execution."""
        await websocket.accept()
        client_id = str(uuid4())
        connection_id = None
        subscription_id = None
        
        try:
            if stream_multiplexer and metrics_aggregator:
                # Check if execution exists
                if execution_id not in metrics_aggregator.parallel_metrics:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Execution {execution_id} not found",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    return
                
                # Add WebSocket connection
                connection_id = await stream_multiplexer.add_connection(
                    websocket=websocket,
                    client_id=client_id,
                    client_info={
                        "type": "execution_metrics", 
                        "execution_id": execution_id,
                        "connected_at": datetime.utcnow()
                    }
                )
                
                # Create execution-specific subscription
                subscription_id = await stream_multiplexer.create_metrics_subscription(
                    client_id=client_id,
                    execution_id=execution_id,
                    include_winner_detection=True,
                    include_metrics=True
                )
                
                # Send initial snapshot
                snapshot = metrics_aggregator.get_execution_snapshot(execution_id)
                await websocket.send_text(json.dumps({
                    "type": "snapshot",
                    "data": snapshot,
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
                logger.info(f"WebSocket execution metrics connected: execution={execution_id}, "
                           f"client={client_id}")
                
                # Keep connection alive
                while True:
                    try:
                        message = await websocket.receive_text()
                        await self._handle_websocket_message(
                            websocket, client_id, subscription_id, message
                        )
                    except WebSocketDisconnect:
                        break
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket execution metrics disconnected: execution={execution_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket execution metrics: {e}")
        finally:
            # Clean up
            if stream_multiplexer:
                if subscription_id:
                    await stream_multiplexer.remove_subscription(subscription_id)
                if connection_id:
                    await stream_multiplexer.remove_connection(connection_id)
    
    async def _handle_websocket_message(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        subscription_id: str, 
        message: str
    ):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            
            elif message_type == "update_subscription":
                # Handle subscription updates
                logger.debug(f"Subscription update request from client {client_id}")
                # Implementation for subscription updates can be added here
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client {client_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    # HTTP endpoints
    
    async def create_subscription(self, request: MetricsSubscriptionRequest):
        """Create a new metrics subscription."""
        if not stream_multiplexer:
            raise HTTPException(status_code=503, detail="Stream multiplexer not available")
        
        try:
            # Convert strategy names to enums
            strategies = None
            if request.strategies:
                strategies = {
                    strategy for strategy in request.strategies
                }
            
            # Convert update types
            update_types = None
            if request.update_types:
                update_types = set(request.update_types)
            
            # Create subscription
            subscription_id = await stream_multiplexer.create_metrics_subscription(
                client_id=request.client_id,
                execution_id=request.execution_id,
                strategies=strategies,
                update_types=update_types,
                min_confidence=request.min_confidence,
                include_winner_detection=request.include_winner_detection,
                include_progress=request.include_progress,
                include_errors=request.include_errors,
                max_message_rate=request.max_message_rate,
                buffer_size=request.buffer_size
            )
            
            return MetricsSubscriptionResponse(
                subscription_id=subscription_id,
                client_id=request.client_id,
                created_at=datetime.utcnow(),
                websocket_url=f"/ws/metrics?subscription_id={subscription_id}",
                settings={
                    "execution_id": request.execution_id,
                    "strategies": request.strategies,
                    "update_types": request.update_types,
                    "min_confidence": request.min_confidence,
                    "include_winner_detection": request.include_winner_detection
                }
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error creating subscription: {e}")
            raise HTTPException(status_code=500, detail="Failed to create subscription")
    
    async def delete_subscription(self, subscription_id: str):
        """Delete a metrics subscription."""
        if not stream_multiplexer:
            raise HTTPException(status_code=503, detail="Stream multiplexer not available")
        
        await stream_multiplexer.remove_subscription(subscription_id)
        return {"message": f"Subscription {subscription_id} deleted"}
    
    async def get_subscription(self, subscription_id: str):
        """Get subscription details."""
        if not stream_multiplexer:
            raise HTTPException(status_code=503, detail="Stream multiplexer not available")
        
        stats = stream_multiplexer.get_subscription_stats(subscription_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        return stats
    
    async def list_executions(self):
        """List all active and recent executions."""
        if not metrics_aggregator:
            raise HTTPException(status_code=503, detail="Metrics aggregator not available")
        
        executions = []
        active_count = 0
        completed_count = 0
        
        for execution_id, parallel_metrics in metrics_aggregator.parallel_metrics.items():
            execution_info = {
                "execution_id": execution_id,
                "start_time": parallel_metrics.start_time.isoformat(),
                "sequence_count": parallel_metrics.sequence_count,
                "status": "active" if parallel_metrics.active_sequences > 0 else "completed",
                "completion_rate": parallel_metrics.completion_rate,
                "best_strategy": parallel_metrics.best_strategy if parallel_metrics.best_strategy else None,
                "significant_difference": parallel_metrics.significant_difference_detected
            }
            
            executions.append(execution_info)
            
            if parallel_metrics.active_sequences > 0:
                active_count += 1
            else:
                completed_count += 1
        
        return ExecutionListResponse(
            executions=executions,
            total_count=len(executions),
            active_count=active_count,
            completed_count=completed_count
        )
    
    async def get_execution_metrics(self, execution_id: str):
        """Get current metrics snapshot for an execution."""
        if not metrics_aggregator:
            raise HTTPException(status_code=503, detail="Metrics aggregator not available")
        
        if execution_id not in metrics_aggregator.parallel_metrics:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        snapshot = metrics_aggregator.get_execution_snapshot(execution_id)
        system_metrics = metrics_aggregator.get_system_metrics()
        
        return MetricsSnapshotResponse(
            execution_id=execution_id,
            timestamp=datetime.utcnow(),
            sequence_metrics=snapshot.get("sequence_metrics", {}),
            parallel_metrics=snapshot.get("parallel_metrics"),
            winner_analysis=snapshot.get("winner_analysis"),
            system_metrics=system_metrics
        )
    
    async def get_winner_analysis(self, execution_id: str):
        """Get winner analysis for an execution."""
        if not metrics_aggregator:
            raise HTTPException(status_code=503, detail="Metrics aggregator not available")
        
        winner_analysis = metrics_aggregator.winner_analyses.get(execution_id)
        if not winner_analysis:
            raise HTTPException(status_code=404, detail="Winner analysis not available")
        
        return winner_analysis.model_dump()
    
    async def get_system_stats(self):
        """Get system-wide statistics."""
        stats = {}
        
        if metrics_aggregator:
            stats["metrics_aggregator_stats"] = metrics_aggregator.get_system_metrics()
        
        if stream_multiplexer:
            stats["stream_multiplexer_stats"] = stream_multiplexer.get_connection_stats()
        
        if parallel_executor:
            stats["parallel_executor_stats"] = parallel_executor.get_performance_summary()
        
        return SystemStatsResponse(
            metrics_aggregator_stats=stats.get("metrics_aggregator_stats", {}),
            stream_multiplexer_stats=stats.get("stream_multiplexer_stats", {}),
            active_executions=stats.get("metrics_aggregator_stats", {}).get("active_executions", 0),
            total_subscriptions=stats.get("stream_multiplexer_stats", {}).get("total_subscriptions", 0),
            uptime_seconds=stats.get("metrics_aggregator_stats", {}).get("uptime_seconds", 0)
        )
    
    async def health_check(self):
        """Health check endpoint."""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "metrics_aggregator": metrics_aggregator is not None,
                "stream_multiplexer": stream_multiplexer is not None,
                "parallel_executor": parallel_executor is not None
            }
        }
        
        # Check if components are actually running
        if metrics_aggregator:
            health["components"]["metrics_aggregator_running"] = metrics_aggregator._running
        
        if stream_multiplexer:
            health["components"]["stream_multiplexer_tasks"] = {
                "cleanup_task": stream_multiplexer.cleanup_task is not None,
                "delivery_task": stream_multiplexer.delivery_task is not None,
                "metrics_task": stream_multiplexer.metrics_streaming_task is not None
            }
        
        return health


# Convenience functions for creating configured API instances

def create_metrics_api(
    title: str = "Open Deep Research Metrics API",
    version: str = "1.0.0",
    enable_cors: bool = True
) -> MetricsAPI:
    """Create a configured metrics API instance."""
    return MetricsAPI(title=title, version=version, enable_cors=enable_cors)


def get_fastapi_app(
    title: str = "Open Deep Research Metrics API",
    version: str = "1.0.0",
    enable_cors: bool = True
) -> FastAPI:
    """Get the FastAPI application instance."""
    api = create_metrics_api(title=title, version=version, enable_cors=enable_cors)
    return api.app


# Production ASGI application
app = get_fastapi_app()

if __name__ == "__main__":
    import uvicorn
    
    # Development server
    uvicorn.run(
        "metrics_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )