"""A2A Protocol implementation for inter-agent communication.

Based on Google's Agent-to-Agent (A2A) protocol for standardized agent communication.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field


# A2A Protocol Data Models
class ContentItem(BaseModel):
    """Individual content item in A2A message."""
    type: str = Field(description="Content type (text, image, structured_data, etc.)")
    text: Optional[str] = Field(default=None, description="Text content")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Structured data content")


class A2AMessage(BaseModel):
    """A2A protocol message structure."""
    task_id: str = Field(description="Unique task identifier")
    content: List[ContentItem] = Field(description="Message content items")
    context: Optional[str] = Field(default=None, description="Compressed context summary")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class A2AResponse(BaseModel):
    """A2A protocol response structure."""
    result: Dict[str, Any] = Field(description="Response result data")
    trace_id: Optional[str] = Field(default=None, description="Trace identifier for observability")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Response metadata")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentCard(BaseModel):
    """Agent capability advertisement card."""
    name: str = Field(description="Agent name/identifier")
    description: str = Field(description="Agent description")
    version: str = Field(description="Agent version")
    capabilities: Dict[str, Any] = Field(description="Agent capabilities and schemas")
    endpoints: Dict[str, str] = Field(description="Agent endpoint URLs")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class Task(BaseModel):
    """Task definition for delegation."""
    id: str = Field(description="Unique task ID")
    description: str = Field(description="Task description")
    context_summary: Optional[str] = Field(default=None, description="Compressed context")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task metadata")


class AgentResult(BaseModel):
    """Result from agent execution."""
    summary: str = Field(description="Result summary")
    artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Result artifacts")
    trace_id: Optional[str] = Field(default=None, description="Execution trace ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Result metadata")


class A2AClient:
    """A2A Protocol client for inter-agent communication."""

    def __init__(self, base_url: Optional[str] = None):
        """Initialize A2A client.

        Args:
            base_url: Base URL for A2A service (optional)
        """
        self.base_url = base_url or "http://localhost:8000"
        self.logger = logging.getLogger(__name__)
        self._session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self):
        """Initialize HTTP client session."""
        if self._session is None:
            self._session = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )

    async def close(self):
        """Close HTTP client session."""
        if self._session:
            await self._session.aclose()
            self._session = None

    async def delegate_task(self, agent_card: AgentCard, task: Task) -> AgentResult:
        """Delegate task to remote agent using A2A protocol.

        Args:
            agent_card: Target agent card
            task: Task to delegate

        Returns:
            AgentResult from the delegated task

        Raises:
            httpx.HTTPError: If HTTP request fails
            ValueError: If response format is invalid
        """
        if not self._session:
            await self.initialize()

        # Construct A2A message
        message = A2AMessage(
            task_id=task.id,
            content=[ContentItem(type="text", text=task.description)],
            context=task.context_summary,
            metadata=task.metadata
        )

        # Get agent endpoint
        agent_base_url = agent_card.endpoints.get("base_url")
        if not agent_base_url:
            raise ValueError(f"No base_url found in agent card: {agent_card.name}")


        try:
            # Send A2A message
            response = await self.send_a2a_message(agent_base_url, message)

            # Parse response
            a2a_response = A2AResponse(**response)

            return AgentResult(
                summary=a2a_response.result.get("summary", ""),
                artifacts=a2a_response.result.get("artifacts", []),
                trace_id=a2a_response.trace_id,
                metadata=a2a_response.metadata
            )

        except httpx.HTTPError as e:
            self.logger.error(f"A2A request failed for agent {agent_card.name}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to parse A2A response from {agent_card.name}: {e}")
            raise ValueError(f"Invalid A2A response format: {e}") from e

    async def send_a2a_message(self, agent_url: str, message: A2AMessage) -> Dict[str, Any]:
        """Send A2A message to agent endpoint.

        Args:
            agent_url: Target agent base URL
            message: A2A message to send

        Returns:
            Raw response dictionary

        Raises:
            httpx.HTTPError: If HTTP request fails
        """
        if not self._session:
            await self.initialize()

        endpoint = f"{agent_url}/a2a/task"

        try:
            response = await self._session.post(
                endpoint,
                json=message.dict(),
                headers={"Content-Type": "application/json"}
            )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            self.logger.error(f"A2A HTTP request failed: {e}")
            raise

    async def get_agent_card(self, agent_url: str) -> AgentCard:
        """Retrieve agent card from agent discovery endpoint.

        Args:
            agent_url: Agent base URL

        Returns:
            AgentCard for the specified agent

        Raises:
            httpx.HTTPError: If HTTP request fails
            ValueError: If response format is invalid
        """
        if not self._session:
            await self.initialize()

        endpoint = f"{agent_url}/a2a/card"

        try:
            response = await self._session.get(endpoint)
            response.raise_for_status()

            card_data = response.json()
            return AgentCard(**card_data)

        except httpx.HTTPError as e:
            self.logger.error(f"Failed to retrieve agent card from {agent_url}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to parse agent card from {agent_url}: {e}")
            raise ValueError(f"Invalid agent card format: {e}") from e

    async def discover_agents(self, discovery_url: str) -> List[AgentCard]:
        """Discover available agents from discovery service.

        Args:
            discovery_url: URL of agent discovery service

        Returns:
            List of available AgentCards

        Raises:
            httpx.HTTPError: If discovery request fails
        """
        if not self._session:
            await self.initialize()

        try:
            response = await self._session.get(discovery_url)
            response.raise_for_status()

            agents_data = response.json()
            return [AgentCard(**agent_data) for agent_data in agents_data]

        except httpx.HTTPError as e:
            self.logger.error(f"Agent discovery failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to parse agent discovery response: {e}")
            return []


class A2AServer:
    """A2A Protocol server for handling incoming agent requests."""

    def __init__(self, agent_name: str, host: str = "0.0.0.0", port: int = 8000):
        """Initialize A2A server.

        Args:
            agent_name: Name of this agent
            host: Server host
            port: Server port
        """
        self.agent_name = agent_name
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        self._app = None
        self._server = None
        self._task_handler = None

    def set_task_handler(self, handler_func):
        """Set function to handle incoming tasks.

        Args:
            handler_func: Async function that takes Task and returns AgentResult
        """
        self._task_handler = handler_func

    def get_agent_card(self) -> AgentCard:
        """Get agent card for this server.

        Returns:
            AgentCard describing this agent's capabilities
        """
        return AgentCard(
            name=self.agent_name,
            description=f"A2A-compatible {self.agent_name} agent",
            version="1.0.0",
            capabilities={
                "skills": [
                    {
                        "name": "process_task",
                        "description": "Process research tasks using A2A protocol",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string"},
                                "content": {"type": "array", "items": {"type": "object"}},
                                "context": {"type": "string"}
                            }
                        },
                        "output_schema": {
                            "type": "object",
                            "properties": {
                                "summary": {"type": "string"},
                                "artifacts": {"type": "array"},
                                "trace_id": {"type": "string"}
                            }
                        }
                    }
                ]
            },
            endpoints={
                "base_url": f"http://{self.host}:{self.port}"
            }
        )

    async def handle_task(self, message: A2AMessage) -> A2AResponse:
        """Handle incoming A2A task.

        Args:
            message: A2A message containing task

        Returns:
            A2AResponse with task result
        """
        if not self._task_handler:
            raise ValueError("No task handler set for A2A server")

        # Convert A2A message to Task
        task = Task(
            id=message.task_id,
            description="\n".join([item.text for item in message.content if item.text]),
            context_summary=message.context,
            metadata=message.metadata
        )

        # Execute task using handler
        result = await self._task_handler(task)

        # Convert result to A2A response
        return A2AResponse(
            result={
                "summary": result.summary,
                "artifacts": result.artifacts
            },
            trace_id=result.trace_id,
            metadata=result.metadata
        )

    async def start(self):
        """Start A2A server."""
        from fastapi import FastAPI, HTTPException

        self._app = FastAPI(title=f"A2A Agent Server - {self.agent_name}")

        @self._app.get("/a2a/card")
        async def get_card():
            """Get agent capability card."""
            return self.get_agent_card().dict()

        @self._app.post("/a2a/task")
        async def process_task(message: dict):
            """Process incoming A2A task."""
            try:
                a2a_message = A2AMessage(**message)
                response = await self.handle_task(a2a_message)
                return response.dict()
            except Exception as e:
                self.logger.error(f"Error processing A2A task: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Start server
        config = uvicorn.Config(
            app=self._app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def stop(self):
        """Stop A2A server."""
        if self._server:
            self._server.should_exit = True


# Utility functions for A2A protocol
def create_task_from_research_query(query: str, context: Optional[str] = None) -> Task:
    """Create Task from research query.

    Args:
        query: Research query string
        context: Optional context summary

    Returns:
        Task object
    """
    import uuid
    return Task(
        id=str(uuid.uuid4()),
        description=query,
        context_summary=context,
        metadata={"type": "research", "timestamp": datetime.utcnow().isoformat()}
    )


def format_task_for_display(task: Task) -> str:
    """Format task for display purposes.

    Args:
        task: Task to format

    Returns:
        Formatted task string
    """
    return f"Task: {task.description} (ID: {task.id})"
