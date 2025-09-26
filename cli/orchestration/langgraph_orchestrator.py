"""LangGraph orchestration engine for research workflows.

This module implements the LangGraph-based orchestration engine from the blueprint,
providing graph-based state management for research workflows.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

try:
    from ..agents.research_agent import ResearchAgent
    from ..core.a2a_client import A2AClient, AgentCard, AgentResult, Task
    from ..core.context_tree import ContextTree, create_research_context_tree
    from ..utils.research_types import ResearchState, RoutedMessage, StreamMessage
except ImportError:
    # For running as standalone module
    from agents.research_agent import ResearchAgent
    from core.a2a_client import A2AClient, Task
    from core.context_tree import create_research_context_tree
    from utils.research_types import ResearchState, StreamMessage


class ResearchResult:
    """Result from research execution."""

    def __init__(self, synthesis: str, papers: List[Dict[str, Any]], trace_id: Optional[str] = None):
        """Initialize research result.

        Args:
            synthesis: Final research synthesis
            papers: Research papers found
            trace_id: Execution trace ID
        """
        self.synthesis = synthesis
        self.papers = papers
        self.trace_id = trace_id


class OrchestrationEngine:
    """LangGraph-based orchestration engine for research workflows."""

    def __init__(self):
        """Initialize orchestration engine."""
        self.logger = logging.getLogger(__name__)
        self.graph = self._build_research_graph()
        self.context_tree = create_research_context_tree()
        self.a2a_client: Optional[A2AClient] = None
        self.agent_registry: Dict[str, ResearchAgent] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}

        # Instance variables for capturing results during execution
        self._current_synthesis: Optional[str] = None
        self._current_papers: List[Dict[str, Any]] = []
        self._current_trace_id: Optional[str] = None

    async def initialize(self):
        """Initialize orchestration engine components."""
        self.a2a_client = A2AClient()
        await self.a2a_client.initialize()
        self.logger.info("Orchestration engine initialized")

    async def close(self):
        """Close orchestration engine resources."""
        if self.a2a_client:
            await self.a2a_client.close()
        self.logger.info("Orchestration engine closed")

    def _build_research_graph(self) -> StateGraph:
        """Build the research workflow graph.

        Returns:
            Configured StateGraph for research workflows
        """
        from langgraph.graph import StateGraph

        # Create workflow graph
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("generate_sequences", self._generate_sequences)
        workflow.add_node("execute_sequences", self._execute_sequences)
        workflow.add_node("synthesize_results", self._synthesize_results)
        workflow.add_node("evaluate_quality", self._evaluate_quality)

        # Define edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "generate_sequences")
        workflow.add_edge("generate_sequences", "execute_sequences")
        workflow.add_edge("execute_sequences", "synthesize_results")
        workflow.add_edge("synthesize_results", "evaluate_quality")
        workflow.add_edge("evaluate_quality", END)

        return workflow.compile()

    async def _analyze_query(self, state: ResearchState, config: RunnableConfig) -> Command:
        """Analyze research query to understand requirements.

        Args:
            state: Current research state
            config: Runtime configuration

        Returns:
            Command to proceed to sequence generation
        """
        query = getattr(state, "query", "")
        self.logger.info(f"Analyzing research query: {query}")

        # Update state with query analysis
        return Command(
            update={
                "query_analysis": {
                    "original_query": query,
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "analysis_status": "completed"
                }
            },
            goto="generate_sequences"
        )

    async def _generate_sequences(self, state: ResearchState, config: RunnableConfig) -> Command:
        """Generate strategic research sequences.

        Args:
            state: Current research state
            config: Runtime configuration

        Returns:
            Command to proceed to sequence execution
        """
        query = getattr(state, "query", "")
        self.logger.info("Generating strategic research sequences")

        # Generate 3 strategic sequences based on available agents
        sequences = await self._create_strategic_sequences(query)

        # Update state with generated sequences
        return Command(
            update={
                "sequences": sequences,
                "sequence_count": len(sequences)
            },
            goto="execute_sequences"
        )

    async def _execute_sequences(self, state: ResearchState, config: RunnableConfig) -> Command:
        """Execute research sequences in parallel.

        Args:
            state: Current research state
            config: Runtime configuration

        Returns:
            Command to proceed to synthesis
        """
        sequences = getattr(state, "sequences", [])
        self.logger.info(f"Executing {len(sequences)} research sequences in parallel")

        # Execute sequences concurrently
        execution_results = await asyncio.gather(
            *[self._execute_single_sequence(seq, state) for seq in sequences],
            return_exceptions=True
        )

        # Process results and handle failures
        successful_results = []
        failed_sequences = []

        for i, result in enumerate(execution_results):
            if isinstance(result, Exception):
                self.logger.error(f"Sequence {i} failed: {result}")
                failed_sequences.append(i)
            else:
                successful_results.append(result)

        return Command(
            update={
                "execution_results": successful_results,
                "failed_sequences": failed_sequences,
                "successful_executions": len(successful_results)
            },
            goto="synthesize_results"
        )

    async def _synthesize_results(self, state: ResearchState, config: RunnableConfig) -> Command:
        """Synthesize results from parallel executions.

        Args:
            state: Current research state
            config: Runtime configuration

        Returns:
            Command to proceed to evaluation
        """
        execution_results = getattr(state, "execution_results", [])
        self.logger.info(f"Synthesizing {len(execution_results)} execution results")

        # Combine all findings
        all_papers = []
        all_insights = []

        for result in execution_results:
            # Handle both dictionary results and AgentResult objects
            if isinstance(result, dict):
                # Legacy format
                if "papers" in result:
                    all_papers.extend(result["papers"])
                if "insights" in result:
                    all_insights.extend(result["insights"])
            elif hasattr(result, 'artifacts'):
                # AgentResult format - extract insights from artifacts
                for artifact in result.artifacts:
                    if isinstance(artifact, dict) and "content" in artifact:
                        # If artifact has content, treat it as an insight
                        all_insights.append(artifact["content"])
                    elif isinstance(artifact, str):
                        # If artifact is a string, treat it as an insight
                        all_insights.append(artifact)

                # Also extract insights from summary if available
                if hasattr(result, 'summary') and result.summary:
                    all_insights.append(result.summary)

        # Create synthesis
        query = getattr(state, "query", "")
        synthesis = await self._create_synthesis(all_papers, all_insights, query)
        self.logger.info(f"Synthesis created: {synthesis[:100]}... (length: {len(synthesis)})")

        # Store synthesis in instance variable for capture during execution
        self._current_synthesis = synthesis
        self._current_papers = all_papers
        self._current_trace_id = f"research-{uuid.uuid4().hex}"

        return Command(
            update={
                "synthesis": synthesis,
                "total_papers": len(all_papers),
                "total_insights": len(all_insights)
            },
            goto="evaluate_quality"
        )

    async def _evaluate_quality(self, state: ResearchState, config: RunnableConfig) -> Command:
        """Evaluate research quality and provide final results.

        Args:
            state: Current research state
            config: Runtime configuration

        Returns:
            Command to end execution
        """
        synthesis = getattr(state, "synthesis", "")
        execution_results = getattr(state, "execution_results", [])

        self.logger.info("Evaluating research quality")

        # Combine all papers from all sequences
        all_papers = []
        for result in execution_results:
            if isinstance(result, dict) and "papers" in result:
                all_papers.extend(result["papers"])

        # Calculate quality metrics
        quality_score = await self._calculate_quality_score(synthesis, all_papers)

        # Create final result
        final_result = ResearchResult(
            synthesis=synthesis,
            papers=all_papers,
            trace_id=f"research-{uuid.uuid4().hex}"
        )

        return Command(
            update={
                "final_result": final_result,
                "quality_score": quality_score,
                "completion_timestamp": datetime.utcnow().isoformat()
            },
            goto=END
        )

    async def _create_strategic_sequences(self, query: str) -> List[Dict[str, Any]]:
        """Create strategic research sequences.

        Args:
            query: Research query

        Returns:
            List of strategic sequences
        """
        # Get available agents
        available_agents = list(self.agent_registry.keys())

        sequences = [
            {
                "name": "Academic Research Focus",
                "agents": available_agents[:min(3, len(available_agents))],
                "focus": "Academic literature and theoretical foundations",
                "confidence": 0.8
            },
            {
                "name": "Technical Analysis Deep-dive",
                "agents": available_agents[1:min(4, len(available_agents))],
                "focus": "Technical implementation and architecture analysis",
                "confidence": 0.7
            },
            {
                "name": "Market Intelligence Survey",
                "agents": available_agents[:min(2, len(available_agents))],
                "focus": "Market trends and competitive analysis",
                "confidence": 0.6
            }
        ]

        return sequences

    async def _execute_single_sequence(self, sequence: Dict[str, Any], state: ResearchState) -> Dict[str, Any]:
        """Execute a single research sequence.

        Args:
            sequence: Sequence definition
            state: Current research state

        Returns:
            Execution results
        """
        sequence_name = sequence.get("name", "unknown")
        agents = sequence.get("agents", [])
        query = getattr(state, "query", "")

        self.logger.info(f"Executing sequence: {sequence_name}")

        # Execute agents in sequence
        sequence_results = []
        combined_insights = []

        for agent_name in agents:
            if agent_name in self.agent_registry:
                agent = self.agent_registry[agent_name]

                # Create task for agent
                task = Task(
                    id=f"{sequence_name}-{agent_name}-{uuid.uuid4().hex}",
                    description=f"Research: {query} (focus: {sequence.get('focus', '')})",
                    context_summary=f"Part of sequence: {sequence_name}"
                )

                # Execute agent task
                try:
                    result = await agent.execute_task(task)
                    if result:
                        sequence_results.append(result)
                        # Extract insights from AgentResult object
                        if hasattr(result, 'artifacts'):
                            # AgentResult format - extract insights from artifacts
                            for artifact in result.artifacts:
                                if isinstance(artifact, dict) and "content" in artifact:
                                    combined_insights.append(artifact["content"])
                                elif isinstance(artifact, str):
                                    combined_insights.append(artifact)
                        # Also extract from summary if available
                        if hasattr(result, 'summary') and result.summary:
                            combined_insights.append(result.summary)
                except Exception as e:
                    self.logger.error(f"Agent {agent_name} failed: {e}")
                    continue

        return {
            "sequence_name": sequence_name,
            "results": sequence_results,
            "insights": combined_insights,
            "papers": [],  # Would be populated by search agents
            "success": len(sequence_results) > 0
        }

    async def _create_synthesis(self, papers: List[Dict[str, Any]], insights: List[str], query: str) -> str:
        """Create synthesis from research results.

        Args:
            papers: Research papers
            insights: Research insights
            query: Original research query

        Returns:
            Research synthesis
        """
        if not papers and not insights:
            return f"No research results found for query: {query}"

        # If we have the synthesis agent available, use it for better synthesis
        if hasattr(self, 'agent_registry') and 'synthesis_agent' in self.agent_registry:
            self.logger.info(f"Attempting to use synthesis agent, available agents: {list(self.agent_registry.keys())}")
            try:
                synthesis_agent = self.agent_registry['synthesis_agent']
                if synthesis_agent.model:
                    self.logger.info("Synthesis agent found and has model, making API call")
                    # Create a comprehensive synthesis using the synthesis agent
                    synthesis_prompt = f"""Perform comprehensive research synthesis for the query: {query}

Available research results:
- Total papers found: {len(papers)}
- Total insights: {len(insights)}

Please provide a comprehensive synthesis that:
1. Summarizes the key findings from all research
2. Identifies major themes and patterns
3. Provides integrated conclusions
4. Highlights any important gaps or limitations
5. Suggests areas for future research

Research insights:
{chr(10).join(f"- {insight}" for insight in insights[:20])}

Please provide a detailed, well-structured synthesis report."""

                    messages = [
                        SystemMessage(content="You are an expert research synthesis agent. Provide comprehensive, well-structured synthesis reports based on research findings."),
                        HumanMessage(content=synthesis_prompt)
                    ]

                    response = await synthesis_agent.model.ainvoke(messages)
                    return f"## Comprehensive Research Synthesis\n\n{response.content}"

            except Exception as e:
                self.logger.error(f"Error in synthesis agent: {e}")

        # Fallback to simple synthesis if synthesis agent fails
        self.logger.info("Using fallback synthesis method")
        synthesis_parts = [f"## Research Synthesis\n\n**Query:** {query}\n"]

        if papers:
            synthesis_parts.append(f"**Papers Found:** {len(papers)}")
            for i, paper in enumerate(papers[:5]):  # Limit to top 5
                synthesis_parts.append(f"- {paper.get('title', 'Unknown Title')}")

        if insights:
            synthesis_parts.append(f"\n**Key Insights:** ({len(insights)} total)")
            for insight in insights[:15]:  # Limit to top 15
                synthesis_parts.append(f"- {insight}")

        return "\n".join(synthesis_parts)

    async def _calculate_quality_score(self, synthesis: str, papers: List[Dict[str, Any]]) -> float:
        """Calculate quality score for research results.

        Args:
            synthesis: Research synthesis
            papers: Research papers

        Returns:
            Quality score between 0 and 1
        """
        score = 0.0

        # Base score for having results
        if synthesis and len(synthesis) > 100:
            score += 0.4

        # Score for number of papers found
        if papers:
            paper_score = min(len(papers) / 10.0, 0.3)  # Up to 0.3 for 10+ papers
            score += paper_score

        # Score for synthesis quality
        if len(synthesis.split()) > 50:  # Decent length synthesis
            score += 0.3

        return min(score, 1.0)

    async def execute_research(self, query: str) -> ResearchResult:
        """Execute complete research workflow.

        Args:
            query: Research query

        Returns:
            ResearchResult with findings
        """
        self.logger.info(f"Starting research execution for: {query}")

        # Initialize research state
        initial_state = ResearchState(
            query=query,
            status="initializing",
            start_time=datetime.utcnow().isoformat()
        )

        # Execute workflow
        final_state = await self.graph.ainvoke(initial_state)

        # Note: Instance variables are already initialized in __init__

        # Extract results - try multiple sources for synthesis
        execution_results = getattr(final_state, "execution_results", [])
        synthesis = getattr(final_state, "synthesis", None)

        # Try to get synthesis from instance variable first (set during workflow)
        if not synthesis:
            synthesis = self._current_synthesis

        # Fallback to extracting from final state
        if not synthesis:
            final_result = getattr(final_state, "final_result", None)
            if final_result and hasattr(final_result, 'synthesis'):
                synthesis = final_result.synthesis

        # If still no synthesis, create a basic one
        if not synthesis:
            if execution_results:
                synthesis = f"Research completed for query: {query}\n\n{len(execution_results)} research sequences executed successfully."
            else:
                synthesis = f"No synthesis available for: {query}"

        # Combine all papers from all sequences
        all_papers = []
        for result in execution_results:
            if isinstance(result, dict):
                all_papers.extend(result.get("papers", []))

        return ResearchResult(
            synthesis=synthesis,
            papers=all_papers,
            trace_id=getattr(final_state, "trace_id", None)
        )

    async def execute_stream(self, query: str) -> AsyncGenerator[StreamMessage, None]:
        """Execute research workflow with streaming output.

        Args:
            query: Research query

        Yields:
            StreamMessage with progress updates
        """
        self.logger.info(f"Starting streaming research execution for: {query}")

        # Initialize state with streaming
        initial_state = ResearchState(
            query=query,
            status="streaming",
            start_time=datetime.utcnow().isoformat()
        )

        # Yield initial progress
        yield StreamMessage(
            message_id=str(uuid.uuid4()),
            sequence_id="main",
            message_type="progress",
            timestamp=int(datetime.utcnow().timestamp() * 1000),
            content={"type": "query_analysis", "status": "started"}
        )

        # Execute with streaming
        async for event in self.graph.astream(initial_state):
            if isinstance(event, dict):
                # Extract meaningful information from the event
                node_name = list(event.keys())[0] if event else "unknown"
                node_data = event[node_name] if event else {}

                # Map node names to meaningful progress events
                if "analyze_query" in node_name:
                    yield StreamMessage(
                        message_id=str(uuid.uuid4()),
                        sequence_id="main",
                        message_type="progress",
                        timestamp=int(datetime.utcnow().timestamp() * 1000),
                        content={"type": "query_analysis", "status": "completed"}
                    )
                elif "generate_sequences" in node_name:
                    yield StreamMessage(
                        message_id=str(uuid.uuid4()),
                        sequence_id="main",
                        message_type="progress",
                        timestamp=int(datetime.utcnow().timestamp() * 1000),
                        content={"type": "sequences_generated", "status": "completed", "count": node_data.get("sequence_count", 0)}
                    )
                elif "execute_sequences" in node_name:
                    yield StreamMessage(
                        message_id=str(uuid.uuid4()),
                        sequence_id="main",
                        message_type="progress",
                        timestamp=int(datetime.utcnow().timestamp() * 1000),
                        content={"type": "parallel_execution", "status": "completed", "results": node_data.get("successful_executions", 0)}
                    )
                elif "synthesize_results" in node_name:
                    yield StreamMessage(
                        message_id=str(uuid.uuid4()),
                        sequence_id="main",
                        message_type="progress",
                        timestamp=int(datetime.utcnow().timestamp() * 1000),
                        content={"type": "synthesis", "status": "completed"}
                    )
                elif "evaluate_quality" in node_name:
                    yield StreamMessage(
                        message_id=str(uuid.uuid4()),
                        sequence_id="main",
                        message_type="progress",
                        timestamp=int(datetime.utcnow().timestamp() * 1000),
                        content={"type": "quality_evaluation", "status": "completed"}
                    )

                # Yield the raw event as well
                yield StreamMessage(
                    message_id=str(uuid.uuid4()),
                    sequence_id="main",
                    message_type="progress",
                    timestamp=int(datetime.utcnow().timestamp() * 1000),
                    content=event
                )

    def register_agent(self, agent: ResearchAgent):
        """Register research agent.

        Args:
            agent: ResearchAgent to register
        """
        self.agent_registry[agent.name] = agent
        self.logger.info(f"Registered research agent: {agent.name}")

    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent names.

        Returns:
            List of agent names
        """
        return list(self.agent_registry.keys())


# Factory functions for different orchestration modes
def create_research_orchestrator() -> OrchestrationEngine:
    """Create orchestration engine optimized for research.

    Returns:
        OrchestrationEngine configured for research workflows
    """
    engine = OrchestrationEngine()
    # Configure for research workflows
    return engine


def create_balanced_orchestrator() -> OrchestrationEngine:
    """Create balanced orchestration engine.

    Returns:
        OrchestrationEngine with balanced configuration
    """
    engine = OrchestrationEngine()
    # Configure for balanced workloads
    return engine


def create_high_throughput_orchestrator() -> OrchestrationEngine:
    """Create high-throughput orchestration engine.

    Returns:
        OrchestrationEngine optimized for throughput
    """
    engine = OrchestrationEngine()
    # Configure for high throughput
    return engine
