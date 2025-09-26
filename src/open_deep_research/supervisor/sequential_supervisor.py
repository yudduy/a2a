"""Sequential supervisor for orchestrating multi-agent workflows with automatic handoffs.

This module provides the core SequentialSupervisor that orchestrates agent execution
using the AgentRegistry, CompletionDetector, and RunningReportBuilder for production-ready
sequential multi-agent workflows.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from open_deep_research.agents.completion_detector import (
    CompletionDetector,
    DetectionStrategy,
)
from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.configuration import Configuration
from open_deep_research.orchestration.report_builder import RunningReportBuilder
from open_deep_research.state import (
    AgentExecutionReport,
    SequentialAgentState,
    SequentialSupervisorState,
)
from open_deep_research.utils import get_all_tools, think_tool

logger = logging.getLogger(__name__)


@dataclass
class SupervisorConfig:
    """Configuration for the Sequential Supervisor."""
    
    # Execution timeouts and limits
    agent_timeout_seconds: float = 600.0  # 10 minutes per agent
    max_agents_per_sequence: int = 10
    max_tool_calls_per_agent: int = 20
    
    # Completion detection settings
    completion_threshold: float = 0.6
    completion_strategy: DetectionStrategy = DetectionStrategy.COMBINED
    
    # Quality thresholds
    min_insights_per_agent: int = 1
    min_research_content_length: int = 100
    
    # Dynamic sequence modification
    allow_dynamic_insertion: bool = True
    max_dynamic_insertions: int = 3
    
    # Error handling
    max_agent_retries: int = 2
    continue_on_agent_failure: bool = True
    
    # Logging and debugging
    debug_mode: bool = False
    log_agent_outputs: bool = True


class SequentialSupervisor:
    """Central orchestrator for sequential multi-agent workflow execution.
    
    This supervisor coordinates the execution of multiple specialized agents in sequence,
    handling automatic completion detection, context sharing between agents, and
    incremental report building throughout the research process.
    """
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        config: Optional[SupervisorConfig] = None,
        system_config: Optional[Configuration] = None
    ):
        """Initialize the Sequential Supervisor.
        
        Args:
            agent_registry: Registry containing available agents
            config: Supervisor-specific configuration
            system_config: System-wide configuration
        """
        self.agent_registry = agent_registry
        self.config = config or SupervisorConfig()
        self.system_config = system_config or Configuration()
        
        # Initialize completion detector
        self.completion_detector = CompletionDetector(debug_mode=self.config.debug_mode)
        self.completion_detector.set_completion_threshold(self.config.completion_threshold)
        
        # Initialize configurable model for agent orchestration
        self.model = init_chat_model(
            configurable_fields=("model", "max_tokens", "api_key", "base_url")
        )
        
        # Track supervisor state
        self._current_workflow_id: Optional[str] = None
        self._execution_stats: Dict[str, Any] = {}
        
        if self.config.debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("SequentialSupervisor initialized in debug mode")
        
        logger.info(f"SequentialSupervisor initialized with {len(agent_registry.list_agents())} available agents")
    
    async def create_workflow_graph(self) -> StateGraph:
        """Build LangGraph StateGraph for sequential agent execution.
        
        Returns:
            Configured StateGraph ready for execution
        """
        logger.info("Building sequential supervisor workflow graph")
        
        # Create the graph
        workflow = StateGraph(SequentialSupervisorState)
        
        # Add supervisor nodes
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("agent_executor", self.agent_executor_node)
        workflow.add_node("finalize_sequence", self.finalize_sequence)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "supervisor",
            self.route_to_next_agent,
            {
                "execute_agent": "agent_executor",
                "finalize": "finalize_sequence",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "agent_executor",
            self._check_agent_completion,
            {
                "continue": "supervisor",
                "retry": "agent_executor",
                "error": END
            }
        )
        
        workflow.add_edge("finalize_sequence", END)
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        return workflow
    
    def _get_state_attr(self, state, attr_name, default=None):
        """Helper to get state attribute handling both dict and object types."""
        if isinstance(state, dict):
            return state.get(attr_name, default)
        else:
            return getattr(state, attr_name, default)
    
    def _set_state_attr(self, state, attr_name, value):
        """Helper to set state attribute handling both dict and object types."""
        if isinstance(state, dict):
            state[attr_name] = value
        else:
            setattr(state, attr_name, value)
    
    async def supervisor_node(self, state: SequentialSupervisorState) -> Command[Literal["supervisor", "agent_executor", "finalize_sequence"]]:
        """Main supervisor logic with automatic handoff detection.
        
        Args:
            state: Current supervisor state
            
        Returns:
            Command for next node execution
        """
        # Handle case where state is a dict instead of SequentialSupervisorState object
        if isinstance(state, dict):
            sequence_position = state.get("sequence_position", 0)
            planned_sequence = state.get("planned_sequence", [])
        else:
            sequence_position = state.sequence_position
            planned_sequence = state.planned_sequence
            
        logger.debug(f"Supervisor node processing: position {sequence_position}/{len(planned_sequence)}")
        
        try:
            # Initialize sequence if needed
            if not self._get_state_attr(state, "sequence_start_time"):
                self._set_state_attr(state, "sequence_start_time", datetime.utcnow())
                
                # Initialize running report
                if not self._get_state_attr(state, "running_report") and planned_sequence:
                    sequence_name = f"sequential_{len(planned_sequence)}_agents"
                    research_topic = self._get_state_attr(state, "research_topic", "")
                    running_report = RunningReportBuilder.initialize_report(
                        research_topic=research_topic,
                        sequence_name=sequence_name,
                        planned_agents=planned_sequence
                    )
                    self._set_state_attr(state, "running_report", running_report)
                    logger.info(f"Initialized running report for topic: {research_topic}")
            
            # Check if we're done with the planned sequence
            if sequence_position >= len(planned_sequence):
                logger.info("Sequence complete - all planned agents executed")
                return Command(goto="finalize_sequence")
            
            # Handle completion of the last agent
            if self._get_state_attr(state, "last_agent_completed"):
                await self._process_completed_agent(state)
                self._set_state_attr(state, "last_agent_completed", None)
                self._set_state_attr(state, "handoff_ready", True)
            
            # Check for dynamic sequence modifications
            handoff_ready = self._get_state_attr(state, "handoff_ready", False)
            if self.config.allow_dynamic_insertion and handoff_ready:
                await self._check_dynamic_sequence_modification(state)
            
            # Prepare for next agent execution
            if handoff_ready and sequence_position < len(planned_sequence):
                next_agent_name = planned_sequence[sequence_position]
                
                # Validate agent exists
                if not self.agent_registry.has_agent(next_agent_name):
                    logger.error(f"Agent '{next_agent_name}' not found in registry")
                    return Command(goto="error")
                
                self._set_state_attr(state, "current_agent", next_agent_name)
                self._set_state_attr(state, "handoff_ready", False)
                
                logger.info(f"Handing off to agent: {next_agent_name} (position {sequence_position + 1}/{len(planned_sequence)})")
                return Command(goto="agent_executor")
            
            # If not ready for handoff, continue processing
            logger.debug("Supervisor continuing - awaiting agent completion")
            return Command(goto="supervisor")
            
        except Exception as e:
            logger.error(f"Error in supervisor node: {e}")
            self._record_error(state, "supervisor_error", str(e))
            return Command(goto="error")
    
    async def route_to_next_agent(self, state: SequentialSupervisorState) -> Literal["execute_agent", "finalize", "error"]:
        """Determine next agent based on completion and sequence state.
        
        Args:
            state: Current supervisor state
            
        Returns:
            Route decision for workflow execution
        """
        try:
            # Error conditions
            executed_agents = self._get_state_attr(state, "executed_agents", [])
            if len(executed_agents) >= self.config.max_agents_per_sequence:
                logger.warning(f"Maximum agent limit reached: {self.config.max_agents_per_sequence}")
                return "finalize"
            
            # Check if sequence is complete
            sequence_position = self._get_state_attr(state, "sequence_position", 0)
            planned_sequence = self._get_state_attr(state, "planned_sequence", [])
            if sequence_position >= len(planned_sequence):
                return "finalize"
            
            # Route to agent execution if we have a current agent
            current_agent = self._get_state_attr(state, "current_agent")
            if current_agent and current_agent not in executed_agents:
                return "execute_agent"
            
            # Finalize if no more agents to execute
            return "finalize"
            
        except Exception as e:
            logger.error(f"Error in routing logic: {e}")
            return "error"
    
    async def agent_executor_node(self, state: SequentialSupervisorState) -> Command[Literal["supervisor", "agent_executor"]]:
        """Execute specific agent with context from previous agents.
        
        Args:
            state: Current supervisor state
            
        Returns:
            Command for next execution step
        """
        if not state.current_agent:
            logger.error("No current agent specified for execution")
            return Command(goto="supervisor")
        
        agent_name = state.current_agent
        logger.info(f"Executing agent: {agent_name}")
        
        try:
            # Get agent configuration
            agent_config = self.agent_registry.get_agent(agent_name)
            if not agent_config:
                logger.error(f"Agent configuration not found: {agent_name}")
                return Command(goto="supervisor")
            
            # Prepare agent execution context
            agent_state = await self._prepare_agent_context(state, agent_name, agent_config)
            
            # Execute the agent
            execution_result = await self._execute_agent(agent_state, agent_config)
            
            # Process execution results
            await self._process_agent_execution(state, execution_result)
            
            logger.info(f"Agent {agent_name} execution completed")
            return Command(goto="supervisor")
            
        except asyncio.TimeoutError:
            logger.error(f"Agent {agent_name} execution timed out after {self.config.agent_timeout_seconds}s")
            self._record_error(state, "agent_timeout", f"Agent {agent_name} timed out")
            return Command(goto="supervisor")
            
        except Exception as e:
            logger.error(f"Error executing agent {agent_name}: {e}")
            self._record_error(state, "agent_execution_error", str(e))
            
            if self.config.continue_on_agent_failure:
                return Command(goto="supervisor")
            else:
                return Command(goto="error")
    
    async def finalize_sequence(self, state: SequentialSupervisorState) -> Command[Literal["finalize_sequence"]]:
        """Complete sequence execution and finalize reports.
        
        Args:
            state: Current supervisor state
            
        Returns:
            Command to end workflow
        """
        logger.info("Finalizing sequential agent execution")
        
        try:
            # Finalize the running report
            running_report = self._get_state_attr(state, 'running_report')
            if running_report:
                finalized_report = RunningReportBuilder.finalize_report(running_report)
                self._set_state_attr(state, 'running_report', finalized_report)
                
                # Generate final report as markdown
                final_report_markdown = RunningReportBuilder.format_report_as_markdown(finalized_report)
                
                # Store in state for downstream processing
                supervisor_messages = self._get_state_attr(state, 'supervisor_messages', [])
                supervisor_messages.append(
                    AIMessage(content=f"# Sequential Research Complete\n\n{final_report_markdown}")
                )
                self._set_state_attr(state, 'supervisor_messages', supervisor_messages)
                
                logger.info(f"Research completed: {finalized_report.total_agents_executed} agents executed, "
                          f"{len(finalized_report.all_insights)} insights generated")
            
            # Update execution statistics
            sequence_start_time = self._get_state_attr(state, 'sequence_start_time')
            total_time = (datetime.utcnow() - sequence_start_time).total_seconds() if sequence_start_time else 0
            executed_agents = self._get_state_attr(state, 'executed_agents', [])
            sequence_modifications = self._get_state_attr(state, 'sequence_modifications', [])
            
            self._execution_stats = {
                "total_execution_time": total_time,
                "agents_executed": len(executed_agents),
                "insights_generated": len(running_report.all_insights) if running_report else 0,
                "sequence_modifications": len(sequence_modifications),
                "completion_status": "success"
            }
            
            logger.info(f"Sequence finalized successfully: {self._execution_stats}")
            return Command(goto=END)
            
        except Exception as e:
            logger.error(f"Error finalizing sequence: {e}")
            self._record_error(state, "finalization_error", str(e))
            return Command(goto=END)
    
    async def _prepare_agent_context(
        self,
        supervisor_state: SequentialSupervisorState,
        agent_name: str,
        agent_config: Dict[str, Any]
    ) -> SequentialAgentState:
        """Prepare context for agent execution from supervisor state.
        
        Args:
            supervisor_state: Current supervisor state
            agent_name: Name of the agent to execute
            agent_config: Configuration for the agent
            
        Returns:
            Prepared agent state with context
        """
        # Collect insights from previous agents
        previous_insights = []
        previous_context = {}
        
        executed_agents = self._get_state_attr(supervisor_state, 'executed_agents', [])
        agent_insights = self._get_state_attr(supervisor_state, 'agent_insights', {})
        agent_context = self._get_state_attr(supervisor_state, 'agent_context', {})
        
        for prev_agent in executed_agents:
            if prev_agent in agent_insights:
                previous_insights.extend(agent_insights[prev_agent])
            
            if prev_agent in agent_context:
                previous_context.update(agent_context[prev_agent])
        
        # Create agent-specific state
        agent_state = SequentialAgentState(
            messages=[
                SystemMessage(content=f"You are {agent_name}, a specialized research agent. "
                                    f"Your expertise: {', '.join(agent_config.get('expertise_areas', []))}")
            ],
            agent_name=agent_name,
            agent_type=agent_config.get('type', 'research_agent'),
            sequence_position=self._get_state_attr(supervisor_state, 'sequence_position', 0),
            research_topic=self._get_state_attr(supervisor_state, 'research_topic', ''),
            assigned_questions=agent_config.get('focus_questions', []),
            previous_agent_insights=previous_insights,
            previous_agent_context=previous_context,
            execution_start_time=datetime.utcnow()
        )
        
        return agent_state
    
    async def _execute_agent(
        self,
        agent_state: SequentialAgentState,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute individual agent with timeout and monitoring.
        
        Args:
            agent_state: Prepared agent state
            agent_config: Agent configuration
            
        Returns:
            Execution result dictionary
        """
        start_time = datetime.utcnow()
        
        try:
            # Prepare agent message with context
            context_message = self._build_agent_context_message(agent_state)
            
            # Get agent tools - all tools by default for agents
            from langchain_core.runnables import RunnableConfig
            config = RunnableConfig()
            agent_tools = await get_all_tools(config)
            agent_tools.append(think_tool)  # Add thinking tool for reflection
            
            # Configure model with tools
            model_with_tools = self.model.bind_tools(agent_tools)
            
            # Execute with timeout
            response = await asyncio.wait_for(
                model_with_tools.ainvoke([context_message]),
                timeout=self.config.agent_timeout_seconds
            )
            
            # Extract insights and completion detection
            completion_result = self.completion_detector.analyze_completion_patterns(
                response,
                custom_indicators=agent_config.get('completion_indicators'),
                strategy=self.config.completion_strategy
            )
            
            # Extract insights from response
            insights = self._extract_insights_from_response(response)
            
            # Build execution result
            execution_duration = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "agent_name": agent_state.agent_name,
                "agent_type": agent_state.agent_type,
                "response": response,
                "completion_result": completion_result,
                "insights": insights,
                "execution_duration": execution_duration,
                "tool_calls_made": len(response.tool_calls) if response.tool_calls else 0,
                "research_content": response.content[:1000] if response.content else "",
                "handoff_context": self._extract_handoff_context(response, agent_state)
            }
            
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise
    
    async def _process_agent_execution(
        self,
        supervisor_state: SequentialSupervisorState,
        execution_result: Dict[str, Any]
    ) -> None:
        """Process results from completed agent execution.
        
        Args:
            supervisor_state: Current supervisor state
            execution_result: Results from agent execution
        """
        agent_name = execution_result["agent_name"]
        
        # Create agent execution report
        agent_report = AgentExecutionReport(
            agent_name=agent_name,
            agent_type=execution_result["agent_type"],
            execution_start=datetime.utcnow() - timedelta(seconds=execution_result["execution_duration"]),
            execution_end=datetime.utcnow(),
            execution_duration=execution_result["execution_duration"],
            insights=execution_result["insights"],
            research_content=execution_result["research_content"],
            questions_addressed=execution_result.get("questions_addressed", []),
            completion_confidence=execution_result["completion_result"].confidence,
            insight_quality_score=self._calculate_insight_quality(execution_result["insights"]),
            research_depth_score=self._calculate_research_depth(execution_result["research_content"]),
            handoff_context=execution_result["handoff_context"],
            suggested_next_questions=execution_result.get("suggested_questions", [])
        )
        
        # Update supervisor state
        executed_agents = self._get_state_attr(supervisor_state, 'executed_agents', [])
        executed_agents.append(agent_name)
        self._set_state_attr(supervisor_state, 'executed_agents', executed_agents)
        
        sequence_position = self._get_state_attr(supervisor_state, 'sequence_position', 0) + 1
        self._set_state_attr(supervisor_state, 'sequence_position', sequence_position)
        
        self._set_state_attr(supervisor_state, 'last_agent_completed', agent_name)
        
        agent_insights = self._get_state_attr(supervisor_state, 'agent_insights', {})
        agent_insights[agent_name] = execution_result["insights"]
        self._set_state_attr(supervisor_state, 'agent_insights', agent_insights)
        
        agent_context = self._get_state_attr(supervisor_state, 'agent_context', {})
        agent_context[agent_name] = execution_result["handoff_context"]
        self._set_state_attr(supervisor_state, 'agent_context', agent_context)
        agent_reports = self._get_state_attr(supervisor_state, 'agent_reports', {})
        agent_reports[agent_name] = agent_report
        self._set_state_attr(supervisor_state, 'agent_reports', agent_reports)
        
        completion_signals = self._get_state_attr(supervisor_state, 'completion_signals', {})
        completion_signals[agent_name] = execution_result["completion_result"].is_complete
        self._set_state_attr(supervisor_state, 'completion_signals', completion_signals)
        
        # Update running report
        running_report = self._get_state_attr(supervisor_state, 'running_report')
        if running_report:
            updated_report = RunningReportBuilder.add_agent_execution(
                running_report, agent_report
            )
            self._set_state_attr(supervisor_state, 'running_report', updated_report)
            
            # Update executive summary periodically
            executed_agents = self._get_state_attr(supervisor_state, 'executed_agents', [])
            if len(executed_agents) % 2 == 0:  # Every 2 agents
                summary_updated_report = RunningReportBuilder.update_executive_summary(
                    updated_report
                )
                self._set_state_attr(supervisor_state, 'running_report', summary_updated_report)
        
        logger.info(f"Processed execution results for agent: {agent_name}")
    
    async def _process_completed_agent(self, state: SequentialSupervisorState) -> None:
        """Process the completion of the last executed agent.
        
        Args:
            state: Current supervisor state
        """
        last_agent_completed = self._get_state_attr(state, 'last_agent_completed')
        if not last_agent_completed:
            return
        
        completed_agent = last_agent_completed
        logger.debug(f"Processing completion of agent: {completed_agent}")
        
        # Agent is already processed in _process_agent_execution
        # This method can be extended for additional post-processing
        pass
    
    async def _check_dynamic_sequence_modification(self, state: SequentialSupervisorState) -> None:
        """Check if sequence should be dynamically modified based on findings.
        
        Args:
            state: Current supervisor state
        """
        if not self.config.allow_dynamic_insertion:
            return
        
        sequence_modifications = self._get_state_attr(state, "sequence_modifications", [])
        if len(sequence_modifications) >= self.config.max_dynamic_insertions:
            logger.debug("Maximum dynamic insertions reached")
            return
        
        # Simple heuristic: if we have gaps in insight coverage, suggest additional agents
        executed_agents = self._get_state_attr(state, 'executed_agents', [])
        if len(executed_agents) >= 2:
            # Analyze insight coverage and suggest additional specialized agents
            # This is a placeholder for more sophisticated logic
            logger.debug("Dynamic sequence modification check completed - no changes needed")
    
    def _check_agent_completion(self, state: SequentialSupervisorState) -> Literal["continue", "retry", "error"]:
        """Check if agent completed successfully and determine next action.
        
        Args:
            state: Current supervisor state
            
        Returns:
            Next action decision
        """
        if not state.current_agent:
            return "error"
        
        agent_name = state.current_agent
        
        # Check if agent is marked as completed
        if agent_name in state.completion_signals:
            if state.completion_signals[agent_name]:
                return "continue"
            else:
                # Check retry count
                retry_count = state.sequence_modifications.count({"type": "retry", "agent": agent_name})
                if retry_count < self.config.max_agent_retries:
                    logger.warning(f"Agent {agent_name} incomplete, retrying ({retry_count + 1}/{self.config.max_agent_retries})")
                    return "retry"
                else:
                    logger.error(f"Agent {agent_name} failed after {self.config.max_agent_retries} retries")
                    if self.config.continue_on_agent_failure:
                        return "continue"
                    else:
                        return "error"
        
        return "continue"
    
    def _build_agent_context_message(self, agent_state: SequentialAgentState) -> HumanMessage:
        """Build context message for agent execution.
        
        Args:
            agent_state: Agent state with context
            
        Returns:
            Formatted message for agent
        """
        context_parts = [
            f"Research Topic: {agent_state.research_topic}",
            f"Your position in sequence: {agent_state.sequence_position + 1}",
            ""
        ]
        
        if agent_state.assigned_questions:
            context_parts.extend([
                "Focus Questions:",
                *[f"- {q}" for q in agent_state.assigned_questions],
                ""
            ])
        
        if agent_state.previous_agent_insights:
            context_parts.extend([
                "Previous Agent Insights:",
                *[f"- {insight}" for insight in agent_state.previous_agent_insights[:5]],
                ""
            ])
        
        context_parts.extend([
            "Please conduct thorough research on this topic from your specialized perspective.",
            "Use tools as needed and provide comprehensive insights when complete."
        ])
        
        return HumanMessage(content="\n".join(context_parts))
    
    def _extract_insights_from_response(self, response: AIMessage) -> List[str]:
        """Extract insights from agent response.
        
        Args:
            response: Agent response message
            
        Returns:
            List of extracted insights
        """
        if not response.content:
            return []
        
        content = response.content
        insights = []
        
        # Look for explicit insight markers
        insight_patterns = [
            r"(?:Key insights?|Important findings?|Main discoveries?)[:\s]*\n?(.+?)(?:\n\n|\n(?=[A-Z])|$)",
            r"(?:In conclusion|To summarize|Key takeaways?)[:\s]*\n?(.+?)(?:\n\n|\n(?=[A-Z])|$)",
            r"(?:^|\n)(?:\d+\.|\-|\*)\s*(.+?)(?:\n|$)"
        ]
        
        import re
        for pattern in insight_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                clean_insight = match.strip()
                if len(clean_insight) > 20 and clean_insight not in insights:
                    insights.append(clean_insight)
        
        # If no structured insights found, extract from content
        if not insights and len(content) > 100:
            sentences = content.split('. ')
            for sentence in sentences:
                if len(sentence) > 50 and any(word in sentence.lower() for word in ['important', 'significant', 'key', 'notable']):
                    insights.append(sentence.strip())
        
        return insights[:10]  # Limit to top 10 insights
    
    def _extract_handoff_context(self, response: AIMessage, agent_state: SequentialAgentState) -> Dict[str, Any]:
        """Extract context to pass to next agent.
        
        Args:
            response: Agent response message
            agent_state: Current agent state
            
        Returns:
            Context dictionary for next agent
        """
        return {
            "agent_type": agent_state.agent_type,
            "execution_time": (datetime.utcnow() - agent_state.execution_start_time).total_seconds() if agent_state.execution_start_time else 0,
            "tool_calls_made": agent_state.tool_calls_made,
            "content_length": len(response.content) if response.content else 0,
            "has_tool_calls": bool(response.tool_calls)
        }
    
    def _calculate_insight_quality(self, insights: List[str]) -> float:
        """Calculate quality score for insights.
        
        Args:
            insights: List of insight strings
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not insights:
            return 0.0
        
        # Simple heuristic based on insight characteristics
        quality_factors = []
        
        for insight in insights:
            factors = [
                len(insight) > 50,  # Sufficient detail
                len(insight.split()) > 8,  # Multiple words
                any(word in insight.lower() for word in ['because', 'therefore', 'however', 'moreover']),  # Reasoning
                insight.count(',') >= 1,  # Complexity
            ]
            quality_factors.append(sum(factors) / len(factors))
        
        return sum(quality_factors) / len(quality_factors)
    
    def _calculate_research_depth(self, research_content: str) -> float:
        """Calculate research depth score.
        
        Args:
            research_content: Research content string
            
        Returns:
            Depth score between 0.0 and 1.0
        """
        if not research_content:
            return 0.0
        
        depth_factors = [
            len(research_content) > self.config.min_research_content_length,
            research_content.count('\n') >= 3,  # Multiple paragraphs
            len(research_content.split()) > 100,  # Substantial content
            research_content.lower().count('research') + research_content.lower().count('study') >= 2  # Domain relevance
        ]
        
        return sum(depth_factors) / len(depth_factors)
    
    def _record_error(self, state: SequentialSupervisorState, error_type: str, error_message: str) -> None:
        """Record error in supervisor state.
        
        Args:
            state: Current supervisor state
            error_type: Type of error
            error_message: Error message
        """
        error_record = {
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self._get_state_attr(state, "current_agent"),
            "sequence_position": self._get_state_attr(state, "sequence_position", 0)
        }
        
        sequence_modifications = self._get_state_attr(state, "sequence_modifications", [])
        sequence_modifications.append(error_record)
        self._set_state_attr(state, "sequence_modifications", sequence_modifications)
        logger.error(f"Recorded error: {error_type} - {error_message}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for the supervisor.
        
        Returns:
            Dictionary with execution statistics
        """
        return self._execution_stats.copy()
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents from registry.
        
        Returns:
            List of agent names
        """
        return self.agent_registry.list_agents()
    
    def validate_sequence(self, planned_sequence: List[str]) -> Dict[str, Any]:
        """Validate a planned agent sequence.
        
        Args:
            planned_sequence: List of agent names to validate
            
        Returns:
            Validation results with errors and warnings
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sequence_length": len(planned_sequence)
        }
        
        # Check sequence length
        if len(planned_sequence) == 0:
            validation_result["valid"] = False
            validation_result["errors"].append("Empty sequence - no agents specified")
        
        if len(planned_sequence) > self.config.max_agents_per_sequence:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Sequence too long: {len(planned_sequence)} > {self.config.max_agents_per_sequence}")
        
        # Check agent existence
        available_agents = set(self.agent_registry.list_agents())
        missing_agents = []
        
        for agent_name in planned_sequence:
            if agent_name not in available_agents:
                missing_agents.append(agent_name)
        
        if missing_agents:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Agents not found in registry: {', '.join(missing_agents)}")
        
        # Check for duplicate agents
        duplicates = set()
        seen = set()
        for agent in planned_sequence:
            if agent in seen:
                duplicates.add(agent)
            seen.add(agent)
        
        if duplicates:
            validation_result["warnings"].append(f"Duplicate agents in sequence: {', '.join(duplicates)}")
        
        return validation_result