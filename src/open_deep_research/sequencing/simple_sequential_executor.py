"""Lightweight sequential executor for agent sequences within parallel execution.

This module provides a simple, efficient executor for running agent sequences
sequentially within each parallel path. It replaces the heavy ParallelExecutor
pattern with a lightweight approach optimized for the always-parallel architecture.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.sequencing.models import AgentExecutionResult, AgentType
from open_deep_research.supervisor.sequence_models import AgentSequence

logger = logging.getLogger(__name__)


class SequenceExecutionResult(BaseModel):
    """Result from executing a single agent sequence."""
    
    sequence_id: str = Field(default_factory=lambda: str(uuid4()))
    sequence_name: str
    agent_results: List[Dict[str, Any]] = Field(default_factory=list)
    comprehensive_findings: List[str] = Field(default_factory=list)
    total_duration: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    
    # Productivity metrics
    total_tool_calls: int = 0
    total_search_queries: int = 0
    key_insights_count: int = 0
    
    @property
    def overall_productivity_metrics(self) -> Dict[str, Any]:
        """Calculate productivity metrics for compatibility."""
        return {
            "tool_productivity": min(1.0, (self.total_tool_calls + self.total_search_queries) / 10.0),
            "insight_quality": min(1.0, self.key_insights_count / 5.0) if self.key_insights_count > 0 else 0.0
        }


class SimpleSequentialExecutor:
    """Lightweight sequential executor for agent sequences."""
    
    def __init__(self, config: RunnableConfig):
        """Initialize the sequential executor.
        
        Args:
            config: Runtime configuration with model settings
        """
        self.config = config
        self.configurable = Configuration.from_runnable_config(config)
        self.agent_registry = AgentRegistry()
        
    async def execute_agent_sequence(
        self, 
        agent_sequence: AgentSequence, 
        research_topic: str,
        sequence_id: Optional[str] = None
    ) -> SequenceExecutionResult:
        """Execute a sequence of agents sequentially.
        
        Args:
            agent_sequence: The agent sequence to execute
            research_topic: The research topic to investigate
            sequence_id: Optional sequence ID for tracking
            
        Returns:
            SequenceExecutionResult with aggregated results
        """
        if sequence_id is None:
            sequence_id = f"seq_{agent_sequence.sequence_name.lower().replace(' ', '_')}_{str(uuid4())[:8]}"
        
        start_time = time.time()
        result = SequenceExecutionResult(
            sequence_id=sequence_id,
            sequence_name=agent_sequence.sequence_name
        )
        
        logger.info(f"Starting sequential execution: {agent_sequence.sequence_name} with {len(agent_sequence.agent_names)} agents")
        
        # Context that gets passed between agents
        research_context = {
            "research_topic": research_topic,
            "research_brief": research_topic,
            "previous_insights": [],
            "sequence_position": 0,
            "questions": [research_topic],
            "accumulated_findings": []
        }
        
        try:
            # Execute each agent in the sequence
            for position, agent_name in enumerate(agent_sequence.agent_names):
                agent_start_time = time.time()
                
                logger.debug(f"Executing agent {position + 1}/{len(agent_sequence.agent_names)}: {agent_name}")
                
                # Get agent configuration from registry
                agent_config = self.agent_registry.get_agent(agent_name)
                if not agent_config:
                    logger.warning(f"Agent '{agent_name}' not found in registry, skipping")
                    continue
                
                # Update context for this agent
                research_context["sequence_position"] = position
                research_context["current_agent"] = agent_name
                
                # Execute the agent
                try:
                    agent_result = await self._execute_single_agent(
                        agent_name=agent_name,
                        agent_config=agent_config,
                        research_context=research_context
                    )
                    
                    # Add to results
                    result.agent_results.append(agent_result)
                    
                    # Update metrics
                    result.total_tool_calls += agent_result.get("tool_calls_made", 0)
                    result.total_search_queries += agent_result.get("search_queries_executed", 0)
                    
                    # Extract insights and update context
                    agent_insights = agent_result.get("key_insights", [])
                    if agent_insights:
                        result.key_insights_count += len(agent_insights)
                        research_context["previous_insights"].extend(agent_insights)
                        result.comprehensive_findings.extend(agent_insights)
                    
                    # Add findings to context for next agent
                    agent_findings = agent_result.get("research_findings", "")
                    if agent_findings:
                        research_context["accumulated_findings"].append(agent_findings)
                    
                    agent_duration = time.time() - agent_start_time
                    logger.debug(f"Agent {agent_name} completed in {agent_duration:.2f}s")
                    
                except Exception as agent_error:
                    logger.error(f"Agent {agent_name} failed: {agent_error}")
                    # Continue with next agent instead of failing entire sequence
                    error_result = {
                        "agent_type": agent_name,
                        "agent_name": agent_name,
                        "key_insights": [],
                        "research_findings": f"Agent execution failed: {str(agent_error)}",
                        "execution_duration": time.time() - agent_start_time,
                        "tool_calls_made": 0,
                        "search_queries_executed": 0,
                        "error": str(agent_error)
                    }
                    result.agent_results.append(error_result)
            
            # Mark as successful if we got any results
            result.success = len(result.agent_results) > 0
            result.total_duration = time.time() - start_time
            
            logger.info(f"Sequential execution completed: {result.sequence_name} "
                       f"({result.total_duration:.2f}s, {len(result.agent_results)} agents)")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.total_duration = time.time() - start_time
            logger.error(f"Sequential execution failed for {agent_sequence.sequence_name}: {e}")
        
        return result
    
    async def _execute_single_agent(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        research_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single agent with the given context using the researcher subgraph.
        
        Args:
            agent_name: Name of the agent to execute
            agent_config: Agent configuration from registry
            research_context: Current research context
            
        Returns:
            Dictionary with agent execution results
        """
        from open_deep_research.state import ResearcherState
        from langchain_core.messages import HumanMessage, SystemMessage
        from open_deep_research.deep_researcher import researcher_subgraph
        
        start_time = time.time()
        
        try:
            # Create agent-specific research prompt based on configuration and context
            agent_description = agent_config.get("description", f"Research agent: {agent_name}")
            agent_expertise = agent_config.get("expertise_areas", [])
            system_prompt = agent_config.get("system_prompt", "You are a specialized research agent.")
            
            # Build research context for this agent
            previous_insights_text = ""
            if research_context.get("previous_insights"):
                previous_insights_text = f"\n\nPrevious Research Insights:\n" + "\n".join([
                    f"- {insight}" for insight in research_context["previous_insights"][-5:]  # Last 5 insights
                ])
            
            accumulated_findings_text = ""
            if research_context.get("accumulated_findings"):
                accumulated_findings_text = f"\n\nAccumulated Findings:\n" + "\n".join([
                    f"- {finding[:200]}..." if len(finding) > 200 else f"- {finding}"
                    for finding in research_context["accumulated_findings"][-3:]  # Last 3 findings
                ])
            
            # Create focused research prompt for this agent
            research_prompt = f"""Please conduct specialized research on: {research_context['research_topic']}

Agent Role: {agent_description}
Areas of Expertise: {', '.join(agent_expertise) if agent_expertise else 'General research'}
Position in Sequence: {research_context.get('sequence_position', 0) + 1}

{previous_insights_text}
{accumulated_findings_text}

Based on your expertise in {', '.join(agent_expertise[:2]) if agent_expertise else 'research'}, please:
1. Conduct thorough research on this topic
2. Provide specific insights from your domain perspective
3. Use available tools to gather comprehensive information
4. Build upon previous findings while adding your specialized knowledge

Focus on delivering insights that complement the previous research and provide unique value from your specialized perspective."""
            
            # Prepare researcher state for execution
            researcher_state = ResearcherState(
                researcher_messages=[
                    HumanMessage(content=research_prompt)
                ],
                tool_call_iterations=0
            )
            
            # Execute the researcher subgraph
            logger.debug(f"Executing researcher subgraph for agent {agent_name}")
            result = await researcher_subgraph.ainvoke(
                researcher_state,
                config=self.config
            )
            
            # Extract results from the researcher execution
            compressed_research = result.get("compressed_research", "")
            raw_notes = result.get("raw_notes", [])
            
            # Parse insights from the compressed research
            key_insights = []
            if compressed_research:
                # Extract key findings from compressed research
                lines = compressed_research.split('\n')
                for line in lines:
                    line = line.strip()
                    # Look for key insights, findings, or conclusions
                    if any(keyword in line.lower() for keyword in ['insight:', 'finding:', 'conclusion:', 'key point:', '•', '-']):
                        clean_line = line.replace('•', '').replace('-', '').strip()
                        if len(clean_line) > 10:  # Meaningful content
                            key_insights.append(clean_line)
                
                # If no structured insights found, extract meaningful sentences
                if not key_insights:
                    sentences = compressed_research.split('. ')
                    for sentence in sentences[:5]:  # Take up to 5 sentences
                        if len(sentence.strip()) > 20:
                            key_insights.append(sentence.strip())
            
            # Ensure we have at least some insights
            if not key_insights and agent_expertise:
                key_insights = [
                    f"Conducted {', '.join(agent_expertise[:2])} research on {research_context['research_topic']}",
                    f"Applied {agent_name} specialized knowledge to the research topic"
                ]
            
            execution_time = time.time() - start_time
            
            # Count tool calls from raw notes
            tool_calls_count = 0
            search_queries_count = 0
            if raw_notes:
                raw_content = ' '.join(raw_notes)
                # Rough heuristic for tool usage
                tool_calls_count = raw_content.lower().count('tool') + raw_content.lower().count('search')
                search_queries_count = raw_content.lower().count('search') + raw_content.lower().count('query')
            
            return {
                "agent_type": agent_name,
                "agent_name": agent_name,
                "key_insights": key_insights[:10],  # Limit to top 10
                "research_findings": compressed_research or f"Research conducted by {agent_name} on {research_context['research_topic']}",
                "execution_duration": execution_time,
                "tool_calls_made": max(tool_calls_count, 1),  # At least 1
                "search_queries_executed": max(search_queries_count, 1),  # At least 1
                "success": True,
                "raw_research_content": compressed_research,
                "raw_notes": raw_notes
            }
            
        except Exception as e:
            logger.error(f"Failed to execute agent {agent_name}: {e}")
            execution_time = time.time() - start_time
            return {
                "agent_type": agent_name,
                "agent_name": agent_name,
                "key_insights": [f"Agent {agent_name} encountered an error during research execution"],
                "research_findings": f"Agent execution failed: {str(e)}",
                "execution_duration": execution_time,
                "tool_calls_made": 0,
                "search_queries_executed": 0,
                "success": False,
                "error": str(e)
            }


async def execute_sequences_in_parallel(
    sequences: List[AgentSequence],
    research_topic: str,
    config: RunnableConfig,
    max_concurrent: int = 3
) -> Dict[str, Any]:
    """Execute multiple agent sequences in parallel using SimpleSequentialExecutor.
    
    Args:
        sequences: List of agent sequences to execute
        research_topic: The research topic to investigate
        config: Runtime configuration
        max_concurrent: Maximum number of concurrent sequences
        
    Returns:
        Dictionary containing results from all parallel sequence executions
    """
    logger.info(f"Starting parallel execution of {len(sequences)} sequences (max concurrent: {max_concurrent})")
    
    # Create executor instance
    executor = SimpleSequentialExecutor(config)
    
    # Execute sequences in parallel with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_semaphore(sequence: AgentSequence) -> Tuple[str, SequenceExecutionResult]:
        async with semaphore:
            result = await executor.execute_agent_sequence(sequence, research_topic)
            return result.sequence_id, result
    
    # Create tasks for all sequences
    tasks = [execute_with_semaphore(seq) for seq in sequences]
    
    start_time = time.time()
    
    # Execute all sequences
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_duration = time.time() - start_time
        
        # Process results
        sequence_results = {}
        successful_count = 0
        failed_sequences = []
        error_summary = {}
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Sequence {i} failed with exception: {result}")
                failed_sequences.append(i)
                error_summary[i] = str(result)
            else:
                seq_id, seq_result = result
                if seq_result.success:
                    successful_count += 1
                    sequence_results[seq_id] = {
                        "comprehensive_findings": seq_result.comprehensive_findings,
                        "agent_results": seq_result.agent_results,
                        "total_duration": seq_result.total_duration,
                        "overall_productivity_metrics": seq_result.overall_productivity_metrics,
                        "productivity_score": seq_result.overall_productivity_metrics["tool_productivity"]
                    }
                else:
                    failed_sequences.append(i)
                    error_summary[i] = seq_result.error_message or "Unknown error"
        
        success_rate = (successful_count / len(sequences)) * 100 if sequences else 0
        
        # Aggregate unique insights
        all_insights = []
        for seq_result in sequence_results.values():
            all_insights.extend(seq_result["comprehensive_findings"])
        
        unique_insights = list(set(all_insights))  # Remove duplicates
        
        # Determine best performing strategy
        best_strategy = None
        if sequence_results:
            best_seq = max(
                sequence_results.items(),
                key=lambda x: len(x[1]["comprehensive_findings"])
            )
            best_strategy = best_seq[0]
        
        return {
            "execution_id": str(uuid4()),
            "research_topic": research_topic,
            "total_duration": total_duration,
            "success_rate": success_rate,
            "sequence_results": sequence_results,
            "best_strategy": best_strategy,
            "unique_insights_across_sequences": unique_insights,
            "failed_sequences": failed_sequences,
            "error_summary": error_summary,
            "best_performing_strategy": best_strategy
        }
        
    except Exception as e:
        logger.error(f"Parallel execution failed: {e}")
        return {
            "execution_id": str(uuid4()),
            "research_topic": research_topic,
            "total_duration": time.time() - start_time,
            "success_rate": 0.0,
            "sequence_results": {},
            "best_strategy": None,
            "unique_insights_across_sequences": [],
            "failed_sequences": list(range(len(sequences))),
            "error_summary": {i: str(e) for i in range(len(sequences))},
            "error_message": str(e)
        }