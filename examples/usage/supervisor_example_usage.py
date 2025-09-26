"""Example usage of the Sequential Supervisor for multi-agent workflows.

This demonstrates how to integrate the SequentialSupervisor with the existing
Open Deep Research codebase for orchestrating sequential agent execution.
"""

import asyncio
import logging
from typing import List

from langchain_core.messages import HumanMessage

from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.configuration import Configuration
from open_deep_research.supervisor import SequentialSupervisor, SupervisorConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def create_research_workflow(
    research_topic: str, 
    planned_agents: List[str],
    debug_mode: bool = False
) -> SequentialSupervisor:
    """Create a configured sequential research workflow.
    
    Args:
        research_topic: The research topic to investigate
        planned_agents: List of agent names to execute in sequence
        debug_mode: Enable debug logging
        
    Returns:
        Configured SequentialSupervisor ready for execution
    """
    logger.info(f"Creating research workflow for: {research_topic}")
    logger.info(f"Planned agent sequence: {', '.join(planned_agents)}")
    
    # Initialize agent registry
    registry = AgentRegistry()
    
    # Create supervisor configuration
    supervisor_config = SupervisorConfig(
        debug_mode=debug_mode,
        agent_timeout_seconds=300.0,  # 5 minutes per agent
        completion_threshold=0.6,
        max_agents_per_sequence=8,
        allow_dynamic_insertion=True,
        continue_on_agent_failure=True
    )
    
    # Create system configuration
    system_config = Configuration()
    
    # Initialize supervisor
    supervisor = SequentialSupervisor(
        agent_registry=registry,
        config=supervisor_config,
        system_config=system_config
    )
    
    # Validate the planned sequence
    validation = supervisor.validate_sequence(planned_agents)
    if not validation["valid"]:
        raise ValueError(f"Invalid agent sequence: {validation['errors']}")
    
    if validation["warnings"]:
        logger.warning(f"Sequence warnings: {validation['warnings']}")
    
    logger.info("✓ Research workflow created successfully")
    return supervisor


async def run_research_sequence(
    supervisor: SequentialSupervisor,
    research_topic: str,
    planned_agents: List[str],
    initial_message: str = None
) -> dict:
    """Execute a sequential research workflow.
    
    Args:
        supervisor: Configured SequentialSupervisor
        research_topic: Research topic
        planned_agents: List of agents to execute
        initial_message: Optional initial message from user
        
    Returns:
        Final state dictionary with results
    """
    logger.info("Starting sequential research execution...")
    
    # Create workflow graph
    workflow = await supervisor.create_workflow_graph()
    
    # Prepare initial state
    initial_state = {
        "messages": [HumanMessage(content=initial_message or f"Research the topic: {research_topic}")],
        "research_topic": research_topic,
        "research_brief": f"Comprehensive research on: {research_topic}",
        "planned_sequence": planned_agents,
        "sequence_position": 0,
        "executed_agents": [],
        "current_agent": None,
        "agent_insights": {},
        "agent_questions": {},
        "agent_context": {},
        "agent_reports": {},
        "running_report": None,
        "report_sections": [],
        "last_agent_completed": None,
        "completion_signals": {},
        "handoff_ready": True,  # Ready to start
        "sequence_start_time": None,
        "sequence_modifications": [],
        "supervisor_messages": [],
        "notes": [],
        "research_iterations": 0,
        "raw_notes": []
    }
    
    # Execute the workflow
    try:
        final_state = await workflow.ainvoke(initial_state)
        
        # Log execution summary
        execution_stats = supervisor.get_execution_stats()
        logger.info(f"Research execution completed: {execution_stats}")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Research execution failed: {e}")
        raise


async def example_academic_research():
    """Example: Academic research using specialized agents."""
    research_topic = "Impact of AI on Academic Publishing and Peer Review"
    
    # Define sequence of specialized agents
    planned_agents = [
        "research_agent",    # Research and literature analysis
        "technical_agent",   # Technical analysis
        "analysis_agent"     # Analysis and synthesis
    ]
    
    try:
        # Create workflow
        supervisor = await create_research_workflow(
            research_topic=research_topic,
            planned_agents=planned_agents,
            debug_mode=True
        )
        
        # Execute research
        results = await run_research_sequence(
            supervisor=supervisor,
            research_topic=research_topic,
            planned_agents=planned_agents,
            initial_message="Please conduct a comprehensive analysis of AI's impact on academic publishing, "
                          "focusing on peer review processes, publication workflows, and research integrity."
        )
        
        # Display results
        if results.get("running_report"):
            report = results["running_report"]
            
            # Show executive summary
            if report.executive_summary:
                pass
        
        
    except Exception:
        pass


async def example_industry_analysis():
    """Example: Industry analysis with dynamic sequence modification."""
    research_topic = "Emerging Trends in Enterprise AI Adoption"
    
    # Start with basic sequence
    planned_agents = [
        "market_agent",
        "technical_agent"
    ]
    
    try:
        # Create workflow with dynamic insertion enabled
        supervisor = await create_research_workflow(
            research_topic=research_topic,
            planned_agents=planned_agents,
            debug_mode=True
        )
        
        # Execute research
        results = await run_research_sequence(
            supervisor=supervisor,
            research_topic=research_topic,
            planned_agents=planned_agents,
            initial_message="Analyze current trends in enterprise AI adoption, "
                          "including market drivers, implementation challenges, and success factors."
        )
        
        # Display results
        if results.get("sequence_modifications"):
            for mod in results["sequence_modifications"]:
                pass
        
        
    except Exception:
        pass


async def example_error_handling():
    """Example: Error handling and recovery in sequential workflows."""
    research_topic = "Blockchain Scalability Solutions"
    
    # Include a non-existent agent to test error handling
    planned_agents = [
        "technical_agent",
        "nonexistent_agent",  # This will fail
        "market_agent"
    ]
    
    try:
        # Create workflow with error continuation enabled
        supervisor = await create_research_workflow(
            research_topic=research_topic,
            planned_agents=planned_agents,
            debug_mode=True
        )
        
        # This should fail validation
        validation = supervisor.validate_sequence(planned_agents)
        if not validation["valid"]:
            
            # Fix the sequence
            fixed_agents = ["technical_agent", "market_agent"]
            validation = supervisor.validate_sequence(fixed_agents)
            
            if validation["valid"]:
                
                # Execute with fixed sequence
                await run_research_sequence(
                    supervisor=supervisor,
                    research_topic=research_topic,
                    planned_agents=fixed_agents
                )
                
        
    except Exception:
        pass


async def main():
    """Run all examples demonstrating Sequential Supervisor usage."""
    try:
        # Run examples
        await example_academic_research()
        await example_industry_analysis()
        await example_error_handling()
        
        
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())