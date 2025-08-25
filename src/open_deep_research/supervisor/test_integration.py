"""Integration test for Sequential Supervisor with existing codebase."""

import asyncio
import logging
from datetime import datetime

from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.supervisor import SequentialSupervisor, SupervisorConfig
from open_deep_research.state import SequentialSupervisorState
from open_deep_research.configuration import Configuration

# Configure logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_supervisor_initialization():
    """Test that supervisor initializes correctly with registry."""
    logger.info("Testing supervisor initialization...")
    
    # Create agent registry (will be empty initially but that's fine for testing)
    registry = AgentRegistry()
    
    # Create supervisor config
    config = SupervisorConfig(
        debug_mode=True,
        agent_timeout_seconds=30.0,
        completion_threshold=0.5
    )
    
    # Create system config
    system_config = Configuration()
    
    # Initialize supervisor
    supervisor = SequentialSupervisor(
        agent_registry=registry,
        config=config,
        system_config=system_config
    )
    
    # Verify initialization
    assert supervisor is not None
    assert supervisor.agent_registry == registry
    assert supervisor.config.debug_mode == True
    assert supervisor.completion_detector is not None
    
    logger.info("✓ Supervisor initialization test passed")


async def test_workflow_graph_creation():
    """Test workflow graph creation."""
    logger.info("Testing workflow graph creation...")
    
    registry = AgentRegistry()
    supervisor = SequentialSupervisor(
        agent_registry=registry,
        config=SupervisorConfig(debug_mode=True)
    )
    
    # Create workflow graph
    workflow = await supervisor.create_workflow_graph()
    
    # Verify graph structure
    assert workflow is not None
    assert "supervisor" in workflow.nodes
    assert "agent_executor" in workflow.nodes
    assert "finalize_sequence" in workflow.nodes
    
    logger.info("✓ Workflow graph creation test passed")


async def test_state_integration():
    """Test integration with state classes."""
    logger.info("Testing state integration...")
    
    # Create initial state dictionary (LangGraph uses TypedDict)
    state = {
        "messages": [],
        "research_topic": "Test research topic",
        "planned_sequence": ["test_agent_1", "test_agent_2"],
        "sequence_position": 0,
        "executed_agents": [],
        "agent_insights": {},
        "agent_questions": {},
        "agent_context": {},
        "agent_reports": {},
        "running_report": None,
        "report_sections": [],
        "last_agent_completed": None,
        "completion_signals": {},
        "handoff_ready": False,
        "sequence_start_time": None,
        "sequence_modifications": [],
        "supervisor_messages": [],
        "notes": [],
        "research_iterations": 0,
        "raw_notes": []
    }
    
    # Verify state fields
    assert state["research_topic"] == "Test research topic"
    assert len(state["planned_sequence"]) == 2
    assert state["sequence_position"] == 0
    assert isinstance(state["executed_agents"], list)
    assert isinstance(state["agent_insights"], dict)
    
    logger.info("✓ State integration test passed")


async def test_supervisor_validation():
    """Test supervisor validation methods."""
    logger.info("Testing supervisor validation...")
    
    registry = AgentRegistry()
    supervisor = SequentialSupervisor(agent_registry=registry)
    
    # Test empty sequence validation
    validation = supervisor.validate_sequence([])
    assert validation["valid"] == False
    assert "Empty sequence" in validation["errors"][0]
    
    # Test sequence with non-existent agents
    validation = supervisor.validate_sequence(["nonexistent_agent"])
    assert validation["valid"] == False
    assert "not found in registry" in validation["errors"][0]
    
    logger.info("✓ Supervisor validation test passed")


async def main():
    """Run all integration tests."""
    logger.info("Starting Sequential Supervisor integration tests...")
    
    try:
        await test_supervisor_initialization()
        await test_workflow_graph_creation()
        await test_state_integration()
        await test_supervisor_validation()
        
        logger.info("🎉 All integration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())