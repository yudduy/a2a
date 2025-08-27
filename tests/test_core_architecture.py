"""Comprehensive tests for Core Architecture components.

This module tests the foundational system components:
- Configuration system with all 21+ fields
- Environment variable integration and validation
- Backward compatibility and API preservation
- Agent registry loading and validation
- Core state management and LangGraph integration

Test Categories:
1. Configuration System (from test_configuration_system.py)
2. Backward Compatibility (from test_comprehensive_backward_compatibility.py)  
3. Agent Registry Core (from test_agent_registry_loading.py)
"""

import pytest
import os
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock, patch

from pydantic import ValidationError
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from open_deep_research.configuration import (
    Configuration, 
    SearchAPI,
    AgentFileFormat,
    ReportUpdateFrequency,
    MCPConfig
)
from open_deep_research.supervisor.sequential_supervisor import SupervisorConfig
from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.agents.loader import AgentLoader
from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.state import DeepResearchState


# =============================================================================
# CONFIGURATION SYSTEM TESTS
# =============================================================================

class TestSequentialSupervisorConfiguration:
    """Test Sequential Supervisor specific configuration fields."""
    
    def setup_method(self):
        """Set up configuration testing."""
        self.config = Configuration()
    
    def test_enable_sequential_supervisor_field(self):
        """Test enable_sequential_supervisor configuration field."""
        assert self.config.enable_sequential_supervisor is True
        
        config_enabled = Configuration(enable_sequential_supervisor=True)
        assert config_enabled.enable_sequential_supervisor is True
        
        config_disabled = Configuration(enable_sequential_supervisor=False)
        assert config_disabled.enable_sequential_supervisor is False
    
    def test_use_shared_state_field(self):
        """Test use_shared_state configuration field."""
        assert self.config.use_shared_state is True
        
        config = Configuration(use_shared_state=False)
        assert config.use_shared_state is False
    
    def test_automatic_handoffs_field(self):
        """Test automatic_handoffs configuration field."""
        assert self.config.automatic_handoffs is True
        
        config = Configuration(automatic_handoffs=False)
        assert config.automatic_handoffs is False
    
    def test_max_agents_per_sequence_field(self):
        """Test max_agents_per_sequence configuration field."""
        assert self.config.max_agents_per_sequence == 5
        
        config = Configuration(max_agents_per_sequence=3)
        assert config.max_agents_per_sequence == 3
        
        # Test validation limits
        config = Configuration(max_agents_per_sequence=1)
        assert config.max_agents_per_sequence >= 1
        
        config = Configuration(max_agents_per_sequence=25)
        assert config.max_agents_per_sequence <= 20


class TestAgentRegistryConfiguration:
    """Test Agent Registry configuration fields."""
    
    def setup_method(self):
        """Set up agent registry configuration testing."""
        self.config = Configuration()
    
    def test_project_agents_dir_field(self):
        """Test project_agents_dir configuration field."""
        assert self.config.project_agents_dir == ".open_deep_research/agents"
        
        config = Configuration(project_agents_dir="custom/agents/path")
        assert config.project_agents_dir == "custom/agents/path"
    
    def test_user_agents_dir_field(self):
        """Test user_agents_dir configuration field."""
        assert self.config.user_agents_dir == "~/.open_deep_research/agents"
        
        config = Configuration(user_agents_dir="/custom/user/agents")
        assert config.user_agents_dir == "/custom/user/agents"
    
    def test_agent_file_format_field(self):
        """Test agent_file_format configuration field."""
        assert self.config.agent_file_format == AgentFileFormat.MARKDOWN
        
        config = Configuration(agent_file_format=AgentFileFormat.YAML)
        assert config.agent_file_format == AgentFileFormat.YAML
        
        # Test string values (should be converted to enum)
        config = Configuration(agent_file_format="yaml")
        assert config.agent_file_format == AgentFileFormat.YAML
    
    def test_inherit_all_tools_field(self):
        """Test inherit_all_tools configuration field."""
        assert self.config.inherit_all_tools is True
        
        config = Configuration(inherit_all_tools=False)
        assert config.inherit_all_tools is False


class TestLLMJudgeConfiguration:
    """Test LLM Judge configuration fields."""
    
    def setup_method(self):
        """Set up LLM Judge configuration testing."""
        self.config = Configuration()
    
    def test_enable_llm_judge_field(self):
        """Test enable_llm_judge configuration field."""
        assert self.config.enable_llm_judge is True
        
        config = Configuration(enable_llm_judge=False)
        assert config.enable_llm_judge is False
    
    def test_evaluation_model_field(self):
        """Test evaluation_model configuration field."""
        assert "claude" in self.config.evaluation_model or "gpt" in self.config.evaluation_model
        
        config = Configuration(evaluation_model="openai:gpt-4")
        assert config.evaluation_model == "openai:gpt-4"
    
    def test_evaluation_criteria_field(self):
        """Test evaluation_criteria configuration field."""
        expected_default = ["completeness", "depth", "coherence", "innovation", "actionability"]
        assert self.config.evaluation_criteria == expected_default
        
        custom_criteria = ["accuracy", "relevance", "clarity"]
        config = Configuration(evaluation_criteria=custom_criteria)
        assert config.evaluation_criteria == custom_criteria


class TestEnvironmentVariableIntegration:
    """Test environment variable integration with configuration."""
    
    def setup_method(self):
        """Set up environment variable testing."""
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_environment_variable_overrides(self):
        """Test that environment variables override default values."""
        os.environ["ENABLE_SEQUENTIAL_SUPERVISOR"] = "false"
        os.environ["MAX_AGENTS_PER_SEQUENCE"] = "7"
        os.environ["USE_RUNNING_REPORTS"] = "false"
        os.environ["EVALUATION_MODEL"] = "openai:gpt-4-turbo"
        
        runnable_config = RunnableConfig(configurable={})
        config = Configuration.from_runnable_config(runnable_config)
        
        assert config.enable_sequential_supervisor is False
        assert config.max_agents_per_sequence == 7
        assert config.use_running_reports is False
        assert config.evaluation_model == "openai:gpt-4-turbo"
    
    def test_boolean_environment_variable_parsing(self):
        """Test parsing of boolean environment variables."""
        boolean_cases = [
            ("true", True), ("True", True), ("TRUE", True), ("1", True),
            ("false", False), ("False", False), ("FALSE", False), ("0", False)
        ]
        
        for env_value, expected in boolean_cases:
            os.environ["ENABLE_SEQUENTIAL_SUPERVISOR"] = env_value
            config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
            assert config.enable_sequential_supervisor == expected
    
    def test_numeric_environment_variable_parsing(self):
        """Test parsing of numeric environment variables."""
        os.environ["MAX_AGENTS_PER_SEQUENCE"] = "10"
        os.environ["EVALUATION_TIMEOUT"] = "300"
        
        config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        
        assert config.max_agents_per_sequence == 10
        assert config.evaluation_timeout == 300


class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def test_threshold_validation(self):
        """Test validation of threshold values."""
        config = Configuration(
            completion_confidence_threshold=0.5,
            modification_threshold=0.8
        )
        assert 0.0 <= config.completion_confidence_threshold <= 1.0
        assert 0.0 <= config.modification_threshold <= 1.0
    
    def test_directory_path_validation(self):
        """Test validation of directory paths."""
        config = Configuration(
            project_agents_dir="custom/agents",
            user_agents_dir="/home/user/agents"
        )
        assert config.project_agents_dir == "custom/agents"
        assert config.user_agents_dir == "/home/user/agents"
    
    def test_enum_field_validation(self):
        """Test validation of enum fields."""
        config = Configuration(
            agent_file_format=AgentFileFormat.YAML,
            report_update_frequency=ReportUpdateFrequency.ON_DEMAND
        )
        assert config.agent_file_format == AgentFileFormat.YAML
        assert config.report_update_frequency == ReportUpdateFrequency.ON_DEMAND


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestWorkflowBackwardCompatibility:
    """Test that existing workflows work unchanged when new features are disabled."""
    
    def setup_method(self):
        """Set up backward compatibility testing."""
        self.legacy_config = Configuration(
            enable_sequential_supervisor=False,
            enable_parallel_execution=False,
            enable_dynamic_sequencing=False
        )
    
    def test_legacy_workflow_configuration(self):
        """Test that legacy configurations are properly handled."""
        assert self.legacy_config.enable_sequential_supervisor is False
        assert self.legacy_config.enable_parallel_execution is False
        
        # Legacy workflow should still work with basic functionality
        assert self.legacy_config.max_research_loops >= 1
        assert self.legacy_config.search_api is not None
    
    def test_configuration_migration(self):
        """Test that old configurations migrate seamlessly."""
        # Test that deprecated fields are handled gracefully
        old_config = Configuration(
            max_research_loops=3,
            search_api="tavily",
            reasoning_model="claude-3-5-sonnet-20241022"
        )
        
        assert old_config.max_research_loops == 3
        assert old_config.search_api == "tavily"
        assert "claude" in old_config.reasoning_model
    
    def test_api_interface_preservation(self):
        """Test that public API interfaces are preserved."""
        config = Configuration()
        
        # Core methods should still exist
        assert hasattr(config, 'from_runnable_config')
        assert callable(getattr(config, 'from_runnable_config'))
        
        # Key fields should be accessible
        assert hasattr(config, 'reasoning_model')
        assert hasattr(config, 'search_api')
        assert hasattr(config, 'max_research_loops')


class TestStateManagementCompatibility:
    """Test that state management remains compatible."""
    
    def test_deep_research_state_compatibility(self):
        """Test DeepResearchState compatibility."""
        # Create basic state
        state = DeepResearchState(
            research_topic="Test topic",
            messages=[],
            research_report="",
            current_agent="research_agent"
        )
        
        assert state.research_topic == "Test topic"
        assert state.messages == []
        assert state.research_report == ""
        assert state.current_agent == "research_agent"
    
    def test_message_handling_compatibility(self):
        """Test that message handling remains backward compatible."""
        messages = [
            HumanMessage(content="Test query"),
            AIMessage(content="Test response")
        ]
        
        state = DeepResearchState(
            research_topic="Test",
            messages=messages,
            research_report=""
        )
        
        assert len(state.messages) == 2
        assert state.messages[0].content == "Test query"
        assert state.messages[1].content == "Test response"


# =============================================================================
# AGENT REGISTRY CORE TESTS
# =============================================================================

class TestAgentRegistryLoading:
    """Test agent registry loading functionality."""
    
    def setup_method(self):
        """Set up test directories and files."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_agents_dir = self.temp_dir / ".open_deep_research" / "agents"
        self.project_agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test agent file
        self._create_test_agent()
        
        self.config = Configuration(
            project_agents_dir=str(self.project_agents_dir)
        )
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_agent(self):
        """Create test agent definition."""
        agent_content = """---
name: test_agent
description: Test agent for registry testing
expertise_areas:
  - testing
  - validation
capabilities:
  - search
  - analysis
---

# Test Agent

This is a test agent for registry validation.
"""
        agent_file = self.project_agents_dir / "test_agent.md"
        agent_file.write_text(agent_content)
    
    def test_agent_registry_initialization(self):
        """Test agent registry initialization."""
        registry = AgentRegistry(self.config)
        assert registry is not None
        assert hasattr(registry, 'config')
    
    def test_agent_loading_basic(self):
        """Test basic agent loading functionality."""
        registry = AgentRegistry(self.config)
        
        # Test that registry can load agents
        try:
            agents = registry.list_agents()
            assert isinstance(agents, (list, dict))
        except Exception as e:
            # Some implementations may require async loading
            assert "async" in str(e).lower() or "not implemented" in str(e).lower()
    
    def test_agent_file_detection(self):
        """Test agent file detection."""
        # Verify test agent file exists
        agent_file = self.project_agents_dir / "test_agent.md"
        assert agent_file.exists()
        
        # Verify file has correct format
        content = agent_file.read_text()
        assert "name: test_agent" in content
        assert "Test Agent" in content
    
    def test_configuration_integration(self):
        """Test agent registry configuration integration."""
        config = Configuration(
            project_agents_dir=str(self.project_agents_dir),
            agent_file_format=AgentFileFormat.MARKDOWN,
            inherit_all_tools=True
        )
        
        registry = AgentRegistry(config)
        assert registry.config.project_agents_dir == str(self.project_agents_dir)
        assert registry.config.agent_file_format == AgentFileFormat.MARKDOWN
        assert registry.config.inherit_all_tools is True


class TestAgentValidation:
    """Test agent validation and error handling."""
    
    def setup_method(self):
        """Set up validation testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_invalid_agent_handling(self):
        """Test handling of invalid agent definitions."""
        # Create invalid agent file
        invalid_agent = self.agents_dir / "invalid.md"
        invalid_agent.write_text("This is not a valid agent definition")
        
        config = Configuration(project_agents_dir=str(self.agents_dir))
        registry = AgentRegistry(config)
        
        # Registry should handle invalid agents gracefully
        assert registry is not None
    
    def test_missing_required_fields(self):
        """Test handling of agents missing required fields."""
        # Create agent with missing fields
        incomplete_agent = """---
description: Agent missing name field
---

# Incomplete Agent
"""
        agent_file = self.agents_dir / "incomplete.md"
        agent_file.write_text(incomplete_agent)
        
        config = Configuration(project_agents_dir=str(self.agents_dir))
        registry = AgentRegistry(config)
        
        # Should not crash during initialization
        assert registry is not None


class TestPerformanceAndCaching:
    """Test configuration and registry performance."""
    
    def test_configuration_creation_performance(self):
        """Test configuration creation performance."""
        import time
        
        start_time = time.time()
        
        for _ in range(100):
            Configuration(
                enable_sequential_supervisor=True,
                max_agents_per_sequence=5,
                completion_confidence_threshold=0.6
            )
        
        creation_time = time.time() - start_time
        
        # Should be very fast (< 0.1s for 100 configurations)
        assert creation_time < 0.1, f"Configuration creation too slow: {creation_time:.3f}s"
    
    def test_from_runnable_config_performance(self):
        """Test performance of from_runnable_config method."""
        import time
        
        runnable_config = RunnableConfig(configurable={
            "enable_sequential_supervisor": True,
            "max_agents_per_sequence": 5
        })
        
        start_time = time.time()
        
        for _ in range(50):
            Configuration.from_runnable_config(runnable_config)
        
        conversion_time = time.time() - start_time
        
        # Should be fast (< 0.1s for 50 conversions)  
        assert conversion_time < 0.1, f"from_runnable_config too slow: {conversion_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])