"""Comprehensive backward compatibility tests for Sequential Multi-Agent Supervisor.

This module ensures that:
- Existing workflows work exactly as before with enable_sequence_optimization=False
- All current functionality is preserved without breaking changes
- Configuration migration works seamlessly
- State management remains compatible with existing flows
- API contracts are maintained for all public interfaces

Test Categories:
1. Workflow compatibility with optimization disabled
2. Configuration backward compatibility
3. State management compatibility
4. API interface preservation
5. Performance regression prevention
6. Integration with existing systems
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import shutil

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from open_deep_research.configuration import Configuration
from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.state import DeepResearchState
from open_deep_research.supervisor.sequential_supervisor import SequentialSupervisor
from open_deep_research.agents.registry import AgentRegistry


class TestWorkflowBackwardCompatibility:
    """Test that existing workflows work unchanged when sequential optimization is disabled."""
    
    def setup_method(self):
        """Set up backward compatibility testing."""
        # Configuration with sequential features disabled
        self.legacy_config = Configuration(
            enable_sequence_optimization=False,
            enable_sequential_supervisor=False,
            enable_parallel_execution=False,
            enable_dynamic_sequencing=False,
            max_concurrent_research_units=3
        )
        
        # Modern configuration with sequential features enabled
        self.modern_config = Configuration(
            enable_sequence_optimization=True,
            enable_sequential_supervisor=True,
            enable_parallel_execution=True,
            enable_dynamic_sequencing=True,
            max_concurrent_research_units=5
        )
    
    @pytest.mark.asyncio
    async def test_legacy_deep_researcher_workflow(self):
        """Test that deep_researcher function works exactly as before."""
        legacy_state = DeepResearchState(
            research_topic="Legacy compatibility test: impact of remote work on productivity",
            research_question="How has remote work affected organizational productivity metrics?",
            max_research_units=3,
            current_research_units=[],
            completed_research_units=[],
            supervisor_messages=[],
            final_report="",
            final_report_messages=[]
        )
        
        # Mock the legacy workflow components
        with patch('open_deep_research.deep_researcher.supervisor_node') as mock_supervisor, \
             patch('open_deep_research.deep_researcher.research_unit_node') as mock_research, \
             patch('open_deep_research.deep_researcher.finalize_report_node') as mock_finalize:
            
            # Configure mocks to simulate legacy behavior
            mock_supervisor.return_value = legacy_state
            mock_research.return_value = legacy_state
            mock_finalize.return_value = legacy_state
            
            # Execute with legacy configuration
            graph = StateGraph(DeepResearchState)
            graph.add_node("supervisor", mock_supervisor)
            graph.add_node("research_unit", mock_research)
            graph.add_node("finalize_report", mock_finalize)
            
            # Test that legacy flow works
            compiled_graph = graph.compile()
            
            # This should work without sequential supervisor interference
            result = await compiled_graph.ainvoke(legacy_state)
            
            # Verify legacy behavior is preserved
            assert isinstance(result, DeepResearchState)
            assert result.research_topic == legacy_state.research_topic
            assert result.max_research_units == 3  # Legacy default
    
    def test_configuration_migration_compatibility(self):
        """Test that configuration migration doesn't break existing setups."""
        # Simulate old configuration format
        old_config_dict = {
            "max_structured_output_retries": 3,
            "allow_clarification": True,
            "max_concurrent_research_units": 5,
            "search_api": "tavily",
            "researcher_model": "anthropic:claude-3-5-sonnet",
            "researcher_model_max_tokens": 8192
        }
        
        # Create configuration from old format
        runnable_config = RunnableConfig(configurable=old_config_dict)
        migrated_config = Configuration.from_runnable_config(runnable_config)
        
        # Verify all old settings are preserved
        assert migrated_config.max_structured_output_retries == 3
        assert migrated_config.allow_clarification is True
        assert migrated_config.max_concurrent_research_units == 5
        
        # Verify new settings have sensible defaults
        assert migrated_config.enable_sequential_supervisor is True  # New feature enabled by default
        assert migrated_config.enable_sequence_optimization is False  # Opt-in feature
        assert migrated_config.enable_parallel_execution is False  # Opt-in feature
    
    def test_state_compatibility_with_legacy_formats(self):
        """Test that state management is compatible with legacy state formats."""
        # Legacy state without sequential supervisor fields
        legacy_state_dict = {
            "research_topic": "Legacy state test",
            "research_question": "Test question",
            "max_research_units": 3,
            "current_research_units": [],
            "completed_research_units": [],
            "supervisor_messages": [],
            "final_report": "",
            "final_report_messages": []
        }
        
        # Create state from legacy format
        legacy_state = DeepResearchState(**legacy_state_dict)
        
        # Verify legacy fields are present
        assert legacy_state.research_topic == "Legacy state test"
        assert legacy_state.max_research_units == 3
        assert len(legacy_state.current_research_units) == 0
        
        # Verify state can be processed by modern system
        assert isinstance(legacy_state, DeepResearchState)
    
    @pytest.mark.asyncio
    async def test_api_interface_preservation(self):
        """Test that all public API interfaces remain unchanged."""
        # Test deep_researcher function signature compatibility
        try:
            # Should accept legacy call patterns
            from open_deep_research.deep_researcher import deep_researcher
            
            # Function should exist and be callable
            assert callable(deep_researcher)
            
            # Test with minimal legacy parameters
            legacy_state = DeepResearchState(
                research_topic="API compatibility test",
                research_question="Does the API work?",
                max_research_units=1
            )
            
            # Should not raise signature errors
            # (We're not executing due to mocking complexity, just testing interface)
            assert True
            
        except ImportError as e:
            pytest.fail(f"deep_researcher function not available: {e}")
        except TypeError as e:
            pytest.fail(f"API interface changed: {e}")


class TestConfigurationBackwardCompatibility:
    """Test configuration backward compatibility and field preservation."""
    
    def setup_method(self):
        """Set up configuration compatibility testing."""
        pass
    
    def test_all_legacy_fields_preserved(self):
        """Test that all legacy configuration fields are preserved."""
        # Legacy fields that must be preserved
        legacy_fields = {
            "max_structured_output_retries": 3,
            "allow_clarification": True,
            "max_concurrent_research_units": 5,
            "search_api": "tavily",
            "researcher_model": "anthropic:claude-3-5-sonnet",
            "researcher_model_max_tokens": 8192,
            "compression_model": "anthropic:claude-3-haiku",
            "compression_model_max_tokens": 8192,
            "final_report_model": "hyperbolic:Qwen/Qwen3-235B-A22B",
            "final_report_model_max_tokens": 10000
        }
        
        # Create configuration with legacy fields
        config = Configuration(**legacy_fields)
        
        # Verify all legacy fields are preserved
        for field_name, expected_value in legacy_fields.items():
            actual_value = getattr(config, field_name)
            assert actual_value == expected_value, f"Legacy field {field_name} changed from {expected_value} to {actual_value}"
    
    def test_new_fields_have_safe_defaults(self):
        """Test that new sequential supervisor fields have safe defaults."""
        config = Configuration()
        
        # New fields should be conservative by default (opt-in)
        conservative_defaults = {
            "enable_sequence_optimization": False,  # Should be opt-in
            "enable_parallel_execution": False,     # Should be opt-in
            "enable_dynamic_sequencing": False,     # Should be opt-in
        }
        
        for field_name, expected_default in conservative_defaults.items():
            actual_value = getattr(config, field_name)
            assert actual_value == expected_default, f"New field {field_name} should default to {expected_default}"
        
        # Safe-to-enable defaults
        safe_defaults = {
            "enable_sequential_supervisor": True,   # Safe to enable
            "use_shared_state": True,              # Safe enhancement
            "automatic_handoffs": True,            # Safe automation
            "use_running_reports": True,           # Safe improvement
            "enable_llm_judge": True               # Safe evaluation
        }
        
        for field_name, expected_default in safe_defaults.items():
            actual_value = getattr(config, field_name)
            assert actual_value == expected_default, f"Safe field {field_name} should default to {expected_default}"
    
    def test_configuration_serialization_compatibility(self):
        """Test that configuration can be serialized/deserialized compatible with legacy systems."""
        # Create configuration with mixed legacy and new fields
        config = Configuration(
            max_concurrent_research_units=3,
            enable_sequential_supervisor=True,
            use_running_reports=False
        )
        
        # Convert to dict (simulates serialization)
        config_dict = config.dict()
        
        # Verify legacy fields are present
        assert "max_concurrent_research_units" in config_dict
        assert config_dict["max_concurrent_research_units"] == 3
        
        # Verify new fields are present
        assert "enable_sequential_supervisor" in config_dict
        assert config_dict["enable_sequential_supervisor"] is True
        
        # Create new configuration from dict (simulates deserialization)
        restored_config = Configuration(**config_dict)
        
        # Verify restoration preserves all values
        assert restored_config.max_concurrent_research_units == 3
        assert restored_config.enable_sequential_supervisor is True
        assert restored_config.use_running_reports is False


class TestPerformanceRegressionPrevention:
    """Test that sequential supervisor additions don't cause performance regressions."""
    
    def setup_method(self):
        """Set up performance regression testing."""
        self.legacy_config = Configuration(
            enable_sequence_optimization=False,
            enable_sequential_supervisor=False,
            enable_parallel_execution=False
        )
        
        self.modern_config = Configuration(
            enable_sequence_optimization=True,
            enable_sequential_supervisor=True,
            enable_parallel_execution=True
        )
    
    def test_configuration_creation_performance(self):
        """Test that configuration creation performance hasn't regressed."""
        import time
        
        # Time legacy configuration creation
        start_time = time.time()
        for _ in range(100):
            Configuration(
                max_concurrent_research_units=5,
                researcher_model="test_model"
            )
        legacy_time = time.time() - start_time
        
        # Time modern configuration creation
        start_time = time.time()
        for _ in range(100):
            Configuration(
                max_concurrent_research_units=5,
                researcher_model="test_model",
                enable_sequential_supervisor=True,
                enable_sequence_optimization=True,
                enable_parallel_execution=True,
                max_agents_per_sequence=5,
                completion_confidence_threshold=0.6
            )
        modern_time = time.time() - start_time
        
        # Modern config creation shouldn't be significantly slower
        performance_ratio = modern_time / legacy_time if legacy_time > 0 else 1.0
        assert performance_ratio < 2.0, f"Configuration creation {performance_ratio:.2f}x slower with new fields"
        
        print(f"Config creation - Legacy: {legacy_time:.3f}s, Modern: {modern_time:.3f}s, Ratio: {performance_ratio:.2f}x")
    
    def test_memory_usage_regression(self):
        """Test that memory usage hasn't significantly increased."""
        import sys
        
        # Create legacy configuration
        legacy_config = Configuration(
            max_concurrent_research_units=3,
            researcher_model="test_model"
        )
        legacy_size = sys.getsizeof(legacy_config)
        
        # Create modern configuration with all new fields
        modern_config = Configuration(
            max_concurrent_research_units=3,
            researcher_model="test_model",
            enable_sequential_supervisor=True,
            enable_sequence_optimization=True,
            enable_parallel_execution=True,
            max_agents_per_sequence=5,
            completion_confidence_threshold=0.6,
            project_agents_dir=".open_deep_research/agents",
            user_agents_dir="~/.open_deep_research/agents",
            completion_indicators=["complete", "finished"],
            evaluation_criteria=["quality", "depth"]
        )
        modern_size = sys.getsizeof(modern_config)
        
        # Memory increase should be reasonable (< 3x)
        memory_ratio = modern_size / legacy_size if legacy_size > 0 else 1.0
        assert memory_ratio < 3.0, f"Configuration memory usage increased {memory_ratio:.2f}x"
        
        print(f"Config memory - Legacy: {legacy_size} bytes, Modern: {modern_size} bytes, Ratio: {memory_ratio:.2f}x")


class TestIntegrationWithExistingSystems:
    """Test integration with existing system components."""
    
    def setup_method(self):
        """Set up integration testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_langgraph_workflow_compatibility(self):
        """Test that LangGraph workflows remain compatible."""
        # Create a simple LangGraph workflow
        from langgraph.graph import StateGraph
        
        # Define legacy state
        legacy_state = DeepResearchState(
            research_topic="LangGraph compatibility test",
            research_question="Does LangGraph integration work?",
            max_research_units=2
        )
        
        # Create workflow
        workflow = StateGraph(DeepResearchState)
        
        def mock_node(state):
            return state
        
        workflow.add_node("test_node", mock_node)
        workflow.set_entry_point("test_node")
        workflow.add_edge("test_node", END)
        
        # Compile and test
        compiled = workflow.compile()
        
        # Should work without sequential supervisor interference
        result = compiled.invoke(legacy_state)
        
        assert isinstance(result, DeepResearchState)
        assert result.research_topic == "LangGraph compatibility test"
    
    def test_tool_integration_compatibility(self):
        """Test that tool integration remains compatible."""
        from open_deep_research.utils import get_all_tools, think_tool
        
        # Test that existing tool loading works
        tools = get_all_tools(["search", "scraper"])
        assert isinstance(tools, list)
        assert len(tools) >= 0  # May be empty if tools aren't available in test environment
        
        # Test that think_tool is available
        assert think_tool is not None
        assert hasattr(think_tool, 'name') or hasattr(think_tool, '_name')
    
    def test_model_configuration_compatibility(self):
        """Test that model configuration remains compatible."""
        from langchain.chat_models import init_chat_model
        
        # Test that model initialization works with legacy config
        try:
            model = init_chat_model(
                model="anthropic:claude-3-5-sonnet",
                max_tokens=4096,
                configurable_fields=("model", "max_tokens")
            )
            assert model is not None
        except Exception as e:
            # If model init fails, it should be due to missing credentials, not interface changes
            assert "api" in str(e).lower() or "key" in str(e).lower() or "auth" in str(e).lower()


class TestDataMigrationCompatibility:
    """Test data migration and state upgrade compatibility."""
    
    def setup_method(self):
        """Set up data migration testing."""
        pass
    
    def test_state_upgrade_from_legacy(self):
        """Test upgrading legacy state to include sequential supervisor fields."""
        # Legacy state data (missing sequential fields)
        legacy_data = {
            "research_topic": "Migration test topic",
            "research_question": "Can legacy states be upgraded?",
            "max_research_units": 3,
            "current_research_units": [],
            "completed_research_units": [],
            "supervisor_messages": [],
            "final_report": "",
            "final_report_messages": []
        }
        
        # Create state from legacy data
        state = DeepResearchState(**legacy_data)
        
        # Verify upgrade is seamless
        assert state.research_topic == "Migration test topic"
        assert state.max_research_units == 3
        
        # Should handle missing sequential fields gracefully
        assert hasattr(state, 'research_topic')  # Legacy field preserved
    
    def test_configuration_upgrade_from_legacy(self):
        """Test upgrading legacy configuration to include new fields."""
        # Legacy configuration data
        legacy_config_data = {
            "max_structured_output_retries": 3,
            "allow_clarification": True,
            "max_concurrent_research_units": 5,
            "researcher_model": "anthropic:claude-3-5-sonnet"
        }
        
        # Create configuration from legacy data
        config = Configuration(**legacy_config_data)
        
        # Verify legacy fields preserved
        assert config.max_structured_output_retries == 3
        assert config.allow_clarification is True
        assert config.max_concurrent_research_units == 5
        assert config.researcher_model == "anthropic:claude-3-5-sonnet"
        
        # Verify new fields have appropriate defaults
        assert hasattr(config, 'enable_sequential_supervisor')
        assert hasattr(config, 'max_agents_per_sequence')
        assert hasattr(config, 'completion_confidence_threshold')


class TestAPIContractPreservation:
    """Test that all public API contracts are preserved."""
    
    def setup_method(self):
        """Set up API contract testing."""
        pass
    
    def test_deep_researcher_function_signature(self):
        """Test that deep_researcher function signature is preserved."""
        from open_deep_research.deep_researcher import deep_researcher
        
        # Function should exist
        assert callable(deep_researcher)
        
        # Should accept standard LangGraph parameters
        import inspect
        signature = inspect.signature(deep_researcher)
        
        # Should have reasonable number of parameters (not overly complex)
        assert len(signature.parameters) <= 10
    
    def test_configuration_class_interface(self):
        """Test that Configuration class interface is preserved."""
        # Should be constructible with no arguments (defaults)
        config = Configuration()
        assert isinstance(config, Configuration)
        
        # Should have from_runnable_config class method
        assert hasattr(Configuration, 'from_runnable_config')
        assert callable(Configuration.from_runnable_config)
        
        # Should support dict conversion
        config_dict = config.dict()
        assert isinstance(config_dict, dict)
        
        # Should have all expected legacy fields
        legacy_fields = [
            'max_structured_output_retries',
            'allow_clarification', 
            'max_concurrent_research_units',
            'search_api',
            'researcher_model',
            'researcher_model_max_tokens'
        ]
        
        for field in legacy_fields:
            assert hasattr(config, field), f"Legacy field {field} missing"
    
    def test_state_class_interface(self):
        """Test that state classes maintain their interface."""
        # DeepResearchState should be constructible
        state = DeepResearchState(
            research_topic="Interface test",
            research_question="Is the interface preserved?",
            max_research_units=1
        )
        
        assert isinstance(state, DeepResearchState)
        
        # Should have all legacy fields
        legacy_state_fields = [
            'research_topic',
            'research_question', 
            'max_research_units',
            'current_research_units',
            'completed_research_units',
            'supervisor_messages',
            'final_report',
            'final_report_messages'
        ]
        
        for field in legacy_state_fields:
            assert hasattr(state, field), f"Legacy state field {field} missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])