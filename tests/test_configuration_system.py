"""Comprehensive tests for Sequential Supervisor configuration system.

This module tests:
- All 21+ configuration fields for sequential supervisor functionality
- Configuration validation and constraints
- Environment variable overrides
- Default values and field constraints
- Configuration integration with supervisor components
- Runtime configuration changes
- Error handling for invalid configurations

Test Categories:
1. Sequential Supervisor configuration fields
2. Agent Registry configuration
3. Completion Detection configuration
4. Running Reports configuration  
5. LLM Judge configuration
6. Environment variable integration
7. Configuration validation and constraints
8. Runtime configuration updates
"""

import pytest
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

from pydantic import ValidationError
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import (
    Configuration, 
    SearchAPI,
    AgentFileFormat,
    ReportUpdateFrequency,
    MCPConfig
)
from open_deep_research.supervisor.sequential_supervisor import SupervisorConfig


class TestSequentialSupervisorConfiguration:
    """Test Sequential Supervisor specific configuration fields."""
    
    def setup_method(self):
        """Set up configuration testing."""
        self.config = Configuration()
    
    def test_enable_sequential_supervisor_field(self):
        """Test enable_sequential_supervisor configuration field."""
        # Default value
        assert self.config.enable_sequential_supervisor is True
        
        # Test explicit values
        config_enabled = Configuration(enable_sequential_supervisor=True)
        assert config_enabled.enable_sequential_supervisor is True
        
        config_disabled = Configuration(enable_sequential_supervisor=False)
        assert config_disabled.enable_sequential_supervisor is False
    
    def test_use_shared_state_field(self):
        """Test use_shared_state configuration field."""
        # Default value
        assert self.config.use_shared_state is True
        
        # Test explicit values
        config = Configuration(use_shared_state=False)
        assert config.use_shared_state is False
    
    def test_automatic_handoffs_field(self):
        """Test automatic_handoffs configuration field."""
        # Default value  
        assert self.config.automatic_handoffs is True
        
        # Test explicit values
        config = Configuration(automatic_handoffs=False)
        assert config.automatic_handoffs is False
    
    def test_allow_dynamic_modification_field(self):
        """Test allow_dynamic_modification configuration field."""
        # Default value
        assert self.config.allow_dynamic_modification is True
        
        # Test explicit values
        config = Configuration(allow_dynamic_modification=False)
        assert config.allow_dynamic_modification is False
    
    def test_max_agents_per_sequence_field(self):
        """Test max_agents_per_sequence configuration field."""
        # Default value
        assert self.config.max_agents_per_sequence == 5
        
        # Test valid values
        config = Configuration(max_agents_per_sequence=3)
        assert config.max_agents_per_sequence == 3
        
        config = Configuration(max_agents_per_sequence=10)
        assert config.max_agents_per_sequence == 10
        
        # Test validation limits
        config = Configuration(max_agents_per_sequence=1)
        assert config.max_agents_per_sequence == 5  # Should use default due to validation
        
        config = Configuration(max_agents_per_sequence=25)
        assert config.max_agents_per_sequence == 20  # Should cap at maximum
    
    def test_modification_threshold_field(self):
        """Test modification_threshold configuration field."""
        # Default value
        assert self.config.modification_threshold == 0.7
        
        # Test valid values
        config = Configuration(modification_threshold=0.5)
        assert config.modification_threshold == 0.5
        
        config = Configuration(modification_threshold=0.9)
        assert config.modification_threshold == 0.9
        
        # Test validation boundaries
        config = Configuration(modification_threshold=-0.1)
        assert 0.0 <= config.modification_threshold <= 1.0
        
        config = Configuration(modification_threshold=1.5)
        assert 0.0 <= config.modification_threshold <= 1.0


class TestAgentRegistryConfiguration:
    """Test Agent Registry configuration fields."""
    
    def setup_method(self):
        """Set up agent registry configuration testing."""
        self.config = Configuration()
    
    def test_project_agents_dir_field(self):
        """Test project_agents_dir configuration field."""
        # Default value
        assert self.config.project_agents_dir == ".open_deep_research/agents"
        
        # Test custom values
        config = Configuration(project_agents_dir="custom/agents/path")
        assert config.project_agents_dir == "custom/agents/path"
        
        # Test validation
        config = Configuration(project_agents_dir="")
        assert config.project_agents_dir == ".open_deep_research/agents"  # Should use default
        
        config = Configuration(project_agents_dir=None)
        assert config.project_agents_dir == ".open_deep_research/agents"  # Should use default
    
    def test_user_agents_dir_field(self):
        """Test user_agents_dir configuration field."""
        # Default value
        assert self.config.user_agents_dir == "~/.open_deep_research/agents"
        
        # Test custom values
        config = Configuration(user_agents_dir="/custom/user/agents")
        assert config.user_agents_dir == "/custom/user/agents"
        
        # Test validation
        config = Configuration(user_agents_dir="")
        assert config.user_agents_dir == "~/.open_deep_research/agents"  # Should use default
    
    def test_agent_file_format_field(self):
        """Test agent_file_format configuration field."""
        # Default value
        assert self.config.agent_file_format == AgentFileFormat.MARKDOWN
        
        # Test enum values
        config = Configuration(agent_file_format=AgentFileFormat.YAML)
        assert config.agent_file_format == AgentFileFormat.YAML
        
        # Test string values (should be converted to enum)
        config = Configuration(agent_file_format="yaml")
        assert config.agent_file_format == AgentFileFormat.YAML
        
        config = Configuration(agent_file_format="markdown")
        assert config.agent_file_format == AgentFileFormat.MARKDOWN
    
    def test_inherit_all_tools_field(self):
        """Test inherit_all_tools configuration field."""
        # Default value
        assert self.config.inherit_all_tools is True
        
        # Test explicit values
        config = Configuration(inherit_all_tools=False)
        assert config.inherit_all_tools is False


class TestCompletionDetectionConfiguration:
    """Test Completion Detection configuration fields."""
    
    def setup_method(self):
        """Set up completion detection configuration testing."""
        self.config = Configuration()
    
    def test_use_automatic_completion_field(self):
        """Test use_automatic_completion configuration field."""
        # Default value
        assert self.config.use_automatic_completion is True
        
        # Test explicit values
        config = Configuration(use_automatic_completion=False)
        assert config.use_automatic_completion is False
    
    def test_completion_confidence_threshold_field(self):
        """Test completion_confidence_threshold configuration field."""
        # Default value
        assert self.config.completion_confidence_threshold == 0.6
        
        # Test valid values
        config = Configuration(completion_confidence_threshold=0.8)
        assert config.completion_confidence_threshold == 0.8
        
        # Test validation boundaries
        config = Configuration(completion_confidence_threshold=-0.1)
        assert 0.0 <= config.completion_confidence_threshold <= 1.0
        
        config = Configuration(completion_confidence_threshold=1.5)
        assert 0.0 <= config.completion_confidence_threshold <= 1.0
    
    def test_completion_indicators_field(self):
        """Test completion_indicators configuration field."""
        # Default value
        expected_default = ["research complete", "analysis complete", "findings summarized", 
                           "investigation finished", "task accomplished"]
        assert self.config.completion_indicators == expected_default
        
        # Test list input
        custom_indicators = ["work done", "task finished", "analysis ready"]
        config = Configuration(completion_indicators=custom_indicators)
        assert config.completion_indicators == custom_indicators
        
        # Test string input (comma-separated)
        string_indicators = "work done, task finished, analysis ready"
        config = Configuration(completion_indicators=string_indicators)
        expected = ["work done", "task finished", "analysis ready"]
        assert config.completion_indicators == expected
        
        # Test empty input handling
        config = Configuration(completion_indicators=[])
        assert len(config.completion_indicators) > 0  # Should use default


class TestRunningReportsConfiguration:
    """Test Running Reports configuration fields."""
    
    def setup_method(self):
        """Set up running reports configuration testing."""
        self.config = Configuration()
    
    def test_use_running_reports_field(self):
        """Test use_running_reports configuration field."""
        # Default value
        assert self.config.use_running_reports is True
        
        # Test explicit values
        config = Configuration(use_running_reports=False)
        assert config.use_running_reports is False
    
    def test_report_update_frequency_field(self):
        """Test report_update_frequency configuration field."""
        # Default value
        assert self.config.report_update_frequency == ReportUpdateFrequency.AFTER_EACH_AGENT
        
        # Test enum values
        config = Configuration(report_update_frequency=ReportUpdateFrequency.AFTER_SEQUENCE)
        assert config.report_update_frequency == ReportUpdateFrequency.AFTER_SEQUENCE
        
        config = Configuration(report_update_frequency=ReportUpdateFrequency.ON_DEMAND)
        assert config.report_update_frequency == ReportUpdateFrequency.ON_DEMAND
        
        # Test string values
        config = Configuration(report_update_frequency="after_sequence")
        assert config.report_update_frequency == ReportUpdateFrequency.AFTER_SEQUENCE
    
    def test_include_agent_metadata_field(self):
        """Test include_agent_metadata configuration field."""
        # Default value
        assert self.config.include_agent_metadata is True
        
        # Test explicit values
        config = Configuration(include_agent_metadata=False)
        assert config.include_agent_metadata is False


class TestLLMJudgeConfiguration:
    """Test LLM Judge configuration fields."""
    
    def setup_method(self):
        """Set up LLM Judge configuration testing."""
        self.config = Configuration()
    
    def test_enable_llm_judge_field(self):
        """Test enable_llm_judge configuration field."""
        # Default value
        assert self.config.enable_llm_judge is True
        
        # Test explicit values
        config = Configuration(enable_llm_judge=False)
        assert config.enable_llm_judge is False
    
    def test_evaluation_model_field(self):
        """Test evaluation_model configuration field."""
        # Default value
        assert self.config.evaluation_model == "anthropic:claude-3-5-sonnet"
        
        # Test custom values
        config = Configuration(evaluation_model="openai:gpt-4")
        assert config.evaluation_model == "openai:gpt-4"
    
    def test_evaluation_model_max_tokens_field(self):
        """Test evaluation_model_max_tokens configuration field."""
        # Default value
        assert self.config.evaluation_model_max_tokens == 8192
        
        # Test custom values
        config = Configuration(evaluation_model_max_tokens=4096)
        assert config.evaluation_model_max_tokens == 4096
    
    def test_evaluation_criteria_field(self):
        """Test evaluation_criteria configuration field."""
        # Default value
        expected_default = ["completeness", "depth", "coherence", "innovation", "actionability"]
        assert self.config.evaluation_criteria == expected_default
        
        # Test list input
        custom_criteria = ["accuracy", "relevance", "clarity"]
        config = Configuration(evaluation_criteria=custom_criteria)
        assert config.evaluation_criteria == custom_criteria
        
        # Test string input (comma-separated)
        string_criteria = "accuracy, relevance, clarity"
        config = Configuration(evaluation_criteria=string_criteria)
        expected = ["accuracy", "relevance", "clarity"]
        assert config.evaluation_criteria == expected
    
    def test_evaluation_timeout_field(self):
        """Test evaluation_timeout configuration field."""
        # Default value
        assert self.config.evaluation_timeout == 120
        
        # Test valid values
        config = Configuration(evaluation_timeout=60)
        assert config.evaluation_timeout == 60
        
        # Test validation boundaries
        config = Configuration(evaluation_timeout=10)
        assert config.evaluation_timeout == 120  # Should use default if below minimum
        
        config = Configuration(evaluation_timeout=700)
        assert config.evaluation_timeout == 600  # Should cap at maximum


class TestParallelExecutionConfiguration:
    """Test Parallel Execution configuration fields."""
    
    def setup_method(self):
        """Set up parallel execution configuration testing."""
        self.config = Configuration()
    
    def test_enable_parallel_execution_field(self):
        """Test enable_parallel_execution configuration field."""
        # Default value
        assert self.config.enable_parallel_execution is False
        
        # Test explicit values
        config = Configuration(enable_parallel_execution=True)
        assert config.enable_parallel_execution is True
    
    def test_max_parallel_sequences_field(self):
        """Test max_parallel_sequences configuration field."""
        # Default value
        assert self.config.max_parallel_sequences == 3
        
        # Test valid values
        config = Configuration(max_parallel_sequences=5)
        assert config.max_parallel_sequences == 5
        
        config = Configuration(max_parallel_sequences=1)
        assert config.max_parallel_sequences == 1
    
    def test_parallel_execution_timeout_field(self):
        """Test parallel_execution_timeout configuration field."""
        # Default value
        assert self.config.parallel_execution_timeout == 3600
        
        # Test custom values
        config = Configuration(parallel_execution_timeout=1800)
        assert config.parallel_execution_timeout == 1800
    
    def test_parallel_retry_attempts_field(self):
        """Test parallel_retry_attempts configuration field."""
        # Default value
        assert self.config.parallel_retry_attempts == 2
        
        # Test custom values
        config = Configuration(parallel_retry_attempts=1)
        assert config.parallel_retry_attempts == 1
        
        config = Configuration(parallel_retry_attempts=5)
        assert config.parallel_retry_attempts == 5


class TestEnvironmentVariableIntegration:
    """Test environment variable integration with configuration."""
    
    def setup_method(self):
        """Set up environment variable testing."""
        # Store original environment
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_environment_variable_overrides(self):
        """Test that environment variables override default values."""
        # Set environment variables
        os.environ["ENABLE_SEQUENTIAL_SUPERVISOR"] = "false"
        os.environ["MAX_AGENTS_PER_SEQUENCE"] = "7"
        os.environ["COMPLETION_CONFIDENCE_THRESHOLD"] = "0.8"
        os.environ["USE_RUNNING_REPORTS"] = "false"
        os.environ["EVALUATION_MODEL"] = "openai:gpt-4-turbo"
        
        # Create config from RunnableConfig (simulating runtime)
        runnable_config = RunnableConfig(configurable={})
        config = Configuration.from_runnable_config(runnable_config)
        
        # Verify environment overrides
        assert config.enable_sequential_supervisor is False
        assert config.max_agents_per_sequence == 7
        assert config.completion_confidence_threshold == 0.8
        assert config.use_running_reports is False
        assert config.evaluation_model == "openai:gpt-4-turbo"
    
    def test_configurable_overrides_environment(self):
        """Test that configurable values override environment variables."""
        # Set environment variable
        os.environ["ENABLE_SEQUENTIAL_SUPERVISOR"] = "false"
        os.environ["MAX_AGENTS_PER_SEQUENCE"] = "3"
        
        # Set configurable values
        runnable_config = RunnableConfig(configurable={
            "enable_sequential_supervisor": True,
            "max_agents_per_sequence": 8
        })
        
        config = Configuration.from_runnable_config(runnable_config)
        
        # Configurable should override environment
        assert config.enable_sequential_supervisor is True
        assert config.max_agents_per_sequence == 8
    
    def test_boolean_environment_variable_parsing(self):
        """Test parsing of boolean environment variables."""
        boolean_cases = [
            ("true", True),
            ("True", True), 
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("", False),
        ]
        
        for env_value, expected in boolean_cases:
            os.environ["ENABLE_SEQUENTIAL_SUPERVISOR"] = env_value
            
            config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
            assert config.enable_sequential_supervisor == expected, \
                f"Environment value '{env_value}' should result in {expected}"
    
    def test_numeric_environment_variable_parsing(self):
        """Test parsing of numeric environment variables."""
        # Integer values
        os.environ["MAX_AGENTS_PER_SEQUENCE"] = "10"
        os.environ["EVALUATION_TIMEOUT"] = "300"
        
        # Float values
        os.environ["COMPLETION_CONFIDENCE_THRESHOLD"] = "0.75"
        os.environ["MODIFICATION_THRESHOLD"] = "0.85"
        
        config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        
        assert config.max_agents_per_sequence == 10
        assert config.evaluation_timeout == 300
        assert config.completion_confidence_threshold == 0.75
        assert config.modification_threshold == 0.85
    
    def test_list_environment_variable_parsing(self):
        """Test parsing of list environment variables."""
        os.environ["COMPLETION_INDICATORS"] = "task done, work finished, analysis ready"
        os.environ["EVALUATION_CRITERIA"] = "accuracy, speed, quality"
        
        config = Configuration.from_runnable_config(RunnableConfig(configurable={}))
        
        expected_indicators = ["task done", "work finished", "analysis ready"]
        expected_criteria = ["accuracy", "speed", "quality"]
        
        assert config.completion_indicators == expected_indicators
        assert config.evaluation_criteria == expected_criteria


class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def setup_method(self):
        """Set up validation testing."""
        pass
    
    def test_threshold_validation(self):
        """Test validation of threshold values."""
        # Valid thresholds
        config = Configuration(
            completion_confidence_threshold=0.5,
            modification_threshold=0.8
        )
        assert 0.0 <= config.completion_confidence_threshold <= 1.0
        assert 0.0 <= config.modification_threshold <= 1.0
        
        # Invalid thresholds should be corrected
        config = Configuration(
            completion_confidence_threshold=-0.1,
            modification_threshold=1.5
        )
        assert 0.0 <= config.completion_confidence_threshold <= 1.0
        assert 0.0 <= config.modification_threshold <= 1.0
    
    def test_directory_path_validation(self):
        """Test validation of directory paths."""
        # Valid paths
        config = Configuration(
            project_agents_dir="custom/agents",
            user_agents_dir="/home/user/agents"
        )
        assert config.project_agents_dir == "custom/agents"
        assert config.user_agents_dir == "/home/user/agents"
        
        # Empty/None paths should use defaults
        config = Configuration(
            project_agents_dir="",
            user_agents_dir=None
        )
        assert config.project_agents_dir == ".open_deep_research/agents"
        assert config.user_agents_dir == "~/.open_deep_research/agents"
    
    def test_timeout_validation(self):
        """Test validation of timeout values."""
        # Valid timeout
        config = Configuration(evaluation_timeout=180)
        assert config.evaluation_timeout == 180
        
        # Too low timeout should use default
        config = Configuration(evaluation_timeout=10)
        assert config.evaluation_timeout == 120
        
        # Too high timeout should be capped
        config = Configuration(evaluation_timeout=800)
        assert config.evaluation_timeout == 600
    
    def test_agents_per_sequence_validation(self):
        """Test validation of max agents per sequence."""
        # Valid values
        config = Configuration(max_agents_per_sequence=3)
        assert config.max_agents_per_sequence == 3
        
        config = Configuration(max_agents_per_sequence=10)
        assert config.max_agents_per_sequence == 10
        
        # Too low should use default
        config = Configuration(max_agents_per_sequence=1)
        assert config.max_agents_per_sequence == 5
        
        # Too high should be capped
        config = Configuration(max_agents_per_sequence=25)
        assert config.max_agents_per_sequence == 20
    
    def test_enum_field_validation(self):
        """Test validation of enum fields."""
        # Valid enum values
        config = Configuration(
            agent_file_format=AgentFileFormat.YAML,
            report_update_frequency=ReportUpdateFrequency.ON_DEMAND
        )
        assert config.agent_file_format == AgentFileFormat.YAML
        assert config.report_update_frequency == ReportUpdateFrequency.ON_DEMAND
        
        # String values should be converted to enums
        config = Configuration(
            agent_file_format="markdown",
            report_update_frequency="after_sequence"
        )
        assert config.agent_file_format == AgentFileFormat.MARKDOWN
        assert config.report_update_frequency == ReportUpdateFrequency.AFTER_SEQUENCE


class TestSupervisorConfigIntegration:
    """Test integration between Configuration and SupervisorConfig."""
    
    def setup_method(self):
        """Set up integration testing."""
        pass
    
    def test_supervisor_config_creation(self):
        """Test creation of SupervisorConfig from Configuration."""
        config = Configuration(
            completion_confidence_threshold=0.75,
            max_agents_per_sequence=7,
            use_automatic_completion=True
        )
        
        # Create supervisor config (manually for testing)
        supervisor_config = SupervisorConfig(
            completion_threshold=config.completion_confidence_threshold,
            max_agents_per_sequence=config.max_agents_per_sequence,
            debug_mode=False
        )
        
        assert supervisor_config.completion_threshold == 0.75
        assert supervisor_config.max_agents_per_sequence == 7
    
    def test_configuration_field_mapping(self):
        """Test mapping between Configuration and SupervisorConfig fields."""
        mapping_tests = [
            ("completion_confidence_threshold", "completion_threshold", 0.8),
            ("max_agents_per_sequence", "max_agents_per_sequence", 6),
            ("allow_dynamic_modification", "allow_dynamic_insertion", True),
        ]
        
        for config_field, supervisor_field, test_value in mapping_tests:
            # Create configuration with test value
            config_kwargs = {config_field: test_value}
            config = Configuration(**config_kwargs)
            
            # Verify the value is set correctly
            assert getattr(config, config_field) == test_value
            
            # Note: In real implementation, this mapping would be automatic
            # Here we're just verifying the fields exist and have correct types
    
    def test_configuration_integration_with_components(self):
        """Test that configuration integrates properly with supervisor components."""
        config = Configuration(
            enable_sequential_supervisor=True,
            use_shared_state=True,
            automatic_handoffs=True,
            use_running_reports=True,
            enable_llm_judge=True
        )
        
        # Verify all sequential supervisor features are enabled
        assert config.enable_sequential_supervisor is True
        assert config.use_shared_state is True
        assert config.automatic_handoffs is True
        assert config.use_running_reports is True
        assert config.enable_llm_judge is True
        
        # This configuration should enable full sequential supervisor functionality
        components_enabled = [
            config.enable_sequential_supervisor,
            config.use_shared_state,
            config.automatic_handoffs,
            config.use_running_reports,
            config.enable_llm_judge
        ]
        
        assert all(components_enabled), "All sequential supervisor components should be enabled"


class TestConfigurationPerformance:
    """Test configuration performance and memory usage."""
    
    def setup_method(self):
        """Set up performance testing."""
        pass
    
    def test_configuration_creation_performance(self):
        """Test configuration creation performance."""
        import time
        
        # Time configuration creation
        start_time = time.time()
        
        for _ in range(100):
            Configuration(
                enable_sequential_supervisor=True,
                max_agents_per_sequence=5,
                completion_confidence_threshold=0.6,
                use_running_reports=True,
                enable_llm_judge=True
            )
        
        creation_time = time.time() - start_time
        
        # Should be very fast (< 0.1s for 100 configurations)
        assert creation_time < 0.1, f"Configuration creation too slow: {creation_time:.3f}s"
        
        print(f"Created 100 configurations in {creation_time:.3f}s")
    
    def test_from_runnable_config_performance(self):
        """Test performance of from_runnable_config method."""
        import time
        
        runnable_config = RunnableConfig(configurable={
            "enable_sequential_supervisor": True,
            "max_agents_per_sequence": 5,
            "completion_confidence_threshold": 0.6
        })
        
        start_time = time.time()
        
        for _ in range(50):
            Configuration.from_runnable_config(runnable_config)
        
        conversion_time = time.time() - start_time
        
        # Should be fast (< 0.1s for 50 conversions)
        assert conversion_time < 0.1, f"from_runnable_config too slow: {conversion_time:.3f}s"
        
        print(f"Converted 50 runnable configs in {conversion_time:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])