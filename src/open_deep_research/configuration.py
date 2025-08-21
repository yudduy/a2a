"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="anthropic:claude-3-haiku",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "anthropic:claude-3-haiku",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="anthropic:claude-3-5-sonnet",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "anthropic:claude-3-5-sonnet",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="anthropic:claude-3-5-sonnet",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "anthropic:claude-3-5-sonnet",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="anthropic:claude-3-5-sonnet",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "anthropic:claude-3-5-sonnet",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )
    
    # Sequential Optimization Configuration
    enable_sequence_optimization: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable sequential agent ordering optimization to test different agent sequences for improved productivity"
            }
        }
    )
    sequence_strategy: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": None,
                "description": "Preferred sequence strategy for agent ordering",
                "options": [
                    {"label": "Auto-select optimal", "value": None},
                    {"label": "Theory First (Academic → Industry → Technical)", "value": "theory_first"},
                    {"label": "Market First (Industry → Academic → Technical)", "value": "market_first"},
                    {"label": "Future Back (Technical → Academic → Industry)", "value": "future_back"}
                ]
            }
        }
    )
    compare_all_sequences: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Compare all sequence strategies and select the best performing one (slower but optimal)"
            }
        }
    )
    sequence_variance_threshold: float = Field(
        default=0.2,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.2,
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
                "description": "Minimum variance threshold for detecting significant productivity differences between sequences (20% = 0.2)"
            }
        }
    )
    
    # Parallel Execution Configuration
    enable_parallel_execution: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable parallel execution of multiple sequence strategies simultaneously with real-time streaming"
            }
        }
    )
    max_parallel_sequences: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 5,
                "step": 1,
                "description": "Maximum number of sequences to execute in parallel (affects resource usage)"
            }
        }
    )
    parallel_execution_timeout: int = Field(
        default=3600,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3600,
                "min": 300,
                "max": 7200,
                "description": "Timeout for parallel sequence execution in seconds (default: 1 hour)"
            }
        }
    )
    parallel_retry_attempts: int = Field(
        default=2,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 2,
                "min": 0,
                "max": 5,
                "step": 1,
                "description": "Number of retry attempts for failed sequences in parallel execution"
            }
        }
    )
    
    # Stream Configuration
    enable_real_time_streaming: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable real-time streaming of parallel execution progress"
            }
        }
    )
    stream_buffer_size: int = Field(
        default=1000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1000,
                "min": 100,
                "max": 10000,
                "description": "Buffer size for streaming messages per subscription"
            }
        }
    )
    max_stream_connections: int = Field(
        default=100,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 100,
                "min": 10,
                "max": 1000,
                "description": "Maximum number of concurrent WebSocket connections for streaming"
            }
        }
    )
    stream_message_rate_limit: Optional[int] = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10,
                "min": 1,
                "max": 100,
                "description": "Maximum messages per second per stream subscription (None for unlimited)"
            }
        }
    )
    
    # Resource Management
    enable_resource_monitoring: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable monitoring of memory and CPU usage during parallel execution"
            }
        }
    )
    memory_usage_threshold_mb: float = Field(
        default=2048.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 2048.0,
                "min": 512.0,
                "max": 16384.0,
                "description": "Memory usage threshold in MB for parallel execution warnings"
            }
        }
    )
    cpu_usage_threshold_percent: float = Field(
        default=80.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 80.0,
                "min": 50.0,
                "max": 95.0,
                "description": "CPU usage threshold in percentage for parallel execution warnings"
            }
        }
    )
    
    # Performance Optimization
    parallel_execution_priority: str = Field(
        default="balanced",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "balanced",
                "description": "Priority mode for parallel execution resource allocation",
                "options": [
                    {"label": "Speed (High Resource Usage)", "value": "speed"},
                    {"label": "Balanced (Moderate Resource Usage)", "value": "balanced"},
                    {"label": "Efficiency (Low Resource Usage)", "value": "efficiency"}
                ]
            }
        }
    )
    enable_execution_caching: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable caching of sequence execution results to avoid redundant API calls"
            }
        }
    )
    cache_expiry_minutes: int = Field(
        default=60,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 60,
                "min": 10,
                "max": 1440,
                "description": "Cache expiry time in minutes for sequence execution results"
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True