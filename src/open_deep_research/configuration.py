"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, validator


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class AgentFileFormat(Enum):
    """Enumeration of supported agent file formats."""
    
    MARKDOWN = "markdown"
    YAML = "yaml"

class ReportUpdateFrequency(Enum):
    """Enumeration of report update frequencies."""
    
    AFTER_EACH_AGENT = "after_each_agent"
    AFTER_SEQUENCE = "after_sequence"
    ON_DEMAND = "on_demand"

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
        default="hyperbolic:Qwen/Qwen3-235B-A22B",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "hyperbolic:Qwen/Qwen3-235B-A22B",
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
        default="hyperbolic:Qwen/Qwen3-235B-A22B",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "hyperbolic:Qwen/Qwen3-235B-A22B",
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
        default="hyperbolic:Qwen/Qwen3-235B-A22B",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "hyperbolic:Qwen/Qwen3-235B-A22B",
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
        default="hyperbolic:Qwen/Qwen3-235B-A22B",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "hyperbolic:Qwen/Qwen3-235B-A22B",
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
    # Planner and Executor Model Configuration for Sequencing
    planner_model: str = Field(
        default="hyperbolic:Qwen/Qwen3-235B-A22B",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "hyperbolic:Qwen/Qwen3-235B-A22B",
                "description": "Model for planning research strategy and generating dynamic sequences"
            }
        }
    )
    planner_model_max_tokens: int = Field(
        default=15000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 15000,
                "description": "Maximum output tokens for planner model"
            }
        }
    )
    executor_model: str = Field(
        default="hyperbolic:Qwen/QwQ-32B",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "hyperbolic:Qwen/QwQ-32B",
                "description": "Model for executing specialized research tasks (sub-agents)"
            }
        }
    )
    executor_model_max_tokens: int = Field(
        default=12000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 12000,
                "description": "Maximum output tokens for executor model"
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
    
    # Dynamic Sequencing Configuration  
    enable_dynamic_sequencing: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable dynamic sequence generation that creates optimal agent orderings based on research topic analysis"
            }
        }
    )
    enable_sequence_optimization: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable sequence optimization and parallel research execution. When disabled, uses standard single-path research."
            }
        }
    )
    max_dynamic_sequences: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 5,
                "step": 1,
                "description": "Maximum number of dynamic sequences to generate and compare for each research topic"
            }
        }
    )
    
    # Parallel Execution Configuration
    enable_parallel_execution: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Execute all 3 LLM-generated sequences in parallel for comprehensive research coverage"
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
    
    # Sequential Supervisor Configuration
    enable_sequential_supervisor: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable the sequential multi-agent supervisor system for orchestrated agent execution"
            }
        }
    )
    use_shared_state: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable shared state management across sequential agents for context passing"
            }
        }
    )
    automatic_handoffs: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable automatic agent handoffs based on completion detection"
            }
        }
    )
    allow_dynamic_modification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Allow dynamic modification of agent sequences during execution"
            }
        }
    )
    max_agents_per_sequence: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 2,
                "max": 10,
                "step": 1,
                "description": "Maximum number of agents allowed in a single sequence"
            }
        }
    )
    modification_threshold: float = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.7,
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
                "description": "Confidence threshold for triggering dynamic sequence modifications (0-1)"
            }
        }
    )
    
    # Agent Registry Configuration
    project_agents_dir: str = Field(
        default=".open_deep_research/agents",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": ".open_deep_research/agents",
                "description": "Directory path for project-specific agent definitions (relative to project root)"
            }
        }
    )
    user_agents_dir: str = Field(
        default="~/.open_deep_research/agents",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "~/.open_deep_research/agents",
                "description": "Directory path for user-global agent definitions (supports ~ expansion)"
            }
        }
    )
    agent_file_format: AgentFileFormat = Field(
        default=AgentFileFormat.MARKDOWN,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "markdown",
                "description": "File format for agent definitions",
                "options": [
                    {"label": "Markdown (.md)", "value": AgentFileFormat.MARKDOWN.value},
                    {"label": "YAML (.yml/.yaml)", "value": AgentFileFormat.YAML.value}
                ]
            }
        }
    )
    inherit_all_tools: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether agents should inherit all available tools by default"
            }
        }
    )
    
    # Completion Detection Configuration
    use_automatic_completion: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable automatic completion detection for agent handoffs"
            }
        }
    )
    completion_confidence_threshold: float = Field(
        default=0.6,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.6,
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
                "description": "Confidence threshold for automatic completion detection (0-1)"
            }
        }
    )
    completion_indicators: List[str] = Field(
        default=["research complete", "analysis complete", "findings summarized", "investigation finished", "task accomplished"],
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "research complete, analysis complete, findings summarized, investigation finished, task accomplished",
                "description": "Comma-separated list of phrases indicating agent completion"
            }
        }
    )
    
    # Running Reports Configuration
    use_running_reports: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable incremental report building during sequential execution"
            }
        }
    )
    report_update_frequency: ReportUpdateFrequency = Field(
        default=ReportUpdateFrequency.AFTER_EACH_AGENT,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "after_each_agent",
                "description": "Frequency for updating the running report",
                "options": [
                    {"label": "After Each Agent", "value": ReportUpdateFrequency.AFTER_EACH_AGENT.value},
                    {"label": "After Complete Sequence", "value": ReportUpdateFrequency.AFTER_SEQUENCE.value},
                    {"label": "On Demand Only", "value": ReportUpdateFrequency.ON_DEMAND.value}
                ]
            }
        }
    )
    include_agent_metadata: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Include agent execution metadata (timing, quality scores) in running reports"
            }
        }
    )
    
    # LLM Judge Configuration
    enable_llm_judge: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable LLM-based evaluation of research reports and agent performance"
            }
        }
    )
    evaluation_model: str = Field(
        default="anthropic:claude-3-5-sonnet",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "anthropic:claude-3-5-sonnet",
                "description": "Model to use for LLM-based evaluation and report scoring"
            }
        }
    )
    evaluation_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for evaluation model"
            }
        }
    )
    evaluation_criteria: List[str] = Field(
        default=["completeness", "depth", "coherence", "innovation", "actionability"],
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "completeness, depth, coherence, innovation, actionability",
                "description": "Comma-separated list of evaluation criteria for report assessment"
            }
        }
    )
    evaluation_timeout: int = Field(
        default=120,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 120,
                "min": 30,
                "max": 600,
                "description": "Timeout for LLM evaluation calls in seconds"
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

    @validator('completion_indicators', pre=True)
    def parse_completion_indicators(cls, v):
        """Parse completion indicators from comma-separated string or list."""
        if isinstance(v, str):
            return [indicator.strip() for indicator in v.split(',') if indicator.strip()]
        return v or ["research complete", "analysis complete", "findings summarized"]
    
    @validator('evaluation_criteria', pre=True)
    def parse_evaluation_criteria(cls, v):
        """Parse evaluation criteria from comma-separated string or list."""
        if isinstance(v, str):
            return [criterion.strip() for criterion in v.split(',') if criterion.strip()]
        return v or ["completeness", "depth", "coherence", "innovation", "actionability"]
    
    @validator('project_agents_dir')
    def validate_project_agents_dir(cls, v):
        """Validate project agents directory path."""
        if not v or not isinstance(v, str):
            return ".open_deep_research/agents"
        return v.strip()
    
    @validator('user_agents_dir')
    def validate_user_agents_dir(cls, v):
        """Validate user agents directory path."""
        if not v or not isinstance(v, str):
            return "~/.open_deep_research/agents"
        return v.strip()
    
    @validator('completion_confidence_threshold', 'modification_threshold')
    def validate_threshold_range(cls, v):
        """Ensure threshold values are between 0 and 1."""
        if not isinstance(v, int | float):
            return 0.6
        return max(0.0, min(1.0, float(v)))
    
    @validator('max_agents_per_sequence')
    def validate_max_agents_per_sequence(cls, v):
        """Ensure max agents per sequence is reasonable."""
        if not isinstance(v, int) or v < 2:
            return 5
        return min(v, 20)  # Cap at reasonable maximum
    
    @validator('evaluation_timeout')
    def validate_evaluation_timeout(cls, v):
        """Ensure evaluation timeout is reasonable."""
        if not isinstance(v, int) or v < 30:
            return 120
        return min(v, 600)  # Cap at 10 minutes

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
        use_enum_values = True