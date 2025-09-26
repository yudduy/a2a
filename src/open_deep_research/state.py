"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Union

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Import AgentSequence for strategic sequences
try:
    from open_deep_research.supervisor.sequence_models import AgentSequence
except ImportError:
    # Define a minimal version if supervisor module isn't available
    class AgentSequence(BaseModel):
        sequence_name: str
        agent_names: List[str]
        rationale: str
        confidence_score: float


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

###################
# Enhanced Message Structures for Frontend Integration
###################

class ThinkingSection(BaseModel):
    """Individual thinking section from reasoning model output."""
    
    id: str = Field(description="Unique identifier for this thinking section")
    content: str = Field(description="The thinking content itself")
    start_pos: int = Field(description="Character position in original text")
    end_pos: int = Field(description="End character position in original text")
    tag_type: str = Field(description="Type of thinking tag (thinking, think, etc.)")
    char_length: int = Field(description="Length of thinking content in characters")
    word_count: int = Field(description="Number of words in thinking content")
    is_collapsed: bool = Field(default=True, description="Whether section starts collapsed")

class ParsedMessageContent(BaseModel):
    """Enhanced message content with preserved thinking sections and metadata."""
    
    # Core content sections
    clean_content: str = Field(description="Message content without thinking tags")
    thinking_sections: List[ThinkingSection] = Field(default_factory=list, description="Extracted thinking sections")
    
    # Metadata for frontend processing
    has_thinking: bool = Field(default=False, description="Whether message contains thinking sections")
    original_content: str = Field(description="Original raw message content")
    section_count: int = Field(default=0, description="Number of thinking sections found")
    total_thinking_chars: int = Field(default=0, description="Total characters in all thinking sections")
    
    # Parallel processing metadata
    sequence_id: Optional[str] = Field(default=None, description="Sequence ID for parallel tab routing")
    sequence_name: Optional[str] = Field(default=None, description="Human-readable sequence name")
    tab_index: Optional[int] = Field(default=None, description="Tab index for frontend display")
    is_parallel_content: bool = Field(default=False, description="Whether this is parallel sequence content")

class EnhancedMessage(BaseModel):
    """Enhanced message structure for frontend integration."""
    
    # Core message fields (compatible with LangGraph SDK)
    id: str = Field(description="Message identifier")
    type: str = Field(description="Message type (human, ai, tool, etc.)")
    content: Union[str, ParsedMessageContent] = Field(description="Message content (string or parsed)")
    
    # Enhanced processing metadata
    parsed_content: Optional[ParsedMessageContent] = Field(default=None, description="Parsed content with thinking sections")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional processing metadata")
    
    # Frontend display metadata
    display_config: Dict[str, Any] = Field(default_factory=dict, description="Frontend display configuration")
    streaming_metadata: Dict[str, Any] = Field(default_factory=dict, description="Streaming and typing animation metadata")

class ParallelSequenceMetadata(BaseModel):
    """Metadata for parallel sequence execution and frontend display."""
    
    sequence_id: str = Field(description="Unique sequence identifier")
    sequence_name: str = Field(description="Human-readable sequence name")
    agent_names: List[str] = Field(description="List of agents in this sequence")
    rationale: str = Field(description="Rationale for this sequence approach")
    research_focus: str = Field(description="Specific research focus area")
    confidence_score: float = Field(description="LLM confidence in this sequence (0-1)")
    
    # Frontend-specific metadata
    tab_index: int = Field(description="Tab index for display ordering")
    status: str = Field(default="initializing", description="Current execution status")
    progress: float = Field(default=0.0, description="Execution progress (0-1)")
    message_count: int = Field(default=0, description="Number of messages in this sequence")
    
    # Timestamps for tracking
    created_at: str = Field(description="ISO timestamp of creation")
    started_at: Optional[str] = Field(default=None, description="ISO timestamp of execution start")
    completed_at: Optional[str] = Field(default=None, description="ISO timestamp of completion")

class SupervisorAnnouncement(BaseModel):
    """Structured supervisor announcement for sequence generation."""
    
    type: str = Field(default="sequences_generated", description="Announcement type")
    research_topic: str = Field(description="The research topic being investigated")
    sequences: List[ParallelSequenceMetadata] = Field(description="Generated sequences for parallel execution")
    
    # Generation metadata
    generation_model: str = Field(description="Model used for sequence generation")
    generation_timestamp: str = Field(description="ISO timestamp of generation")
    total_sequences: int = Field(description="Total number of sequences generated")
    recommended_sequence: Optional[int] = Field(default=None, description="Index of recommended sequence")
    
    # Display metadata
    announcement_title: str = Field(default="Research Sequences Generated", description="Title for UI display")
    announcement_description: str = Field(
        default="Based on research plan and subagent registry, here are sequences generated:",
        description="Description for UI display"
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    
    # NEW: Support for LLM-generated strategic sequences and parallel execution
    strategic_sequences: Optional[List[AgentSequence]] = None
    parallel_sequence_results: Optional[Dict[str, Any]] = None
    
    # NEW: LLM Judge evaluation results for orchestration analysis
    evaluation_result: Optional[Dict[str, Any]] = None
    orchestration_insights: Optional[Dict[str, Any]] = None
    
    # NEW: Enhanced message processing for frontend integration
    enhanced_messages: Annotated[List[EnhancedMessage], override_reducer] = []
    supervisor_announcement: Optional[SupervisorAnnouncement] = None
    parallel_sequences_metadata: Annotated[List[ParallelSequenceMetadata], override_reducer] = []

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


###################
# Sequential Multi-Agent Supervisor State
###################

class AgentExecutionReport(BaseModel):
    """Report from a single agent execution in sequential workflow."""
    
    agent_name: str = Field(description="Name of the agent that executed")
    agent_type: str = Field(description="Type/specialization of the agent") 
    execution_start: datetime = Field(description="When agent execution started")
    execution_end: datetime = Field(description="When agent execution completed")
    execution_duration: float = Field(description="Execution time in seconds")
    
    # Agent outputs
    insights: List[str] = Field(description="Key insights generated by the agent")
    research_content: str = Field(description="Main research content/findings")
    questions_addressed: List[str] = Field(description="Questions the agent addressed")
    
    # Quality metrics
    completion_confidence: float = Field(description="Confidence that agent completed task (0-1)")
    insight_quality_score: float = Field(description="Quality score for insights (0-1)")
    research_depth_score: float = Field(description="Depth of research achieved (0-1)")
    
    # Context for next agent
    handoff_context: Dict[str, Any] = Field(description="Context to pass to next agent")
    suggested_next_questions: List[str] = Field(description="Questions for next agent to explore")


class RunningReport(BaseModel):
    """Incrementally built report from sequential agent execution."""
    
    research_topic: str = Field(description="Main research topic")
    sequence_name: str = Field(description="Name of the agent sequence used")
    start_time: datetime = Field(description="When research sequence started")
    
    # Agent execution reports
    agent_reports: List[AgentExecutionReport] = Field(description="Reports from each agent")
    
    # Cumulative insights
    all_insights: List[str] = Field(description="All insights from all agents")
    insight_connections: List[Dict[str, str]] = Field(description="Connections between agent insights")
    
    # Report sections
    executive_summary: str = Field(description="High-level summary of findings")
    detailed_findings: List[str] = Field(description="Detailed findings organized by theme")
    recommendations: List[str] = Field(description="Actionable recommendations")
    
    # Metadata
    total_agents_executed: int = Field(description="Number of agents that executed")
    total_execution_time: float = Field(description="Total time for all agents")
    completion_status: str = Field(description="Status: running, completed, failed")


def merge_dict_values(current_value: Dict, new_value: Dict) -> Dict:
    """Reducer function that merges dictionary values intelligently."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    
    result = current_value.copy() if current_value else {}
    if isinstance(new_value, dict):
        for key, value in new_value.items():
            if key in result:
                # Merge lists
                if isinstance(result[key], list) and isinstance(value, list):
                    result[key] = result[key] + value
                # Update other values
                else:
                    result[key] = value
            else:
                result[key] = value
    return result


class SequentialSupervisorState(MessagesState):
    """Enhanced supervisor state for sequential multi-agent execution.
    
    This extends the existing supervisor functionality with agent-specific
    context management, sequence tracking, and running report building.
    """
    
    # Core research context (backward compatible)
    research_topic: str = ""
    research_brief: Optional[str] = None
    
    # LLM-generated strategic sequences
    strategic_sequences: Optional[List[AgentSequence]] = Field(
        default=None, description="LLM-generated strategic agent sequences"
    )
    
    # Sequence management
    planned_sequence: List[str] = Field(default_factory=list, description="Planned sequence of agent names")
    executed_agents: List[str] = Field(default_factory=list, description="Agents that have completed execution")
    current_agent: Optional[str] = Field(default=None, description="Currently executing agent")
    sequence_position: int = Field(default=0, description="Position in the planned sequence")
    
    # Agent-specific context (shared across agents)
    agent_insights: Annotated[Dict[str, List[str]], merge_dict_values] = Field(
        default_factory=dict, description="Insights keyed by agent name"
    )
    agent_questions: Annotated[Dict[str, List[str]], merge_dict_values] = Field(
        default_factory=dict, description="Questions addressed by each agent"
    )
    agent_context: Annotated[Dict[str, Dict], merge_dict_values] = Field(
        default_factory=dict, description="Context passed between agents"
    )
    agent_reports: Annotated[Dict[str, AgentExecutionReport], merge_dict_values] = Field(
        default_factory=dict, description="Execution reports from each agent"
    )
    
    # Running report (builds incrementally)
    running_report: Optional[RunningReport] = Field(default=None, description="Incrementally built research report")
    report_sections: Annotated[List[Dict[str, str]], operator.add] = Field(
        default_factory=list, description="Individual report sections as they're built"
    )
    
    # Completion tracking
    last_agent_completed: Optional[str] = Field(default=None, description="Last agent that completed")
    completion_signals: Annotated[Dict[str, bool], merge_dict_values] = Field(
        default_factory=dict, description="Completion status for each agent"
    )
    handoff_ready: bool = Field(default=False, description="Whether ready for next agent handoff")
    
    # Sequence metadata
    sequence_start_time: Optional[datetime] = Field(default=None, description="When sequence execution started")
    sequence_modifications: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list, description="Record of dynamic sequence modifications"
    )
    
    # Backward compatibility with existing workflow
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer] = Field(default_factory=list)
    notes: Annotated[list[str], override_reducer] = Field(default_factory=list)
    research_iterations: int = Field(default=0)
    raw_notes: Annotated[list[str], override_reducer] = Field(default_factory=list)


class SequentialAgentState(MessagesState):
    """State for individual agents in sequential execution.
    
    Provides context from previous agents and tracks individual agent execution.
    """
    
    # Agent identity and context
    agent_name: str
    agent_type: str
    sequence_position: int
    
    # Research context
    research_topic: str
    assigned_questions: List[str] = Field(default_factory=list)
    previous_agent_insights: List[str] = Field(default_factory=list)
    previous_agent_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution tracking
    execution_start_time: Optional[datetime] = Field(default=None)
    tool_calls_made: int = Field(default=0)
    completion_detected: bool = Field(default=False)
    completion_confidence: float = Field(default=0.0)
    
    # Agent outputs
    generated_insights: List[str] = Field(default_factory=list)
    research_findings: str = Field(default="")
    handoff_context: Dict[str, Any] = Field(default_factory=dict)
    next_agent_questions: List[str] = Field(default_factory=list)
    
    # Quality metrics
    insight_quality_scores: List[float] = Field(default_factory=list)
    research_depth_score: float = Field(default=0.0)
    
    # Message handling (backward compatible)
    agent_messages: Annotated[list[MessageLikeRepresentation], operator.add] = Field(default_factory=list)