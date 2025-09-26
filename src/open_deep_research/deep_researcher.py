"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Literal, Optional

# Configure logger for this module
logger = logging.getLogger(__name__)

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.core.sequence_generator import (
    AgentSequence,
    SequenceGenerationInput,
    UnifiedSequenceGenerator,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    clean_reasoning_model_output,
    get_all_tools,
    get_api_key_for_model,
    get_model_config_for_provider,
    get_model_token_limit,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    parse_reasoning_model_output,
    remove_up_to_last_ai_message,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "base_url"),
)


def create_cleaned_structured_output(model, output_schema):
    """Create a structured output model that cleans reasoning model outputs.
    
    This wrapper handles QwQ and other reasoning models that output thinking tags
    by cleaning the output before structured parsing while preserving thinking content
    for frontend display when needed.
    """
    import json

    from langchain_core.output_parsers import PydanticOutputParser
    
    # Create a wrapper that has the same interface as with_structured_output
    class StructuredWrapper:
        def __init__(self, model, output_schema):
            self.model = model
            self.output_schema = output_schema
            
        async def cleaned_structured_invoke(self, messages):
            # Get raw response from the configured model
            response = await self.model.ainvoke(messages)
            
            # Parse reasoning model output to extract thinking sections and clean content
            parsed_output = parse_reasoning_model_output(response.content)
            cleaned_content = parsed_output['clean_content']
            
            # Store thinking sections in response metadata for potential frontend use
            if hasattr(response, 'response_metadata'):
                response.response_metadata['thinking_sections'] = parsed_output['thinking_sections']
                response.response_metadata['has_thinking'] = parsed_output['has_thinking']
            
            # Parse as JSON and create the structured output
            try:
                parsed_json = json.loads(cleaned_content)
                return self.output_schema.model_validate(parsed_json)
            except (json.JSONDecodeError, Exception) as e:
                # Add logging to understand what content is being produced
                import logging
                logging.warning(f"JSON parsing failed for {self.output_schema.__name__}: {str(e)[:100]}...")
                logging.debug(f"Cleaned content: {cleaned_content[:200]}...")
                
                # Special handling for ResearchQuestion objects when JSON parsing fails
                if self.output_schema.__name__ == "ResearchQuestion":
                    # Create a default ResearchQuestion with the cleaned content as research_brief
                    return self.output_schema(research_brief=cleaned_content)
                
                # For other schemas, try the original structured output as fallback
                try:
                    parser = PydanticOutputParser(pydantic_object=self.output_schema)
                    return parser.parse(cleaned_content)
                except Exception as fallback_error:
                    logging.error(f"PydanticOutputParser fallback also failed: {fallback_error}")
                    # As a last resort, create a basic instance for ResearchQuestion
                    if self.output_schema.__name__ == "ResearchQuestion":
                        return self.output_schema(research_brief="Unable to parse research brief from model output.")
                    raise fallback_error
            
        async def ainvoke(self, messages):
            return await self.cleaned_structured_invoke(messages)
            
        def with_retry(self, **kwargs):
            # Return self to maintain chain interface
            return self
            
        def with_config(self, config):
            # Update the underlying model config
            self.model = self.model.with_config(config)
            return self
    
    return StructuredWrapper(model, output_schema)

def create_enhanced_message_with_thinking(response, message_type: str = "ai", sequence_metadata: dict = None):
    """Create enhanced message structure with preserved thinking sections.
    
    Args:
        response: Model response with potential thinking content
        message_type: Type of message (ai, human, tool, etc.)
        sequence_metadata: Optional metadata for parallel sequence routing
        
    Returns:
        Enhanced message with parsed thinking sections and metadata
    """
    from open_deep_research.state import (
        EnhancedMessage,
        ParsedMessageContent,
        ThinkingSection,
    )
    
    # Parse the response content for thinking sections
    if hasattr(response, 'content') and response.content:
        parsed_output = parse_reasoning_model_output(response.content)
        
        # Convert thinking sections to Pydantic models
        thinking_sections = [
            ThinkingSection(**section) for section in parsed_output['thinking_sections']
        ]
        
        # Create parsed content structure
        parsed_content = ParsedMessageContent(
            clean_content=parsed_output['clean_content'],
            thinking_sections=thinking_sections,
            has_thinking=parsed_output['has_thinking'],
            original_content=parsed_output['original_content'],
            section_count=parsed_output['section_count'],
            total_thinking_chars=parsed_output['total_thinking_chars'],
            sequence_id=sequence_metadata.get('sequence_id') if sequence_metadata else None,
            sequence_name=sequence_metadata.get('sequence_name') if sequence_metadata else None,
            tab_index=sequence_metadata.get('tab_index') if sequence_metadata else None,
            is_parallel_content=bool(sequence_metadata)
        )
        
        # Create enhanced message
        enhanced_message = EnhancedMessage(
            id=getattr(response, 'id', f"msg_{int(time.time()*1000)}"),
            type=message_type,
            content=parsed_content.clean_content,  # Use clean content for compatibility
            parsed_content=parsed_content,
            processing_metadata={
                'model_used': getattr(response, 'response_metadata', {}).get('model_name', 'unknown'),
                'processing_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime()),
                'has_thinking': parsed_content.has_thinking,
                'thinking_sections_count': len(thinking_sections)
            },
            display_config={
                'show_thinking_collapsed': True,
                'enable_typing_animation': True,
                'typing_speed': 20
            }
        )
        
        return enhanced_message
    
    # Fallback for responses without content
    return EnhancedMessage(
        id=f"msg_{int(time.time()*1000)}",
        type=message_type,
        content=str(response) if response else "",
        processing_metadata={'processing_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())}
    )

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = get_model_config_for_provider(
        model_name=configurable.research_model,
        api_key=get_api_key_for_model(configurable.research_model, config),
        max_tokens=configurable.research_model_max_tokens,
        tags=["clarification", "user_interaction"]
    )
    
    # Configure model with cleaned structured output for reasoning models
    clarification_model = create_cleaned_structured_output(
        configurable_model, ClarifyWithUser
    ).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(model_config)
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Transform user messages into a structured research brief for the supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. The supervisor will handle sequence planning
    and execution separately.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Dictionary with research brief and supervisor initialization
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = get_model_config_for_provider(
        model_name=configurable.research_model,
        api_key=get_api_key_for_model(configurable.research_model, config),
        max_tokens=configurable.research_model_max_tokens,
        tags=["research_brief", "topic_analysis"]
    )
    
    # Configure model for structured research question generation with cleaning
    research_model = create_cleaned_structured_output(
        configurable_model, ResearchQuestion
    ).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    # Return clean dictionary focused on research brief
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": {
            "type": "override",
            "value": [
                SystemMessage(content=supervisor_system_prompt),
                HumanMessage(content=response.research_brief)
            ]
        }
    }





async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = get_model_config_for_provider(
        model_name=configurable.research_model,
        api_key=get_api_key_for_model(configurable.research_model, config),
        max_tokens=configurable.research_model_max_tokens,
        tags=["researcher", "tool_execution", "data_gathering"]
    )
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Enhanced message processing with thinking section preservation
    enhanced_message = None
    if hasattr(response, 'content') and response.content:
        # Parse for thinking sections while cleaning for processing
        parsed_output = parse_reasoning_model_output(response.content)
        
        # Update response content to cleaned version for processing
        response.content = parsed_output['clean_content']
        
        # Create enhanced message if thinking sections are present
        if parsed_output['has_thinking']:
            enhanced_message = create_enhanced_message_with_thinking(response, "ai")
    
    # Step 4: Update state and proceed to tool execution
    update_dict = {
        "researcher_messages": [response],
        "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
    }
    
    # Add enhanced message if thinking content was found
    if enhanced_message:
        update_dict["enhanced_messages"] = [enhanced_message]
    
    return Command(
        goto="researcher_tools",
        update=update_dict
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    compression_model_config = get_model_config_for_provider(
        model_name=configurable.compression_model,
        api_key=get_api_key_for_model(configurable.compression_model, config),
        max_tokens=configurable.compression_model_max_tokens,
        tags=["compression", "synthesis", "report_generation"]
    )
    synthesizer_model = configurable_model.with_config(compression_model_config)
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Clean reasoning model output to remove thinking tags
            if hasattr(response, 'content') and response.content:
                response.content = clean_reasoning_model_output(response.content)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with LLM Judge evaluation.
    
    This enhanced function now includes LLM Judge evaluation of different sequence reports
    to determine the best orchestration approach and provide scoring/ranking.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report, evaluation results, and cleared state
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: Extract sequence reports for evaluation
    sequence_reports = await extract_sequence_reports_for_evaluation(state)
    evaluation_result = None
    
    # Step 3: Run LLM Judge evaluation if we have multiple sequence reports
    if len(sequence_reports) > 1:
        logger.info(f"Running LLM Judge evaluation on {len(sequence_reports)} sequence reports")
        try:
            from open_deep_research.evaluation.llm_judge import LLMJudge
            
            judge = LLMJudge(config=config)
            evaluation_result = await judge.evaluate_reports(
                reports=sequence_reports,
                research_topic=state.get("research_brief", "Unknown research topic"),
                sequence_names=list(sequence_reports.keys())
            )
            
            logger.info(f"LLM Judge evaluation completed. Winner: {evaluation_result.winning_sequence}")
            
        except Exception as e:
            logger.error(f"LLM Judge evaluation failed: {e}")
            evaluation_result = None
    else:
        logger.info("Skipping LLM Judge evaluation: insufficient sequence reports for comparison")
    
    # Step 4: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = get_model_config_for_provider(
        model_name=configurable.final_report_model,
        api_key=get_api_key_for_model(configurable.final_report_model, config),
        max_tokens=configurable.final_report_model_max_tokens,
        tags=["final_report", "document_generation", "writing"]
    )
    
    # Step 5: Generate enhanced report with evaluation insights if available
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with evaluation insights
            enhanced_prompt = create_enhanced_final_report_prompt(
                state=state,
                findings=findings,
                evaluation_result=evaluation_result
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=enhanced_prompt)
            ])
            
            # Clean reasoning model output to remove thinking tags
            if hasattr(final_report, 'content') and final_report.content:
                final_report.content = clean_reasoning_model_output(final_report.content)
            
            # Step 6: Prepare enhanced result with evaluation data
            result = {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
            # Add evaluation results to the state for frontend access
            if evaluation_result:
                result.update({
                    "evaluation_result": {
                        "winning_sequence": evaluation_result.winning_sequence,
                        "winning_score": evaluation_result.winning_sequence_score,
                        "sequence_rankings": [
                            {
                                "sequence_name": eval.sequence_name,
                                "overall_score": eval.overall_score,
                                "key_strengths": eval.key_strengths[:3],
                                "executive_summary": eval.executive_summary
                            }
                            for eval in sorted(evaluation_result.individual_evaluations, 
                                             key=lambda x: x.overall_score, reverse=True)
                        ],
                        "key_differentiators": evaluation_result.key_differentiators,
                        "performance_gaps": evaluation_result.performance_gaps,
                        "evaluation_model": evaluation_result.evaluation_model,
                        "processing_time": evaluation_result.processing_time
                    },
                    "orchestration_insights": create_orchestration_insights(evaluation_result)
                })
            
            return result
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 7: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }


async def extract_sequence_reports_for_evaluation(state: AgentState) -> Dict[str, str]:
    """Extract sequence reports from state for LLM Judge evaluation.
    
    This function examines both parallel and sequential execution results to extract
    individual sequence reports that can be compared by the LLM Judge.
    
    Args:
        state: Current agent state containing execution results
        
    Returns:
        Dictionary mapping sequence names to their report content
    """
    import logging
    logger = logging.getLogger(__name__)
    
    sequence_reports = {}
    
    # Extract from parallel sequence results if available
    parallel_results = state.get("parallel_sequence_results", {})
    if parallel_results and "sequence_results" in parallel_results:
        logger.info("Extracting reports from parallel sequence results")
        
        for seq_id, result in parallel_results["sequence_results"].items():
            # Create comprehensive report from parallel execution
            report_content = f"# Sequence Report: {seq_id}\n\n"
            
            # Add comprehensive findings
            if result.get("comprehensive_findings"):
                report_content += "## Key Findings\n"
                for finding in result["comprehensive_findings"]:
                    report_content += f"- {finding}\n"
                report_content += "\n"
            
            # Add agent-specific insights
            if result.get("agent_results"):
                report_content += "## Agent Insights\n"
                for agent_result in result["agent_results"]:
                    agent_type = agent_result.get("agent_type", "unknown")
                    insights = agent_result.get("key_insights", [])
                    if insights:
                        report_content += f"### {agent_type}\n"
                        for insight in insights:
                            report_content += f"- {insight}\n"
                        report_content += "\n"
            
            # Add performance metrics
            report_content += "## Performance Metrics\n"
            report_content += f"- Duration: {result.get('total_duration', 0.0):.1f} seconds\n"
            report_content += f"- Productivity Score: {result.get('productivity_score', 0.0):.2f}\n"
            
            sequence_reports[seq_id] = report_content
    
    # Extract from sequential execution results if no parallel results
    elif state.get("running_report") or state.get("notes"):
        logger.info("Extracting reports from sequential execution results")
        
        # Try to extract from running report first
        running_report = state.get("running_report")
        if running_report and hasattr(running_report, 'executive_summary'):
            sequence_name = getattr(running_report, 'sequence_name', 'sequential_execution')
            report_content = f"# Sequential Research Report: {sequence_name}\n\n"
            
            if running_report.executive_summary:
                report_content += f"## Executive Summary\n{running_report.executive_summary}\n\n"
            
            if hasattr(running_report, 'detailed_findings') and running_report.detailed_findings:
                report_content += "## Detailed Findings\n"
                for finding in running_report.detailed_findings:
                    report_content += f"- {finding}\n"
                report_content += "\n"
            
            if hasattr(running_report, 'recommendations') and running_report.recommendations:
                report_content += "## Recommendations\n"
                for i, rec in enumerate(running_report.recommendations, 1):
                    report_content += f"{i}. {rec}\n"
            
            sequence_reports[sequence_name] = report_content
        
        # Fallback to using notes and strategic sequences
        else:
            strategic_sequences = state.get("strategic_sequences", [])
            notes = state.get("notes", [])
            
            if strategic_sequences:
                # Create reports for each strategic sequence based on available notes
                for i, sequence in enumerate(strategic_sequences[:min(3, len(strategic_sequences))]):
                    sequence_name = sequence.sequence_name
                    report_content = f"# Strategic Sequence Report: {sequence_name}\n\n"
                    report_content += f"## Research Focus\n{sequence.research_focus}\n\n"
                    report_content += f"## Approach\n{sequence.approach_description}\n\n"
                    
                    # Distribute notes across sequences (simplified approach)
                    sequence_notes = notes[i::len(strategic_sequences)] if notes else []
                    if sequence_notes:
                        report_content += "## Research Findings\n"
                        for note in sequence_notes:
                            report_content += f"- {note}\n"
                    
                    sequence_reports[sequence_name] = report_content
            else:
                # Final fallback - create a single report from all available content
                report_content = "# Research Report\n\n"
                if notes:
                    report_content += "## Research Findings\n"
                    for note in notes:
                        report_content += f"- {note}\n"
                
                sequence_reports["primary_research"] = report_content
    
    logger.info(f"Extracted {len(sequence_reports)} sequence reports for evaluation")
    return sequence_reports


def create_enhanced_final_report_prompt(
    state: AgentState,
    findings: str,
    evaluation_result: Optional[Any] = None
) -> str:
    """Create an enhanced final report prompt including evaluation insights.
    
    Args:
        state: Current agent state
        findings: Research findings string
        evaluation_result: LLM Judge evaluation results if available
        
    Returns:
        Enhanced prompt string for final report generation
    """
    # Start with the base prompt
    base_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        messages=get_buffer_string(state.get("messages", [])),
        findings=findings,
        date=get_today_str()
    )
    
    # Add evaluation insights if available
    if evaluation_result:
        evaluation_section = "\n\n## ORCHESTRATION EVALUATION INSIGHTS\n\n"
        evaluation_section += f"Based on LLM Judge evaluation of {len(evaluation_result.individual_evaluations)} research sequences:\n\n"
        
        # Add winning sequence information
        evaluation_section += f"**Best Performing Approach:** {evaluation_result.winning_sequence}\n"
        evaluation_section += f"**Score:** {evaluation_result.winning_sequence_score:.1f}/100\n\n"
        
        # Add key differentiators
        if evaluation_result.key_differentiators:
            evaluation_section += "**Key Success Factors:**\n"
            for differentiator in evaluation_result.key_differentiators[:3]:
                evaluation_section += f"- {differentiator}\n"
            evaluation_section += "\n"
        
        # Add sequence comparison insights
        if hasattr(evaluation_result, 'comparative_analysis'):
            comp_analysis = evaluation_result.comparative_analysis
            if hasattr(comp_analysis, 'best_sequence_reasoning'):
                evaluation_section += f"**Why This Approach Worked Best:** {comp_analysis.best_sequence_reasoning}\n\n"
        
        # Add performance gaps information
        if evaluation_result.performance_gaps:
            evaluation_section += "**Performance Analysis:**\n"
            for seq_name, gap in evaluation_result.performance_gaps.items():
                evaluation_section += f"- {seq_name}: {gap:.1f} points behind best approach\n"
        
        evaluation_section += "\nPlease incorporate these orchestration insights into your final report, highlighting what made the research approach effective and providing guidance for future research on similar topics.\n"
        
        # Append to the base prompt
        enhanced_prompt = base_prompt + evaluation_section
        return enhanced_prompt
    
    return base_prompt


def create_orchestration_insights(evaluation_result: Any) -> Dict[str, Any]:
    """Create structured orchestration insights for frontend display.
    
    Args:
        evaluation_result: LLM Judge evaluation results
        
    Returns:
        Dictionary containing orchestration insights and recommendations
    """
    insights = {
        "summary": f"Evaluated {len(evaluation_result.individual_evaluations)} research approaches",
        "best_approach": {
            "name": evaluation_result.winning_sequence,
            "score": evaluation_result.winning_sequence_score,
            "advantages": []
        },
        "key_learnings": [],
        "recommendations": {},
        "methodology_effectiveness": {}
    }
    
    # Extract advantages of the winning sequence
    winning_eval = next(
        (eval for eval in evaluation_result.individual_evaluations 
         if eval.sequence_name == evaluation_result.winning_sequence),
        None
    )
    
    if winning_eval:
        insights["best_approach"]["advantages"] = winning_eval.key_strengths[:3]
        insights["key_learnings"] = [
            f"Superior {criterion}: {getattr(winning_eval, criterion).score:.1f}/10"
            for criterion in ["completeness", "depth", "coherence", "innovation", "actionability"]
            if getattr(winning_eval, criterion).score >= 8.0
        ]
    
    # Add recommendations for each sequence
    for eval in evaluation_result.individual_evaluations:
        if eval.key_strengths:
            insights["recommendations"][eval.sequence_name] = f"Best for: {eval.key_strengths[0]}"
    
    # Add methodology effectiveness insights
    if hasattr(evaluation_result, 'comparative_analysis'):
        comp_analysis = evaluation_result.comparative_analysis
        if hasattr(comp_analysis, 'criteria_leaders'):
            insights["methodology_effectiveness"] = comp_analysis.criteria_leaders
    
    return insights




async def emit_sequences_to_frontend(
    strategic_sequences: List[AgentSequence], 
    research_topic: str, 
    configurable: Configuration
) -> None:
    """Emit strategic sequences to frontend for real-time display."""
    try:
        # Create structured parallel sequence metadata
        parallel_metadata = []
        for i, seq in enumerate(strategic_sequences):
            metadata = {
                "sequence_id": f"seq_{i}_{int(time.time()*1000)}",
                "sequence_name": seq.sequence_name,
                "agent_names": seq.agent_names,
                "rationale": seq.rationale,
                "research_focus": getattr(seq, 'approach_description', seq.research_focus),
                "confidence_score": seq.confidence_score,
                "tab_index": i,
                "status": "initializing",
                "progress": 0.0,
                "message_count": 0,
                "created_at": time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime()),
                "approach_description": getattr(seq, 'approach_description', seq.research_focus),
                "expected_outcomes": getattr(seq, 'expected_outcomes', [])
            }
            parallel_metadata.append(metadata)
        
        # Create structured supervisor announcement
        from open_deep_research.state import (
            ParallelSequenceMetadata,
            SupervisorAnnouncement,
        )
        supervisor_announcement = SupervisorAnnouncement(
            research_topic=research_topic,
            sequences=[ParallelSequenceMetadata(**metadata) for metadata in parallel_metadata],
            generation_model=configurable.research_model,
            generation_timestamp=time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime()),
            total_sequences=len(strategic_sequences),
            recommended_sequence=0  # Default to first sequence
        )
        
        # Prepare frontend-compatible format
        frontend_sequences = {
            "type": "sequences_generated",
            "sequences": parallel_metadata,
            "announcement": {
                "title": supervisor_announcement.announcement_title,
                "description": supervisor_announcement.announcement_description,
                "research_topic": supervisor_announcement.research_topic,
                "total_sequences": supervisor_announcement.total_sequences
            }
        }
        
        # Emit to frontend via stream writer
        writer = get_stream_writer()
        if writer:
            writer(frontend_sequences)
            logger.info(f"Successfully emitted {len(strategic_sequences)} sequences to frontend")
        else:
            logger.warning("Stream writer not available - frontend may not receive sequence data")
            
    except Exception as e:
        logger.error(f"Failed to emit sequences to frontend: {e}")


async def sequence_research_supervisor(state: AgentState, config: RunnableConfig) -> Command[Literal["final_report_generation"]]:
    """Always-parallel research supervisor with agent-aware sequence generation.
    
    This function:
    1. Initializes agent registry to discover available specialized agents
    2. Generates 3 strategic sequences based on discovered agents and research topic
    3. Executes all 3 sequences in parallel, each as sequential agent workflows
    4. Aggregates results from all parallel executions
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        configurable = Configuration.from_runnable_config(config)
        research_topic = state.get("research_brief", "")
        
        # Step 1: Initialize agent registry to discover available specialized agents
        logger.info("Initializing agent registry to discover specialized agents")
        agent_registry = await initialize_agent_registry(config)
        
        if not agent_registry or len(agent_registry.list_agents()) == 0:
            logger.error("No agents found in registry - cannot generate strategic sequences")
            return Command(
                goto="final_report_generation",
                update={
                    "notes": ["Research failed: No agents available in registry. Please ensure agent definitions exist in .open_deep_research/agents/ directory."],
                    "research_brief": research_topic,
                    "final_report": "Research could not be completed: No specialized agents found in agent registry."
                }
            )
        
        logger.info(f"Found {len(agent_registry.list_agents())} specialized agents in registry")
        
        # Step 2: Generate 3 strategic sequences based on discovered agents
        logger.info("Generating strategic research sequences based on available agents")
        strategic_sequences = await generate_strategic_sequences(
            research_topic=research_topic,
            config=config
        )
        
        if not strategic_sequences or len(strategic_sequences) == 0:
            logger.error("Failed to generate strategic sequences - cannot proceed with research")
            return Command(
                goto="final_report_generation",
                update={
                    "notes": ["Research failed: Unable to generate strategic sequences from available agents."],
                    "research_brief": research_topic,
                    "final_report": "Research could not be completed: Failed to generate strategic research sequences."
                }
            )
        
        logger.info(f"Generated {len(strategic_sequences)} strategic sequences for parallel execution")
        for i, seq in enumerate(strategic_sequences):
            logger.info(f"Sequence {i+1}: '{seq.sequence_name}' using agents {seq.agent_names}")
        
        # Step 3: Emit sequences to frontend for real-time display
        await emit_sequences_to_frontend(strategic_sequences, research_topic, configurable)
        
        # Step 4: Always execute all sequences in parallel
        logger.info("Executing all strategic sequences in parallel")
        try:
            parallel_results = await execute_parallel_sequences(
                strategic_sequences=strategic_sequences,
                config=config,
                research_topic=research_topic
            )
            
            if parallel_results.get("success_rate", 0) > 0:
                # Convert parallel results to AgentState format
                result_state = convert_parallel_results_to_agent_state(parallel_results, state)
                result_state["strategic_sequences"] = strategic_sequences  # Preserve sequences for frontend
                
                logger.info(f"Parallel execution completed with {parallel_results.get('success_rate', 0):.1f}% success rate")
                
                return Command(
                    goto="final_report_generation",
                    update=result_state
                )
            else:
                logger.error("All parallel sequences failed - no research results available")
                return Command(
                    goto="final_report_generation",
                    update={
                        "notes": [f"All parallel sequences failed: {parallel_results.get('error_message', 'Unknown error')}"],
                        "research_brief": research_topic,
                        "final_report": "Research could not be completed: All strategic sequences failed to execute.",
                        "strategic_sequences": strategic_sequences
                    }
                )
                
        except Exception as parallel_error:
            logger.error(f"Parallel execution failed: {parallel_error}")
            return Command(
                goto="final_report_generation",
                update={
                    "notes": [f"Parallel execution failed: {str(parallel_error)}"],
                    "research_brief": research_topic,
                    "final_report": f"Research could not be completed due to parallel execution error: {str(parallel_error)}",
                    "strategic_sequences": strategic_sequences
                }
            )
        
    except Exception as e:
        logger.error(f"Research supervisor failed: {e}", exc_info=True)
        return Command(
            goto="final_report_generation",
            update={
                "notes": [f"Research failed due to supervisor error: {str(e)}"],
                "research_brief": state.get("research_brief", ""),
                "final_report": f"Research could not be completed due to error: {str(e)}"
            }
        )


# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("sequence_research_supervisor", sequence_research_supervisor)  # Sequential multi-agent execution
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
# clarify_with_user routes automatically via Command to either write_research_brief or END
deep_researcher_builder.add_edge("write_research_brief", "sequence_research_supervisor")  # Flows to supervisor after emitting UpdateEvent
deep_researcher_builder.add_edge("sequence_research_supervisor", "final_report_generation") # Sequential research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

async def execute_parallel_sequences(
    strategic_sequences: List[AgentSequence],
    config: RunnableConfig,
    research_topic: str
) -> Dict[str, Any]:
    """Execute all strategic sequences in parallel using SimpleSequentialExecutor.
    
    This function now uses the lightweight SimpleSequentialExecutor which executes
    agents sequentially within each parallel path, replacing the heavy ParallelExecutor.
    
    Args:
        strategic_sequences: List of LLM-generated strategic sequences
        config: Runtime configuration with model settings
        research_topic: The research topic to investigate
        
    Returns:
        Dictionary containing results from all parallel sequence executions
    """
    import logging

    from open_deep_research.sequencing.simple_sequential_executor import (
        execute_sequences_in_parallel,
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Get configuration for parallel execution
        configurable = Configuration.from_runnable_config(config)
        
        # Limit sequences to max allowed
        max_sequences = min(len(strategic_sequences), configurable.max_parallel_sequences)
        sequences_to_execute = strategic_sequences[:max_sequences]
        
        logger.info(f"Starting parallel execution of {len(sequences_to_execute)} sequences using SimpleSequentialExecutor")
        
        # Execute sequences in parallel using the new lightweight executor
        parallel_results = await execute_sequences_in_parallel(
            sequences=sequences_to_execute,
            research_topic=research_topic,
            config=config,
            max_concurrent=max_sequences
        )
        
        logger.info(f"Parallel execution completed with {parallel_results.get('success_rate', 0):.1f}% success rate")
        
        return parallel_results
            
    except Exception as e:
        logger.error(f"Parallel sequence execution failed: {e}")
        # Return a minimal error result compatible with existing error handling
        return {
            "execution_id": "failed",
            "research_topic": research_topic,
            "total_duration": 0.0,
            "success_rate": 0.0,
            "sequence_results": {},
            "error_message": str(e),
            "failed_sequences": list(range(len(strategic_sequences))),
            "error_summary": {i: str(e) for i in range(len(strategic_sequences))},
            "best_strategy": None,
            "unique_insights_across_sequences": []
        }


def convert_parallel_results_to_agent_state(
    parallel_results: Dict[str, Any],
    original_state: AgentState
) -> Dict[str, Any]:
    """Convert parallel execution results back to AgentState format.
    
    Args:
        parallel_results: Results from parallel sequence execution
        original_state: Original agent state for field preservation
        
    Returns:
        Dictionary with AgentState compatible updates
    """
    # Aggregate results from all successful sequences
    all_notes = []
    all_raw_notes = []
    final_reports = []
    
    # Handle successful sequence results
    for sequence_id, result in parallel_results.get("sequence_results", {}).items():
        # Add findings as notes
        if result.get("comprehensive_findings"):
            all_notes.extend([f"[{sequence_id}] {finding}" for finding in result["comprehensive_findings"]])
        
        # Add agent insights as raw notes
        for agent_result in result.get("agent_results", []):
            agent_type = agent_result.get("agent_type", "unknown")
            insights = agent_result.get("key_insights", [])
            if insights:
                all_raw_notes.extend([f"[{sequence_id}-{agent_type}] {insight}" for insight in insights])
        
        # Create sequence-specific report section
        if result.get("comprehensive_findings"):
            sequence_report = f"## Sequence: {sequence_id}\n\n"
            sequence_report += "\n".join([f"- {finding}" for finding in result["comprehensive_findings"]])
            sequence_report += f"\n\n**Productivity Score:** {result.get('productivity_score', 0.0):.2f}\n"
            sequence_report += f"**Duration:** {result.get('total_duration', 0.0):.1f} seconds\n"
            final_reports.append(sequence_report)
    
    # Handle failed sequences
    failed_sequences = parallel_results.get("failed_sequences", [])
    error_summary = parallel_results.get("error_summary", {})
    if failed_sequences:
        all_notes.append(f"Failed sequences: {len(failed_sequences)} out of {len(failed_sequences) + len(parallel_results.get('sequence_results', {}))}")
        for seq_id in failed_sequences:
            error_msg = error_summary.get(seq_id, "Unknown error")
            all_raw_notes.append(f"[FAILED-seq_{seq_id}] {error_msg}")
    
    # Create combined final report
    if final_reports:
        combined_report = "# Parallel Research Execution Results\n\n"
        combined_report += f"**Research Topic:** {parallel_results.get('research_topic', 'Unknown')}\n\n"
        combined_report += "**Execution Summary:**\n"
        combined_report += f"- Success Rate: {parallel_results.get('success_rate', 0.0):.1f}%\n"
        combined_report += f"- Total Duration: {parallel_results.get('total_duration', 0.0):.1f} seconds\n"
        combined_report += f"- Best Strategy: {parallel_results.get('best_strategy', 'None')}\n\n"
        
        if parallel_results.get("unique_insights"):
            combined_report += "**Unique Insights Across All Sequences:**\n"
            for insight in parallel_results["unique_insights"][:10]:  # Limit to top 10
                combined_report += f"- {insight}\n"
            combined_report += "\n"
        
        combined_report += "\n".join(final_reports)
    else:
        combined_report = f"Parallel research execution failed: {parallel_results.get('error_message', 'Unknown error')}"
    
    # Create AgentState update dictionary
    result_state = {
        "notes": {"type": "override", "value": all_notes},
        "raw_notes": {"type": "override", "value": all_raw_notes},
        "final_report": combined_report,
        "research_brief": original_state.get("research_brief", ""),
        "supervisor_messages": original_state.get("supervisor_messages", []),
        "messages": original_state.get("messages", []),
        "parallel_sequence_results": parallel_results,  # Store full parallel results for analysis
        "strategic_sequences": original_state.get("strategic_sequences", [])  # Preserve sequences for frontend
    }
    
    return result_state


async def generate_strategic_sequences(
    research_topic: str,
    config: RunnableConfig
) -> List[AgentSequence]:
    """Generate strategic agent sequences using LLM reasoning.
    
    Args:
        research_topic: The research topic to generate sequences for
        config: Runtime configuration with model settings
        
    Returns:
        List of strategic agent sequences, empty list if generation fails
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize agent registry to get available agents
        agent_registry = await initialize_agent_registry(config)
        if not agent_registry:
            logger.warning("No agent registry available for sequence generation, using fallback")
            return await create_fallback_strategic_sequences([], research_topic)
        
        # Get agent capabilities
        from open_deep_research.supervisor.agent_capability_mapper import (
            AgentCapabilityMapper,
        )
        capability_mapper = AgentCapabilityMapper(agent_registry)
        agent_capabilities = capability_mapper.get_all_agent_capabilities()
        
        if not agent_capabilities:
            logger.warning("No agent capabilities found for sequence generation, using fallback")
            return await create_fallback_strategic_sequences([], research_topic)
        
        # Validate all agent capabilities to ensure they're properly formed
        validated_capabilities = []
        for i, cap in enumerate(agent_capabilities):
            try:
                # Check if it's already a proper AgentCapability object
                from open_deep_research.supervisor.sequence_models import (
                    AgentCapability,
                )
                
                if isinstance(cap, AgentCapability):
                    # Validate by ensuring all required fields are present
                    if cap.name and cap.expertise_areas and cap.description:
                        validated_capabilities.append(cap)
                    else:
                        logger.warning(f"AgentCapability {i} missing required fields: name={cap.name}, expertise_areas={cap.expertise_areas}, description={cap.description}")
                elif isinstance(cap, dict):
                    # Try to construct AgentCapability from dict
                    try:
                        agent_cap = AgentCapability(**cap)
                        validated_capabilities.append(agent_cap)
                        logger.debug(f"Successfully converted dict to AgentCapability for agent {agent_cap.name}")
                    except Exception as dict_error:
                        logger.warning(f"Failed to convert dict to AgentCapability: {dict_error}")
                        logger.debug(f"Problematic dict: {cap}")
                else:
                    # Try to convert other types
                    if hasattr(cap, 'name') and hasattr(cap, 'expertise_areas') and hasattr(cap, 'description'):
                        # Convert to dict first, then to AgentCapability
                        cap_dict = {
                            'name': cap.name,
                            'expertise_areas': cap.expertise_areas or [],
                            'description': cap.description or '',
                            'typical_use_cases': getattr(cap, 'typical_use_cases', []),
                            'strength_summary': getattr(cap, 'strength_summary', ''),
                            'core_responsibilities': getattr(cap, 'core_responsibilities', []),
                            'completion_indicators': getattr(cap, 'completion_indicators', [])
                        }
                        agent_cap = AgentCapability(**cap_dict)
                        validated_capabilities.append(agent_cap)
                        logger.debug(f"Successfully converted object to AgentCapability for agent {agent_cap.name}")
                    else:
                        logger.warning(f"Invalid capability object found, skipping: {type(cap)} - {cap}")
            except Exception as e:
                logger.warning(f"Failed to validate agent capability {i}: {e}")
                logger.debug(f"Capability data: {cap}")
        
        if not validated_capabilities:
            logger.warning("No valid agent capabilities found for sequence generation, using fallback")
            return await create_fallback_strategic_sequences([], research_topic)
        
        # Limit to first 10 agents for performance and LLM context window
        if len(validated_capabilities) > 10:
            validated_capabilities = validated_capabilities[:10]
            logger.info("Limited agent capabilities to first 10 agents for LLM processing")
        
        agent_capabilities = validated_capabilities
        
        # Initialize LLM sequence generator with configuration
        configurable = Configuration.from_runnable_config(config)
        model_config = get_model_config_for_provider(
            model_name=configurable.research_model,
            api_key=get_api_key_for_model(configurable.research_model, config),
            max_tokens=configurable.research_model_max_tokens,
            tags=["sequence_generation", "strategic_planning"]
        )
        
        generator = UnifiedSequenceGenerator(agent_registry, model_config)
        
        # Prepare input for sequence generation
        generation_input = SequenceGenerationInput(
            research_topic=research_topic,
            research_brief=None,  # Could be enhanced in future
            available_agents=agent_capabilities,
            research_type=None,  # Could be extracted from research topic in future
            constraints={"max_agents_per_sequence": 4},
            generation_mode="hybrid",  # Use hybrid mode for best results
            num_sequences=3
        )
        
        logger.info(f"Generating strategic sequences for research topic: {research_topic}")
        logger.debug(f"Using {len(agent_capabilities)} validated agent capabilities for LLM generation")
        
        # Generate sequences asynchronously
        try:
            result = await generator.generate_sequences(generation_input)
            
            if result.success and result.output.sequences:
                logger.info(f"Successfully generated {len(result.output.sequences)} strategic sequences")
                # Log details about each sequence
                for i, seq in enumerate(result.output.sequences):
                    logger.info(f"Sequence {i+1}: {seq.sequence_name} - {seq.research_focus} (confidence: {seq.confidence_score:.2f})")
                
                # Identify recommended sequence
                recommended_idx = getattr(result.output, 'recommended_sequence', 0)
                if 0 <= recommended_idx < len(result.output.sequences):
                    logger.info(f"LLM recommends sequence {recommended_idx + 1}: {result.output.sequences[recommended_idx].sequence_name}")
                
                return result.output.sequences
            else:
                error_details = "Unknown error"
                if hasattr(result, 'metadata') and hasattr(result.metadata, 'error_details'):
                    error_details = result.metadata.error_details
                elif hasattr(result, 'metadata') and hasattr(result.metadata, 'fallback_used') and result.metadata.fallback_used:
                    error_details = "LLM generation used fallback mode"
                logger.warning(f"LLM sequence generation failed: {error_details}")
                logger.info("Creating fallback sequences using agent capabilities")
                return await create_fallback_strategic_sequences(agent_capabilities, research_topic)
        
        except Exception as generation_error:
            logger.error(f"Exception during LLM sequence generation: {generation_error}")
            logger.info("Creating fallback sequences due to generation exception")
            return await create_fallback_strategic_sequences(agent_capabilities, research_topic)
            
    except Exception as e:
        logger.error(f"Failed to generate strategic sequences: {e}")
        # Always provide fallback sequences instead of empty list
        try:
            agent_registry = await initialize_agent_registry(config)
            if agent_registry:
                capability_mapper = AgentCapabilityMapper(agent_registry)
                agent_capabilities = capability_mapper.get_all_agent_capabilities()
                return await create_fallback_strategic_sequences(agent_capabilities, research_topic)
        except:
            pass
        
        # Final fallback with generic agents
        return await create_fallback_strategic_sequences([], research_topic)


async def create_fallback_strategic_sequences(
    agent_capabilities: List[Any],
    research_topic: str
) -> List[AgentSequence]:
    """Create fallback strategic sequences when LLM generation fails.
    
    Args:
        agent_capabilities: List of available agent capabilities
        research_topic: The research topic to create sequences for
        
    Returns:
        List of fallback AgentSequence objects
    """
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("Creating fallback strategic sequences")
    
    # Extract agent names from capabilities
    if agent_capabilities:
        agent_names = [cap.name for cap in agent_capabilities]
    else:
        # Use default generic agent names
        agent_names = ["general_researcher", "academic_analyst", "industry_expert"]
    
    sequences = []
    
    # Sequence 1: Foundational Academic Research - designed to trigger 'foundational' theme
    if agent_names:
        seq1_agents = agent_names[:min(3, len(agent_names))]
        sequences.append(AgentSequence(
            sequence_name="Foundational Academic Research",
            agent_names=seq1_agents,
            rationale="Systematic academic research approach building comprehensive theoretical foundations through scholarly analysis and evidence-based methodology",
            approach_description="Academic rigor with systematic literature review and theoretical framework development for comprehensive foundational understanding",
            expected_outcomes=[
                "Theoretical framework establishment",
                "Academic literature synthesis", 
                "Evidence-based foundational insights"
            ],
            confidence_score=0.8,
            research_focus="Academic foundation and theoretical understanding"
        ))
    
    # Sequence 2: Technical Implementation Analysis - designed to trigger 'technical' theme  
    if len(agent_names) >= 2:
        seq2_agents = agent_names[1:min(4, len(agent_names))]
        if len(seq2_agents) == 0 and agent_names:
            seq2_agents = [agent_names[0]]
        
        sequences.append(AgentSequence(
            sequence_name="Technical Implementation Deep-Dive", 
            agent_names=seq2_agents,
            rationale="Engineering-focused technical analysis examining system architecture, implementation feasibility, and performance optimization through specialized technical expertise",
            approach_description="Technical architecture analysis with engineering feasibility assessment and system design optimization",
            expected_outcomes=[
                "Technical architecture insights",
                "Implementation feasibility analysis",
                "Performance optimization recommendations"
            ],
            confidence_score=0.7,
            research_focus="Technical implementation and system architecture"
        ))
    
    # Sequence 3: Market Intelligence Analysis - designed to trigger 'market' theme
    if agent_names:
        # Use first and last agents if available, or just first
        seq3_agents = []
        if len(agent_names) >= 1:
            seq3_agents.append(agent_names[0])
        if len(agent_names) >= 3:
            seq3_agents.append(agent_names[-1])
        
        sequences.append(AgentSequence(
            sequence_name="Market Intelligence Analysis",
            agent_names=seq3_agents or [agent_names[0]] if agent_names else [],
            rationale="Business-oriented market research focusing on commercial viability, competitive landscape, and industry trends to inform strategic business decisions",
            approach_description="Market dynamics analysis with competitive intelligence and commercial opportunity assessment",
            expected_outcomes=[
                "Market opportunity identification",
                "Competitive landscape mapping",
                "Commercial viability assessment"
            ],
            confidence_score=0.6,
            research_focus="Market trends and commercial intelligence"
        ))
    
    # Ensure we have exactly 3 sequences - critical for supervisor functionality
    fallback_themes = [
        {
            "name": "Investigative Research Analysis", 
            "rationale": "Comprehensive investigative research approach examining deep patterns and uncovering detailed insights through thorough investigation",
            "approach": "Investigative analysis with comprehensive examination and detailed exploration",
            "focus": "Investigation and comprehensive analysis",
            "outcomes": ["Deep investigative findings", "Comprehensive examination", "Detailed insights"]
        },
        {
            "name": "Experimental Innovation Research",
            "rationale": "Innovative experimental research exploring cutting-edge developments and future trends through experimental methodology", 
            "approach": "Experimental innovation analysis with emerging technology exploration",
            "focus": "Innovation and experimental research",
            "outcomes": ["Innovation insights", "Experimental findings", "Future trends"]
        },
        {
            "name": "Rapid Assessment Research",
            "rationale": "Quick rapid research approach optimized for immediate insights and actionable findings through efficient analysis",
            "approach": "Rapid assessment with quick analysis and immediate decision support", 
            "focus": "Rapid analysis and quick insights",
            "outcomes": ["Immediate insights", "Quick assessment", "Actionable findings"]
        }
    ]
    
    while len(sequences) < 3:
        theme_idx = len(sequences) - 1 if len(sequences) > 0 else 0
        theme = fallback_themes[theme_idx % len(fallback_themes)]
        fallback_agents = agent_names[:1] if agent_names else ["general_researcher"]
        
        sequences.append(AgentSequence(
            sequence_name=theme["name"],
            agent_names=fallback_agents,
            rationale=theme["rationale"],
            approach_description=theme["approach"],
            expected_outcomes=theme["outcomes"],
            confidence_score=0.3,
            research_focus=theme["focus"]
        ))
    
    # Guarantee we never return empty sequences
    if not sequences:
        sequences = [AgentSequence(
            sequence_name="Emergency Research Sequence",
            agent_names=["emergency_researcher"],
            rationale="Emergency fallback to ensure supervisor can execute research",
            approach_description="Emergency research execution when no other options available",
            expected_outcomes=["Basic research attempt"],
            confidence_score=0.1,
            research_focus="Emergency research"
        )]
    
    logger.info(f"Created {len(sequences)} fallback strategic sequences")
    return sequences[:3]  # Ensure exactly 3 sequences


async def initialize_agent_registry(config: RunnableConfig) -> Optional[Any]:
    """Initialize agent registry with existing tools integration.
    
    Args:
        config: Runtime configuration
        
    Returns:
        Initialized AgentRegistry or None if initialization fails
    """
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    try:
        from open_deep_research.agents.registry import AgentRegistry
        
        # Initialize registry with current project root
        project_root = Path.cwd()
        agent_registry = AgentRegistry(project_root=str(project_root))
        
        # Check if agents directory exists and has agents
        stats = agent_registry.get_registry_stats()
        logger.info(f"Agent registry initialized: {stats['total_agents']} agents found")
        
        if stats['total_agents'] == 0:
            logger.warning("No agents found in registry")
            # Create agent directories if they don't exist
            agent_registry.create_agent_directories()
            return None
        
        # Validate all agents
        validation_results = agent_registry.validate_all_agents()
        if validation_results:
            logger.warning(f"Agent validation warnings: {validation_results}")
        
        # Get all available tools for inheritance
        available_tools = await get_all_tools(config)
        tool_names = [getattr(tool, 'name', 'unknown_tool') for tool in available_tools]
        logger.info(f"Available tools for agents: {tool_names}")
        
        return agent_registry
        
    except ImportError as e:
        logger.error(f"Failed to import AgentRegistry: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize agent registry: {e}")
        return None




# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()