"""Integration layer for the sequencing framework with OpenDeepResearch.

This module provides integration points to incorporate sequence optimization
into the main OpenDeepResearch workflow while maintaining backward compatibility.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.types import Command

from open_deep_research.configuration import Configuration
from open_deep_research.state import AgentState
from open_deep_research.sequencing import (
    SequenceOptimizationEngine,
    ParallelSequenceExecutor
)

logger = logging.getLogger(__name__)


class SequencingConfiguration(Configuration):
    """Simplified configuration for dynamic sequencing framework."""
    
    max_dynamic_sequences: int = 3
    """Maximum number of dynamic sequences to generate per research topic"""


async def dynamic_sequence_research_supervisor(state: AgentState, config: RunnableConfig) -> Command:
    """Dynamic research supervisor that generates optimal sequences based on topic.
    
    This function uses the sequence optimization framework to generate custom
    sequences tailored to the research topic.
    
    Args:
        state: Current agent state
        config: Runtime configuration
        
    Returns:
        Command to continue with dynamically generated sequences
    """
    configurable = SequencingConfiguration.from_runnable_config(config)
    
    try:
        # Initialize sequence optimization engine
        engine = SequenceOptimizationEngine(config)
        research_topic = state.get("research_brief", "")
        
        if not research_topic:
            logger.warning("No research brief found, using standard approach")
            return Command(goto="standard_supervisor")
        
        logger.info(f"Generating dynamic sequences for: {research_topic}")
        
        # Generate multiple dynamic sequences using topic analysis
        from .sequence_selector import SequenceAnalyzer
        analyzer = SequenceAnalyzer()
        
        # Generate custom sequences based on topic
        dynamic_sequences = analyzer.generate_dynamic_sequences(
            research_topic, 
            num_sequences=configurable.max_dynamic_sequences
        )
        
        if not dynamic_sequences:
            logger.warning("No dynamic sequences generated, using standard approach")
            return Command(goto="standard_supervisor")
        
        # Execute sequences in parallel
        executor = ParallelSequenceExecutor(config)
        comparison = await executor.execute_sequences_parallel(dynamic_sequences, research_topic)
        
        # Create research report from best performing sequence
        research_synthesis = _create_dynamic_synthesis(comparison)
        notes = _extract_dynamic_notes(comparison)
        
        return Command(
            goto="__end__",
            update={
                "notes": notes,
                "research_brief": research_topic,
                "dynamic_sequence_result": comparison
            }
        )
    
    except Exception as e:
        logger.error(f"Dynamic sequence generation failed: {e}")
        logger.info("Falling back to standard research supervisor")
        return Command(goto="standard_supervisor")


def _create_dynamic_synthesis(comparison_result) -> str:
    """Create research synthesis from dynamic sequence comparison."""
    # Extract best performing sequence results
    if hasattr(comparison_result, 'best_result'):
        best_result = comparison_result.best_result
        return best_result.final_research_synthesis
    return "Dynamic sequence research completed."

def _extract_dynamic_notes(comparison_result) -> List[str]:
    """Extract notes from dynamic sequence comparison."""
    notes = []
    if hasattr(comparison_result, 'all_results'):
        for result in comparison_result.all_results:
            if hasattr(result, 'comprehensive_findings'):
                notes.extend(result.comprehensive_findings)
    return notes[:20]  # Limit to prevent overwhelming output


def _create_comparison_synthesis(comparison) -> str:
    """Create a synthesis report from sequence comparison."""
    
    synthesis_parts = [
        f"# Sequence Optimization Analysis: {comparison.research_topic}",
        "",
        "## Executive Summary",
        f"This analysis compared {len(comparison.compared_sequences)} different agent sequence strategies "
        f"to optimize research productivity. The analysis {'**detected significant**' if comparison.significant_difference_detected else '**found limited**'} "
        f"variance in productivity outcomes across different agent orderings.",
        "",
        f"**Key Finding:** {comparison.productivity_variance:.1%} variance detected between sequences",
        f"**Optimal Sequence:** {comparison.highest_productivity_sequence.value}",
        f"**Productivity Advantage:** {comparison.productivity_advantage:.1f}%",
        "",
        "## Sequence Performance Rankings",
        ""
    ]
    
    # Add performance rankings
    sorted_rankings = sorted(
        comparison.productivity_rankings.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (strategy, score) in enumerate(sorted_rankings, 1):
        synthesis_parts.append(f"{i}. **{strategy}**: {score:.3f} (Tool Productivity)")
    
    synthesis_parts.extend([
        "",
        "## Research Findings by Sequence",
        ""
    ])
    
    # Add findings from each sequence
    for result in comparison.compared_sequences:
        strategy_name = result.sequence_pattern.strategy
        synthesis_parts.extend([
            f"### {strategy_name} Sequence Results",
            f"- **Execution Time:** {result.total_duration:.1f} seconds",
            f"- **Unique Insights:** {result.unique_insights_generated}",
            f"- **Research Quality:** {result.overall_productivity_metrics.research_quality_score:.3f}",
            "",
            "**Key Findings:**"
        ])
        
        for finding in result.comprehensive_findings[:5]:  # Top 5 findings
            synthesis_parts.append(f"- {finding}")
        
        synthesis_parts.append("")
    
    # Add sequence-specific advantages
    synthesis_parts.extend([
        "## Sequence-Specific Advantages",
        ""
    ])
    
    for strategy, advantages in comparison.sequence_specific_advantages.items():
        synthesis_parts.extend([
            f"### {strategy}",
            ""
        ])
        for advantage in advantages:
            synthesis_parts.append(f"- {advantage}")
        synthesis_parts.append("")
    
    # Add methodology note
    synthesis_parts.extend([
        "## Methodology",
        "This analysis used the Sequential Agent Ordering Optimization Framework, which:",
        "- Executes specialized agents (Academic, Industry, Technical Trends) in different orders",
        "- Measures Tool Productivity as Research Quality divided by Agent Calls",
        "- Prevents cognitive offloading through dynamic question generation",
        "- Uses linear context passing to ensure each agent builds on previous insights",
        f"- Applies a {comparison.productivity_variance:.1%} variance threshold for significance testing"
    ])
    
    return "\n".join(synthesis_parts)


def _extract_comparison_notes(comparison) -> List[str]:
    """Extract notes from sequence comparison for standard workflow."""
    notes = []
    
    # Add summary note
    notes.append(f"Sequence optimization analysis completed for '{comparison.research_topic}'. "
                f"{'Significant' if comparison.significant_difference_detected else 'Limited'} "
                f"productivity variance detected ({comparison.productivity_variance:.1%}).")
    
    # Add findings from best performing sequence
    best_result = None
    for result in comparison.compared_sequences:
        if result.sequence_pattern.strategy == comparison.highest_productivity_sequence:
            best_result = result
            break
    
    if best_result:
        notes.append(f"Optimal sequence ({comparison.highest_productivity_sequence.value}) findings:")
        notes.extend(best_result.comprehensive_findings)
    
    # Add comparative insights
    notes.append("Sequence-specific insights identified:")
    for strategy, insights in comparison.unique_insights_by_sequence.items():
        if insights:
            notes.append(f"{strategy}: {len(insights)} unique insights")
            notes.extend(insights[:3])  # Top 3 unique insights
    
    return notes


def _extract_sequence_notes(result) -> List[str]:
    """Extract notes from single sequence result for standard workflow."""
    notes = []
    
    # Add sequence summary
    strategy_name = result.sequence_pattern.strategy
    notes.append(f"Research completed using {strategy_name} sequence strategy. "
                f"Tool Productivity: {result.overall_productivity_metrics.tool_productivity:.3f}, "
                f"Quality Score: {result.overall_productivity_metrics.research_quality_score:.3f}")
    
    # Add agent-specific findings
    for agent_result in result.agent_results:
        agent_name = agent_result.agent_type.value.replace('_', ' ').title()
        notes.append(f"{agent_name} Agent ({agent_result.execution_duration:.1f}s, "
                    f"{agent_result.tool_calls_made} calls): {len(agent_result.key_insights)} insights")
        notes.extend(agent_result.key_insights)
    
    # Add comprehensive findings
    notes.extend(result.comprehensive_findings)
    
    return notes


def create_enhanced_deep_researcher():
    """Create an enhanced deep researcher graph with sequence optimization."""
    
    from open_deep_research.deep_researcher import (
        deep_researcher_builder,
        clarify_with_user,
        write_research_brief,
        final_report_generation,
        supervisor_subgraph
    )
    
    # Create enhanced builder
    enhanced_builder = StateGraph(AgentState, input=AgentInputState, config_schema=SequencingConfiguration)
    
    # Add all standard nodes
    enhanced_builder.add_node("clarify_with_user", clarify_with_user)
    enhanced_builder.add_node("write_research_brief", write_research_brief)
    enhanced_builder.add_node("sequence_research_supervisor", sequence_research_supervisor)
    enhanced_builder.add_node("standard_supervisor", supervisor_subgraph)
    enhanced_builder.add_node("final_report_generation", final_report_generation)
    
    # Define routing logic
    def route_to_supervisor(state: AgentState, config: RunnableConfig) -> str:
        """Route to either sequence optimization or standard supervisor."""
        configurable = SequencingConfiguration.from_runnable_config(config)
        if configurable.enable_sequence_optimization:
            return "sequence_research_supervisor"
        else:
            return "standard_supervisor"
    
    # Add edges
    enhanced_builder.add_edge("clarify_with_user", "write_research_brief")
    enhanced_builder.add_conditional_edges(
        "write_research_brief",
        route_to_supervisor,
        {
            "sequence_research_supervisor": "sequence_research_supervisor",
            "standard_supervisor": "standard_supervisor"
        }
    )
    enhanced_builder.add_edge("sequence_research_supervisor", "final_report_generation")
    enhanced_builder.add_edge("standard_supervisor", "final_report_generation")
    enhanced_builder.add_edge("final_report_generation", "__end__")
    
    return enhanced_builder.compile()


# Usage example configuration
SEQUENCE_CONFIG_EXAMPLE = {
    "configurable": {
        # Standard OpenDeepResearch configuration
        "research_model": "openai:gpt-4.1",
        "research_model_max_tokens": 8192,
        "compression_model": "openai:gpt-4.1", 
        "compression_model_max_tokens": 4096,
        "final_report_model": "openai:gpt-4.1",
        "final_report_model_max_tokens": 8192,
        "max_react_tool_calls": 8,
        "max_researcher_iterations": 4,
        "max_concurrent_research_units": 3,
        "search_api": "tavily",
        
        # Sequence optimization configuration
        "enable_sequence_optimization": True,
        "sequence_strategy": "theory_first",  # or "market_first", "future_back", None for auto
        "compare_all_sequences": False,  # Set to True to compare all strategies
        "sequence_variance_threshold": 0.2
    }
}