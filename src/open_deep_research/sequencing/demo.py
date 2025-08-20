"""Demonstration script for the Sequential Agent Ordering Optimization Framework.

This script provides examples of how to use the sequencing framework to prove
that different agent orderings produce measurably different productivity outcomes.
"""

import asyncio
import logging
from typing import Dict, List

from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.sequencing import (
    SequenceOptimizationEngine,
    SequenceStrategy,
    SEQUENCE_PATTERNS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequencingDemo:
    """Demonstration class for the sequencing framework."""
    
    def __init__(self, config: RunnableConfig):
        """Initialize the demo with configuration."""
        self.config = config
        self.engine = SequenceOptimizationEngine(config)
    
    async def single_sequence_demo(self, research_topic: str, strategy: SequenceStrategy):
        """Demonstrate execution of a single sequence strategy."""
        logger.info(f"=== Single Sequence Demo: {strategy.value} ===")
        logger.info(f"Research Topic: {research_topic}")
        
        # Get the sequence pattern
        pattern = SEQUENCE_PATTERNS[strategy]
        logger.info(f"Agent Order: {' → '.join([a.value for a in pattern.agent_order])}")
        
        # Execute the sequence
        result = await self.engine.execute_sequence(pattern, research_topic)
        
        # Display results
        logger.info(f"Execution completed in {result.total_duration:.1f} seconds")
        logger.info(f"Tool Productivity: {result.overall_productivity_metrics.tool_productivity:.3f}")
        logger.info(f"Research Quality: {result.overall_productivity_metrics.research_quality_score:.3f}")
        logger.info(f"Agent Efficiency: {result.overall_productivity_metrics.agent_efficiency:.3f}")
        logger.info(f"Unique Insights: {result.unique_insights_generated}")
        
        # Show agent-specific results
        for agent_result in result.agent_results:
            logger.info(f"  {agent_result.agent_type.value}: "
                       f"{len(agent_result.key_insights)} insights, "
                       f"{agent_result.tool_calls_made} tool calls, "
                       f"{agent_result.execution_duration:.1f}s")
        
        return result
    
    async def sequence_comparison_demo(self, research_topic: str):
        """Demonstrate comparison of all three sequence strategies."""
        logger.info(f"=== Sequence Comparison Demo ===")
        logger.info(f"Research Topic: {research_topic}")
        logger.info("Comparing all three sequence strategies...")
        
        # Execute comparison
        comparison = await self.engine.compare_sequences(research_topic)
        
        # Display comparison results
        logger.info(f"\n=== COMPARISON RESULTS ===")
        logger.info(f"Productivity Variance: {comparison.productivity_variance:.3f}")
        logger.info(f"Significant Difference Detected: {comparison.significant_difference_detected}")
        logger.info(f"Best Performing Sequence: {comparison.highest_productivity_sequence.value}")
        logger.info(f"Productivity Advantage: {comparison.productivity_advantage:.1f}%")
        
        # Show detailed rankings
        logger.info("\n=== PRODUCTIVITY RANKINGS ===")
        sorted_rankings = sorted(
            comparison.productivity_rankings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (strategy, score) in enumerate(sorted_rankings, 1):
            logger.info(f"{i}. {strategy.value}: {score:.3f}")
        
        # Show unique insights
        logger.info("\n=== SEQUENCE-SPECIFIC INSIGHTS ===")
        for strategy, insights in comparison.unique_insights_by_sequence.items():
            logger.info(f"{strategy.value}: {len(insights)} unique insights")
            for insight in insights[:2]:  # Show first 2
                logger.info(f"  - {insight[:100]}...")
        
        return comparison
    
    async def batch_analysis_demo(self, research_topics: List[str]):
        """Demonstrate batch analysis across multiple research topics."""
        logger.info(f"=== Batch Analysis Demo ===")
        logger.info(f"Analyzing {len(research_topics)} research topics...")
        
        comparisons = await self.engine.batch_sequence_analysis(research_topics)
        
        # Aggregate results
        significant_differences = sum(1 for c in comparisons if c.significant_difference_detected)
        avg_variance = sum(c.productivity_variance for c in comparisons) / len(comparisons)
        
        # Count best performing strategies
        best_strategies = {}
        for comparison in comparisons:
            strategy = comparison.highest_productivity_sequence
            best_strategies[strategy] = best_strategies.get(strategy, 0) + 1
        
        logger.info(f"\n=== BATCH ANALYSIS RESULTS ===")
        logger.info(f"Topics with Significant Differences: {significant_differences}/{len(research_topics)}")
        logger.info(f"Average Productivity Variance: {avg_variance:.3f}")
        
        logger.info("\n=== STRATEGY PERFORMANCE SUMMARY ===")
        for strategy, count in best_strategies.items():
            percentage = (count / len(research_topics)) * 100
            logger.info(f"{strategy.value}: Best in {count}/{len(research_topics)} topics ({percentage:.1f}%)")
        
        return comparisons
    
    def display_framework_overview(self):
        """Display an overview of the framework capabilities."""
        logger.info("=== SEQUENTIAL AGENT ORDERING OPTIMIZATION FRAMEWORK ===")
        logger.info("")
        logger.info("This framework proves that different sequential orderings of")
        logger.info("specialized agents produce measurably different productivity outcomes.")
        logger.info("")
        logger.info("KEY COMPONENTS:")
        logger.info("1. Three Specialized Agents:")
        logger.info("   - Academic Agent: Theory & research focus")
        logger.info("   - Industry Agent: Market & business focus")
        logger.info("   - Technical Trends Agent: Implementation & future trends")
        logger.info("")
        logger.info("2. Three Strategic Sequences:")
        logger.info("   - Theory First: Academic → Industry → Technical")
        logger.info("   - Market First: Industry → Academic → Technical")
        logger.info("   - Future Back: Technical → Academic → Industry")
        logger.info("")
        logger.info("3. Core Metrics:")
        logger.info("   - Tool Productivity (TP) = Research Quality / Agent Calls")
        logger.info("   - Agent Efficiency = Useful Insights / Total Agent Calls")
        logger.info("   - Context Efficiency = Relevant Context Used / Total Available")
        logger.info("   - Time to Value = Time to first significant insight")
        logger.info("")
        logger.info("4. Key Features:")
        logger.info("   - Dynamic question generation based on previous insights")
        logger.info("   - Linear context passing (not cumulative)")
        logger.info("   - Cognitive offloading prevention")
        logger.info("   - Adaptive learning from insight transitions")
        logger.info("   - >20% variance detection for statistical significance")
        logger.info("")


async def run_complete_demo():
    """Run a complete demonstration of the framework."""
    # Create configuration (you would typically load this from environment)
    config = RunnableConfig(configurable={
        "research_model": "openai:gpt-4.1",
        "research_model_max_tokens": 8192,
        "compression_model": "openai:gpt-4.1",
        "compression_model_max_tokens": 4096,
        "max_react_tool_calls": 8,
        "max_researcher_iterations": 4,
        "max_concurrent_research_units": 3,
        "search_api": "tavily"
    })
    
    demo = SequencingDemo(config)
    
    # Display framework overview
    demo.display_framework_overview()
    
    # Demo research topics
    research_topics = [
        "Quantum computing applications in drug discovery",
        "AI safety alignment with human values", 
        "Renewable energy storage solutions",
        "Autonomous vehicle ethics frameworks"
    ]
    
    try:
        # 1. Single sequence demonstration
        logger.info("\n" + "="*60)
        result = await demo.single_sequence_demo(
            research_topics[0], 
            SequenceStrategy.THEORY_FIRST
        )
        
        # 2. Sequence comparison demonstration
        logger.info("\n" + "="*60)
        comparison = await demo.sequence_comparison_demo(research_topics[1])
        
        # 3. Batch analysis demonstration
        logger.info("\n" + "="*60)
        batch_results = await demo.batch_analysis_demo(research_topics)
        
        # 4. Performance summary
        logger.info("\n" + "="*60)
        performance_summary = demo.engine.get_performance_summary()
        logger.info("=== OVERALL PERFORMANCE SUMMARY ===")
        for key, value in performance_summary.items():
            logger.info(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        logger.error("Note: This demo requires proper API keys and configuration.")
        logger.error("Please ensure your environment is configured with:")
        logger.error("- OpenAI API key (or other model provider)")
        logger.error("- Tavily API key (or other search provider)")
        logger.error("- Proper model configurations in your .env file")


if __name__ == "__main__":
    """Run the demo when script is executed directly."""
    print("Sequential Agent Ordering Optimization Framework Demo")
    print("====================================================")
    print()
    print("This demo showcases how different agent orderings affect research productivity.")
    print("It requires proper API keys and configuration to run fully.")
    print()
    
    # Run the demo
    asyncio.run(run_complete_demo())