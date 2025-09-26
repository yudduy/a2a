"""Example usage of SequenceGenerator with Sequential Supervisor.

This example demonstrates how to integrate the SequenceGenerator with the
Sequential Multi-Agent Supervisor for optimal agent execution ordering.
"""

import asyncio
import logging

from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.core.sequence_generator import (
    UnifiedSequenceGenerator,
)
from open_deep_research.state import SequentialSupervisorState
from open_deep_research.supervisor.sequential_supervisor import (
    SequentialSupervisor,
    SupervisorConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequenceGeneratorIntegration:
    """Integration example showing SequenceGenerator with Sequential Supervisor."""
    
    def __init__(self, project_root: str = None):
        """Initialize the integration components.
        
        Args:
            project_root: Root directory for agent registry
        """
        # Initialize agent registry
        self.agent_registry = AgentRegistry(project_root)
        
        # Initialize sequence generator
        self.sequence_generator = UnifiedSequenceGenerator(
            agent_registry=self.agent_registry,
            debug_mode=True
        )
        
        # Initialize sequential supervisor
        supervisor_config = SupervisorConfig(
            agent_timeout_seconds=300.0,
            max_agents_per_sequence=5,
            allow_dynamic_insertion=True,
            debug_mode=True
        )
        
        self.supervisor = SequentialSupervisor(
            agent_registry=self.agent_registry,
            config=supervisor_config
        )
        
        logger.info("SequenceGenerator integration initialized")
    
    async def generate_and_execute_optimal_sequence(
        self,
        research_topic: str,
        num_sequence_options: int = 3
    ) -> dict:
        """Generate optimal sequence and execute with supervisor.
        
        Args:
            research_topic: The research topic to investigate
            num_sequence_options: Number of sequence alternatives to generate
            
        Returns:
            Dictionary with execution results and sequence information
        """
        logger.info("Starting sequence generation and execution for: %s", research_topic[:100])
        
        try:
            # Step 1: Generate optimal sequences
            sequences = self.sequence_generator.generate_sequences(
                research_topic=research_topic,
                num_sequences=num_sequence_options
            )
            
            if not sequences:
                raise RuntimeError("No sequences generated")
            
            # Step 2: Select the best sequence
            best_sequence = sequences[0]  # Already ranked by fitness score
            
            logger.info("Selected optimal sequence: %s (score: %.2f)",
                       best_sequence.sequence_name, best_sequence.score)
            logger.info("Agent sequence: %s", " → ".join(best_sequence.agents))
            logger.info("Strategy: %s", best_sequence.strategy.value)
            logger.info("Rationale: %s", best_sequence.rationale)
            
            # Step 3: Validate the sequence
            validation_result = self.supervisor.validate_sequence(best_sequence.agents)
            
            if not validation_result["valid"]:
                logger.error("Sequence validation failed: %s", validation_result["errors"])
                raise RuntimeError(f"Invalid sequence: {validation_result['errors']}")
            
            # Log any warnings
            for warning in validation_result.get("warnings", []):
                logger.warning("Sequence validation warning: %s", warning)
            
            # Step 4: Create supervisor state with the optimal sequence
            supervisor_state = SequentialSupervisorState()
            supervisor_state.research_topic = research_topic
            supervisor_state.planned_sequence = best_sequence.agents.copy()
            supervisor_state.sequence_position = 0
            supervisor_state.handoff_ready = True
            
            # Step 5: Build and execute the workflow
            workflow = await self.supervisor.create_workflow_graph()
            compiled_workflow = workflow.compile()
            
            # Execute the workflow
            logger.info("Executing optimal agent sequence...")
            final_state = await compiled_workflow.ainvoke(supervisor_state)
            
            # Step 6: Return execution results
            execution_stats = self.supervisor.get_execution_stats()
            
            return {
                "success": True,
                "sequence_info": {
                    "name": best_sequence.sequence_name,
                    "strategy": best_sequence.strategy.value,
                    "agents": best_sequence.agents,
                    "score": best_sequence.score,
                    "confidence": best_sequence.confidence,
                    "rationale": best_sequence.rationale,
                    "estimated_duration": best_sequence.estimated_duration
                },
                "alternative_sequences": [
                    {
                        "name": seq.sequence_name,
                        "strategy": seq.strategy.value,
                        "agents": seq.agents,
                        "score": seq.score
                    }
                    for seq in sequences[1:]  # All other sequences
                ],
                "execution_results": {
                    "agents_executed": len(final_state.executed_agents),
                    "insights_generated": len(final_state.running_report.all_insights) if final_state.running_report else 0,
                    "execution_time": execution_stats.get("total_execution_time", 0),
                    "completion_status": execution_stats.get("completion_status", "unknown")
                },
                "final_report": final_state.running_report
            }
            
        except Exception as e:
            logger.error("Sequence generation and execution failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "sequence_info": None,
                "execution_results": None
            }
    
    def analyze_topic_and_recommend_strategy(self, research_topic: str) -> dict:
        """Analyze topic characteristics and recommend sequence strategy.
        
        Args:
            research_topic: The research topic to analyze
            
        Returns:
            Dictionary with topic analysis and strategy recommendations
        """
        logger.info("Analyzing topic characteristics: %s", research_topic[:100])
        
        # Analyze topic
        topic_analysis = self.sequence_generator.analyze_topic_characteristics(research_topic)
        
        # Generate multiple sequences to compare strategies
        sequences = self.sequence_generator.generate_sequences(
            research_topic=research_topic,
            num_sequences=5
        )
        
        return {
            "topic_analysis": {
                "topic_type": topic_analysis.topic_type.value,
                "complexity_score": topic_analysis.complexity_score,
                "scope_breadth": topic_analysis.scope_breadth,
                "estimated_agents_needed": topic_analysis.estimated_agents_needed,
                "priority_areas": topic_analysis.priority_areas,
                "domain_indicators": topic_analysis.domain_indicators,
                "market_relevance": topic_analysis.market_relevance,
                "technical_complexity": topic_analysis.technical_complexity,
                "data_intensity": topic_analysis.data_intensity,
                "time_sensitivity": topic_analysis.time_sensitivity
            },
            "strategy_recommendations": [
                {
                    "sequence_name": seq.sequence_name,
                    "strategy": seq.strategy.value,
                    "agents": seq.agents,
                    "score": seq.score,
                    "confidence": seq.confidence,
                    "rationale": seq.rationale,
                    "estimated_duration": seq.estimated_duration,
                    "scoring_breakdown": {
                        "topic_fit": seq.topic_fit_score,
                        "coverage": seq.coverage_score,
                        "efficiency": seq.efficiency_score,
                        "expertise_match": seq.expertise_match_score
                    }
                }
                for seq in sequences
            ]
        }
    
    def get_available_agents_summary(self) -> dict:
        """Get summary of available agents and their capabilities.
        
        Returns:
            Dictionary with agent registry information
        """
        agents_detailed = self.agent_registry.list_agents_detailed()
        registry_stats = self.agent_registry.get_registry_stats()
        
        return {
            "registry_stats": registry_stats,
            "agents": [
                {
                    "name": agent["name"],
                    "description": agent["description"][:100] + "..." if len(agent["description"]) > 100 else agent["description"],
                    "expertise_areas": agent["expertise_areas"][:5],  # Limit to first 5
                    "source": agent["source"]
                }
                for agent in agents_detailed
            ]
        }


async def example_academic_research():
    """Example: Academic research topic with theory-first approach."""
    integration = SequenceGeneratorIntegration()
    
    research_topic = """
    Conduct a comprehensive systematic literature review on explainable artificial intelligence (XAI) 
    methods in medical diagnosis applications. The research should analyze peer-reviewed publications 
    from 2020-2024, focusing on interpretability techniques for deep learning models used in medical 
    image analysis, clinical decision support systems, and diagnostic prediction models. The review 
    should evaluate the effectiveness of different XAI approaches, identify research gaps, and provide 
    recommendations for future development of transparent AI systems in healthcare.
    """
    
    
    # Analyze topic characteristics
    analysis_result = integration.analyze_topic_and_recommend_strategy(research_topic)
    
    for i, strategy in enumerate(analysis_result['strategy_recommendations'][:3], 1):
        pass


async def example_market_analysis():
    """Example: Market analysis topic with market-first approach."""
    integration = SequenceGeneratorIntegration()
    
    research_topic = """
    Conduct a comprehensive market opportunity analysis for AI-powered customer service solutions 
    in the financial services sector. Evaluate market size, growth projections, competitive landscape, 
    and customer adoption patterns. Analyze key players, pricing strategies, and differentiation 
    opportunities. Assess regulatory considerations, integration challenges, and ROI potential for 
    financial institutions. Provide strategic recommendations for market entry timing, target customer 
    segments, and go-to-market approaches for AI customer service platforms.
    """
    
    
    # Analyze and execute optimal sequence
    execution_result = await integration.generate_and_execute_optimal_sequence(
        research_topic=research_topic,
        num_sequence_options=3
    )
    
    if execution_result["success"]:
        execution_result["sequence_info"]
        
        execution_result["execution_results"]
    else:
        pass


async def main():
    """Run example demonstrations of SequenceGenerator integration."""
    # Show available agents
    integration = SequenceGeneratorIntegration()
    agent_summary = integration.get_available_agents_summary()
    
    for agent in agent_summary['agents'][:5]:  # Show first 5
        pass
    
    # Run examples
    try:
        await example_academic_research()
        await example_market_analysis()
        
    except Exception as e:
        logger.error("Example execution failed: %s", e)


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())