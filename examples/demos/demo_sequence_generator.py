"""Simple demonstration of SequenceGenerator functionality.

This script shows how the SequenceGenerator analyzes topics and creates
optimal agent sequences without requiring the full supervisor workflow.
"""

import logging

from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.core.sequence_generator import (
    UnifiedSequenceGenerator,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_topic_analysis():
    """Demonstrate topic analysis capabilities."""
    # Initialize components
    registry = AgentRegistry()
    generator = UnifiedSequenceGenerator(registry, debug_mode=False)
    
    # Test different types of research topics
    test_topics = {
        "Academic Research": """
        Conduct a systematic literature review on machine learning interpretability methods
        in healthcare applications. The research should analyze peer-reviewed publications,
        evaluate different explainable AI techniques, and identify research gaps in the field.
        """,
        
        "Market Analysis": """
        Analyze the market opportunity for AI-powered customer service solutions in fintech.
        Evaluate competitive landscape, pricing strategies, customer segments, and revenue
        potential. Assess market size, growth projections, and go-to-market strategies.
        """,
        
        "Technical Evaluation": """
        Design a scalable microservices architecture for real-time data processing systems.
        Evaluate technology stack options, assess performance requirements, and define
        implementation strategies for high-throughput distributed computing platforms.
        """,
        
        "Complex Multi-Domain": """
        Comprehensive analysis of blockchain technology adoption in supply chain management.
        Research academic literature, analyze market trends, evaluate technical implementation
        challenges, and assess business model implications across different industries.
        """
    }
    
    for topic_type, topic in test_topics.items():
        
        # Analyze topic characteristics
        analysis = generator.analyze_topic_characteristics(topic)
        
        
        # Show domain confidence scores
        for domain, score in analysis.domain_indicators.items():
            if score > 0.1:  # Only show significant indicators
                pass


def demonstrate_sequence_generation():
    """Demonstrate sequence generation for different topic types."""
    # Initialize components
    registry = AgentRegistry()
    generator = UnifiedSequenceGenerator(registry, debug_mode=False)
    
    # Test topic
    research_topic = """
    Analyze the impact of artificial intelligence on healthcare delivery systems.
    Research current applications, evaluate effectiveness, assess implementation
    challenges, analyze market opportunities, and provide strategic recommendations
    for healthcare organizations considering AI adoption.
    """
    
    
    try:
        # Generate multiple sequence options
        sequences = generator.generate_sequences(
            research_topic=research_topic,
            num_sequences=4
        )
        
        
        for i, sequence in enumerate(sequences, 1):
            
            # Show detailed scoring
            pass
            
    except Exception:
        logger.error("Sequence generation failed", exc_info=True)


def demonstrate_strategy_comparison():
    """Demonstrate how different strategies work for the same topic."""
    # Initialize components
    registry = AgentRegistry()
    generator = UnifiedSequenceGenerator(registry, debug_mode=False)
    
    # Business-focused research topic
    business_topic = """
    Market entry strategy analysis for AI-powered educational technology platforms
    in emerging markets. Evaluate competitive landscape, assess customer needs,
    analyze regulatory requirements, and develop go-to-market recommendations.
    """
    
    
    # Generate sequences with specific strategies
    from open_deep_research.orchestration.sequence_generator import SequenceStrategy
    
    strategies_to_test = [
        SequenceStrategy.MARKET_FIRST,
        SequenceStrategy.THEORY_FIRST, 
        SequenceStrategy.TECHNICAL_FIRST
    ]
    
    available_agents = registry.list_agents()
    topic_analysis = generator.analyze_topic_characteristics(business_topic)
    
    
    for strategy in strategies_to_test:
        try:
            if strategy == SequenceStrategy.MARKET_FIRST:
                generator.create_market_first_sequence(available_agents, topic_analysis)
            elif strategy == SequenceStrategy.THEORY_FIRST:
                generator.create_theory_first_sequence(available_agents, topic_analysis)
            elif strategy == SequenceStrategy.TECHNICAL_FIRST:
                generator.create_technical_first_sequence(available_agents, topic_analysis)
            
            
            # Show rationale for each strategy
            
            
        except Exception:
            pass


def show_agent_registry_info():
    """Display information about available agents."""
    registry = AgentRegistry()
    registry.get_registry_stats()
    
    
    # Show available agents
    agents = registry.list_agents_detailed()
    if agents:
        for agent in agents:
            expertise_summary = ', '.join(agent['expertise_areas'][:3])
            if len(agent['expertise_areas']) > 3:
                expertise_summary += f" (+ {len(agent['expertise_areas']) - 3} more)"
            
    else:
        pass


def main():
    """Run all demonstrations."""
    try:
        # Show registry information first
        show_agent_registry_info()
        
        # Run demonstrations only if agents are available
        registry = AgentRegistry()
        if registry.list_agents():
            demonstrate_topic_analysis()
            demonstrate_sequence_generation()
            demonstrate_strategy_comparison()
        else:
            pass
    
    except Exception:
        logger.error("Demonstration failed", exc_info=True)


if __name__ == "__main__":
    main()