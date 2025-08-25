"""Simple demonstration of SequenceGenerator functionality.

This script shows how the SequenceGenerator analyzes topics and creates
optimal agent sequences without requiring the full supervisor workflow.
"""

import logging
from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.core.sequence_generator import UnifiedSequenceGenerator, TopicType, SequenceGenerationInput, AgentCapability

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_topic_analysis():
    """Demonstrate topic analysis capabilities."""
    print("=== Topic Analysis Demonstration ===")
    
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
        print(f"\n--- {topic_type} Topic ---")
        print(f"Topic: {topic.strip()[:100]}...")
        
        # Analyze topic characteristics
        analysis = generator.analyze_topic_characteristics(topic)
        
        print(f"Detected Type: {analysis.topic_type.value}")
        print(f"Complexity Score: {analysis.complexity_score:.2f}")
        print(f"Scope Breadth: {analysis.scope_breadth:.2f}")
        print(f"Estimated Agents Needed: {analysis.estimated_agents_needed}")
        print(f"Priority Areas: {', '.join(analysis.priority_areas)}")
        print(f"Market Relevance: {analysis.market_relevance:.2f}")
        print(f"Technical Complexity: {analysis.technical_complexity:.2f}")
        print(f"Keywords: {', '.join(analysis.keywords[:8])}...")
        
        # Show domain confidence scores
        print("Domain Indicators:")
        for domain, score in analysis.domain_indicators.items():
            if score > 0.1:  # Only show significant indicators
                print(f"  {domain}: {score:.2f}")


def demonstrate_sequence_generation():
    """Demonstrate sequence generation for different topic types."""
    print("\n=== Sequence Generation Demonstration ===")
    
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
    
    print(f"Research Topic: {research_topic.strip()}")
    print()
    
    try:
        # Generate multiple sequence options
        sequences = generator.generate_sequences(
            research_topic=research_topic,
            num_sequences=4
        )
        
        print(f"Generated {len(sequences)} sequence options:")
        print()
        
        for i, sequence in enumerate(sequences, 1):
            print(f"{i}. {sequence.sequence_name}")
            print(f"   Strategy: {sequence.strategy.value}")
            print(f"   Agents: {' → '.join(sequence.agents)}")
            print(f"   Score: {sequence.score:.3f} (confidence: {sequence.confidence:.2f})")
            print(f"   Rationale: {sequence.rationale}")
            print(f"   Estimated Duration: {sequence.estimated_duration:.1f} hours")
            
            # Show detailed scoring
            print(f"   Scoring Details:")
            print(f"     Topic Fit: {sequence.topic_fit_score:.2f}")
            print(f"     Coverage: {sequence.coverage_score:.2f}")
            print(f"     Efficiency: {sequence.efficiency_score:.2f}")
            print(f"     Expertise Match: {sequence.expertise_match_score:.2f}")
            print()
            
    except Exception as e:
        print(f"Error generating sequences: {e}")
        logger.error("Sequence generation failed", exc_info=True)


def demonstrate_strategy_comparison():
    """Demonstrate how different strategies work for the same topic."""
    print("=== Strategy Comparison Demonstration ===")
    
    # Initialize components
    registry = AgentRegistry()
    generator = UnifiedSequenceGenerator(registry, debug_mode=False)
    
    # Business-focused research topic
    business_topic = """
    Market entry strategy analysis for AI-powered educational technology platforms
    in emerging markets. Evaluate competitive landscape, assess customer needs,
    analyze regulatory requirements, and develop go-to-market recommendations.
    """
    
    print(f"Topic: {business_topic.strip()}")
    print()
    
    # Generate sequences with specific strategies
    from open_deep_research.orchestration.sequence_generator import SequenceStrategy
    
    strategies_to_test = [
        SequenceStrategy.MARKET_FIRST,
        SequenceStrategy.THEORY_FIRST, 
        SequenceStrategy.TECHNICAL_FIRST
    ]
    
    available_agents = registry.list_agents()
    topic_analysis = generator.analyze_topic_characteristics(business_topic)
    
    print("Strategy Comparison Results:")
    
    for strategy in strategies_to_test:
        try:
            if strategy == SequenceStrategy.MARKET_FIRST:
                agents = generator.create_market_first_sequence(available_agents, topic_analysis)
            elif strategy == SequenceStrategy.THEORY_FIRST:
                agents = generator.create_theory_first_sequence(available_agents, topic_analysis)
            elif strategy == SequenceStrategy.TECHNICAL_FIRST:
                agents = generator.create_technical_first_sequence(available_agents, topic_analysis)
            
            print(f"\n{strategy.value.replace('_', ' ').title()} Strategy:")
            print(f"  Agents: {' → '.join(agents) if agents else 'None available'}")
            print(f"  Agent Count: {len(agents)}")
            
            # Show rationale for each strategy
            strategy_rationales = {
                SequenceStrategy.MARKET_FIRST: "Start with market opportunity assessment, then validate with technical and research insights",
                SequenceStrategy.THEORY_FIRST: "Build academic foundation first, then apply market and technical analysis",
                SequenceStrategy.TECHNICAL_FIRST: "Assess technical feasibility first, then validate with research and market analysis"
            }
            
            print(f"  Rationale: {strategy_rationales.get(strategy, 'Custom strategy')}")
            
        except Exception as e:
            print(f"\n{strategy.value} Strategy: Error - {e}")


def show_agent_registry_info():
    """Display information about available agents."""
    print("=== Agent Registry Information ===")
    
    registry = AgentRegistry()
    stats = registry.get_registry_stats()
    
    print(f"Total Agents: {stats['total_agents']}")
    print(f"Project Agents: {stats['project_agents']}")
    print(f"User Agents: {stats['user_agents']}")
    print(f"Project Directory: {stats['project_agents_dir']}")
    print(f"User Directory: {stats['user_agents_dir']}")
    print()
    
    # Show available agents
    agents = registry.list_agents_detailed()
    if agents:
        print("Available Agents:")
        for agent in agents:
            expertise_summary = ', '.join(agent['expertise_areas'][:3])
            if len(agent['expertise_areas']) > 3:
                expertise_summary += f" (+ {len(agent['expertise_areas']) - 3} more)"
            
            print(f"  • {agent['name']}: {expertise_summary}")
    else:
        print("No agents found in registry.")
        print("To use the SequenceGenerator, create agent definitions in:")
        print(f"  {stats['project_agents_dir']}")


def main():
    """Run all demonstrations."""
    print("SequenceGenerator Demonstration")
    print("=" * 50)
    
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
            print("\nSkipping demonstrations - no agents available.")
            print("Create agent definitions in the project agents directory to run full demonstrations.")
    
    except Exception as e:
        logger.error("Demonstration failed", exc_info=True)
        print(f"\nDemonstration error: {e}")


if __name__ == "__main__":
    main()