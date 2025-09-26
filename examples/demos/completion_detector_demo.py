#!/usr/bin/env python3
"""Demonstration script for CompletionDetector integration with sequential multi-agent workflows.

This script shows how the completion detector works with agent registry completion indicators
and provides examples of robust completion detection in production scenarios.
"""

import logging

from completion_detector import CompletionDetector, CompletionPattern, DetectionStrategy
from langchain_core.messages import AIMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_completion_detection():
    """Demonstrate basic completion detection functionality."""
    detector = CompletionDetector(debug_mode=False)
    
    # Test messages with varying completion confidence
    test_messages = [
        ("Strong completion signal", 
         "The research is complete. All findings have been summarized and the investigation has concluded."),
        
        ("Conclusion with summary", 
         "In conclusion, after extensive analysis of the market trends and technical feasibility, "
         "I can summarize that the technology shows strong potential. The research objectives have been met."),
        
        ("Partial work indication", 
         "I'm still analyzing the data and need to conduct additional searches to gather more information."),
        
        ("Handoff readiness", 
         "The analysis is ready for handoff to the next phase. All key aspects have been covered."),
        
        ("Work completion", 
         "The comprehensive analysis has been completed. No more sources are available for this topic."),
    ]
    
    for description, content in test_messages:
        message = AIMessage(content=content)
        
        # Analyze with combined strategy
        detector.analyze_completion_patterns(message)
        


def demo_custom_agent_indicators():
    """Demonstrate custom completion indicators per agent type."""
    detector = CompletionDetector()
    
    # Simulate different agent types with custom completion indicators
    agent_scenarios = [
        {
            "agent_type": "academic_researcher",
            "custom_indicators": [
                r"literature\s+review\s+complete",
                r"theoretical\s+analysis\s+concluded",
                r"research\s+gaps\s+identified"
            ],
            "message": "The literature review is complete and all theoretical analysis has been concluded. "
                      "Key research gaps have been identified for future investigation."
        },
        {
            "agent_type": "market_analyst", 
            "custom_indicators": [
                r"market\s+assessment\s+finished",
                r"competitive\s+analysis\s+done",
                r"business\s+case\s+evaluated"
            ],
            "message": "Market assessment is finished with comprehensive competitive analysis done. "
                      "The business case has been thoroughly evaluated."
        },
        {
            "agent_type": "technical_specialist",
            "custom_indicators": [
                r"implementation\s+feasibility\s+assessed",
                r"technical\s+constraints\s+analyzed",
                r"architecture\s+review\s+complete"
            ],
            "message": "Implementation feasibility has been assessed and technical constraints analyzed. "
                      "The architecture review is complete with recommendations."
        }
    ]
    
    for scenario in agent_scenarios:
        message = AIMessage(content=scenario["message"])
        
        # Test without custom indicators
        detector.analyze_completion_patterns(message)
        
        # Test with custom indicators
        detector.analyze_completion_patterns(
            message, 
            custom_indicators=scenario["custom_indicators"]
        )
        


def demo_detection_strategies():
    """Demonstrate different detection strategies and their effectiveness."""
    detector = CompletionDetector()
    
    # Test message with mixed signals
    test_content = """
    After conducting extensive research and analysis, I have gathered comprehensive data 
    on the market trends. The investigation has been thorough and systematic.
    
    In conclusion, the findings reveal significant opportunities in the emerging technology 
    sector. The analysis is complete and all relevant sources have been consulted.
    
    Final thoughts: The research objectives have been met and the deliverable is ready 
    for the next phase of the project.
    """
    
    message = AIMessage(content=test_content)
    
    strategies = [
        DetectionStrategy.CONTENT_PATTERNS,
        DetectionStrategy.TOOL_USAGE_PATTERNS, 
        DetectionStrategy.MESSAGE_STRUCTURE,
        DetectionStrategy.COMBINED
    ]
    
    
    for strategy in strategies:
        detector.analyze_completion_patterns(message, strategy=strategy)
        


def demo_agent_registry_integration():
    """Demonstrate integration with agent registry completion indicators."""
    # This would normally load from actual agent files, but we'll simulate
    mock_agent_configs = {
        "research_specialist": {
            "name": "research_specialist",
            "completion_indicators": [
                "research methodology established",
                "data collection complete", 
                "analysis framework ready"
            ]
        },
        "synthesis_agent": {
            "name": "synthesis_agent", 
            "completion_indicators": [
                "synthesis complete",
                "insights consolidated",
                "final report ready"
            ]
        }
    }
    
    detector = CompletionDetector()
    
    test_cases = [
        {
            "agent": "research_specialist",
            "message": "The research methodology has been established and data collection is complete. "
                      "The analysis framework is ready for the next phase."
        },
        {
            "agent": "synthesis_agent",
            "message": "Synthesis is complete with all insights consolidated. The final report is ready "
                      "for review and distribution."
        }
    ]
    
    for case in test_cases:
        agent_config = mock_agent_configs[case["agent"]]
        message = AIMessage(content=case["message"])
        
        # Use agent-specific completion indicators
        detector.analyze_completion_patterns(
            message,
            custom_indicators=agent_config["completion_indicators"]
        )
        


def demo_production_usage():
    """Demonstrate production-ready usage patterns."""
    detector = CompletionDetector()
    
    # Add custom pattern for production environment
    detector.add_custom_pattern(CompletionPattern(
        pattern=r"handoff\s+to\s+next\s+agent",
        weight=0.8,
        description="Explicit agent handoff signal"
    ))
    
    # Adjust threshold for production requirements
    detector.set_completion_threshold(0.6)  # Higher threshold for production
    
    
    # Simulate production workflow
    workflow_messages = [
        "Starting research on market analysis for AI technologies.",
        "Conducting searches and gathering relevant data sources.",
        "Analysis in progress with preliminary findings emerging.",
        "Research complete. Comprehensive analysis finished with all findings summarized. "
        "Ready for handoff to next agent.",
        "Next agent can proceed with the implementation phase."
    ]
    
    for i, content in enumerate(workflow_messages, 1):
        message = AIMessage(content=content)
        detector.analyze_completion_patterns(message)
        
    


def main():
    """Run all demonstration scenarios."""
    demo_basic_completion_detection()
    demo_custom_agent_indicators()
    demo_detection_strategies()
    demo_agent_registry_integration()
    demo_production_usage()
    


if __name__ == "__main__":
    main()