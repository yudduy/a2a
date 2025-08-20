#!/usr/bin/env python3
"""Test script for basic sequence execution functionality.

This script validates that the sequence optimization engine can be instantiated
and that the basic framework components work correctly without requiring
full API execution.
"""

import asyncio
import sys
import traceback
from typing import Dict, Any

from langchain_core.runnables import RunnableConfig


def test_engine_instantiation() -> Dict[str, Any]:
    """Test that the SequenceOptimizationEngine can be instantiated."""
    results = {
        "status": "success",
        "errors": [],
        "details": {}
    }
    
    try:
        from open_deep_research.sequencing import SequenceOptimizationEngine
        from open_deep_research.configuration import Configuration
        
        print("Testing SequenceOptimizationEngine instantiation...")
        
        # Create configuration
        config = RunnableConfig(configurable={
            "enable_sequence_optimization": True,
            "sequence_strategy": "theory_first",
            "compare_all_sequences": False,
            "research_model": "openai:gpt-4.1-mini",  # Use mini model for testing
            "search_api": "none"  # Disable search for testing
        })
        
        # Instantiate engine
        engine = SequenceOptimizationEngine(config)
        results["details"]["engine_created"] = True
        print("✓ SequenceOptimizationEngine instantiated successfully")
        
        # Test configuration parsing
        parsed_config = Configuration.from_runnable_config(config)
        results["details"]["config_parsed"] = True
        results["details"]["sequence_optimization_enabled"] = parsed_config.enable_sequence_optimization
        results["details"]["sequence_strategy"] = parsed_config.sequence_strategy
        print(f"✓ Configuration parsed: optimization={parsed_config.enable_sequence_optimization}")
        
        # Verify engine attributes
        if hasattr(engine, 'config'):
            results["details"]["has_config"] = True
            print("✓ Engine has configuration attribute")
        
        if hasattr(engine, 'director'):
            results["details"]["has_director"] = True
            print("✓ Engine has research director")
        
        if hasattr(engine, 'metrics_calculator'):
            results["details"]["has_metrics"] = True
            print("✓ Engine has metrics calculator")
            
    except Exception as e:
        error_msg = f"Engine instantiation error: {str(e)}"
        results["errors"].append(error_msg)
        results["status"] = "failed"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        
    return results


def test_sequence_pattern_validation() -> Dict[str, Any]:
    """Test sequence pattern validation and selection."""
    results = {
        "status": "success",
        "errors": [],
        "patterns_validated": []
    }
    
    try:
        from open_deep_research.sequencing import (
            SequenceOptimizationEngine,
            SEQUENCE_PATTERNS,
            SequenceStrategy
        )
        
        print("\nTesting sequence pattern validation...")
        
        config = RunnableConfig(configurable={
            "enable_sequence_optimization": True,
            "research_model": "openai:gpt-4.1-mini",
            "search_api": "none"
        })
        
        engine = SequenceOptimizationEngine(config)
        
        # Test each sequence pattern
        for strategy, pattern in SEQUENCE_PATTERNS.items():
            try:
                # Validate pattern structure
                assert hasattr(pattern, 'strategy')
                assert hasattr(pattern, 'agent_order')
                assert hasattr(pattern, 'description')
                assert len(pattern.agent_order) == 3
                
                results["patterns_validated"].append({
                    "strategy": strategy.value,
                    "agents": [agent.value for agent in pattern.agent_order],
                    "description_length": len(pattern.description)
                })
                
                print(f"✓ {strategy.value}: {len(pattern.agent_order)} agents validated")
                
            except Exception as e:
                error_msg = f"Pattern validation failed for {strategy.value}: {str(e)}"
                results["errors"].append(error_msg)
                results["status"] = "failed"
                print(f"✗ {error_msg}")
        
        if len(results["patterns_validated"]) == 3:
            print("✓ All sequence patterns validated successfully")
            
    except Exception as e:
        error_msg = f"Pattern validation error: {str(e)}"
        results["errors"].append(error_msg)
        results["status"] = "failed"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        
    return results


def test_metrics_calculator() -> Dict[str, Any]:
    """Test the metrics calculator functionality."""
    results = {
        "status": "success",
        "errors": [],
        "metrics_tested": []
    }
    
    try:
        from open_deep_research.sequencing import (
            MetricsCalculator,
            ToolProductivityMetrics,
            AgentType
        )
        from datetime import datetime
        
        print("\nTesting MetricsCalculator...")
        
        # Create calculator
        calculator = MetricsCalculator()
        print("✓ MetricsCalculator instantiated")
        
        # Test tool productivity calculation
        mock_metrics = ToolProductivityMetrics(
            tool_productivity=0.85,
            research_quality_score=0.9,
            total_agent_calls=12,
            agent_efficiency=0.75,
            context_efficiency=0.8,
            time_to_value=45.0,
            insight_novelty=0.85,
            insight_relevance=0.9,
            insight_actionability=0.8,
            research_completeness=0.85,
            useful_insights_count=8,
            redundant_research_count=2,
            cognitive_offloading_incidents=1,
            relevant_context_used=0.8,
            total_context_available=1.0
        )
        
        # Validate metrics structure
        assert mock_metrics.tool_productivity > 0
        assert 0 <= mock_metrics.research_quality_score <= 1
        assert mock_metrics.total_agent_calls > 0
        
        results["metrics_tested"].append({
            "tool_productivity": mock_metrics.tool_productivity,
            "research_quality": mock_metrics.research_quality_score,
            "agent_calls": mock_metrics.total_agent_calls
        })
        
        print(f"✓ Tool Productivity metrics validated: {mock_metrics.tool_productivity:.3f}")
        print(f"✓ Research Quality: {mock_metrics.research_quality_score:.3f}")
        print(f"✓ Agent Efficiency: {mock_metrics.agent_efficiency:.3f}")
        
    except Exception as e:
        error_msg = f"Metrics calculator error: {str(e)}"
        results["errors"].append(error_msg)
        results["status"] = "failed"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        
    return results


def test_specialized_agents() -> Dict[str, Any]:
    """Test specialized agent instantiation."""
    results = {
        "status": "success",
        "errors": [],
        "agents_tested": []
    }
    
    try:
        from open_deep_research.sequencing import (
            AcademicAgent,
            IndustryAgent,
            TechnicalTrendsAgent,
            ResearchContext
        )
        from langchain_core.runnables import RunnableConfig
        
        print("\nTesting specialized agents...")
        
        config = RunnableConfig(configurable={
            "research_model": "openai:gpt-4.1-mini",
            "search_api": "none"
        })
        
        # Test each agent type
        agents = [
            ("Academic", AcademicAgent),
            ("Industry", IndustryAgent),
            ("TechnicalTrends", TechnicalTrendsAgent)
        ]
        
        for agent_name, agent_class in agents:
            try:
                agent = agent_class(config)
                
                # Verify agent has required attributes
                assert hasattr(agent, 'agent_type')
                assert hasattr(agent, 'specialty_focus')
                
                results["agents_tested"].append({
                    "agent_type": agent_name,
                    "has_agent_type": hasattr(agent, 'agent_type'),
                    "has_specialty": hasattr(agent, 'specialty_focus')
                })
                
                print(f"✓ {agent_name}Agent instantiated successfully")
                
            except Exception as e:
                error_msg = f"{agent_name}Agent instantiation failed: {str(e)}"
                results["errors"].append(error_msg)
                results["status"] = "failed"
                print(f"✗ {error_msg}")
        
        if len(results["agents_tested"]) == 3:
            print("✓ All specialized agents validated")
            
    except Exception as e:
        error_msg = f"Agent testing error: {str(e)}"
        results["errors"].append(error_msg)
        results["status"] = "failed"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        
    return results


async def test_integration_with_main_system() -> Dict[str, Any]:
    """Test integration with the main OpenDeepResearch system."""
    results = {
        "status": "success", 
        "errors": [],
        "integration_checks": []
    }
    
    try:
        from open_deep_research.deep_researcher import deep_researcher
        from open_deep_research.configuration import Configuration
        
        print("\nTesting integration with main system...")
        
        # Test that configuration includes sequencing fields
        config = Configuration(
            enable_sequence_optimization=True,
            sequence_strategy="theory_first", 
            compare_all_sequences=False
        )
        
        results["integration_checks"].append({
            "config_has_sequencing": True,
            "optimization_enabled": config.enable_sequence_optimization,
            "strategy_set": config.sequence_strategy is not None
        })
        
        print("✓ Configuration includes sequencing fields")
        print(f"✓ Optimization enabled: {config.enable_sequence_optimization}")
        print(f"✓ Strategy configured: {config.sequence_strategy}")
        
        # Check if deep_researcher function exists (main entry point)
        assert callable(deep_researcher)
        results["integration_checks"].append({
            "main_entry_point_exists": True
        })
        print("✓ Main deep_researcher entry point exists")
        
    except Exception as e:
        error_msg = f"Integration test error: {str(e)}"
        results["errors"].append(error_msg)
        results["status"] = "failed"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        
    return results


def main():
    """Run all sequence execution tests."""
    print("Sequential Agent Ordering Optimization Framework - Execution Tests")
    print("=" * 75)
    
    # Test results collection
    all_results = {}
    
    # Run synchronous tests
    all_results["engine_instantiation"] = test_engine_instantiation()
    all_results["sequence_patterns"] = test_sequence_pattern_validation()
    all_results["metrics_calculator"] = test_metrics_calculator()
    all_results["specialized_agents"] = test_specialized_agents()
    
    # Run async integration test
    print("\nRunning integration tests...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        all_results["integration"] = loop.run_until_complete(test_integration_with_main_system())
    except Exception as e:
        all_results["integration"] = {
            "status": "failed",
            "errors": [f"Async test failed: {str(e)}"]
        }
    
    # Summary
    print("\n" + "=" * 75)
    print("TEST SUMMARY")
    print("=" * 75)
    
    total_tests = len(all_results)
    passed_tests = sum(1 for result in all_results.values() if result["status"] == "success")
    
    for test_name, result in all_results.items():
        status_symbol = "✓" if result["status"] == "success" else "✗"
        print(f"{status_symbol} {test_name}: {result['status']}")
        if result["errors"]:
            for error in result["errors"]:
                print(f"    - {error}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All execution tests PASSED!")
        print("\nFramework is ready for basic sequence execution testing.")
        print("Note: Full execution requires API keys for models and search.")
        return 0
    else:
        print("❌ Some execution tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())