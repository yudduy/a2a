#!/usr/bin/env python3
"""Test script for validating sequential agent ordering optimization imports.

This script validates that all sequencing framework components can be imported
and that the integration with OpenDeepResearch works correctly.
"""

import sys
import traceback
from typing import Dict, List, Any

def test_basic_imports() -> Dict[str, Any]:
    """Test basic module imports for the sequencing framework."""
    results = {
        "status": "success",
        "errors": [],
        "imported_modules": [],
        "missing_modules": []
    }
    
    try:
        # Test core sequencing imports
        print("Testing core sequencing imports...")
        
        from open_deep_research.sequencing import (
            SequenceOptimizationEngine,
            SEQUENCE_PATTERNS,
            THEORY_FIRST_PATTERN,
            MARKET_FIRST_PATTERN,
            FUTURE_BACK_PATTERN
        )
        results["imported_modules"].extend([
            "SequenceOptimizationEngine",
            "SEQUENCE_PATTERNS", 
            "THEORY_FIRST_PATTERN",
            "MARKET_FIRST_PATTERN",
            "FUTURE_BACK_PATTERN"
        ])
        print("✓ Core engine imports successful")
        
        # Test model imports
        from open_deep_research.sequencing import (
            AgentType,
            SequencePattern,
            SequenceResult,
            SequenceStrategy,
            ToolProductivityMetrics,
            InsightTransition
        )
        results["imported_modules"].extend([
            "AgentType",
            "SequencePattern",
            "SequenceResult", 
            "SequenceStrategy",
            "ToolProductivityMetrics",
            "InsightTransition"
        ])
        print("✓ Model imports successful")
        
        # Test specialized agent imports
        from open_deep_research.sequencing import (
            AcademicAgent,
            IndustryAgent,
            TechnicalTrendsAgent,
            SpecializedAgent,
            ResearchContext
        )
        results["imported_modules"].extend([
            "AcademicAgent",
            "IndustryAgent",
            "TechnicalTrendsAgent",
            "SpecializedAgent",
            "ResearchContext"
        ])
        print("✓ Specialized agent imports successful")
        
        # Test director and metrics imports
        from open_deep_research.sequencing import (
            SupervisorResearchDirector,
            MetricsCalculator
        )
        results["imported_modules"].extend([
            "SupervisorResearchDirector",
            "MetricsCalculator"
        ])
        print("✓ Director and metrics imports successful")
        
        # Test configuration imports
        from open_deep_research.configuration import Configuration
        results["imported_modules"].append("Configuration")
        print("✓ Configuration import successful")
        
        print(f"✓ All {len(results['imported_modules'])} modules imported successfully")
        
    except ImportError as e:
        error_msg = f"Import error: {str(e)}"
        results["errors"].append(error_msg)
        results["status"] = "failed"
        print(f"✗ Import failed: {error_msg}")
        traceback.print_exc()
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        results["errors"].append(error_msg)
        results["status"] = "failed"
        print(f"✗ Unexpected error: {error_msg}")
        traceback.print_exc()
        
    return results

def test_sequence_patterns() -> Dict[str, Any]:
    """Test that sequence patterns are correctly defined."""
    results = {
        "status": "success",
        "errors": [],
        "patterns_tested": []
    }
    
    try:
        from open_deep_research.sequencing import SEQUENCE_PATTERNS, SequenceStrategy
        
        print("\nTesting sequence patterns...")
        
        # Verify all three patterns exist using enum values
        expected_strategies = [
            SequenceStrategy.THEORY_FIRST,
            SequenceStrategy.MARKET_FIRST,
            SequenceStrategy.FUTURE_BACK
        ]
        
        for strategy in expected_strategies:
            if strategy in SEQUENCE_PATTERNS:
                pattern = SEQUENCE_PATTERNS[strategy]
                results["patterns_tested"].append({
                    "strategy": strategy.value,
                    "description": pattern.description,
                    "sequence_length": len(pattern.agent_order),
                    "agents": [str(agent.value) for agent in pattern.agent_order]
                })
                print(f"✓ {strategy.value}: {len(pattern.agent_order)} agents")
                print(f"   Order: {[agent.value for agent in pattern.agent_order]}")
            else:
                error_msg = f"Missing sequence pattern: {strategy.value}"
                results["errors"].append(error_msg)
                results["status"] = "failed"
                print(f"✗ {error_msg}")
                
        if len(results["patterns_tested"]) == 3:
            print("✓ All sequence patterns validated")
        
    except Exception as e:
        error_msg = f"Pattern validation error: {str(e)}"
        results["errors"].append(error_msg)
        results["status"] = "failed"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        
    return results

def test_configuration_fields() -> Dict[str, Any]:
    """Test that configuration fields are properly defined."""
    results = {
        "status": "success", 
        "errors": [],
        "fields_found": []
    }
    
    try:
        from open_deep_research.configuration import Configuration
        
        print("\nTesting configuration fields...")
        
        # Test that new fields exist
        config = Configuration()
        
        expected_fields = [
            "enable_sequence_optimization",
            "sequence_strategy", 
            "compare_all_sequences",
            "sequence_variance_threshold"
        ]
        
        for field_name in expected_fields:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                results["fields_found"].append({
                    "field": field_name,
                    "default_value": value,
                    "type": type(value).__name__
                })
                print(f"✓ {field_name}: {value} ({type(value).__name__})")
            else:
                error_msg = f"Missing configuration field: {field_name}"
                results["errors"].append(error_msg)
                results["status"] = "failed" 
                print(f"✗ {error_msg}")
                
        if len(results["fields_found"]) == 4:
            print("✓ All configuration fields validated")
            
    except Exception as e:
        error_msg = f"Configuration validation error: {str(e)}"
        results["errors"].append(error_msg)
        results["status"] = "failed"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        
    return results

def main():
    """Run all import tests."""
    print("Sequential Agent Ordering Optimization Framework - Import Tests")
    print("=" * 70)
    
    # Test results collection
    all_results = {}
    
    # Run tests
    all_results["basic_imports"] = test_basic_imports()
    all_results["sequence_patterns"] = test_sequence_patterns() 
    all_results["configuration_fields"] = test_configuration_fields()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
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
        print("🎉 All import tests PASSED!")
        return 0
    else:
        print("❌ Some import tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())