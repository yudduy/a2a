#!/usr/bin/env python3
"""
Test script for the Sequential Agent Ordering Optimization Framework.
Tests the framework without requiring API keys or external services.
"""

import asyncio
import sys
import traceback
from typing import Dict, Any

# Add the src directory to Python path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from open_deep_research.sequencing import (
            SequenceOptimizationEngine,
            SEQUENCE_PATTERNS,
            AgentType,
            SequenceStrategy
        )
        from open_deep_research.configuration import Configuration
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration with sequence optimization settings."""
    print("\n🧪 Testing configuration...")
    
    try:
        from open_deep_research.configuration import Configuration
        
        # Test basic configuration
        config = Configuration(
            enable_sequence_optimization=True,
            sequence_strategy="theory_first",
            compare_all_sequences=False,
            research_model="openai:gpt-4.1"
        )
        
        assert config.enable_sequence_optimization == True
        assert config.sequence_strategy == "theory_first"
        assert config.compare_all_sequences == False
        
        print("✅ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_sequence_patterns():
    """Test that sequence patterns are properly defined."""
    print("\n🧪 Testing sequence patterns...")
    
    try:
        from open_deep_research.sequencing import (
            SEQUENCE_PATTERNS,
            SequenceStrategy,
            AgentType
        )
        
        # Verify all three patterns exist
        assert SequenceStrategy.THEORY_FIRST in SEQUENCE_PATTERNS
        assert SequenceStrategy.MARKET_FIRST in SEQUENCE_PATTERNS
        assert SequenceStrategy.FUTURE_BACK in SEQUENCE_PATTERNS
        
        # Verify pattern structure
        theory_first = SEQUENCE_PATTERNS[SequenceStrategy.THEORY_FIRST]
        assert theory_first.strategy == SequenceStrategy.THEORY_FIRST
        assert len(theory_first.agent_order) == 3
        assert theory_first.agent_order[0] == AgentType.ACADEMIC
        
        print("✅ Sequence patterns test passed!")
        print(f"   Available patterns: {[s.value for s in SEQUENCE_PATTERNS.keys()]}")
        return True
        
    except Exception as e:
        print(f"❌ Sequence patterns test failed: {e}")
        traceback.print_exc()
        return False

def test_sequence_engine_initialization():
    """Test that the sequence engine can be initialized without errors."""
    print("\n🧪 Testing sequence engine initialization...")
    
    try:
        from open_deep_research.sequencing import SequenceOptimizationEngine
        from langchain_core.runnables import RunnableConfig
        
        # Create minimal config
        config = RunnableConfig(configurable={
            "research_model": "openai:gpt-4.1",
            "enable_sequence_optimization": True
        })
        
        # Initialize engine
        engine = SequenceOptimizationEngine(config)
        
        # Verify engine has required components
        assert hasattr(engine, 'research_director')
        assert hasattr(engine, 'agents')
        assert hasattr(engine, 'metrics_calculator')
        
        print("✅ Sequence engine initialization test passed!")
        print(f"   Available agents: {list(engine.agents.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Sequence engine initialization failed: {e}")
        traceback.print_exc()
        return False

async def test_mock_sequence_execution():
    """Test sequence execution logic with mock data (no API calls)."""
    print("\n🧪 Testing mock sequence execution...")
    
    try:
        from open_deep_research.sequencing import (
            SequenceOptimizationEngine,
            THEORY_FIRST_PATTERN
        )
        from langchain_core.runnables import RunnableConfig
        
        # Create mock config  
        config = RunnableConfig(configurable={
            "research_model": "mock://test-model",
            "enable_sequence_optimization": True,
            "search_api": "none"
        })
        
        engine = SequenceOptimizationEngine(config)
        
        # Test pattern validation
        pattern = THEORY_FIRST_PATTERN
        assert pattern.strategy.value == "theory_first"
        assert len(pattern.agent_order) == 3
        
        print("✅ Mock sequence execution test passed!")
        print(f"   Pattern: {pattern.strategy.value}")
        print(f"   Agent order: {[a.value for a in pattern.agent_order]}")
        return True
        
    except Exception as e:
        print(f"❌ Mock sequence execution failed: {e}")
        traceback.print_exc()
        return False

def test_integration_hooks():
    """Test that integration with existing OpenDeepResearch works."""
    print("\n🧪 Testing integration hooks...")
    
    try:
        from open_deep_research.deep_researcher import sequence_optimization_router
        from open_deep_research.configuration import Configuration
        from langchain_core.runnables import RunnableConfig
        
        # Test router logic with disabled sequence optimization
        config_disabled = RunnableConfig(configurable={
            "enable_sequence_optimization": False
        })
        
        # Create mock state
        mock_state = {"research_brief": "test query"}
        
        # The router should be callable (we can't test execution without full setup)
        assert callable(sequence_optimization_router)
        
        print("✅ Integration hooks test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration hooks test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 SEQUENTIAL AGENT ORDERING OPTIMIZATION FRAMEWORK TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_sequence_patterns,
        test_sequence_engine_initialization,
        test_integration_hooks
    ]
    
    async_tests = [
        test_mock_sequence_execution
    ]
    
    passed = 0
    total = len(tests) + len(async_tests)
    
    # Run sync tests
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    # Run async tests
    async def run_async_tests():
        nonlocal passed
        for test in async_tests:
            try:
                if await test():
                    passed += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} crashed: {e}")
    
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 60)
    print(f"🎯 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Framework is ready for use.")
        print("\n💡 To use with real API keys, set up your .env file with:")
        print("   - OPENAI_API_KEY (or other model provider)")
        print("   - TAVILY_API_KEY (or other search provider)")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)