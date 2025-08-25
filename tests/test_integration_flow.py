#!/usr/bin/env python3
"""Integration flow testing for LLM sequence generation system.

This module tests the integration of LLM sequence generation with the
main deep research workflow.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_deep_research.deep_researcher import (
    generate_strategic_sequences,
    initialize_agent_registry
)
from open_deep_research.supervisor.llm_sequence_generator import LLMSequenceGenerator
from open_deep_research.supervisor.agent_capability_mapper import AgentCapabilityMapper
from open_deep_research.supervisor.sequence_models import SequenceGenerationInput
from open_deep_research.configuration import Configuration
from langchain_core.runnables import RunnableConfig


class IntegrationTestRunner:
    """Test runner for integration flow testing."""
    
    def __init__(self):
        """Initialize test runner."""
        self.test_results = {}
        
        # Test scenarios
        self.integration_scenarios = [
            {
                "name": "Agent Registry Initialization",
                "description": "Test initialization of agent registry with real configurations"
            },
            {
                "name": "Strategic Sequence Generation",
                "description": "Test LLM-based strategic sequence generation with real agents"
            },
            {
                "name": "Capability Mapping Integration",
                "description": "Test integration of agent capability mapping with sequence generation"
            },
            {
                "name": "End-to-End Workflow",
                "description": "Test complete flow from research topic to strategic sequences"
            }
        ]
    
    def print_header(self, title: str):
        """Print formatted header."""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    
    def print_subheader(self, title: str):
        """Print formatted subheader."""
        print(f"\n--- {title} ---")
    
    async def test_agent_registry_initialization(self):
        """Test agent registry initialization."""
        self.print_subheader("Testing Agent Registry Initialization")
        
        try:
            # Create minimal config
            config = RunnableConfig()
            
            # Test registry initialization
            start_time = time.time()
            registry = await initialize_agent_registry(config)
            init_time = time.time() - start_time
            
            if registry:
                agent_names = registry.list_agents()
                print(f"✓ Registry initialized successfully in {init_time:.3f}s")
                print(f"  Found {len(agent_names)} agents: {agent_names}")
                
                # Test registry stats
                stats = registry.get_registry_stats()
                print(f"  Registry stats: {stats}")
                
                return {
                    "success": True,
                    "agent_count": len(agent_names),
                    "init_time": init_time,
                    "stats": stats
                }
            else:
                print("❌ Registry initialization failed or no agents found")
                return {
                    "success": False,
                    "error": "Registry initialization failed"
                }
                
        except Exception as e:
            print(f"❌ Agent registry initialization error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_strategic_sequence_generation(self):
        """Test strategic sequence generation with real agents."""
        self.print_subheader("Testing Strategic Sequence Generation")
        
        try:
            config = RunnableConfig()
            research_topic = "AI governance frameworks for healthcare applications"
            
            print(f"Research Topic: {research_topic}")
            
            # Test sequence generation
            start_time = time.time()
            sequences = await generate_strategic_sequences(research_topic, config)
            generation_time = time.time() - start_time
            
            print(f"✓ Sequence generation completed in {generation_time:.3f}s")
            print(f"  Generated {len(sequences)} sequences")
            
            # Display sequences
            for i, seq in enumerate(sequences, 1):
                print(f"\n  Sequence {i}: {seq.sequence_name}")
                print(f"    Agents: {' → '.join(seq.agent_names)}")
                print(f"    Confidence: {seq.confidence_score:.2f}")
                print(f"    Rationale: {seq.rationale[:100]}...")
            
            return {
                "success": len(sequences) > 0,
                "sequence_count": len(sequences),
                "generation_time": generation_time,
                "sequences": [
                    {
                        "name": seq.sequence_name,
                        "agents": seq.agent_names,
                        "confidence": seq.confidence_score
                    }
                    for seq in sequences
                ]
            }
            
        except Exception as e:
            print(f"❌ Strategic sequence generation error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_capability_mapping_integration(self):
        """Test capability mapping integration."""
        self.print_subheader("Testing Capability Mapping Integration")
        
        try:
            # Initialize registry and mapper
            config = RunnableConfig()
            registry = await initialize_agent_registry(config)
            
            if not registry:
                print("❌ Registry not available for capability mapping test")
                return {"success": False, "error": "No registry available"}
            
            mapper = AgentCapabilityMapper(registry)
            
            # Test capability mapping
            start_time = time.time()
            capabilities = mapper.get_all_agent_capabilities()
            mapping_time = time.time() - start_time
            
            print(f"✓ Capability mapping completed in {mapping_time:.3f}s")
            print(f"  Mapped {len(capabilities)} agent capabilities")
            
            # Display capabilities
            for cap in capabilities:
                print(f"\n  Agent: {cap.name}")
                print(f"    Expertise: {', '.join(cap.expertise_areas)}")
                print(f"    Description: {cap.description}")
                print(f"    Use Cases: {', '.join(cap.typical_use_cases[:2])}...")
            
            # Test expertise summary
            expertise_summary = mapper.get_expertise_summary(capabilities)
            print(f"\n  Expertise Distribution:")
            for expertise, count in sorted(expertise_summary.items()):
                print(f"    {expertise}: {count} agents")
            
            return {
                "success": True,
                "capability_count": len(capabilities),
                "mapping_time": mapping_time,
                "expertise_distribution": expertise_summary
            }
            
        except Exception as e:
            print(f"❌ Capability mapping integration error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        self.print_subheader("Testing End-to-End Workflow")
        
        try:
            config = RunnableConfig()
            
            # Step 1: Initialize components
            print("Step 1: Initializing components...")
            registry = await initialize_agent_registry(config)
            if not registry:
                print("❌ No registry available for end-to-end test")
                return {"success": False, "error": "No registry available"}
            
            mapper = AgentCapabilityMapper(registry)
            generator = LLMSequenceGenerator()
            
            # Step 2: Get capabilities
            print("Step 2: Getting agent capabilities...")
            capabilities = mapper.get_all_agent_capabilities()
            print(f"  Found {len(capabilities)} agent capabilities")
            
            # Step 3: Test different research scenarios
            research_scenarios = [
                {
                    "topic": "The impact of machine learning on climate change predictions",
                    "type": "academic"
                },
                {
                    "topic": "Implementing microservices architecture for large-scale applications",
                    "type": "technical"
                },
                {
                    "topic": "Consumer adoption trends for electric vehicles in emerging markets",
                    "type": "market"
                }
            ]
            
            scenario_results = []
            
            for i, scenario in enumerate(research_scenarios, 1):
                print(f"\nStep 3.{i}: Testing scenario - {scenario['type']} research")
                print(f"  Topic: {scenario['topic']}")
                
                # Create input
                input_data = SequenceGenerationInput(
                    research_topic=scenario['topic'],
                    research_brief=f"Comprehensive {scenario['type']} research needed",
                    available_agents=capabilities,
                    research_type=scenario['type']
                )
                
                # Generate sequences (using sync version for consistency)
                start_time = time.time()
                result = generator.generate_sequences_sync(input_data)
                scenario_time = time.time() - start_time
                
                print(f"    ✓ Generated {len(result.output.sequences)} sequences in {scenario_time:.3f}s")
                print(f"    Success: {result.success}, Fallback: {result.metadata.fallback_used}")
                
                scenario_results.append({
                    "scenario": scenario['type'],
                    "success": result.success,
                    "sequence_count": len(result.output.sequences),
                    "generation_time": scenario_time,
                    "fallback_used": result.metadata.fallback_used
                })
            
            # Step 4: Analyze results
            print("\nStep 4: Analyzing workflow results...")
            total_scenarios = len(scenario_results)
            successful_scenarios = sum(1 for r in scenario_results if r['success'])
            avg_time = sum(r['generation_time'] for r in scenario_results) / total_scenarios
            
            print(f"  Success rate: {successful_scenarios}/{total_scenarios} ({successful_scenarios/total_scenarios*100:.1f}%)")
            print(f"  Average generation time: {avg_time:.3f}s")
            
            return {
                "success": successful_scenarios > 0,
                "total_scenarios": total_scenarios,
                "successful_scenarios": successful_scenarios,
                "success_rate": successful_scenarios / total_scenarios,
                "average_generation_time": avg_time,
                "scenario_results": scenario_results
            }
            
        except Exception as e:
            print(f"❌ End-to-end workflow error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    async def run_all_integration_tests(self):
        """Run all integration tests."""
        self.print_header("INTEGRATION FLOW TESTING")
        print("Testing the complete integration of LLM sequence generation system")
        
        # Test 1: Agent Registry Initialization
        registry_result = await self.test_agent_registry_initialization()
        self.test_results["agent_registry"] = registry_result
        
        # Test 2: Strategic Sequence Generation
        sequence_result = await self.test_strategic_sequence_generation()
        self.test_results["sequence_generation"] = sequence_result
        
        # Test 3: Capability Mapping Integration
        mapping_result = await self.test_capability_mapping_integration()
        self.test_results["capability_mapping"] = mapping_result
        
        # Test 4: End-to-End Workflow
        e2e_result = await self.test_end_to_end_workflow()
        self.test_results["end_to_end"] = e2e_result
        
        # Generate summary
        self.generate_integration_summary()
    
    def generate_integration_summary(self):
        """Generate integration test summary."""
        self.print_header("INTEGRATION TEST SUMMARY")
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        print(f"✓ Integration Test Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        
        # Detailed results
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result.get('success', False) else "✗ FAIL"
            print(f"\n{status} {test_name.replace('_', ' ').title()}")
            
            if result.get('success', False):
                # Print success metrics
                if 'agent_count' in result:
                    print(f"  - Found {result['agent_count']} agents")
                if 'sequence_count' in result:
                    print(f"  - Generated {result['sequence_count']} sequences")
                if 'generation_time' in result:
                    print(f"  - Generation time: {result['generation_time']:.3f}s")
                if 'success_rate' in result:
                    print(f"  - Success rate: {result['success_rate']*100:.1f}%")
            else:
                # Print error details
                if 'error' in result:
                    print(f"  - Error: {result['error']}")
        
        # Integration health assessment
        print(f"\n🔍 Integration Health Assessment:")
        
        if successful_tests == total_tests:
            print("  🟢 EXCELLENT: All integration tests passing")
        elif successful_tests >= total_tests * 0.75:
            print("  🟡 GOOD: Most integration tests passing, minor issues detected")
        elif successful_tests >= total_tests * 0.5:
            print("  🟠 MODERATE: Half of integration tests passing, significant issues detected")
        else:
            print("  🔴 CRITICAL: Most integration tests failing, major integration issues")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if not self.test_results.get('agent_registry', {}).get('success', False):
            print("  - Check agent registry configuration and ensure .open_deep_research/agents/ directory exists")
        
        if not self.test_results.get('sequence_generation', {}).get('success', False):
            print("  - Verify LLM API keys are configured for sequence generation")
        
        if not self.test_results.get('capability_mapping', {}).get('success', False):
            print("  - Review agent configuration files for proper formatting")
        
        if not self.test_results.get('end_to_end', {}).get('success', False):
            print("  - Check end-to-end workflow for integration issues")
        
        if successful_tests == total_tests:
            print("  - All integration tests passing! System ready for production use.")


async def main():
    """Main entry point for integration testing."""
    # Set up environment
    os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent.parent / "src"))
    
    # Run integration tests
    runner = IntegrationTestRunner()
    await runner.run_all_integration_tests()


if __name__ == "__main__":
    asyncio.run(main())