#!/usr/bin/env python3
"""Manual testing script for LLM sequence generation system.

This script tests the complete LLM sequence generation workflow with real
agent configurations and provides detailed output for analysis.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_deep_research.supervisor.llm_sequence_generator import LLMSequenceGenerator
from open_deep_research.supervisor.agent_capability_mapper import AgentCapabilityMapper
from open_deep_research.supervisor.sequence_models import (
    SequenceGenerationInput,
    SequenceGenerationResult
)
from open_deep_research.agents.registry import AgentRegistry


class LLMSequenceTestRunner:
    """Test runner for manual LLM sequence generation testing."""
    
    def __init__(self):
        """Initialize test runner."""
        self.generator = LLMSequenceGenerator()
        self.mapper = AgentCapabilityMapper()
        
        # Research scenarios to test
        self.test_scenarios = [
            {
                "name": "Academic Research",
                "topic": "The impact of machine learning on climate change predictions",
                "brief": "Comprehensive academic investigation needed to understand how ML techniques are being applied to improve climate modeling accuracy, identify current limitations, and explore future research directions.",
                "research_type": "academic",
                "expected_focus": ["literature review", "academic sources", "research methodology"]
            },
            {
                "name": "Technical Implementation",
                "topic": "Implementing microservices architecture for large-scale applications",
                "brief": "Technical analysis required for understanding best practices, architectural patterns, performance considerations, and practical implementation challenges for microservices at scale.",
                "research_type": "technical",
                "expected_focus": ["technical feasibility", "implementation details", "architecture"]
            },
            {
                "name": "Market Analysis",
                "topic": "Consumer adoption trends for electric vehicles in emerging markets",
                "brief": "Market research to understand adoption patterns, barriers to entry, consumer preferences, and market dynamics in developing economies for electric vehicle penetration.",
                "research_type": "market",
                "expected_focus": ["market dynamics", "consumer behavior", "economic factors"]
            },
            {
                "name": "Interdisciplinary Study",
                "topic": "AI governance frameworks for healthcare applications",
                "brief": "Complex interdisciplinary research combining technical AI capabilities, regulatory requirements, ethical considerations, and healthcare industry needs for governance framework development.",
                "research_type": "interdisciplinary",
                "expected_focus": ["regulatory compliance", "technical implementation", "ethical considerations"]
            },
            {
                "name": "Innovation Research",
                "topic": "Emerging trends in quantum computing applications for cryptography",
                "brief": "Cutting-edge research into quantum computing developments, their implications for current cryptographic systems, and the timeline for practical quantum-resistant solutions.",
                "research_type": "innovation",
                "expected_focus": ["emerging technologies", "future implications", "security analysis"]
            }
        ]
        
        self.results = {}
        self.performance_metrics = {}
    
    def print_header(self, title: str):
        """Print formatted header."""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    
    def print_subheader(self, title: str):
        """Print formatted subheader."""
        print(f"\n--- {title} ---")
    
    def setup_test_environment(self):
        """Set up test environment and verify configurations."""
        self.print_header("SETTING UP TEST ENVIRONMENT")
        
        # Check for agent configurations
        agents_dir = Path(".open_deep_research/agents")
        if not agents_dir.exists():
            print(f"❌ Agent directory not found: {agents_dir}")
            print("Creating test agent directory...")
            agents_dir.mkdir(parents=True, exist_ok=True)
            return False
        
        # List available agents
        agent_files = list(agents_dir.glob("*.md"))
        print(f"✓ Found {len(agent_files)} agent configuration files:")
        for agent_file in agent_files:
            print(f"  - {agent_file.name}")
        
        # Test agent registry loading
        try:
            registry = AgentRegistry()
            available_agents = registry.list_agents()
            print(f"✓ Agent registry loaded successfully with {len(available_agents)} agents:")
            for agent in available_agents:
                print(f"  - {agent}")
            
            # Test capability mapping
            capabilities = self.mapper.get_all_agent_capabilities()
            print(f"✓ Generated capabilities for {len(capabilities)} agents")
            
            return len(capabilities) > 0
            
        except Exception as e:
            print(f"❌ Failed to load agent registry: {e}")
            return False
    
    def test_agent_capability_mapping(self):
        """Test agent capability mapping functionality."""
        self.print_header("TESTING AGENT CAPABILITY MAPPING")
        
        try:
            capabilities = self.mapper.get_all_agent_capabilities()
            
            print(f"✓ Successfully mapped {len(capabilities)} agents to capabilities")
            
            # Display each capability
            for cap in capabilities:
                self.print_subheader(f"Agent: {cap.name}")
                print(f"Description: {cap.description}")
                print(f"Expertise Areas: {', '.join(cap.expertise_areas)}")
                print(f"Strength Summary: {cap.strength_summary}")
                print(f"Typical Use Cases: {', '.join(cap.typical_use_cases)}")
            
            # Test expertise summary
            expertise_summary = self.mapper.get_expertise_summary(capabilities)
            self.print_subheader("Expertise Distribution")
            for expertise, count in sorted(expertise_summary.items()):
                print(f"  {expertise}: {count} agents")
            
            return capabilities
            
        except Exception as e:
            print(f"❌ Agent capability mapping failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def test_scenario(self, scenario: Dict[str, Any], capabilities: List) -> Dict[str, Any]:
        """Test a single research scenario."""
        self.print_subheader(f"Testing: {scenario['name']}")
        
        # Create input
        input_data = SequenceGenerationInput(
            research_topic=scenario['topic'],
            research_brief=scenario['brief'],
            available_agents=capabilities,
            research_type=scenario['research_type']
        )
        
        print(f"Research Topic: {scenario['topic']}")
        print(f"Research Type: {scenario['research_type']}")
        print(f"Available Agents: {len(capabilities)}")
        
        # Time the generation
        start_time = time.time()
        
        try:
            # Test both sync and async versions
            print("\n🔄 Testing synchronous generation...")
            sync_result = self.generator.generate_sequences_sync(input_data)
            sync_time = time.time() - start_time
            
            print(f"✓ Sync generation completed in {sync_time:.2f}s")
            print(f"  Success: {sync_result.success}")
            print(f"  Fallback used: {sync_result.metadata.fallback_used}")
            if sync_result.metadata.error_details:
                print(f"  Error details: {sync_result.metadata.error_details}")
            
            # Analyze sequences
            sequences = sync_result.output.sequences
            print(f"  Generated {len(sequences)} sequences")
            
            # Display each sequence
            for i, seq in enumerate(sequences, 1):
                print(f"\n  Sequence {i}: {seq.sequence_name}")
                print(f"    Agents: {' → '.join(seq.agent_names)}")
                print(f"    Confidence: {seq.confidence_score:.2f}")
                print(f"    Focus: {seq.research_focus}")
                print(f"    Rationale: {seq.rationale[:100]}...")
            
            # Check sequence diversity
            sequence_names = {seq.sequence_name for seq in sequences}
            agent_combinations = {tuple(seq.agent_names) for seq in sequences}
            
            diversity_score = len(sequence_names) / 3.0  # Expect 3 unique names
            combination_diversity = len(agent_combinations) / 3.0  # Expect 3 unique combinations
            
            print(f"\n  Analysis:")
            print(f"    Name diversity: {diversity_score:.2f} (3 unique names: {len(sequence_names) == 3})")
            print(f"    Agent combination diversity: {combination_diversity:.2f}")
            print(f"    Recommended sequence: {sync_result.output.recommended_sequence}")
            
            # Store results
            return {
                "success": sync_result.success,
                "generation_time": sync_time,
                "fallback_used": sync_result.metadata.fallback_used,
                "sequence_count": len(sequences),
                "diversity_score": diversity_score,
                "combination_diversity": combination_diversity,
                "token_usage": {
                    "input_tokens": sync_result.metadata.input_token_count,
                    "output_tokens": sync_result.metadata.output_token_count
                },
                "sequences": [
                    {
                        "name": seq.sequence_name,
                        "agents": seq.agent_names,
                        "confidence": seq.confidence_score,
                        "rationale": seq.rationale
                    }
                    for seq in sequences
                ],
                "error_details": sync_result.metadata.error_details
            }
            
        except Exception as e:
            print(f"❌ Scenario test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "generation_time": time.time() - start_time
            }
    
    def test_error_scenarios(self, capabilities: List):
        """Test error handling scenarios."""
        self.print_header("TESTING ERROR HANDLING SCENARIOS")
        
        error_scenarios = [
            {
                "name": "Empty Agent List",
                "input": SequenceGenerationInput(
                    research_topic="Test topic",
                    research_brief="Test brief",
                    available_agents=[],
                    research_type="test"
                )
            },
            {
                "name": "Empty Research Topic",
                "input": SequenceGenerationInput(
                    research_topic="",
                    research_brief="Test brief",
                    available_agents=capabilities[:2] if capabilities else [],
                    research_type="test"
                )
            },
            {
                "name": "Very Long Research Topic",
                "input": SequenceGenerationInput(
                    research_topic="This is a very long research topic " * 100,
                    research_brief="Test brief",
                    available_agents=capabilities[:3] if capabilities else [],
                    research_type="test"
                )
            }
        ]
        
        error_results = {}
        
        for scenario in error_scenarios:
            self.print_subheader(f"Testing: {scenario['name']}")
            
            try:
                start_time = time.time()
                result = self.generator.generate_sequences_sync(scenario['input'])
                test_time = time.time() - start_time
                
                print(f"✓ Handled gracefully in {test_time:.2f}s")
                print(f"  Success: {result.success}")
                print(f"  Fallback used: {result.metadata.fallback_used}")
                print(f"  Sequences generated: {len(result.output.sequences)}")
                
                error_results[scenario['name']] = {
                    "handled_gracefully": True,
                    "success": result.success,
                    "fallback_used": result.metadata.fallback_used,
                    "generation_time": test_time
                }
                
            except Exception as e:
                print(f"❌ Error not handled gracefully: {e}")
                error_results[scenario['name']] = {
                    "handled_gracefully": False,
                    "error": str(e)
                }
        
        return error_results
    
    def test_performance_scenarios(self, capabilities: List):
        """Test performance with various loads."""
        self.print_header("TESTING PERFORMANCE SCENARIOS")
        
        if not capabilities:
            print("❌ No capabilities available for performance testing")
            return {}
        
        performance_tests = [
            {
                "name": "Single Agent",
                "agents": capabilities[:1],
                "topic": "Simple research topic"
            },
            {
                "name": "Multiple Agents",
                "agents": capabilities[:3],
                "topic": "Multi-agent research coordination"
            },
            {
                "name": "All Available Agents",
                "agents": capabilities,
                "topic": "Comprehensive research requiring all available expertise"
            },
            {
                "name": "Complex Topic",
                "agents": capabilities[:3],
                "topic": "Complex interdisciplinary research requiring deep analysis of technological, social, economic, and environmental factors with extensive stakeholder consideration and multi-phase implementation planning"
            }
        ]
        
        performance_results = {}
        
        for test in performance_tests:
            self.print_subheader(f"Performance Test: {test['name']}")
            
            input_data = SequenceGenerationInput(
                research_topic=test['topic'],
                available_agents=test['agents']
            )
            
            # Run multiple iterations for average timing
            times = []
            successful_runs = 0
            
            for i in range(3):  # 3 iterations
                try:
                    start_time = time.time()
                    result = self.generator.generate_sequences_sync(input_data)
                    run_time = time.time() - start_time
                    times.append(run_time)
                    if result.success:
                        successful_runs += 1
                    print(f"  Run {i+1}: {run_time:.2f}s (Success: {result.success})")
                except Exception as e:
                    print(f"  Run {i+1}: Failed with {e}")
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                print(f"  Average: {avg_time:.2f}s")
                print(f"  Range: {min_time:.2f}s - {max_time:.2f}s")
                print(f"  Success rate: {successful_runs}/3")
                
                performance_results[test['name']] = {
                    "average_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "success_rate": successful_runs / 3,
                    "agent_count": len(test['agents'])
                }
        
        return performance_results
    
    def run_all_tests(self):
        """Run all test scenarios."""
        print("🚀 Starting Manual LLM Sequence Generation Testing")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Setup
        setup_success = self.setup_test_environment()
        if not setup_success:
            print("❌ Setup failed, using fallback mode for testing")
        
        # Test capability mapping
        capabilities = self.test_agent_capability_mapping()
        
        # Test main scenarios
        self.print_header("TESTING RESEARCH SCENARIOS")
        
        for scenario in self.test_scenarios:
            result = self.test_scenario(scenario, capabilities)
            self.results[scenario['name']] = result
            
            # Brief pause between scenarios
            time.sleep(1)
        
        # Test error handling
        error_results = self.test_error_scenarios(capabilities)
        
        # Test performance
        performance_results = self.test_performance_scenarios(capabilities)
        
        # Generate summary
        self.generate_test_summary(error_results, performance_results)
    
    def generate_test_summary(self, error_results: Dict, performance_results: Dict):
        """Generate comprehensive test summary."""
        self.print_header("TEST SUMMARY AND ANALYSIS")
        
        # Overall success metrics
        total_scenarios = len(self.results)
        successful_scenarios = sum(1 for r in self.results.values() if r.get('success', False))
        fallback_usage = sum(1 for r in self.results.values() if r.get('fallback_used', False))
        
        print(f"✓ Scenario Success Rate: {successful_scenarios}/{total_scenarios} ({successful_scenarios/total_scenarios*100:.1f}%)")
        print(f"📊 Fallback Usage: {fallback_usage}/{total_scenarios} scenarios")
        
        # Performance analysis
        if self.results:
            generation_times = [r.get('generation_time', 0) for r in self.results.values() if 'generation_time' in r]
            if generation_times:
                avg_time = sum(generation_times) / len(generation_times)
                print(f"⏱️  Average Generation Time: {avg_time:.2f}s")
                print(f"⏱️  Fastest: {min(generation_times):.2f}s, Slowest: {max(generation_times):.2f}s")
        
        # Quality analysis
        print(f"\n📈 Quality Metrics:")
        diversity_scores = [r.get('diversity_score', 0) for r in self.results.values() if 'diversity_score' in r]
        if diversity_scores:
            avg_diversity = sum(diversity_scores) / len(diversity_scores)
            print(f"  Sequence Name Diversity: {avg_diversity:.2f}/1.0")
        
        # Token usage analysis
        print(f"\n🔢 Token Usage:")
        total_input_tokens = sum(r.get('token_usage', {}).get('input_tokens', 0) or 0 for r in self.results.values())
        total_output_tokens = sum(r.get('token_usage', {}).get('output_tokens', 0) or 0 for r in self.results.values())
        if total_input_tokens > 0:
            print(f"  Total Input Tokens: {total_input_tokens:,}")
            print(f"  Total Output Tokens: {total_output_tokens:,}")
            print(f"  Average per scenario: {total_input_tokens/len(self.results):.0f} input, {total_output_tokens/len(self.results):.0f} output")
        
        # Error handling analysis
        print(f"\n🛡️  Error Handling:")
        graceful_errors = sum(1 for r in error_results.values() if r.get('handled_gracefully', False))
        print(f"  Graceful error handling: {graceful_errors}/{len(error_results)}")
        
        # Performance analysis
        if performance_results:
            print(f"\n⚡ Performance Analysis:")
            for test_name, perf in performance_results.items():
                print(f"  {test_name}: {perf['average_time']:.2f}s avg ({perf['agent_count']} agents)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if fallback_usage > total_scenarios / 2:
            print("  - High fallback usage detected. Check LLM API configuration.")
        if avg_time > 10:
            print("  - Generation times are high. Consider optimizing prompts or model selection.")
        if avg_diversity < 0.8:
            print("  - Sequence diversity could be improved. Review prompt engineering.")
        if graceful_errors < len(error_results):
            print("  - Some error scenarios not handled gracefully. Improve error handling.")
        
        print(f"\n✅ Testing completed successfully!")
        
        # Save detailed results
        self.save_detailed_results(error_results, performance_results)
    
    def save_detailed_results(self, error_results: Dict, performance_results: Dict):
        """Save detailed test results to file."""
        results_file = Path("test_results_llm_sequence_generation.json")
        
        detailed_results = {
            "timestamp": datetime.now().isoformat(),
            "scenario_results": self.results,
            "error_handling_results": error_results,
            "performance_results": performance_results,
            "summary": {
                "total_scenarios": len(self.results),
                "successful_scenarios": sum(1 for r in self.results.values() if r.get('success', False)),
                "fallback_usage": sum(1 for r in self.results.values() if r.get('fallback_used', False))
            }
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            print(f"\n📄 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def main():
    """Main entry point for manual testing."""
    # Set up environment
    os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent.parent / "src"))
    
    # Run tests
    runner = LLMSequenceTestRunner()
    runner.run_all_tests()


if __name__ == "__main__":
    main()