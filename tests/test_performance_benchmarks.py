"""Performance benchmarking tests for Sequential Multi-Agent Supervisor.

This module provides comprehensive performance testing and benchmarking for:
- Handoff timing requirements (<3 seconds)
- Agent loading and registry operations
- Completion detection performance
- Memory usage patterns
- Concurrent operation performance
- Scalability validation

Test Categories:
1. Handoff performance benchmarks
2. Agent registry performance
3. Completion detection speed
4. Memory usage validation
5. Concurrent operations testing
6. Scalability benchmarks
7. Performance regression detection
"""

import pytest
import asyncio
import time
import psutil
import gc
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from datetime import datetime
import tempfile
import shutil

from langchain_core.messages import AIMessage

from open_deep_research.supervisor.sequential_supervisor import SequentialSupervisor, SupervisorConfig
from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.agents.completion_detector import CompletionDetector
from open_deep_research.configuration import Configuration
from open_deep_research.state import SequentialSupervisorState


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result container."""
    test_name: str
    execution_times: List[float]
    memory_usage_mb: List[float]
    success_rate: float
    avg_time: float
    p95_time: float
    max_time: float
    min_time: float
    memory_increase_mb: float


class PerformanceBenchmarkRunner:
    """Utility class for running and reporting performance benchmarks."""
    
    def __init__(self):
        self.results: List[PerformanceBenchmark] = []
        self.process = psutil.Process()
    
    def benchmark_function(self, test_name: str, func, *args, iterations: int = 10, **kwargs) -> PerformanceBenchmark:
        """Benchmark a function's performance over multiple iterations."""
        execution_times = []
        memory_usage = []
        successes = 0
        
        # Initial memory baseline
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        for i in range(iterations):
            try:
                # Measure execution time
                start_time = time.time()
                if asyncio.iscoroutinefunction(func):
                    asyncio.run(func(*args, **kwargs))
                else:
                    func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                execution_times.append(execution_time)
                
                # Measure memory usage
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_usage.append(current_memory)
                
                successes += 1
                
            except Exception as e:
                print(f"Iteration {i} failed: {e}")
                continue
        
        # Final memory measurement
        gc.collect()
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        if not execution_times:
            return PerformanceBenchmark(
                test_name=test_name,
                execution_times=[],
                memory_usage_mb=[],
                success_rate=0.0,
                avg_time=0.0,
                p95_time=0.0,
                max_time=0.0,
                min_time=0.0,
                memory_increase_mb=0.0
            )
        
        # Calculate statistics
        avg_time = statistics.mean(execution_times)
        p95_time = statistics.quantiles(execution_times, n=20)[18] if len(execution_times) >= 20 else max(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        success_rate = successes / iterations
        memory_increase = final_memory - initial_memory
        
        benchmark = PerformanceBenchmark(
            test_name=test_name,
            execution_times=execution_times,
            memory_usage_mb=memory_usage,
            success_rate=success_rate,
            avg_time=avg_time,
            p95_time=p95_time,
            max_time=max_time,
            min_time=min_time,
            memory_increase_mb=memory_increase
        )
        
        self.results.append(benchmark)
        return benchmark
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance benchmark report."""
        report = []
        report.append("# Sequential Multi-Agent Supervisor Performance Benchmark Report")
        report.append(f"Generated at: {datetime.now().isoformat()}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        total_tests = len(self.results)
        passing_tests = len([r for r in self.results if r.success_rate >= 0.8])
        
        report.append(f"- Total Performance Tests: {total_tests}")
        report.append(f"- Tests Meeting Performance Requirements: {passing_tests}/{total_tests} ({passing_tests/total_tests*100:.1f}%)")
        
        # Performance requirements check
        handoff_tests = [r for r in self.results if "handoff" in r.test_name.lower()]
        if handoff_tests:
            handoff_compliant = [r for r in handoff_tests if r.p95_time < 3.0]
            report.append(f"- Handoff Timing Requirement (<3s): {len(handoff_compliant)}/{len(handoff_tests)} tests compliant")
        
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Performance Results")
        report.append("")
        
        for result in self.results:
            report.append(f"### {result.test_name}")
            report.append("")
            report.append(f"- **Success Rate**: {result.success_rate:.1%}")
            report.append(f"- **Average Time**: {result.avg_time:.3f}s")
            report.append(f"- **95th Percentile**: {result.p95_time:.3f}s")
            report.append(f"- **Min/Max Time**: {result.min_time:.3f}s / {result.max_time:.3f}s")
            report.append(f"- **Memory Impact**: {result.memory_increase_mb:.2f}MB")
            
            # Performance assessment
            status = "✅ PASS"
            if "handoff" in result.test_name.lower() and result.p95_time >= 3.0:
                status = "❌ FAIL - Exceeds 3s handoff requirement"
            elif result.success_rate < 0.8:
                status = "❌ FAIL - Low success rate"
            elif result.memory_increase_mb > 100:
                status = "⚠️  WARN - High memory usage"
            
            report.append(f"- **Status**: {status}")
            report.append("")
        
        # Performance Summary Table
        report.append("## Performance Summary Table")
        report.append("")
        report.append("| Test Name | Avg Time (s) | P95 Time (s) | Success Rate | Memory (MB) | Status |")
        report.append("|-----------|--------------|--------------|--------------|-------------|--------|")
        
        for result in self.results:
            status_symbol = "✅" if result.success_rate >= 0.8 and result.p95_time < 5.0 else "❌"
            report.append(f"| {result.test_name} | {result.avg_time:.3f} | {result.p95_time:.3f} | {result.success_rate:.1%} | {result.memory_increase_mb:.1f} | {status_symbol} |")
        
        report.append("")
        
        # Recommendations
        report.append("## Performance Recommendations")
        report.append("")
        
        # Analyze results and provide recommendations
        slow_tests = [r for r in self.results if r.p95_time > 2.0]
        if slow_tests:
            report.append("### Timing Optimizations Needed")
            for test in slow_tests:
                report.append(f"- {test.test_name}: P95 time {test.p95_time:.3f}s exceeds optimal threshold")
        
        memory_heavy_tests = [r for r in self.results if r.memory_increase_mb > 50]
        if memory_heavy_tests:
            report.append("### Memory Usage Optimizations Needed")
            for test in memory_heavy_tests:
                report.append(f"- {test.test_name}: Memory increase {test.memory_increase_mb:.1f}MB is high")
        
        failing_tests = [r for r in self.results if r.success_rate < 0.9]
        if failing_tests:
            report.append("### Reliability Improvements Needed")
            for test in failing_tests:
                report.append(f"- {test.test_name}: Success rate {test.success_rate:.1%} below 90%")
        
        if not slow_tests and not memory_heavy_tests and not failing_tests:
            report.append("All performance benchmarks are within acceptable ranges. ✅")
        
        return "\n".join(report)


class TestHandoffPerformanceBenchmarks:
    """Benchmark handoff timing performance requirements."""
    
    def setup_method(self):
        """Set up handoff performance testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / ".open_deep_research" / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test agent
        agent_content = """# Performance Test Agent
## Description
Agent optimized for performance testing
## Expertise Areas
- Performance testing
## Tools
- search
## Completion Indicators
- Task complete
"""
        (self.agents_dir / "perf_agent.md").write_text(agent_content)
        
        self.registry = AgentRegistry(project_root=str(self.temp_dir))
        self.config = SupervisorConfig(debug_mode=False, agent_timeout_seconds=30.0)
        
        with patch('open_deep_research.supervisor.sequential_supervisor.init_chat_model'):
            self.supervisor = SequentialSupervisor(
                agent_registry=self.registry,
                config=self.config
            )
        
        self.benchmark_runner = PerformanceBenchmarkRunner()
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_supervisor_node_handoff_timing(self):
        """Benchmark supervisor node handoff timing."""
        state = SequentialSupervisorState(
            research_topic="Performance test topic",
            planned_sequence=["perf_agent"],
            sequence_position=0,
            handoff_ready=True,
            executed_agents=[],
            agent_insights={},
            agent_context={},
            agent_reports={},
            completion_signals={},
            supervisor_messages=[],
            sequence_modifications=[],
            sequence_start_time=datetime.utcnow(),
            running_report=None,
            last_agent_completed=None,
            current_agent=None
        )
        
        async def handoff_test():
            with patch.object(self.supervisor.model, 'bind_tools') as mock_bind, \
                 patch('open_deep_research.supervisor.sequential_supervisor.get_all_tools'), \
                 patch('open_deep_research.supervisor.sequential_supervisor.think_tool'):
                
                mock_model = AsyncMock()
                mock_bind.return_value = mock_model
                mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Quick response complete."))
                
                return await self.supervisor.supervisor_node(state)
        
        benchmark = self.benchmark_runner.benchmark_function(
            "Supervisor Node Handoff Timing",
            handoff_test,
            iterations=20
        )
        
        # Verify handoff timing requirement
        assert benchmark.p95_time < 3.0, f"Handoff P95 time {benchmark.p95_time:.3f}s exceeds 3s requirement"
        assert benchmark.avg_time < 2.0, f"Handoff average time {benchmark.avg_time:.3f}s is too high"
        assert benchmark.success_rate >= 0.95, f"Handoff success rate {benchmark.success_rate:.1%} too low"
    
    @pytest.mark.asyncio 
    async def test_agent_executor_node_timing(self):
        """Benchmark agent executor node timing."""
        state = SequentialSupervisorState(
            research_topic="Agent execution test",
            planned_sequence=["perf_agent"],
            sequence_position=0,
            current_agent="perf_agent",
            executed_agents=[],
            agent_insights={},
            agent_context={},
            agent_reports={},
            completion_signals={},
            supervisor_messages=[],
            sequence_modifications=[],
            sequence_start_time=datetime.utcnow(),
            running_report=None,
            last_agent_completed=None,
            handoff_ready=False
        )
        
        async def executor_test():
            with patch.object(self.supervisor.model, 'bind_tools') as mock_bind, \
                 patch('open_deep_research.supervisor.sequential_supervisor.get_all_tools'), \
                 patch('open_deep_research.supervisor.sequential_supervisor.think_tool'):
                
                mock_model = AsyncMock()
                mock_bind.return_value = mock_model
                mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Agent execution complete."))
                
                return await self.supervisor.agent_executor_node(state)
        
        benchmark = self.benchmark_runner.benchmark_function(
            "Agent Executor Node Timing",
            executor_test,
            iterations=15
        )
        
        # Agent execution should be reasonable but can be slower than handoffs
        assert benchmark.p95_time < 5.0, f"Agent execution P95 time {benchmark.p95_time:.3f}s too high"
        assert benchmark.success_rate >= 0.90, f"Agent execution success rate {benchmark.success_rate:.1%} too low"


class TestAgentRegistryPerformance:
    """Benchmark agent registry loading and operations."""
    
    def setup_method(self):
        """Set up agent registry performance testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / ".open_deep_research" / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create multiple agents for performance testing
        for i in range(50):
            agent_content = f"""# Performance Agent {i:02d}
## Description
Performance test agent number {i}
## Expertise Areas
- Performance testing {i}
- Agent {i} expertise
## Tools
- search
- agent_{i}_tool
## Completion Indicators
- Agent {i} task complete
"""
            (self.agents_dir / f"perf_agent_{i:02d}.md").write_text(agent_content)
        
        self.benchmark_runner = PerformanceBenchmarkRunner()
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_agent_registry_loading_performance(self):
        """Benchmark agent registry loading performance."""
        def loading_test():
            return AgentRegistry(project_root=str(self.temp_dir))
        
        benchmark = self.benchmark_runner.benchmark_function(
            "Agent Registry Loading (50 agents)",
            loading_test,
            iterations=10
        )
        
        # Loading should be fast even with many agents
        assert benchmark.avg_time < 2.0, f"Agent loading average time {benchmark.avg_time:.3f}s too slow"
        assert benchmark.p95_time < 3.0, f"Agent loading P95 time {benchmark.p95_time:.3f}s too slow"
        assert benchmark.success_rate >= 0.95, f"Agent loading success rate {benchmark.success_rate:.1%} too low"
    
    def test_agent_search_performance(self):
        """Benchmark agent search performance."""
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        def search_test():
            results = registry.search_agents("performance")
            return len(results)
        
        benchmark = self.benchmark_runner.benchmark_function(
            "Agent Search Performance",
            search_test,
            iterations=100
        )
        
        # Search should be very fast
        assert benchmark.avg_time < 0.1, f"Agent search average time {benchmark.avg_time:.3f}s too slow"
        assert benchmark.p95_time < 0.2, f"Agent search P95 time {benchmark.p95_time:.3f}s too slow"
        assert benchmark.success_rate >= 0.98, f"Agent search success rate {benchmark.success_rate:.1%} too low"
    
    def test_agent_retrieval_performance(self):
        """Benchmark individual agent retrieval performance."""
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        def retrieval_test():
            agent = registry.get_agent("perf_agent_25")
            return agent is not None
        
        benchmark = self.benchmark_runner.benchmark_function(
            "Agent Retrieval Performance",
            retrieval_test,
            iterations=200
        )
        
        # Retrieval should be extremely fast
        assert benchmark.avg_time < 0.01, f"Agent retrieval average time {benchmark.avg_time:.3f}s too slow"
        assert benchmark.p95_time < 0.05, f"Agent retrieval P95 time {benchmark.p95_time:.3f}s too slow"
        assert benchmark.success_rate >= 0.99, f"Agent retrieval success rate {benchmark.success_rate:.1%} too low"


class TestCompletionDetectionPerformance:
    """Benchmark completion detection performance."""
    
    def setup_method(self):
        """Set up completion detection performance testing."""
        self.detector = CompletionDetector(debug_mode=False)
        self.benchmark_runner = PerformanceBenchmarkRunner()
    
    def test_completion_detection_speed(self):
        """Benchmark completion detection speed for various message sizes."""
        message_sizes = [100, 500, 1000, 2000, 5000]
        
        for size in message_sizes:
            content = "This is test content. " * (size // 20)
            content += "Research complete and analysis finished."
            message = AIMessage(content=content)
            
            def detection_test():
                return self.detector.analyze_completion_patterns(message)
            
            benchmark = self.benchmark_runner.benchmark_function(
                f"Completion Detection ({size} chars)",
                detection_test,
                iterations=50
            )
            
            # Detection should be fast regardless of message size
            assert benchmark.avg_time < 0.5, f"Detection avg time {benchmark.avg_time:.3f}s too slow for {size} chars"
            assert benchmark.p95_time < 1.0, f"Detection P95 time {benchmark.p95_time:.3f}s too slow for {size} chars"
            assert benchmark.success_rate >= 0.95, f"Detection success rate {benchmark.success_rate:.1%} too low"


class TestConcurrentOperationsPerformance:
    """Benchmark concurrent operations performance."""
    
    def setup_method(self):
        """Set up concurrent operations testing."""
        self.benchmark_runner = PerformanceBenchmarkRunner()
    
    @pytest.mark.asyncio
    async def test_concurrent_completion_detection(self):
        """Benchmark concurrent completion detection operations."""
        detector = CompletionDetector(debug_mode=False)
        
        messages = [
            AIMessage(content=f"Test message {i}. Research complete.") 
            for i in range(20)
        ]
        
        async def concurrent_detection_test():
            tasks = [
                detector.analyze_completion_patterns(msg) 
                for msg in messages
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return len([r for r in results if not isinstance(r, Exception)])
        
        benchmark = self.benchmark_runner.benchmark_function(
            "Concurrent Completion Detection (20 parallel)",
            concurrent_detection_test,
            iterations=5
        )
        
        # Concurrent operations should scale well
        assert benchmark.avg_time < 2.0, f"Concurrent detection avg time {benchmark.avg_time:.3f}s too slow"
        assert benchmark.success_rate >= 0.8, f"Concurrent detection success rate {benchmark.success_rate:.1%} too low"


class TestMemoryUsageValidation:
    """Validate memory usage patterns and detect leaks."""
    
    def setup_method(self):
        """Set up memory usage testing."""
        self.benchmark_runner = PerformanceBenchmarkRunner()
    
    def test_agent_registry_memory_usage(self):
        """Test memory usage of agent registry operations."""
        temp_dir = Path(tempfile.mkdtemp())
        agents_dir = temp_dir / ".open_deep_research" / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create agents for memory testing
        for i in range(20):
            (agents_dir / f"memory_agent_{i}.md").write_text(f"""# Memory Agent {i}
## Description
Agent for memory testing
## Expertise Areas
- Memory test {i}
## Tools
- search
""")
        
        def memory_test():
            # Create and destroy registry multiple times
            for _ in range(5):
                registry = AgentRegistry(project_root=str(temp_dir))
                agents = registry.list_agents()
                del registry
            gc.collect()
        
        benchmark = self.benchmark_runner.benchmark_function(
            "Agent Registry Memory Usage",
            memory_test,
            iterations=10
        )
        
        # Memory increase should be minimal
        assert benchmark.memory_increase_mb < 50, f"Memory increase {benchmark.memory_increase_mb:.1f}MB too high"
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_completion_detector_memory_stability(self):
        """Test completion detector memory stability."""
        detector = CompletionDetector(debug_mode=False)
        
        def memory_stability_test():
            # Process many messages
            for i in range(100):
                message = AIMessage(content=f"Test message {i}. Analysis complete.")
                result = detector.analyze_completion_patterns(message)
                del result  # Explicit cleanup
            gc.collect()
        
        benchmark = self.benchmark_runner.benchmark_function(
            "Completion Detector Memory Stability",
            memory_stability_test,
            iterations=5
        )
        
        # Memory should remain stable
        assert benchmark.memory_increase_mb < 20, f"Memory increase {benchmark.memory_increase_mb:.1f}MB indicates leak"


class TestScalabilityBenchmarks:
    """Test system scalability with increasing loads."""
    
    def setup_method(self):
        """Set up scalability testing."""
        self.benchmark_runner = PerformanceBenchmarkRunner()
    
    def test_agent_registry_scalability(self):
        """Test agent registry performance with increasing agent counts."""
        agent_counts = [10, 25, 50, 100]
        
        for count in agent_counts:
            temp_dir = Path(tempfile.mkdtemp())
            agents_dir = temp_dir / ".open_deep_research" / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            
            # Create specified number of agents
            for i in range(count):
                (agents_dir / f"scale_agent_{i}.md").write_text(f"""# Scale Agent {i}
## Description
Scalability test agent {i}
## Expertise Areas
- Scalability {i}
## Tools
- search
""")
            
            def scalability_test():
                registry = AgentRegistry(project_root=str(temp_dir))
                agents = registry.list_agents()
                return len(agents)
            
            benchmark = self.benchmark_runner.benchmark_function(
                f"Agent Registry Scalability ({count} agents)",
                scalability_test,
                iterations=5
            )
            
            # Performance should scale reasonably
            expected_max_time = 0.5 + (count / 100.0)  # Allow more time for larger counts
            assert benchmark.avg_time < expected_max_time, \
                f"Registry loading with {count} agents too slow: {benchmark.avg_time:.3f}s"
            
            shutil.rmtree(temp_dir)


def generate_performance_report(benchmark_runner: PerformanceBenchmarkRunner) -> str:
    """Generate and save performance benchmark report."""
    report = benchmark_runner.generate_report()
    
    # Save report to file
    report_path = Path("performance_benchmark_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Performance benchmark report saved to: {report_path}")
    return report


# Integration test that runs all benchmarks and generates report
@pytest.mark.slow
def test_comprehensive_performance_benchmark():
    """Run comprehensive performance benchmarks and generate report."""
    benchmark_runner = PerformanceBenchmarkRunner()
    
    # This would run all the individual benchmark tests
    # For now, just create a sample report
    sample_benchmark = PerformanceBenchmark(
        test_name="Sample Performance Test",
        execution_times=[0.1, 0.12, 0.09, 0.11, 0.10],
        memory_usage_mb=[100, 102, 101, 103, 102],
        success_rate=1.0,
        avg_time=0.104,
        p95_time=0.12,
        max_time=0.12,
        min_time=0.09,
        memory_increase_mb=2.0
    )
    
    benchmark_runner.results.append(sample_benchmark)
    
    # Generate comprehensive report
    report = generate_performance_report(benchmark_runner)
    
    # Verify report was generated
    assert len(report) > 1000
    assert "Performance Benchmark Report" in report
    assert "Executive Summary" in report
    assert sample_benchmark.test_name in report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])