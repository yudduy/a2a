"""Performance and memory usage tests for the dynamic sequence system.

This test module validates performance characteristics, memory usage patterns,
and scalability of the meta-sequence optimizer system under various loads
and usage patterns.
"""

import pytest
import asyncio
import time
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from unittest.mock import Mock, AsyncMock, patch

from langchain_core.runnables import RunnableConfig

from open_deep_research.sequencing.models import (
    DynamicSequencePattern,
    AgentType,
    SequenceResult,
    AgentExecutionResult
)
from open_deep_research.sequencing.sequence_engine import SequenceOptimizationEngine
from open_deep_research.sequencing.sequence_selector import SequenceAnalyzer


class TestDynamicSequenceGenerationPerformance:
    """Test performance characteristics of dynamic sequence generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()
    
    def test_single_sequence_generation_timing(self):
        """Test timing for single sequence generation."""
        topic = "Machine learning applications in autonomous vehicle navigation systems"
        
        start_time = time.time()
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should complete within reasonable time
        assert generation_time < 2.0  # Less than 2 seconds
        assert len(sequences) == 3
        
        # Verify quality isn't sacrificed for speed
        for seq in sequences:
            assert len(seq.description) > 20
            assert len(seq.reasoning) > 50
            assert len(seq.expected_advantages) >= 2
    
    def test_batch_generation_performance(self):
        """Test performance with batch generation of sequences."""
        topics = [
            "Artificial intelligence in medical diagnosis",
            "Blockchain technology for supply chain transparency",
            "Quantum computing algorithms for optimization",
            "Internet of Things security frameworks",
            "5G network slicing for industrial applications"
        ]
        
        start_time = time.time()
        all_sequences = []
        
        for topic in topics:
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
            all_sequences.extend(sequences)
        
        end_time = time.time()
        batch_time = end_time - start_time
        
        # Should handle batch processing efficiently
        assert batch_time < 10.0  # Less than 10 seconds for 5 topics × 3 sequences
        assert len(all_sequences) == 15  # 5 topics × 3 sequences each
        
        # Average time per sequence should be reasonable
        avg_time_per_sequence = batch_time / 15
        assert avg_time_per_sequence < 1.0  # Less than 1 second per sequence
    
    def test_large_sequence_count_performance(self):
        """Test performance when generating many sequences for one topic."""
        topic = "Sustainable energy systems and smart grid technologies"
        
        start_time = time.time()
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=10)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should scale reasonably with sequence count
        assert generation_time < 5.0  # Less than 5 seconds for 10 sequences
        assert len(sequences) <= 10  # May limit to reasonable number
        
        # Sequences should still be diverse and high quality
        agent_orders = [tuple(seq.agent_order) for seq in sequences]
        unique_orders = set(agent_orders)
        assert len(unique_orders) >= 2  # At least 2 different orderings
    
    def test_complex_topic_processing_performance(self):
        """Test performance with complex, long topics."""
        complex_topic = (
            "Comprehensive interdisciplinary analysis of the integration of artificial intelligence, "
            "machine learning, and deep learning technologies with Internet of Things devices, "
            "edge computing infrastructure, and 5G/6G wireless networks for the development of "
            "smart city applications including traffic management, energy optimization, waste "
            "management, public safety, and citizen services, considering technical feasibility, "
            "economic viability, regulatory compliance, privacy concerns, security implications, "
            "environmental impact, and social acceptance across diverse urban environments and "
            "cultural contexts in both developed and emerging economies"
        )
        
        start_time = time.time()
        sequences = self.analyzer.generate_dynamic_sequences(complex_topic, num_sequences=5)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should handle complex topics without significant performance degradation
        assert processing_time < 3.0  # Less than 3 seconds even for complex topics
        assert len(sequences) == 5
        
        # Quality should remain high for complex topics
        for seq in sequences:
            assert seq.confidence_score > 0.4
            assert len(seq.reasoning) > 100


class TestSequenceExecutionPerformance:
    """Test performance characteristics of sequence execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=RunnableConfig)
        self.mock_config.get = Mock(return_value={})
        
        with patch('open_deep_research.sequencing.sequence_engine.SupervisorResearchDirector'), \
             patch('open_deep_research.sequencing.sequence_engine.MetricsCalculator'), \
             patch('open_deep_research.sequencing.sequence_engine.SequenceAnalyzer'):
            self.engine = SequenceOptimizationEngine(
                config=self.mock_config,
                enable_real_time_metrics=False  # Disable for performance testing
            )
    
    @pytest.mark.asyncio
    async def test_single_dynamic_sequence_execution_timing(self):
        """Test execution timing for single dynamic sequence."""
        dynamic_pattern = DynamicSequencePattern(
            agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
            description="Performance test sequence",
            reasoning="Testing execution performance",
            confidence_score=0.8
        )
        
        research_topic = "Performance testing topic"
        
        # Mock fast agent execution
        mock_agents = [AsyncMock() for _ in range(3)]
        for i, agent in enumerate(mock_agents):
            mock_result = Mock(spec=AgentExecutionResult)
            mock_result.agent_type = dynamic_pattern.agent_order[i]
            mock_result.execution_duration = 0.1  # Fast execution
            mock_result.key_insights = [f"Insight {i}"]
            mock_result.research_findings = f"Findings {i}"
            mock_result.refined_insights = [f"Refined {i}"]
            agent.execute_research.return_value = mock_result
        
        with patch.object(self.engine, '_get_agent') as mock_get_agent, \
             patch.object(self.engine.metrics_calculator, 'calculate_sequence_productivity') as mock_calc:
            
            mock_get_agent.side_effect = mock_agents
            mock_calc.return_value = Mock()
            
            # Mock research director
            self.engine.research_director.direct_next_investigation = AsyncMock(
                return_value=Mock(questions=["Fast question"])
            )
            self.engine.research_director.track_insight_productivity = AsyncMock(
                return_value=Mock(insight_types=["RESEARCH_GAP"], transition_quality=0.8)
            )
            
            start_time = time.time()
            result = await self.engine.execute_sequence(dynamic_pattern, research_topic)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Should complete quickly with mocked agents
            assert execution_time < 1.0  # Less than 1 second
            assert isinstance(result, SequenceResult)
            assert len(result.agent_results) == 3
    
    @pytest.mark.asyncio
    async def test_variable_length_sequence_performance(self):
        """Test performance scaling with sequence length."""
        sequences_by_length = [
            # Length 1
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC],
                description="Single agent sequence",
                reasoning="Performance test - length 1",
                confidence_score=0.8
            ),
            # Length 3
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
                description="Standard three agent sequence",
                reasoning="Performance test - length 3",
                confidence_score=0.8
            ),
            # Length 5
            DynamicSequencePattern(
                agent_order=[
                    AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS,
                    AgentType.ACADEMIC, AgentType.INDUSTRY
                ],
                description="Extended five agent sequence",
                reasoning="Performance test - length 5",
                confidence_score=0.8
            )
        ]
        
        research_topic = "Performance scaling test"
        execution_times = []
        
        # Mock agents
        mock_agents = [AsyncMock() for _ in range(9)]  # Enough for all sequences
        agent_counter = 0
        
        for i, sequence in enumerate(sequences_by_length):
            seq_agents = []
            for j, agent_type in enumerate(sequence.agent_order):
                mock_result = Mock(spec=AgentExecutionResult)
                mock_result.agent_type = agent_type
                mock_result.execution_duration = 0.1
                mock_result.key_insights = [f"Insight {j}"]
                mock_result.research_findings = f"Findings {j}"
                mock_result.refined_insights = [f"Refined {j}"]
                
                mock_agents[agent_counter].execute_research.return_value = mock_result
                seq_agents.append(mock_agents[agent_counter])
                agent_counter += 1
            
            with patch.object(self.engine, '_get_agent') as mock_get_agent, \
                 patch.object(self.engine.metrics_calculator, 'calculate_sequence_productivity') as mock_calc:
                
                mock_get_agent.side_effect = seq_agents
                mock_calc.return_value = Mock()
                
                # Mock research director
                self.engine.research_director.direct_next_investigation = AsyncMock(
                    return_value=Mock(questions=["Test question"])
                )
                self.engine.research_director.track_insight_productivity = AsyncMock(
                    return_value=Mock(insight_types=["RESEARCH_GAP"], transition_quality=0.8)
                )
                
                start_time = time.time()
                result = await self.engine.execute_sequence(sequence, research_topic)
                end_time = time.time()
                
                execution_time = end_time - start_time
                execution_times.append((len(sequence.agent_order), execution_time))
                
                assert isinstance(result, SequenceResult)
                assert len(result.agent_results) == len(sequence.agent_order)
        
        # Verify reasonable scaling
        lengths = [length for length, _ in execution_times]
        times = [time for _, time in execution_times]
        
        # Execution time should scale roughly linearly with sequence length
        for i in range(1, len(times)):
            time_ratio = times[i] / times[0]
            length_ratio = lengths[i] / lengths[0]
            # Time scaling should be reasonable (not exponential)
            assert time_ratio <= length_ratio * 2  # Allow some overhead


class TestMemoryUsagePatterns:
    """Test memory usage patterns and resource management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()
        # Force garbage collection before tests
        gc.collect()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def test_sequence_generation_memory_usage(self):
        """Test memory usage during sequence generation."""
        initial_memory = self.get_memory_usage()
        
        # Generate multiple batches of sequences
        topics = [f"Research topic {i} for memory testing" for i in range(10)]
        all_sequences = []
        
        for topic in topics:
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=5)
            all_sequences.extend(sequences)
            
            # Check memory periodically
            current_memory = self.get_memory_usage()
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 100  # Less than 100MB increase
        
        final_memory = self.get_memory_usage()
        total_memory_increase = final_memory - initial_memory
        
        # Total memory increase should be bounded
        assert total_memory_increase < 150  # Less than 150MB for all sequences
        assert len(all_sequences) == 50  # 10 topics × 5 sequences
        
        # Clean up and verify memory release
        del all_sequences
        gc.collect()
        cleanup_memory = self.get_memory_usage()
        
        # Memory should be released (with some tolerance for Python's memory management)
        memory_after_cleanup = cleanup_memory - initial_memory
        assert memory_after_cleanup < total_memory_increase * 0.5  # At least 50% should be released
    
    def test_large_sequence_memory_efficiency(self):
        """Test memory efficiency with large numbers of sequences."""
        initial_memory = self.get_memory_usage()
        
        # Generate many sequences
        topic = "Memory efficiency test topic"
        sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=20)
        
        peak_memory = self.get_memory_usage()
        memory_per_sequence = (peak_memory - initial_memory) / len(sequences)
        
        # Memory per sequence should be reasonable
        assert memory_per_sequence < 5.0  # Less than 5MB per sequence
        
        # Verify sequences are valid
        assert len(sequences) <= 20
        assert all(isinstance(seq, DynamicSequencePattern) for seq in sequences)
    
    def test_sequence_object_size(self):
        """Test the size of sequence objects in memory."""
        import sys
        
        # Create various sequence patterns
        patterns = [
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC],
                description="Short pattern",
                reasoning="Brief reasoning",
                confidence_score=0.8
            ),
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
                description="Standard pattern with longer description and more detailed explanation",
                reasoning="More comprehensive reasoning with detailed explanation of approach and methodology",
                confidence_score=0.85,
                expected_advantages=[
                    "Detailed advantage 1 with comprehensive explanation",
                    "Detailed advantage 2 with thorough analysis",
                    "Detailed advantage 3 with extensive justification"
                ]
            ),
            DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC] * 10,  # Long sequence
                description="Extended pattern with many repetitions",
                reasoning="Extended reasoning for long sequence pattern",
                confidence_score=0.75
            )
        ]
        
        # Check object sizes
        for i, pattern in enumerate(patterns):
            size_bytes = sys.getsizeof(pattern)
            size_kb = size_bytes / 1024
            
            # Object size should be reasonable
            assert size_kb < 10.0  # Less than 10KB per pattern object
            
            # More complex patterns should not be dramatically larger
            if i > 0:
                prev_size = sys.getsizeof(patterns[i-1]) / 1024
                size_ratio = size_kb / prev_size
                assert size_ratio < 5.0  # No more than 5x larger


class TestConcurrencyAndScalability:
    """Test concurrency handling and scalability characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()
    
    def test_concurrent_sequence_generation(self):
        """Test concurrent generation of sequences."""
        topics = [
            "Concurrent topic 1: AI in healthcare",
            "Concurrent topic 2: Blockchain in finance",
            "Concurrent topic 3: IoT in manufacturing",
            "Concurrent topic 4: 5G in telecommunications",
            "Concurrent topic 5: Quantum computing in cryptography"
        ]
        
        def generate_sequences(topic):
            """Generate sequences for a topic."""
            return self.analyzer.generate_dynamic_sequences(topic, num_sequences=3)
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_sequences, topic) for topic in topics]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # Verify all results
        assert len(results) == 5
        total_sequences = sum(len(result) for result in results)
        assert total_sequences == 15  # 5 topics × 3 sequences
        
        # Compare with sequential execution
        start_time = time.time()
        sequential_results = []
        for topic in topics:
            sequential_results.append(generate_sequences(topic))
        end_time = time.time()
        sequential_time = end_time - start_time
        
        # Concurrent execution should not be significantly slower
        # (May not be faster due to Python GIL, but should not be much slower)
        assert concurrent_time <= sequential_time * 1.5  # Allow 50% overhead
    
    def test_thread_safety(self):
        """Test thread safety of sequence generation."""
        topic = "Thread safety test topic"
        num_threads = 10
        sequences_per_thread = 2
        
        all_sequences = []
        lock = threading.Lock()
        
        def thread_worker():
            """Worker function for thread."""
            sequences = self.analyzer.generate_dynamic_sequences(topic, num_sequences=sequences_per_thread)
            with lock:
                all_sequences.extend(sequences)
        
        # Create and start threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=thread_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        expected_total = num_threads * sequences_per_thread
        assert len(all_sequences) == expected_total
        
        # All sequences should be valid
        assert all(isinstance(seq, DynamicSequencePattern) for seq in all_sequences)
        
        # All sequences should have unique IDs (thread safety check)
        sequence_ids = [seq.sequence_id for seq in all_sequences]
        assert len(set(sequence_ids)) == len(sequence_ids)  # All unique
    
    @pytest.mark.asyncio
    async def test_async_execution_scalability(self):
        """Test scalability of async sequence execution."""
        mock_config = Mock(spec=RunnableConfig)
        mock_config.get = Mock(return_value={})
        
        with patch('open_deep_research.sequencing.sequence_engine.SupervisorResearchDirector'), \
             patch('open_deep_research.sequencing.sequence_engine.MetricsCalculator'), \
             patch('open_deep_research.sequencing.sequence_engine.SequenceAnalyzer'):
            engine = SequenceOptimizationEngine(mock_config, enable_real_time_metrics=False)
        
        # Create multiple sequences
        sequences = []
        for i in range(5):
            seq = DynamicSequencePattern(
                agent_order=[AgentType.ACADEMIC, AgentType.INDUSTRY],
                description=f"Scalability test sequence {i}",
                reasoning=f"Testing scalability - sequence {i}",
                confidence_score=0.8
            )
            sequences.append(seq)
        
        research_topic = "Async scalability test"
        
        # Mock agents
        mock_agents = [AsyncMock() for _ in range(10)]  # 5 sequences × 2 agents
        for i, agent in enumerate(mock_agents):
            mock_result = Mock(spec=AgentExecutionResult)
            mock_result.agent_type = AgentType.ACADEMIC if i % 2 == 0 else AgentType.INDUSTRY
            mock_result.execution_duration = 0.1
            mock_result.key_insights = [f"Insight {i}"]
            mock_result.research_findings = f"Findings {i}"
            mock_result.refined_insights = [f"Refined {i}"]
            agent.execute_research.return_value = mock_result
        
        with patch.object(engine, '_get_agent') as mock_get_agent, \
             patch.object(engine.metrics_calculator, 'calculate_sequence_productivity') as mock_calc:
            
            mock_get_agent.side_effect = mock_agents
            mock_calc.return_value = Mock()
            
            # Mock research director
            engine.research_director.direct_next_investigation = AsyncMock(
                return_value=Mock(questions=["Test question"])
            )
            engine.research_director.track_insight_productivity = AsyncMock(
                return_value=Mock(insight_types=["RESEARCH_GAP"], transition_quality=0.8)
            )
            
            # Execute all sequences concurrently
            start_time = time.time()
            tasks = [engine.execute_sequence(seq, research_topic) for seq in sequences]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Should complete efficiently
            assert execution_time < 2.0  # Less than 2 seconds for 5 sequences
            assert len(results) == 5
            assert all(isinstance(result, SequenceResult) for result in results)


if __name__ == "__main__":
    pytest.main([__file__])