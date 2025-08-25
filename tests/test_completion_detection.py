"""Comprehensive tests for automatic completion detection and handoff mechanisms.

This module tests:
- CompletionDetector accuracy and reliability
- Different detection strategies (pattern-based, semantic, combined)
- Custom completion indicators for specialized agents
- Confidence scoring and threshold management
- Performance under various message types and lengths
- Edge cases and error handling

Test Categories:
1. Pattern-based completion detection
2. Semantic completion analysis
3. Combined detection strategies
4. Custom indicator handling
5. Confidence scoring validation
6. Performance and reliability testing
7. Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, patch
import time
from typing import List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from open_deep_research.agents.completion_detector import (
    CompletionDetector, 
    DetectionStrategy,
    CompletionResult
)


class TestPatternBasedCompletionDetection:
    """Test pattern-based completion detection mechanisms."""
    
    def setup_method(self):
        """Set up completion detector for testing."""
        self.detector = CompletionDetector(debug_mode=True)
        self.detector.set_completion_threshold(0.6)
    
    def test_explicit_completion_indicators(self):
        """Test detection of explicit completion phrases."""
        test_cases = [
            # Strong completion indicators
            {
                "content": "After thorough analysis, my research is complete. Research complete.",
                "expected_complete": True,
                "min_confidence": 0.9
            },
            {
                "content": "The investigation has concluded. Analysis complete.",
                "expected_complete": True,
                "min_confidence": 0.8
            },
            {
                "content": "Task accomplished successfully with comprehensive findings.",
                "expected_complete": True,
                "min_confidence": 0.7
            },
            {
                "content": "Findings summarized and ready for handoff to next agent.",
                "expected_complete": True,
                "min_confidence": 0.75
            },
            {
                "content": "Investigation finished. All questions addressed comprehensively.",
                "expected_complete": True,
                "min_confidence": 0.8
            },
            # Weak or no completion indicators
            {
                "content": "I am continuing to research this topic and need more information.",
                "expected_complete": False,
                "max_confidence": 0.4
            },
            {
                "content": "The analysis is ongoing and preliminary results suggest further investigation.",
                "expected_complete": False,
                "max_confidence": 0.3
            },
            {
                "content": "More work is needed to fully understand the implications.",
                "expected_complete": False,
                "max_confidence": 0.4
            }
        ]
        
        for i, case in enumerate(test_cases):
            message = AIMessage(content=case["content"])
            result = self.detector.analyze_completion_patterns(message, strategy=DetectionStrategy.PATTERN_BASED)
            
            assert result.is_complete == case["expected_complete"], \
                f"Case {i}: Expected {case['expected_complete']}, got {result.is_complete} for: {case['content'][:50]}..."
            
            if case["expected_complete"]:
                assert result.confidence >= case["min_confidence"], \
                    f"Case {i}: Confidence {result.confidence} below minimum {case['min_confidence']}"
            else:
                assert result.confidence <= case.get("max_confidence", 1.0), \
                    f"Case {i}: Confidence {result.confidence} above maximum {case.get('max_confidence', 1.0)}"
    
    def test_custom_completion_indicators(self):
        """Test detection with custom agent-specific completion indicators."""
        custom_indicators = [
            "market research finished",
            "competitive analysis done", 
            "business insights documented",
            "industry trends identified",
            "strategic recommendations ready"
        ]
        
        test_cases = [
            {
                "content": "The comprehensive market research finished with detailed competitive insights.",
                "expected_complete": True,
                "min_confidence": 0.8
            },
            {
                "content": "All business insights documented and strategic recommendations ready for review.",
                "expected_complete": True,
                "min_confidence": 0.8
            },
            {
                "content": "Industry trends identified through extensive analysis. Competitive analysis done.",
                "expected_complete": True,
                "min_confidence": 0.8
            },
            {
                "content": "Still gathering market data and analyzing competitive positioning.",
                "expected_complete": False,
                "max_confidence": 0.5
            }
        ]
        
        for i, case in enumerate(test_cases):
            message = AIMessage(content=case["content"])
            result = self.detector.analyze_completion_patterns(
                message,
                custom_indicators=custom_indicators,
                strategy=DetectionStrategy.PATTERN_BASED
            )
            
            assert result.is_complete == case["expected_complete"], \
                f"Custom Case {i}: Expected {case['expected_complete']}, got {result.is_complete}"
            
            if case["expected_complete"]:
                assert result.confidence >= case["min_confidence"], \
                    f"Custom Case {i}: Confidence {result.confidence} below minimum"
    
    def test_pattern_confidence_scoring(self):
        """Test confidence scoring for pattern-based detection."""
        # Multiple strong indicators should increase confidence
        message_strong = AIMessage(content="Research complete. Analysis finished. Task accomplished. Investigation concluded.")
        result_strong = self.detector.analyze_completion_patterns(message_strong, strategy=DetectionStrategy.PATTERN_BASED)
        
        # Single indicator should have lower confidence
        message_single = AIMessage(content="The task is complete.")
        result_single = self.detector.analyze_completion_patterns(message_single, strategy=DetectionStrategy.PATTERN_BASED)
        
        # Weak indicator should have even lower confidence
        message_weak = AIMessage(content="This work is done for now.")
        result_weak = self.detector.analyze_completion_patterns(message_weak, strategy=DetectionStrategy.PATTERN_BASED)
        
        assert result_strong.confidence > result_single.confidence
        assert result_single.confidence > result_weak.confidence
        assert result_strong.is_complete
        assert result_single.is_complete
    
    def test_negation_handling(self):
        """Test handling of negated completion indicators."""
        negation_cases = [
            {
                "content": "The research is not complete yet. More investigation needed.",
                "expected_complete": False
            },
            {
                "content": "I haven't finished the analysis. Still working on it.",
                "expected_complete": False
            },
            {
                "content": "The task is not accomplished. Need more time.",
                "expected_complete": False
            },
            {
                "content": "Analysis incomplete. Findings not yet summarized.",
                "expected_complete": False
            }
        ]
        
        for case in negation_cases:
            message = AIMessage(content=case["content"])
            result = self.detector.analyze_completion_patterns(message, strategy=DetectionStrategy.PATTERN_BASED)
            
            assert result.is_complete == case["expected_complete"], \
                f"Negation test failed for: {case['content']}"
            assert result.confidence < 0.6  # Should have low confidence for incomplete


class TestSemanticCompletionAnalysis:
    """Test semantic-based completion analysis."""
    
    def setup_method(self):
        """Set up semantic testing."""
        self.detector = CompletionDetector(debug_mode=True)
        self.detector.set_completion_threshold(0.6)
    
    def test_semantic_completion_signals(self):
        """Test semantic analysis of completion signals."""
        semantic_test_cases = [
            {
                "content": "Based on my comprehensive analysis, I have gathered sufficient evidence to conclude that the proposed solution is viable. The research demonstrates clear benefits and minimal risks.",
                "expected_complete": True,
                "description": "Conclusive analysis"
            },
            {
                "content": "Having examined all available data sources and consulted relevant literature, I can provide definitive recommendations for the next steps in this project.",
                "expected_complete": True, 
                "description": "Definitive recommendations"
            },
            {
                "content": "The investigation remains ongoing. Several questions require further exploration before I can reach conclusive findings.",
                "expected_complete": False,
                "description": "Ongoing investigation"
            },
            {
                "content": "Initial findings suggest interesting possibilities, but more research is needed to validate these hypotheses.",
                "expected_complete": False,
                "description": "Initial findings needing validation"
            }
        ]
        
        for case in semantic_test_cases:
            message = AIMessage(content=case["content"])
            result = self.detector.analyze_completion_patterns(message, strategy=DetectionStrategy.SEMANTIC_BASED)
            
            assert result.is_complete == case["expected_complete"], \
                f"Semantic test '{case['description']}' failed: expected {case['expected_complete']}, got {result.is_complete}"
    
    def test_semantic_context_understanding(self):
        """Test understanding of semantic context for completion."""
        # Test with different contexts
        contexts = [
            {
                "agent_type": "academic_researcher",
                "content": "The literature review is comprehensive and all relevant papers have been analyzed. Theoretical framework established.",
                "expected_complete": True
            },
            {
                "agent_type": "market_analyst", 
                "content": "Market segmentation analysis reveals three distinct customer groups. Competitive positioning assessed.",
                "expected_complete": True
            },
            {
                "agent_type": "technical_specialist",
                "content": "System architecture designed and performance benchmarks established. Implementation roadmap created.",
                "expected_complete": True
            },
            {
                "agent_type": "research_agent",
                "content": "Data collection phase ongoing. Several data sources still being processed.",
                "expected_complete": False
            }
        ]
        
        for context in contexts:
            message = AIMessage(content=context["content"])
            # Note: In real implementation, agent_type might be passed as context
            result = self.detector.analyze_completion_patterns(message, strategy=DetectionStrategy.SEMANTIC_BASED)
            
            assert result.is_complete == context["expected_complete"], \
                f"Context test for {context['agent_type']} failed"


class TestCombinedDetectionStrategies:
    """Test combined detection strategies that use multiple approaches."""
    
    def setup_method(self):
        """Set up combined strategy testing."""
        self.detector = CompletionDetector(debug_mode=True)
        self.detector.set_completion_threshold(0.6)
    
    def test_combined_strategy_accuracy(self):
        """Test accuracy of combined detection strategy."""
        combined_test_cases = [
            {
                "content": "Research complete. After comprehensive analysis of all available data, I have reached definitive conclusions with high confidence.",
                "expected_complete": True,
                "min_confidence": 0.8,
                "description": "Strong pattern + semantic signals"
            },
            {
                "content": "Task accomplished! The investigation demonstrates clear evidence supporting the proposed approach.",
                "expected_complete": True,
                "min_confidence": 0.7,
                "description": "Pattern + conclusive language"
            },
            {
                "content": "Work continues on this analysis. Some interesting findings but need more investigation.",
                "expected_complete": False,
                "max_confidence": 0.5,
                "description": "No completion patterns, ongoing work"
            },
            {
                "content": "Analysis complete. However, the results are inconclusive and require additional research.",
                "expected_complete": False,
                "max_confidence": 0.6,
                "description": "Conflicting signals - pattern vs semantic"
            }
        ]
        
        for case in combined_test_cases:
            message = AIMessage(content=case["content"])
            result = self.detector.analyze_completion_patterns(message, strategy=DetectionStrategy.COMBINED)
            
            assert result.is_complete == case["expected_complete"], \
                f"Combined test '{case['description']}' failed: expected {case['expected_complete']}, got {result.is_complete}"
            
            if case["expected_complete"] and "min_confidence" in case:
                assert result.confidence >= case["min_confidence"], \
                    f"Combined test confidence {result.confidence} below minimum {case['min_confidence']}"
            elif not case["expected_complete"] and "max_confidence" in case:
                assert result.confidence <= case["max_confidence"], \
                    f"Combined test confidence {result.confidence} above maximum {case['max_confidence']}"
    
    def test_strategy_comparison(self):
        """Test comparison of different detection strategies."""
        test_message = AIMessage(content="Research complete. Based on comprehensive analysis, all objectives have been accomplished with definitive results.")
        
        # Test all strategies
        pattern_result = self.detector.analyze_completion_patterns(test_message, strategy=DetectionStrategy.PATTERN_BASED)
        semantic_result = self.detector.analyze_completion_patterns(test_message, strategy=DetectionStrategy.SEMANTIC_BASED)
        combined_result = self.detector.analyze_completion_patterns(test_message, strategy=DetectionStrategy.COMBINED)
        
        # All should detect completion for this strong case
        assert pattern_result.is_complete
        assert semantic_result.is_complete 
        assert combined_result.is_complete
        
        # Combined should have highest or equal confidence
        assert combined_result.confidence >= max(pattern_result.confidence, semantic_result.confidence) * 0.9  # Allow small variance
    
    def test_conflicting_signals_resolution(self):
        """Test resolution of conflicting signals between strategies."""
        conflicting_cases = [
            {
                "content": "Task complete. But I'm not sure if this is the right approach and need more validation.",
                "description": "Pattern positive, semantic uncertain"
            },
            {
                "content": "The comprehensive analysis demonstrates conclusive evidence, though the work is not finished.",
                "description": "Semantic positive, pattern negative"
            },
            {
                "content": "Research finished successfully. However, additional investigation may be beneficial.",
                "description": "Mixed signals requiring resolution"
            }
        ]
        
        for case in conflicting_cases:
            message = AIMessage(content=case["content"])
            result = self.detector.analyze_completion_patterns(message, strategy=DetectionStrategy.COMBINED)
            
            # Combined strategy should provide reasoned resolution
            assert isinstance(result.is_complete, bool)
            assert 0.0 <= result.confidence <= 1.0
            assert len(result.reasoning) > 0
            
            print(f"Conflicting case '{case['description']}': Complete={result.is_complete}, Confidence={result.confidence:.2f}")


class TestCompletionDetectorConfiguration:
    """Test completion detector configuration and customization."""
    
    def setup_method(self):
        """Set up configuration testing."""
        self.detector = CompletionDetector(debug_mode=False)
    
    def test_threshold_configuration(self):
        """Test completion threshold configuration."""
        test_message = AIMessage(content="The work is mostly done.")
        
        # Test with different thresholds
        thresholds = [0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            self.detector.set_completion_threshold(threshold)
            result = self.detector.analyze_completion_patterns(test_message)
            
            # Completion decision should respect threshold
            if result.confidence >= threshold:
                assert result.is_complete
            else:
                assert not result.is_complete
    
    def test_debug_mode_output(self):
        """Test debug mode provides detailed information."""
        debug_detector = CompletionDetector(debug_mode=True)
        regular_detector = CompletionDetector(debug_mode=False)
        
        test_message = AIMessage(content="Analysis complete with comprehensive findings.")
        
        debug_result = debug_detector.analyze_completion_patterns(test_message)
        regular_result = regular_detector.analyze_completion_patterns(test_message)
        
        # Debug mode should provide more detailed reasoning
        assert len(debug_result.reasoning) >= len(regular_result.reasoning)
        assert debug_result.is_complete == regular_result.is_complete
        assert abs(debug_result.confidence - regular_result.confidence) < 0.1
    
    def test_custom_patterns_configuration(self):
        """Test configuration with custom completion patterns."""
        custom_indicators = [
            "evaluation finished",
            "assessment concluded",
            "review completed",
            "validation done",
            "testing accomplished"
        ]
        
        test_message = AIMessage(content="After thorough testing, the validation done and evaluation finished.")
        
        # Test with custom indicators
        result = self.detector.analyze_completion_patterns(
            test_message,
            custom_indicators=custom_indicators,
            strategy=DetectionStrategy.PATTERN_BASED
        )
        
        assert result.is_complete
        assert result.confidence > 0.7
        assert any("validation done" in reason or "evaluation finished" in reason 
                  for reason in result.reasoning)


class TestCompletionDetectorPerformance:
    """Test completion detector performance and reliability."""
    
    def setup_method(self):
        """Set up performance testing."""
        self.detector = CompletionDetector(debug_mode=False)
        self.detector.set_completion_threshold(0.6)
    
    def test_detection_speed(self):
        """Test detection speed for various message lengths."""
        message_lengths = [100, 500, 1000, 2000, 5000]
        
        for length in message_lengths:
            # Create message of specified length
            content = "This is a test message. " * (length // 25)
            content += "Research complete and analysis finished."
            
            message = AIMessage(content=content)
            
            # Time the detection
            start_time = time.time()
            result = self.detector.analyze_completion_patterns(message)
            detection_time = time.time() - start_time
            
            # Detection should be fast (< 1 second for all lengths)
            assert detection_time < 1.0, f"Detection took {detection_time:.3f}s for {length} char message"
            assert result.is_complete  # Should detect completion despite length
            
            print(f"Message length {length}: {detection_time:.3f}s")
    
    def test_consistency_across_runs(self):
        """Test consistency of detection across multiple runs."""
        test_cases = [
            AIMessage(content="Research complete. All objectives accomplished."),
            AIMessage(content="Still working on the analysis. More time needed."),
            AIMessage(content="Task accomplished with definitive results."),
            AIMessage(content="Investigation ongoing. Preliminary findings available.")
        ]
        
        for message in test_cases:
            results = []
            
            # Run detection multiple times
            for _ in range(10):
                result = self.detector.analyze_completion_patterns(message)
                results.append((result.is_complete, result.confidence))
            
            # Results should be consistent
            completion_values = [r[0] for r in results]
            confidence_values = [r[1] for r in results]
            
            assert all(c == completion_values[0] for c in completion_values), \
                "Completion detection inconsistent across runs"
            
            # Confidence should be very similar (within 0.05)
            max_confidence_diff = max(confidence_values) - min(confidence_values)
            assert max_confidence_diff < 0.05, \
                f"Confidence variance {max_confidence_diff:.3f} too high"
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable across many detections."""
        import gc
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run many detections
        test_message = AIMessage(content="Analysis complete with comprehensive findings and recommendations.")
        
        for i in range(100):
            self.detector.analyze_completion_patterns(test_message)
            
            # Force garbage collection every 20 iterations
            if i % 20 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (< 10MB)
        memory_mb = memory_increase / 1024 / 1024
        assert memory_mb < 10, f"Memory increased by {memory_mb:.2f}MB after 100 detections"


class TestCompletionDetectorEdgeCases:
    """Test edge cases and error handling for completion detector."""
    
    def setup_method(self):
        """Set up edge case testing."""
        self.detector = CompletionDetector(debug_mode=True)
    
    def test_empty_message_handling(self):
        """Test handling of empty or null messages."""
        # Empty message
        empty_message = AIMessage(content="")
        result = self.detector.analyze_completion_patterns(empty_message)
        
        assert not result.is_complete
        assert result.confidence == 0.0
        assert len(result.reasoning) > 0
        
        # Whitespace only message
        whitespace_message = AIMessage(content="   \n\t   ")
        result = self.detector.analyze_completion_patterns(whitespace_message)
        
        assert not result.is_complete
        assert result.confidence == 0.0
    
    def test_very_long_message_handling(self):
        """Test handling of very long messages."""
        # Create very long message (10KB)
        long_content = "This is a detailed research analysis. " * 250
        long_content += "Research complete and analysis finished."
        
        long_message = AIMessage(content=long_content)
        result = self.detector.analyze_completion_patterns(long_message)
        
        # Should still work correctly
        assert result.is_complete
        assert result.confidence > 0.6
        assert len(result.reasoning) > 0
    
    def test_non_english_content_handling(self):
        """Test handling of non-English content."""
        # Note: This is a basic test - real implementation might need i18n support
        non_english_cases = [
            AIMessage(content="La recherche est terminée. Research complete."),  # French + English
            AIMessage(content="研究完了。Analysis finished."),  # Japanese + English
            AIMessage(content="Forschung abgeschlossen. Task accomplished.")  # German + English
        ]
        
        for message in non_english_cases:
            result = self.detector.analyze_completion_patterns(message)
            
            # Should detect English completion indicators
            assert result.is_complete
            assert result.confidence > 0.5
    
    def test_malformed_input_handling(self):
        """Test handling of malformed inputs."""
        # Test with None content (should be handled gracefully)
        try:
            none_message = AIMessage(content=None)
            result = self.detector.analyze_completion_patterns(none_message)
            
            assert not result.is_complete
            assert result.confidence == 0.0
        except Exception as e:
            # If exception is raised, it should be a reasonable one
            assert "content" in str(e).lower() or "none" in str(e).lower()
    
    def test_special_characters_handling(self):
        """Test handling of messages with special characters and formatting."""
        special_cases = [
            AIMessage(content="## Research Complete ##\n\n**Analysis finished** with *comprehensive* findings!"),
            AIMessage(content="Task accomplished ✅ Research complete 🎉"),
            AIMessage(content="RESEARCH COMPLETE!!! 🔬📊 Analysis finished!!! 📈📋"),
            AIMessage(content="Research complete... though... more could be done... but sufficient for now.")
        ]
        
        for message in special_cases:
            result = self.detector.analyze_completion_patterns(message)
            
            # Should detect completion despite special formatting
            assert result.is_complete
            assert result.confidence > 0.6
    
    def test_invalid_strategy_handling(self):
        """Test handling of invalid detection strategies."""
        test_message = AIMessage(content="Research complete.")
        
        # Test with invalid strategy (should fall back to default)
        try:
            result = self.detector.analyze_completion_patterns(test_message, strategy="invalid_strategy")
            # Should either work with fallback or raise appropriate error
            if result:
                assert isinstance(result, CompletionResult)
        except ValueError as e:
            assert "strategy" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])