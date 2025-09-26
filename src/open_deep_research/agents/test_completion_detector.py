#!/usr/bin/env python3
"""Comprehensive test suite for CompletionDetector.

This test suite validates the completion detection system for sequential
multi-agent workflows, ensuring robust detection of task completion
without requiring explicit handoff tools.
"""

import re
import unittest

from completion_detector import (
    CompletionDetector,
    CompletionPattern,
    DetectionResult,
    DetectionStrategy,
)
from langchain_core.messages import AIMessage


class TestCompletionDetector(unittest.TestCase):
    """Test suite for CompletionDetector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = CompletionDetector(debug_mode=False)
        self.test_messages = {
            'strong_completion': AIMessage(
                content="The research is complete. All findings have been summarized and the investigation has concluded."
            ),
            'weak_completion': AIMessage(
                content="I am still analyzing the data and need to conduct more research."
            ),
            'conclusion_structure': AIMessage(
                content="In conclusion, after extensive analysis, all objectives have been met. The research is finished."
            ),
            'handoff_signal': AIMessage(
                content="Analysis complete. Ready to hand off to the next agent for further processing."
            ),
            'empty_message': AIMessage(content=""),
            'no_completion': AIMessage(
                content="The data shows interesting patterns that require further investigation."
            )
        }
    
    def test_basic_completion_detection(self):
        """Test basic completion detection functionality."""
        # Strong completion signals should be detected
        result = self.detector.analyze_completion_patterns(self.test_messages['strong_completion'])
        self.assertTrue(result.is_complete)
        self.assertGreater(result.confidence, 0.5)
        self.assertGreater(len(result.matched_patterns), 0)
        
        # Weak signals should not be detected as complete
        result = self.detector.analyze_completion_patterns(self.test_messages['weak_completion'])
        self.assertFalse(result.is_complete)
        self.assertLess(result.confidence, 0.5)
    
    def test_is_agent_complete_method(self):
        """Test the is_agent_complete convenience method."""
        self.assertTrue(
            self.detector.is_agent_complete(self.test_messages['strong_completion'])
        )
        self.assertFalse(
            self.detector.is_agent_complete(self.test_messages['weak_completion'])
        )
    
    def test_get_completion_confidence_method(self):
        """Test the get_completion_confidence method."""
        confidence = self.detector.get_completion_confidence(
            self.test_messages['strong_completion']
        )
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_custom_completion_indicators(self):
        """Test custom completion indicators functionality."""
        custom_indicators = ['analysis finished', 'work done']
        
        # Test message that matches custom indicators
        custom_message = AIMessage(content="The analysis is finished and all work is done.")
        result = self.detector.analyze_completion_patterns(
            custom_message, 
            custom_indicators=custom_indicators
        )
        
        self.assertTrue(result.is_complete)
        # Should have custom pattern matches
        custom_matches = [p for p in result.matched_patterns if 'Custom:' in p]
        self.assertGreater(len(custom_matches), 0)
    
    def test_detection_strategies(self):
        """Test different detection strategies."""
        test_message = self.test_messages['conclusion_structure']
        
        strategies = [
            DetectionStrategy.CONTENT_PATTERNS,
            DetectionStrategy.TOOL_USAGE_PATTERNS,
            DetectionStrategy.MESSAGE_STRUCTURE,
            DetectionStrategy.COMBINED
        ]
        
        results = {}
        for strategy in strategies:
            result = self.detector.analyze_completion_patterns(test_message, strategy=strategy)
            results[strategy] = result
            
            # All results should be valid DetectionResult objects
            self.assertIsInstance(result, DetectionResult)
            self.assertIsInstance(result.confidence, float)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
        
        # Combined strategy should generally perform well
        combined_result = results[DetectionStrategy.COMBINED]
        self.assertIsInstance(combined_result.matched_patterns, list)
    
    def test_tool_usage_pattern_detection(self):
        """Test tool usage pattern detection."""
        # Message without tool calls
        no_tools_msg = AIMessage(content="This is a substantial analysis without any tool calls.")
        result = self.detector.analyze_completion_patterns(
            no_tools_msg, 
            strategy=DetectionStrategy.TOOL_USAGE_PATTERNS
        )
        
        # Should detect no tool calls as potential completion indicator
        self.assertGreater(result.confidence, 0.0)
        self.assertIn("No tool calls", ' '.join(result.matched_patterns))
    
    def test_message_structure_detection(self):
        """Test message structure-based detection."""
        structured_message = AIMessage(
            content="After extensive research, I can provide the following analysis. "
                   "The data reveals important patterns. In conclusion, all objectives "
                   "have been achieved and the research is complete."
        )
        
        result = self.detector.analyze_completion_patterns(
            structured_message,
            strategy=DetectionStrategy.MESSAGE_STRUCTURE
        )
        
        # Should detect conclusion structure
        self.assertGreater(result.confidence, 0.0)
        self.assertIsInstance(result.analysis_details, dict)
    
    def test_custom_pattern_addition(self):
        """Test adding custom completion patterns."""
        initial_count = len(self.detector._default_patterns)
        
        # Add custom pattern
        custom_pattern = CompletionPattern(
            pattern=r"mission\s+accomplished",
            weight=0.95,
            description="Mission completion indicator"
        )
        
        self.detector.add_custom_pattern(custom_pattern)
        
        # Should have more patterns
        self.assertEqual(len(self.detector._default_patterns), initial_count + 1)
        
        # Test the new pattern
        test_message = AIMessage(content="The mission is accomplished successfully.")
        result = self.detector.analyze_completion_patterns(test_message)
        
        # Should detect the custom pattern
        self.assertIn("Mission completion indicator", result.matched_patterns)
    
    def test_threshold_configuration(self):
        """Test completion threshold configuration."""
        original_threshold = self.detector.completion_threshold
        
        # Set higher threshold
        new_threshold = 0.8
        self.detector.set_completion_threshold(new_threshold)
        self.assertEqual(self.detector.completion_threshold, new_threshold)
        
        # Test with threshold that's too high/low
        with self.assertRaises(ValueError):
            self.detector.set_completion_threshold(1.5)
        
        with self.assertRaises(ValueError):
            self.detector.set_completion_threshold(-0.1)
        
        # Restore original threshold
        self.detector.set_completion_threshold(original_threshold)
    
    def test_empty_and_invalid_messages(self):
        """Test handling of empty and invalid messages."""
        # Empty message
        result = self.detector.analyze_completion_patterns(self.test_messages['empty_message'])
        self.assertFalse(result.is_complete)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.reasoning, "Empty message content")
        
        # Invalid message type
        result = self.detector.analyze_completion_patterns("not a message")
        self.assertFalse(result.is_complete)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("Invalid message type", result.reasoning)
    
    def test_pattern_statistics(self):
        """Test pattern statistics functionality."""
        stats = self.detector.get_pattern_statistics()
        
        # Should contain expected keys
        expected_keys = [
            'total_patterns', 
            'compiled_patterns', 
            'strategy_distribution',
            'pattern_descriptions',
            'completion_threshold',
            'weights'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Should have reasonable values
        self.assertGreater(stats['total_patterns'], 0)
        self.assertGreater(stats['compiled_patterns'], 0)
        self.assertIsInstance(stats['strategy_distribution'], dict)
        self.assertIsInstance(stats['pattern_descriptions'], list)
    
    def test_regex_pattern_compilation(self):
        """Test regex pattern compilation and matching."""
        # Test that patterns compile correctly
        for i, pattern_config in enumerate(self.detector._default_patterns):
            pattern_key = f"default_{i}"
            if pattern_key in self.detector._compiled_patterns:
                compiled_pattern = self.detector._compiled_patterns[pattern_key]
                
                # Should be a compiled regex pattern
                self.assertIsInstance(compiled_pattern, re.Pattern)
                
                # Should not raise exceptions when searching
                try:
                    compiled_pattern.search("test content")
                except Exception as e:
                    self.fail(f"Pattern {pattern_key} failed to search: {e}")
    
    def test_custom_indicator_patterns(self):
        """Test custom indicator pattern matching."""
        # Test literal string matching
        literal_indicators = ['exact match phrase']
        message = AIMessage(content="This contains an exact match phrase in the text.")
        
        result = self.detector.analyze_completion_patterns(message, literal_indicators)
        custom_matches = [p for p in result.matched_patterns if 'Custom:' in p]
        self.assertGreater(len(custom_matches), 0)
        
        # Test regex pattern matching
        regex_indicators = [r'task\s+\d+\s+complete']
        message = AIMessage(content="Task 42 is complete and ready for review.")
        
        result = self.detector.analyze_completion_patterns(message, regex_indicators)
        custom_matches = [p for p in result.matched_patterns if 'Custom:' in p]
        self.assertGreater(len(custom_matches), 0)
    
    def test_confidence_score_bounds(self):
        """Test that confidence scores are properly bounded."""
        test_messages = [
            self.test_messages['strong_completion'],
            self.test_messages['weak_completion'],
            self.test_messages['conclusion_structure'],
            self.test_messages['no_completion']
        ]
        
        for message in test_messages:
            result = self.detector.analyze_completion_patterns(message)
            
            # Confidence should always be between 0 and 1
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
    
    def test_detection_result_structure(self):
        """Test that DetectionResult objects have proper structure."""
        result = self.detector.analyze_completion_patterns(
            self.test_messages['strong_completion']
        )
        
        # Should have all required fields
        self.assertIsInstance(result.is_complete, bool)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.detection_strategy, DetectionStrategy)
        self.assertIsInstance(result.matched_patterns, list)
        self.assertIsInstance(result.analysis_details, dict)
        self.assertIsInstance(result.reasoning, str)
        
        # Reasoning should be non-empty
        self.assertTrue(result.reasoning.strip())


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for real-world scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.detector = CompletionDetector()
    
    def test_academic_research_workflow(self):
        """Test completion detection in academic research workflow."""
        academic_indicators = [
            'literature review complete',
            'methodology established', 
            'data analysis finished'
        ]
        
        workflow_messages = [
            "Beginning systematic literature review on machine learning applications.",
            "Literature review is complete with 50+ papers analyzed. Now establishing methodology.",
            "Methodology has been established. Beginning data collection and analysis phase.",
            "Data analysis is finished. All research objectives have been met and findings compiled."
        ]
        
        completion_detected = []
        for content in workflow_messages:
            message = AIMessage(content=content)
            is_complete = self.detector.is_agent_complete(message, academic_indicators)
            completion_detected.append(is_complete)
        
        # Only the last message should indicate completion
        self.assertEqual(completion_detected, [False, False, False, True])
    
    def test_market_analysis_workflow(self):
        """Test completion detection in market analysis workflow."""
        market_indicators = [
            'market assessment complete',
            'competitor analysis done',
            'business case validated'
        ]
        
        final_message = AIMessage(
            content="Market assessment is complete with comprehensive competitor analysis done. "
                   "The business case has been validated and is ready for stakeholder review."
        )
        
        result = self.detector.analyze_completion_patterns(final_message, market_indicators)
        
        self.assertTrue(result.is_complete)
        self.assertGreater(result.confidence, 0.7)  # Should be high confidence
        
        # Should match multiple custom indicators
        custom_matches = [p for p in result.matched_patterns if 'Custom:' in p]
        self.assertGreaterEqual(len(custom_matches), 2)
    
    def test_technical_feasibility_workflow(self):
        """Test completion detection in technical feasibility workflow."""
        technical_indicators = [
            'architecture review complete',
            'implementation plan ready',
            'risk assessment done'
        ]
        
        # Message with mixed completion signals
        mixed_message = AIMessage(
            content="The architecture review is complete and implementation plan is ready. "
                   "However, we still need to conduct additional risk assessment for the deployment phase."
        )
        
        result = self.detector.analyze_completion_patterns(mixed_message, technical_indicators)
        
        # Should detect partial completion but maybe not full completion
        # depending on threshold and mixed signals
        self.assertGreater(result.confidence, 0.3)
        
        # Should match some but not all indicators
        custom_matches = [p for p in result.matched_patterns if 'Custom:' in p]
        self.assertGreater(len(custom_matches), 0)


def run_tests():
    """Run the complete test suite."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_suite.addTest(unittest.makeSuite(TestCompletionDetector))
    test_suite.addTest(unittest.makeSuite(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    
    success = run_tests()
    
    if success:
        pass
    else:
        pass
        
    exit(0 if success else 1)