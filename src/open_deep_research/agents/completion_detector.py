"""Completion detection system for sequential multi-agent workflows.

This module provides robust detection of agent task completion without requiring
explicit handoff tools, supporting custom completion indicators and multiple
detection strategies for production-ready sequential agent workflows.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Pattern, Union

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class DetectionStrategy(Enum):
    """Available completion detection strategies."""
    
    CONTENT_PATTERNS = "content_patterns"
    TOOL_USAGE_PATTERNS = "tool_usage_patterns"
    MESSAGE_STRUCTURE = "message_structure"
    COMBINED = "combined"


class CompletionConfidence(Enum):
    """Completion confidence levels."""
    
    VERY_LOW = 0.0
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 1.0


@dataclass
class DetectionResult:
    """Result of completion detection analysis."""
    
    is_complete: bool
    confidence: float
    detection_strategy: DetectionStrategy
    matched_patterns: List[str]
    analysis_details: Dict[str, Union[str, float, int]]
    reasoning: str


@dataclass
class CompletionPattern:
    """Configuration for a completion detection pattern."""
    
    pattern: Union[str, Pattern]
    weight: float = 1.0
    strategy: DetectionStrategy = DetectionStrategy.CONTENT_PATTERNS
    case_sensitive: bool = False
    requires_context: bool = False
    description: str = ""


class CompletionDetector:
    """Robust completion detection for sequential multi-agent workflows.
    
    This detector analyzes AI messages to determine when an agent has completed
    its task using multiple detection strategies including content patterns,
    tool usage analysis, and message structure evaluation.
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the completion detector.
        
        Args:
            debug_mode: Enable detailed logging for debugging completion detection
        """
        self.debug_mode = debug_mode
        self._compiled_patterns: Dict[str, Pattern] = {}
        
        # Initialize default completion patterns
        self._default_patterns = self._build_default_patterns()
        self._compile_patterns()
        
        # Detection thresholds
        self.completion_threshold = 0.5  # Minimum confidence for completion
        self.high_confidence_threshold = 0.75  # High confidence threshold
        
        # Message analysis weights
        self.weights = {
            'explicit_completion': 0.5,
            'conclusion_patterns': 0.2,
            'content_structure': 0.1,
            'tool_usage_patterns': 0.2
        }
        
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("CompletionDetector initialized in debug mode")
    
    def _build_default_patterns(self) -> List[CompletionPattern]:
        """Build the default set of completion detection patterns."""
        patterns = [
            # Explicit completion phrases (high confidence)
            CompletionPattern(
                pattern=r'\b(?:research|analysis|investigation|study|examination)\s+(?:is\s+)?(?:complete|completed|finished|done|concluded)\b',
                weight=0.9,
                strategy=DetectionStrategy.CONTENT_PATTERNS,
                description="Explicit completion statements"
            ),
            CompletionPattern(
                pattern=r'\b(?:findings|results|conclusions)\s+(?:have been\s+)?(?:summarized|compiled|presented|delivered)\b',
                weight=0.85,
                strategy=DetectionStrategy.CONTENT_PATTERNS,
                description="Results delivery statements"
            ),
            CompletionPattern(
                pattern=r'\b(?:no\s+(?:more|additional|further)|all\s+(?:available|relevant))\s+(?:sources|information|data|research)\b',
                weight=0.8,
                strategy=DetectionStrategy.CONTENT_PATTERNS,
                description="Exhaustive research indicators"
            ),
            CompletionPattern(
                pattern=r'\b(?:investigation|analysis|research)\s+(?:has\s+)?concluded\b',
                weight=0.85,
                strategy=DetectionStrategy.CONTENT_PATTERNS,
                description="Investigation conclusion statements"
            ),
            
            # Conclusion and summary patterns (medium-high confidence)
            CompletionPattern(
                pattern=r'\b(?:in\s+conclusion|to\s+summarize|in\s+summary|finally|lastly)\b',
                weight=0.7,
                strategy=DetectionStrategy.MESSAGE_STRUCTURE,
                description="Conclusion transition phrases"
            ),
            CompletionPattern(
                pattern=r'\b(?:final\s+(?:thoughts|analysis|assessment|findings)|overall\s+(?:conclusion|assessment))\b',
                weight=0.75,
                strategy=DetectionStrategy.MESSAGE_STRUCTURE,
                description="Final assessment indicators"
            ),
            CompletionPattern(
                pattern=r'\b(?:comprehensive\s+(?:review|analysis|examination)|thorough\s+(?:investigation|research))\s+(?:has\s+been\s+)?(?:completed|conducted|performed)\b',
                weight=0.8,
                strategy=DetectionStrategy.CONTENT_PATTERNS,
                description="Comprehensive work completion"
            ),
            
            # Research state indicators (medium confidence)
            CompletionPattern(
                pattern=r'\b(?:all\s+(?:key|relevant|major|important)\s+(?:aspects|areas|topics|questions)\s+(?:have\s+been\s+)?(?:covered|addressed|explored|examined))\b',
                weight=0.65,
                strategy=DetectionStrategy.CONTENT_PATTERNS,
                description="Comprehensive coverage indicators"
            ),
            CompletionPattern(
                pattern=r'\b(?:sufficient|adequate|comprehensive)\s+(?:information|data|evidence|research)\s+(?:has\s+been\s+)?(?:gathered|collected|obtained)\b',
                weight=0.6,
                strategy=DetectionStrategy.CONTENT_PATTERNS,
                description="Information sufficiency indicators"
            ),
            
            # Handoff and transition patterns (medium confidence)
            CompletionPattern(
                pattern=r'\b(?:ready\s+(?:for|to)|prepared\s+(?:for|to)|can\s+now\s+(?:proceed|move|advance))\s+(?:next\s+(?:phase|stage|step)|handoff|transition)\b',
                weight=0.65,
                strategy=DetectionStrategy.CONTENT_PATTERNS,
                description="Transition readiness indicators"
            ),
            CompletionPattern(
                pattern=r'\b(?:passing|handing\s+off|transferring|delegating)\s+(?:to|control|responsibility)\b',
                weight=0.7,
                strategy=DetectionStrategy.CONTENT_PATTERNS,
                description="Explicit handoff language"
            ),
        ]
        
        return patterns
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self._compiled_patterns.clear()
        
        for i, pattern_config in enumerate(self._default_patterns):
            pattern_key = f"default_{i}"
            
            if isinstance(pattern_config.pattern, str):
                flags = 0 if pattern_config.case_sensitive else re.IGNORECASE
                try:
                    compiled = re.compile(pattern_config.pattern, flags)
                    self._compiled_patterns[pattern_key] = compiled
                    
                    if self.debug_mode:
                        logger.debug(f"Compiled pattern {pattern_key}: {pattern_config.description}")
                        
                except re.error as e:
                    logger.warning(f"Failed to compile pattern {pattern_key}: {e}")
            else:
                self._compiled_patterns[pattern_key] = pattern_config.pattern
    
    def is_agent_complete(
        self, 
        message: AIMessage, 
        custom_indicators: Optional[List[str]] = None,
        strategy: DetectionStrategy = DetectionStrategy.COMBINED
    ) -> bool:
        """Determine if an agent has completed its task.
        
        Args:
            message: The AI message to analyze for completion indicators
            custom_indicators: Custom completion patterns specific to the agent
            strategy: Detection strategy to use for analysis
            
        Returns:
            True if the agent appears to have completed its task, False otherwise
        """
        result = self.analyze_completion_patterns(message, custom_indicators, strategy)
        
        is_complete = result.confidence >= self.completion_threshold
        
        if self.debug_mode:
            logger.debug(
                f"Completion check: {is_complete} (confidence: {result.confidence:.3f}, "
                f"threshold: {self.completion_threshold})"
            )
        
        return is_complete
    
    def get_completion_confidence(
        self, 
        message: AIMessage, 
        custom_indicators: Optional[List[str]] = None,
        strategy: DetectionStrategy = DetectionStrategy.COMBINED
    ) -> float:
        """Get confidence score for completion detection.
        
        Args:
            message: The AI message to analyze
            custom_indicators: Custom completion patterns specific to the agent
            strategy: Detection strategy to use for analysis
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        result = self.analyze_completion_patterns(message, custom_indicators, strategy)
        return result.confidence
    
    def analyze_completion_patterns(
        self, 
        message: AIMessage, 
        custom_indicators: Optional[List[str]] = None,
        strategy: DetectionStrategy = DetectionStrategy.COMBINED
    ) -> DetectionResult:
        """Comprehensive analysis of completion patterns in a message.
        
        Args:
            message: The AI message to analyze
            custom_indicators: Custom completion patterns specific to the agent
            strategy: Detection strategy to use for analysis
            
        Returns:
            DetectionResult with detailed analysis of completion indicators
        """
        if not isinstance(message, AIMessage):
            logger.warning(f"Expected AIMessage, got {type(message)}")
            return DetectionResult(
                is_complete=False,
                confidence=0.0,
                detection_strategy=strategy,
                matched_patterns=[],
                analysis_details={"error": "Invalid message type"},
                reasoning="Message is not an AIMessage instance"
            )
        
        content = message.content if message.content else ""
        if not content.strip():
            return DetectionResult(
                is_complete=False,
                confidence=0.0,
                detection_strategy=strategy,
                matched_patterns=[],
                analysis_details={"content_length": 0},
                reasoning="Empty message content"
            )
        
        # Strategy-specific analysis
        if strategy == DetectionStrategy.CONTENT_PATTERNS:
            return self._analyze_content_patterns(content, custom_indicators, message)
        elif strategy == DetectionStrategy.TOOL_USAGE_PATTERNS:
            return self._analyze_tool_usage_patterns(message)
        elif strategy == DetectionStrategy.MESSAGE_STRUCTURE:
            return self._analyze_message_structure(content, message)
        else:  # COMBINED strategy
            return self._analyze_combined_patterns(content, custom_indicators, message)
    
    def _analyze_content_patterns(
        self, 
        content: str, 
        custom_indicators: Optional[List[str]], 
        message: AIMessage
    ) -> DetectionResult:
        """Analyze content-based completion patterns."""
        matched_patterns = []
        pattern_scores = []
        
        # Check default patterns
        for i, pattern_config in enumerate(self._default_patterns):
            if pattern_config.strategy != DetectionStrategy.CONTENT_PATTERNS:
                continue
                
            pattern_key = f"default_{i}"
            if pattern_key in self._compiled_patterns:
                compiled_pattern = self._compiled_patterns[pattern_key]
                
                if compiled_pattern.search(content):
                    matched_patterns.append(pattern_config.description)
                    pattern_scores.append(pattern_config.weight)
                    
                    if self.debug_mode:
                        logger.debug(f"Matched pattern: {pattern_config.description} (weight: {pattern_config.weight})")
        
        # Check custom indicators
        if custom_indicators:
            custom_weight = 0.9  # High weight for custom indicators
            for indicator in custom_indicators:
                if self._check_custom_indicator(content, indicator):
                    matched_patterns.append(f"Custom: {indicator}")
                    pattern_scores.append(custom_weight)
                    
                    if self.debug_mode:
                        logger.debug(f"Matched custom indicator: {indicator}")
        
        # Calculate confidence - use max score approach with normalization
        if pattern_scores:
            # Take the highest scoring pattern and apply diminishing returns for additional patterns
            max_score = max(pattern_scores)
            bonus_scores = sum(score * 0.1 for score in pattern_scores[1:])  # Additional patterns add small bonus
            confidence = min(max_score + bonus_scores, 1.0)
        else:
            confidence = 0.0
        
        return DetectionResult(
            is_complete=confidence >= self.completion_threshold,
            confidence=confidence,
            detection_strategy=DetectionStrategy.CONTENT_PATTERNS,
            matched_patterns=matched_patterns,
            analysis_details={
                "content_length": len(content),
                "pattern_scores": pattern_scores,
                "max_score": max(pattern_scores) if pattern_scores else 0.0,
                "patterns_checked": len([p for p in self._default_patterns if p.strategy == DetectionStrategy.CONTENT_PATTERNS]) + (len(custom_indicators) if custom_indicators else 0)
            },
            reasoning=f"Content pattern analysis yielded {len(matched_patterns)} matches with confidence {confidence:.3f}"
        )
    
    def _analyze_content_patterns_comprehensive(
        self, 
        content: str, 
        custom_indicators: Optional[List[str]], 
        message: AIMessage
    ) -> DetectionResult:
        """Comprehensive content analysis including all pattern types."""
        matched_patterns = []
        pattern_scores = []
        
        # Check ALL patterns (not just content patterns)
        for i, pattern_config in enumerate(self._default_patterns):
            pattern_key = f"default_{i}"
            if pattern_key in self._compiled_patterns:
                compiled_pattern = self._compiled_patterns[pattern_key]
                
                if compiled_pattern.search(content):
                    matched_patterns.append(pattern_config.description)
                    pattern_scores.append(pattern_config.weight)
                    
                    if self.debug_mode:
                        logger.debug(f"Matched pattern: {pattern_config.description} (weight: {pattern_config.weight})")
        
        # Check custom indicators
        if custom_indicators:
            custom_weight = 0.9  # High weight for custom indicators
            for indicator in custom_indicators:
                if self._check_custom_indicator(content, indicator):
                    matched_patterns.append(f"Custom: {indicator}")
                    pattern_scores.append(custom_weight)
                    
                    if self.debug_mode:
                        logger.debug(f"Matched custom indicator: {indicator}")
        
        # Calculate confidence - use max score approach with normalization
        if pattern_scores:
            # Take the highest scoring pattern and apply diminishing returns for additional patterns
            max_score = max(pattern_scores)
            bonus_scores = sum(score * 0.1 for score in pattern_scores[1:])  # Additional patterns add small bonus
            confidence = min(max_score + bonus_scores, 1.0)
        else:
            confidence = 0.0
        
        return DetectionResult(
            is_complete=confidence >= self.completion_threshold,
            confidence=confidence,
            detection_strategy=DetectionStrategy.CONTENT_PATTERNS,
            matched_patterns=matched_patterns,
            analysis_details={
                "content_length": len(content),
                "pattern_scores": pattern_scores,
                "max_score": max(pattern_scores) if pattern_scores else 0.0,
                "patterns_checked": len(self._default_patterns) + (len(custom_indicators) if custom_indicators else 0)
            },
            reasoning=f"Comprehensive content analysis yielded {len(matched_patterns)} matches with confidence {confidence:.3f}"
        )
    
    def _analyze_tool_usage_patterns(self, message: AIMessage) -> DetectionResult:
        """Analyze tool usage patterns for completion detection."""
        analysis_details = {
            "has_tool_calls": bool(message.tool_calls),
            "tool_call_count": len(message.tool_calls) if message.tool_calls else 0,
            "content_length": len(message.content) if message.content else 0
        }
        
        matched_patterns = []
        confidence = 0.0
        
        # No tool calls suggests completion of interactive phase
        if not message.tool_calls:
            matched_patterns.append("No tool calls (potential completion)")
            confidence += 0.4
            
            # Higher confidence if substantial content exists
            if analysis_details["content_length"] > 200:
                matched_patterns.append("Substantial content without tool calls")
                confidence += 0.3
        
        # Check for completion-indicating tool usage patterns
        if message.tool_calls:
            analysis_details["tool_names"] = [tc.get("name", "unknown") for tc in message.tool_calls]
            
            # think_tool usage often indicates reflective conclusion
            if any("think" in tc.get("name", "").lower() for tc in message.tool_calls):
                matched_patterns.append("Reflective thinking tool usage")
                confidence += 0.2
        
        return DetectionResult(
            is_complete=confidence >= self.completion_threshold,
            confidence=confidence,
            detection_strategy=DetectionStrategy.TOOL_USAGE_PATTERNS,
            matched_patterns=matched_patterns,
            analysis_details=analysis_details,
            reasoning=f"Tool usage analysis: {len(matched_patterns)} indicators, confidence {confidence:.3f}"
        )
    
    def _analyze_message_structure(self, content: str, message: AIMessage) -> DetectionResult:
        """Analyze message structure for completion indicators."""
        matched_patterns = []
        confidence = 0.0
        
        # Structural analysis
        lines = content.split('\n')
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        
        analysis_details = {
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        }
        
        # Check for conclusion structure patterns
        for pattern_config in self._default_patterns:
            if pattern_config.strategy != DetectionStrategy.MESSAGE_STRUCTURE:
                continue
                
            f"structure_{id(pattern_config)}"
            
            # Compile on-demand for structure patterns
            if isinstance(pattern_config.pattern, str):
                try:
                    compiled = re.compile(pattern_config.pattern, re.IGNORECASE)
                    if compiled.search(content):
                        matched_patterns.append(pattern_config.description)
                        confidence += pattern_config.weight * 0.7  # Slightly lower weight for structure
                except re.error:
                    continue
        
        # Structural completion indicators
        if len(paragraphs) >= 3:
            # Look for conclusion-like final paragraph
            final_paragraph = paragraphs[-1].lower()
            conclusion_words = ['conclusion', 'summary', 'overall', 'finally', 'therefore', 'thus']
            
            if any(word in final_paragraph for word in conclusion_words):
                matched_patterns.append("Conclusion-structured final paragraph")
                confidence += 0.3
        
        # Long-form comprehensive content suggests completion
        if analysis_details["sentence_count"] > 10 and analysis_details["paragraph_count"] > 2:
            matched_patterns.append("Comprehensive structured content")
            confidence += 0.2
        
        confidence = min(confidence, 1.0)
        
        return DetectionResult(
            is_complete=confidence >= self.completion_threshold,
            confidence=confidence,
            detection_strategy=DetectionStrategy.MESSAGE_STRUCTURE,
            matched_patterns=matched_patterns,
            analysis_details=analysis_details,
            reasoning=f"Structure analysis: {len(matched_patterns)} indicators, confidence {confidence:.3f}"
        )
    
    def _analyze_combined_patterns(
        self, 
        content: str, 
        custom_indicators: Optional[List[str]], 
        message: AIMessage
    ) -> DetectionResult:
        """Combined analysis using all detection strategies."""
        # Run individual analyses (content analysis includes all pattern types)
        content_result = self._analyze_content_patterns_comprehensive(content, custom_indicators, message)
        tool_result = self._analyze_tool_usage_patterns(message)
        structure_result = self._analyze_message_structure(content, message)
        
        # Combine results with weighted scoring
        combined_confidence = (
            content_result.confidence * self.weights['explicit_completion'] +
            structure_result.confidence * self.weights['conclusion_patterns'] +
            tool_result.confidence * self.weights['tool_usage_patterns'] +
            # Content structure component from combined analysis
            (content_result.confidence * structure_result.confidence) * self.weights['content_structure']
        )
        
        # Aggregate matched patterns
        all_matched_patterns = (
            content_result.matched_patterns + 
            tool_result.matched_patterns + 
            structure_result.matched_patterns
        )
        
        # Combine analysis details
        combined_details = {
            "content_analysis": content_result.analysis_details,
            "tool_analysis": tool_result.analysis_details,
            "structure_analysis": structure_result.analysis_details,
            "combined_confidence": combined_confidence,
            "individual_confidences": {
                "content": content_result.confidence,
                "tool_usage": tool_result.confidence,
                "structure": structure_result.confidence
            }
        }
        
        reasoning = (
            f"Combined analysis: content={content_result.confidence:.3f}, "
            f"tools={tool_result.confidence:.3f}, structure={structure_result.confidence:.3f}, "
            f"final={combined_confidence:.3f}"
        )
        
        return DetectionResult(
            is_complete=combined_confidence >= self.completion_threshold,
            confidence=combined_confidence,
            detection_strategy=DetectionStrategy.COMBINED,
            matched_patterns=all_matched_patterns,
            analysis_details=combined_details,
            reasoning=reasoning
        )
    
    def _check_custom_indicator(self, content: str, indicator: str) -> bool:
        """Check if a custom completion indicator is present in content."""
        try:
            # First try as literal substring (case insensitive)
            if indicator.lower() in content.lower():
                return True
            
            # Then try as regex pattern
            try:
                pattern = re.compile(indicator, re.IGNORECASE)
                return pattern.search(content) is not None
            except re.error:
                # If regex compilation fails, fall back to substring match
                return indicator.lower() in content.lower()
                
        except Exception:
            # Safety fallback for any unexpected errors
            return False
    
    def add_custom_pattern(self, pattern: CompletionPattern) -> None:
        """Add a custom completion pattern to the detector.
        
        Args:
            pattern: CompletionPattern configuration to add
        """
        # Add to patterns list
        self._default_patterns.append(pattern)
        
        # Compile the new pattern
        pattern_key = f"custom_{len(self._default_patterns)}"
        
        if isinstance(pattern.pattern, str):
            flags = 0 if pattern.case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(pattern.pattern, flags)
                self._compiled_patterns[pattern_key] = compiled
                
                if self.debug_mode:
                    logger.debug(f"Added custom pattern {pattern_key}: {pattern.description}")
                    
            except re.error as e:
                logger.warning(f"Failed to compile custom pattern {pattern_key}: {e}")
        else:
            self._compiled_patterns[pattern_key] = pattern.pattern
    
    def set_completion_threshold(self, threshold: float) -> None:
        """Set the completion confidence threshold.
        
        Args:
            threshold: Confidence threshold between 0.0 and 1.0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.completion_threshold = threshold
        
        if self.debug_mode:
            logger.debug(f"Updated completion threshold to {threshold}")
    
    def get_pattern_statistics(self) -> Dict[str, Union[int, List[str]]]:
        """Get statistics about loaded completion patterns.
        
        Returns:
            Dictionary with pattern statistics and descriptions
        """
        pattern_descriptions = [pattern.description for pattern in self._default_patterns]
        strategy_counts = {}
        
        for pattern in self._default_patterns:
            strategy = pattern.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_patterns": len(self._default_patterns),
            "compiled_patterns": len(self._compiled_patterns),
            "strategy_distribution": strategy_counts,
            "pattern_descriptions": pattern_descriptions,
            "completion_threshold": self.completion_threshold,
            "weights": self.weights
        }