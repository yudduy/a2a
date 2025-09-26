"""Completion analyzer for intelligent handoff decisions in sequential workflows.

This module analyzes agent completion signals and provides insights for
dynamic sequence modification and next agent selection.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from open_deep_research.agents import CompletionDetector
from open_deep_research.state import SequentialSupervisorState

logger = logging.getLogger(__name__)


class CompletionAnalysis:
    """Results from completion analysis."""
    
    def __init__(
        self,
        is_complete: bool,
        confidence: float,
        completion_indicators: List[str],
        insights_extracted: List[str],
        quality_score: float,
        research_depth: float,
        next_questions: List[str],
        handoff_context: Dict[str, Any],
        completion_reason: str
    ):
        self.is_complete = is_complete
        self.confidence = confidence
        self.completion_indicators = completion_indicators
        self.insights_extracted = insights_extracted
        self.quality_score = quality_score
        self.research_depth = research_depth
        self.next_questions = next_questions
        self.handoff_context = handoff_context
        self.completion_reason = completion_reason


class CompletionAnalyzer:
    """Analyze agent completion and provide handoff intelligence."""
    
    def __init__(self, completion_detector: Optional[CompletionDetector] = None):
        """Initialize the completion analyzer.
        
        Args:
            completion_detector: Optional completion detector instance
        """
        self.completion_detector = completion_detector or CompletionDetector()
        
        # Analysis thresholds
        self.min_insight_quality = 0.5
        self.min_research_depth = 0.4
        self.min_content_length = 100
        
    def analyze_agent_completion(
        self,
        agent_name: str,
        agent_message: AIMessage,
        agent_config: Dict[str, Any],
        execution_duration: float,
        previous_insights: List[str]
    ) -> CompletionAnalysis:
        """Analyze if an agent has completed its task and extract insights.
        
        Args:
            agent_name: Name of the agent being analyzed
            agent_message: Latest message from the agent
            agent_config: Agent configuration from registry
            execution_duration: How long the agent has been executing
            previous_insights: Insights from previous agents in sequence
            
        Returns:
            CompletionAnalysis with detailed analysis results
        """
        logger.debug(f"Analyzing completion for agent '{agent_name}'")
        
        # Get custom completion indicators from agent config
        custom_indicators = agent_config.get("completion_indicators", [])
        
        # Basic completion detection
        is_complete = self.completion_detector.is_agent_complete(
            agent_message, custom_indicators
        )
        confidence = self.completion_detector.get_completion_confidence(
            agent_message, custom_indicators
        )
        
        # Extract insights from agent message
        insights = self._extract_insights(agent_message.content)
        
        # Calculate quality scores
        quality_score = self._calculate_quality_score(agent_message.content, insights)
        research_depth = self._calculate_research_depth(agent_message.content, execution_duration)
        
        # Generate next questions based on findings
        next_questions = self._generate_next_questions(
            insights, previous_insights, agent_config
        )
        
        # Create handoff context
        handoff_context = self._create_handoff_context(
            agent_name, insights, agent_message.content, agent_config
        )
        
        # Determine completion reason
        completion_reason = self._determine_completion_reason(
            is_complete, confidence, quality_score, research_depth, execution_duration
        )
        
        # Find completion indicators that triggered detection
        triggered_indicators = []
        if is_complete:
            all_indicators = custom_indicators + self.completion_detector.DEFAULT_INDICATORS
            content_lower = agent_message.content.lower()
            triggered_indicators = [
                indicator for indicator in all_indicators
                if indicator.lower() in content_lower
            ]
        
        analysis = CompletionAnalysis(
            is_complete=is_complete,
            confidence=confidence,
            completion_indicators=triggered_indicators,
            insights_extracted=insights,
            quality_score=quality_score,
            research_depth=research_depth,
            next_questions=next_questions,
            handoff_context=handoff_context,
            completion_reason=completion_reason
        )
        
        logger.info(f"Completion analysis for '{agent_name}': "
                   f"Complete={is_complete}, Confidence={confidence:.2f}, "
                   f"Quality={quality_score:.2f}, Insights={len(insights)}")
        
        return analysis
    
    def _extract_insights(self, content: str) -> List[str]:
        """Extract key insights from agent content.
        
        Args:
            content: Agent message content
            
        Returns:
            List of extracted insights
        """
        insights = []
        
        # Look for insight indicators
        insight_patterns = [
            "key finding:", "important discovery:", "critical insight:",
            "significant finding:", "major discovery:", "key observation:",
            "important note:", "critical point:", "significant insight:",
            "conclusion:", "finding:", "insight:", "observation:",
            "discovery:", "result:", "outcome:"
        ]
        
        lines = content.split('\n')
        for line in lines:
            line_lower = line.strip().lower()
            
            # Check if line contains insight indicators
            for pattern in insight_patterns:
                if pattern in line_lower:
                    # Extract the insight (remove the indicator)
                    insight = line.strip()
                    for p in insight_patterns:
                        insight = insight.replace(p.capitalize(), "").replace(p.upper(), "").strip()
                    
                    if len(insight) > 20:  # Minimum insight length
                        insights.append(insight)
                    break
            
            # Also look for bullet points and numbered lists
            if (line_lower.startswith('- ') or 
                line_lower.startswith('* ') or
                (len(line_lower) > 3 and line_lower[0].isdigit() and line_lower[1] in '. ')
               ):
                insight = line.strip().lstrip('- *0123456789. ')
                if len(insight) > 20:
                    insights.append(insight)
        
        # Deduplicate and limit insights
        unique_insights = []
        for insight in insights:
            if insight not in unique_insights and len(unique_insights) < 10:
                unique_insights.append(insight)
        
        return unique_insights
    
    def _calculate_quality_score(self, content: str, insights: List[str]) -> float:
        """Calculate quality score for agent output.
        
        Args:
            content: Agent message content
            insights: Extracted insights
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Content length factor (0.2 weight)
        if len(content) > self.min_content_length:
            score += 0.2 * min(len(content) / 1000, 1.0)
        
        # Insights factor (0.3 weight)
        if insights:
            insights_score = min(len(insights) / 5, 1.0)  # Normalize to 5 insights
            score += 0.3 * insights_score
        
        # Structure factor (0.2 weight)
        structure_score = 0.0
        if '\n' in content:  # Has line breaks
            structure_score += 0.3
        if any(marker in content.lower() for marker in ['conclusion', 'summary', 'findings']):
            structure_score += 0.4
        if any(marker in content for marker in ['- ', '* ', '1. ', '2. ']):  # Has lists
            structure_score += 0.3
        score += 0.2 * min(structure_score, 1.0)
        
        # Depth factor (0.2 weight)
        depth_indicators = ['analysis', 'evaluation', 'assessment', 'comparison', 'framework', 'methodology']
        depth_count = sum(1 for indicator in depth_indicators if indicator in content.lower())
        depth_score = min(depth_count / 3, 1.0)
        score += 0.2 * depth_score
        
        # Specificity factor (0.1 weight)
        specific_indicators = ['research shows', 'data indicates', 'studies suggest', 'evidence supports']
        specific_count = sum(1 for indicator in specific_indicators if indicator in content.lower())
        specific_score = min(specific_count / 2, 1.0)
        score += 0.1 * specific_score
        
        return min(score, 1.0)
    
    def _calculate_research_depth(self, content: str, execution_duration: float) -> float:
        """Calculate research depth score.
        
        Args:
            content: Agent message content
            execution_duration: Time spent by agent
            
        Returns:
            Research depth score between 0.0 and 1.0
        """
        depth_score = 0.0
        
        # Time factor (0.3 weight) - assumes longer time = more research
        time_score = min(execution_duration / 120, 1.0)  # Normalize to 2 minutes
        depth_score += 0.3 * time_score
        
        # Citation/source factor (0.4 weight)
        source_indicators = ['according to', 'source:', 'study by', 'research from', 'published', 'journal']
        source_count = sum(1 for indicator in source_indicators if indicator in content.lower())
        source_score = min(source_count / 3, 1.0)
        depth_score += 0.4 * source_score
        
        # Detail factor (0.3 weight)
        detail_indicators = ['specifically', 'particularly', 'detailed', 'comprehensive', 'thorough']
        detail_count = sum(1 for indicator in detail_indicators if indicator in content.lower())
        detail_score = min(detail_count / 2, 1.0)
        depth_score += 0.3 * detail_score
        
        return min(depth_score, 1.0)
    
    def _generate_next_questions(
        self,
        current_insights: List[str],
        previous_insights: List[str],
        agent_config: Dict[str, Any]
    ) -> List[str]:
        """Generate questions for the next agent based on current findings.
        
        Args:
            current_insights: Insights from current agent
            previous_insights: Insights from previous agents
            agent_config: Current agent configuration
            
        Returns:
            List of questions for next agent
        """
        questions = []
        
        # Extract key topics from insights
        all_insights = current_insights + previous_insights
        insight_text = " ".join(all_insights).lower()
        
        # Generate questions based on agent type and insights
        agent_config.get("expertise_areas", [])
        
        # Generic follow-up questions
        if current_insights:
            questions.extend([
                f"How can we build upon the finding that {current_insights[0][:100]}?",
                "What are the implications of these discoveries for practical implementation?",
                "What additional validation or research is needed for these insights?"
            ])
        
        # Domain-specific questions based on keywords
        if any(keyword in insight_text for keyword in ['market', 'business', 'commercial']):
            questions.extend([
                "What are the market opportunities and commercial viability?",
                "Who are the key stakeholders and what are their needs?"
            ])
        
        if any(keyword in insight_text for keyword in ['technical', 'implementation', 'architecture']):
            questions.extend([
                "What technical approaches would be most effective?",
                "What are the implementation challenges and solutions?"
            ])
        
        if any(keyword in insight_text for keyword in ['research', 'academic', 'theory']):
            questions.extend([
                "What does the academic literature reveal about this topic?",
                "What theoretical frameworks are most applicable?"
            ])
        
        # Limit and deduplicate questions
        unique_questions = []
        for question in questions:
            if question not in unique_questions and len(unique_questions) < 6:
                unique_questions.append(question)
        
        return unique_questions
    
    def _create_handoff_context(
        self,
        agent_name: str,
        insights: List[str],
        content: str,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create context for handoff to next agent.
        
        Args:
            agent_name: Name of current agent
            insights: Extracted insights
            content: Full agent content
            agent_config: Agent configuration
            
        Returns:
            Handoff context dictionary
        """
        context = {
            "source_agent": agent_name,
            "agent_type": agent_config.get("description", "Unknown"),
            "expertise_areas": agent_config.get("expertise_areas", []),
            "key_insights": insights,
            "research_summary": content[:500] + "..." if len(content) > 500 else content,
            "handoff_timestamp": datetime.utcnow().isoformat(),
        }
        
        # Extract key topics and themes
        content_lower = content.lower()
        topics = []
        
        # Common research topics
        topic_keywords = {
            "market_analysis": ["market", "competition", "business", "commercial"],
            "technical_analysis": ["technical", "technology", "implementation", "architecture"],
            "academic_research": ["research", "study", "academic", "theory", "literature"],
            "user_experience": ["user", "customer", "experience", "interface", "usability"],
            "security": ["security", "privacy", "protection", "vulnerability"],
            "performance": ["performance", "speed", "efficiency", "optimization"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        context["identified_topics"] = topics
        
        return context
    
    def _determine_completion_reason(
        self,
        is_complete: bool,
        confidence: float,
        quality_score: float,
        research_depth: float,
        execution_duration: float
    ) -> str:
        """Determine the reason for completion decision.
        
        Args:
            is_complete: Whether agent is detected as complete
            confidence: Completion confidence score
            quality_score: Quality score of output
            research_depth: Research depth score
            execution_duration: Time spent executing
            
        Returns:
            Reason for completion decision
        """
        if not is_complete:
            if confidence < 0.3:
                return "Agent appears to still be working on the task"
            elif quality_score < self.min_insight_quality:
                return "Output quality below threshold, agent likely still working"
            elif research_depth < self.min_research_depth:
                return "Research depth insufficient, agent needs more time"
            else:
                return "Agent not showing clear completion signals"
        
        # Agent is complete - determine primary reason
        reasons = []
        
        if confidence > 0.7:
            reasons.append("Strong completion signals detected")
        elif confidence > 0.5:
            reasons.append("Moderate completion signals detected")
        
        if quality_score > 0.7:
            reasons.append("High quality output achieved")
        
        if research_depth > 0.6:
            reasons.append("Sufficient research depth reached")
        
        if execution_duration > 180:  # 3 minutes
            reasons.append("Extended execution time indicates thorough work")
        
        if not reasons:
            reasons.append("Basic completion criteria met")
        
        return "; ".join(reasons)
    
    def analyze_sequence_progress(
        self,
        state: SequentialSupervisorState
    ) -> Dict[str, Any]:
        """Analyze overall sequence progress and suggest modifications.
        
        Args:
            state: Current supervisor state
            
        Returns:
            Analysis of sequence progress with suggestions
        """
        analysis = {
            "total_agents_executed": len(state.executed_agents),
            "current_position": state.sequence_position,
            "remaining_agents": len(state.planned_sequence) - state.sequence_position,
            "total_insights": sum(len(insights) for insights in state.agent_insights.values()),
            "sequence_efficiency": 0.0,
            "quality_trend": "stable",
            "suggested_modifications": []
        }
        
        # Calculate sequence efficiency
        if state.executed_agents:
            total_insights = analysis["total_insights"]
            total_agents = len(state.executed_agents)
            analysis["sequence_efficiency"] = total_insights / max(total_agents, 1)
        
        # Analyze quality trend
        if len(state.agent_reports) >= 2:
            recent_scores = [
                report.insight_quality_score for report in 
                list(state.agent_reports.values())[-2:]
            ]
            if recent_scores[1] > recent_scores[0] + 0.1:
                analysis["quality_trend"] = "improving"
            elif recent_scores[1] < recent_scores[0] - 0.1:
                analysis["quality_trend"] = "declining"
        
        # Suggest modifications
        if analysis["sequence_efficiency"] < 2.0:  # Less than 2 insights per agent
            analysis["suggested_modifications"].append(
                "Consider adding more specialized agents to increase insight generation"
            )
        
        if analysis["quality_trend"] == "declining":
            analysis["suggested_modifications"].append(
                "Quality declining - consider adjusting remaining agent sequence"
            )
        
        return analysis