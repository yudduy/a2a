"""Context manager for intelligent context sharing between sequential agents.

This module manages the flow of context, insights, and knowledge between agents
in a sequential workflow, ensuring each agent receives optimal context.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from open_deep_research.state import (
    SequentialSupervisorState, 
    SequentialAgentState,
    AgentExecutionReport
)

logger = logging.getLogger(__name__)


class ContextPackage:
    """Package of context prepared for an agent."""
    
    def __init__(
        self,
        target_agent: str,
        research_topic: str,
        assigned_questions: List[str],
        previous_insights: List[str],
        relevant_context: Dict[str, Any],
        handoff_notes: List[str],
        priority_areas: List[str],
        context_summary: str
    ):
        self.target_agent = target_agent
        self.research_topic = research_topic
        self.assigned_questions = assigned_questions
        self.previous_insights = previous_insights
        self.relevant_context = relevant_context
        self.handoff_notes = handoff_notes
        self.priority_areas = priority_areas
        self.context_summary = context_summary
        self.prepared_at = datetime.utcnow()


class ContextManager:
    """Manage context flow between sequential agents."""
    
    def __init__(self):
        """Initialize the context manager."""
        
        # Context filtering strategies
        self.max_context_items = 10  # Limit context to prevent overload
        self.relevance_threshold = 0.3  # Minimum relevance score
        
        # Agent expertise mapping for context relevance
        self.agent_interests = {
            "research-agent": [
                "academic", "literature", "studies", "research", "methodology",
                "theory", "framework", "evidence", "validation", "peer-reviewed"
            ],
            "analysis-agent": [
                "data", "analysis", "pattern", "trend", "statistics", "metrics",
                "interpretation", "correlation", "comparison", "evaluation"
            ],
            "market-agent": [
                "market", "business", "commercial", "competitive", "customer",
                "revenue", "strategy", "opportunity", "stakeholder", "economic"
            ],
            "technical-agent": [
                "technical", "implementation", "architecture", "system", "design",
                "development", "infrastructure", "scalability", "performance", "security"
            ],
            "synthesis-agent": [
                "integration", "synthesis", "recommendations", "strategy", "roadmap",
                "priorities", "decisions", "framework", "governance", "planning"
            ]
        }
    
    def prepare_agent_context(
        self,
        state: SequentialSupervisorState,
        target_agent: str,
        agent_config: Dict[str, Any]
    ) -> ContextPackage:
        """Prepare context package for the target agent.
        
        Args:
            state: Current supervisor state
            target_agent: Name of agent receiving context
            agent_config: Configuration of target agent
            
        Returns:
            ContextPackage with relevant context for the agent
        """
        logger.debug(f"Preparing context package for '{target_agent}'")
        
        # Generate questions for this agent
        assigned_questions = self._generate_agent_questions(
            state, target_agent, agent_config
        )
        
        # Filter previous insights for relevance
        relevant_insights = self._filter_relevant_insights(
            state, target_agent
        )
        
        # Extract relevant context from previous agents
        relevant_context = self._extract_relevant_context(
            state, target_agent
        )
        
        # Generate handoff notes
        handoff_notes = self._generate_handoff_notes(
            state, target_agent
        )
        
        # Identify priority areas
        priority_areas = self._identify_priority_areas(
            state, target_agent, agent_config
        )
        
        # Create context summary
        context_summary = self._create_context_summary(
            state, target_agent, relevant_insights, priority_areas
        )
        
        context_package = ContextPackage(
            target_agent=target_agent,
            research_topic=state.research_topic,
            assigned_questions=assigned_questions,
            previous_insights=relevant_insights,
            relevant_context=relevant_context,
            handoff_notes=handoff_notes,
            priority_areas=priority_areas,
            context_summary=context_summary
        )
        
        logger.info(f"Context package prepared for '{target_agent}': "
                   f"{len(assigned_questions)} questions, "
                   f"{len(relevant_insights)} insights, "
                   f"{len(priority_areas)} priority areas")
        
        return context_package
    
    def _generate_agent_questions(
        self,
        state: SequentialSupervisorState,
        target_agent: str,
        agent_config: Dict[str, Any]
    ) -> List[str]:
        """Generate specific questions for the target agent.
        
        Args:
            state: Current supervisor state
            target_agent: Name of target agent
            agent_config: Configuration of target agent
            
        Returns:
            List of questions for the agent
        """
        questions = []
        
        # Get agent expertise areas
        expertise_areas = agent_config.get("expertise_areas", [])
        agent_description = agent_config.get("description", "")
        
        # Base questions for the research topic
        research_topic = state.research_topic
        
        # Agent-specific question templates
        if "research" in target_agent.lower():
            questions.extend([
                f"What does current academic research reveal about {research_topic}?",
                f"What are the key theoretical frameworks relevant to {research_topic}?",
                f"What gaps exist in current knowledge about {research_topic}?",
                f"What research methodologies would be most appropriate for studying {research_topic}?"
            ])
        
        elif "analysis" in target_agent.lower():
            questions.extend([
                f"What patterns can be identified in the data related to {research_topic}?",
                f"How can we quantitatively evaluate {research_topic}?",
                f"What are the key variables and relationships in {research_topic}?",
                f"What analytical frameworks would be most useful for {research_topic}?"
            ])
        
        elif "market" in target_agent.lower():
            questions.extend([
                f"What market opportunities exist for {research_topic}?",
                f"Who are the key stakeholders and competitors related to {research_topic}?",
                f"What is the commercial viability of {research_topic}?",
                f"What business models could be applied to {research_topic}?"
            ])
        
        elif "technical" in target_agent.lower():
            questions.extend([
                f"What are the technical requirements for implementing {research_topic}?",
                f"What architectural approaches would be most suitable for {research_topic}?",
                f"What are the key technical challenges and solutions for {research_topic}?",
                f"How can {research_topic} be scaled and optimized?"
            ])
        
        elif "synthesis" in target_agent.lower():
            questions.extend([
                f"How can all findings about {research_topic} be integrated into a coherent strategy?",
                f"What are the key recommendations for {research_topic}?",
                f"What implementation roadmap would be most effective for {research_topic}?",
                f"What are the priorities and next steps for {research_topic}?"
            ])
        
        # Add questions based on previous agent findings
        if state.agent_insights:
            all_insights = []
            for insights in state.agent_insights.values():
                all_insights.extend(insights)
            
            # Extract topics from insights that might need this agent's expertise
            insight_text = " ".join(all_insights).lower()
            agent_keywords = self.agent_interests.get(target_agent, [])
            
            relevant_topics = [
                keyword for keyword in agent_keywords
                if keyword in insight_text
            ]
            
            if relevant_topics:
                questions.append(
                    f"Based on previous findings about {', '.join(relevant_topics[:3])}, "
                    f"what additional insights can you provide about {research_topic}?"
                )
        
        # Limit questions to avoid overload
        return questions[:6]
    
    def _filter_relevant_insights(
        self,
        state: SequentialSupervisorState,
        target_agent: str
    ) -> List[str]:
        """Filter previous insights for relevance to target agent.
        
        Args:
            state: Current supervisor state
            target_agent: Name of target agent
            
        Returns:
            List of relevant insights
        """
        relevant_insights = []
        agent_keywords = self.agent_interests.get(target_agent, [])
        
        # Get all insights from previous agents
        all_insights = []
        for agent_name, insights in state.agent_insights.items():
            for insight in insights:
                all_insights.append((agent_name, insight))
        
        # Score insights for relevance
        scored_insights = []
        for agent_name, insight in all_insights:
            relevance_score = self._calculate_relevance_score(
                insight, agent_keywords
            )
            if relevance_score >= self.relevance_threshold:
                scored_insights.append((relevance_score, agent_name, insight))
        
        # Sort by relevance and take top insights
        scored_insights.sort(reverse=True, key=lambda x: x[0])
        
        for score, agent_name, insight in scored_insights[:self.max_context_items]:
            relevant_insights.append(f"[{agent_name.upper()}] {insight}")
        
        return relevant_insights
    
    def _calculate_relevance_score(
        self,
        insight: str,
        agent_keywords: List[str]
    ) -> float:
        """Calculate relevance score of insight to agent.
        
        Args:
            insight: Insight text to score
            agent_keywords: Keywords relevant to target agent
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        insight_lower = insight.lower()
        matches = sum(
            1 for keyword in agent_keywords
            if keyword in insight_lower
        )
        
        # Normalize by number of keywords
        relevance_score = matches / max(len(agent_keywords), 1)
        
        # Boost score for longer insights (more likely to be substantial)
        length_bonus = min(len(insight) / 200, 0.2)  # Up to 0.2 bonus
        
        return min(relevance_score + length_bonus, 1.0)
    
    def _extract_relevant_context(
        self,
        state: SequentialSupervisorState,
        target_agent: str
    ) -> Dict[str, Any]:
        """Extract relevant context from previous agent execution.
        
        Args:
            state: Current supervisor state
            target_agent: Name of target agent
            
        Returns:
            Dictionary of relevant context
        """
        context = {
            "sequence_position": state.sequence_position,
            "total_agents_planned": len(state.planned_sequence),
            "agents_completed": len(state.executed_agents),
            "research_duration": 0.0,
            "key_topics": [],
            "emerging_themes": [],
            "quality_trend": "stable"
        }
        
        # Calculate research duration
        if state.sequence_start_time:
            duration = (datetime.utcnow() - state.sequence_start_time).total_seconds()
            context["research_duration"] = duration
        
        # Extract key topics from all insights
        all_insights_text = ""
        for insights in state.agent_insights.values():
            all_insights_text += " " + " ".join(insights)
        
        # Simple topic extraction based on frequency
        words = all_insights_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only significant words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        context["key_topics"] = [word for word, freq in sorted_words[:5]]
        
        # Analyze quality trend if we have multiple agent reports
        if len(state.agent_reports) >= 2:
            recent_scores = [
                report.insight_quality_score 
                for report in list(state.agent_reports.values())[-2:]
            ]
            if recent_scores[1] > recent_scores[0] + 0.1:
                context["quality_trend"] = "improving"
            elif recent_scores[1] < recent_scores[0] - 0.1:
                context["quality_trend"] = "declining"
        
        return context
    
    def _generate_handoff_notes(
        self,
        state: SequentialSupervisorState,
        target_agent: str
    ) -> List[str]:
        """Generate handoff notes for the target agent.
        
        Args:
            state: Current supervisor state
            target_agent: Name of target agent
            
        Returns:
            List of handoff notes
        """
        notes = []
        
        # Note about sequence position
        position = state.sequence_position + 1
        total = len(state.planned_sequence)
        notes.append(f"You are agent {position} of {total} in this research sequence")
        
        # Note about previous agents
        if state.executed_agents:
            previous = ", ".join(state.executed_agents)
            notes.append(f"Previous agents: {previous}")
        
        # Note about research progress
        total_insights = sum(len(insights) for insights in state.agent_insights.values())
        if total_insights > 0:
            notes.append(f"Total insights so far: {total_insights}")
        
        # Note about any sequence modifications
        if state.sequence_modifications:
            notes.append(f"Sequence has been modified {len(state.sequence_modifications)} times")
        
        # Note about quality expectations
        if state.agent_reports:
            avg_quality = sum(
                report.insight_quality_score 
                for report in state.agent_reports.values()
            ) / len(state.agent_reports)
            notes.append(f"Average quality score so far: {avg_quality:.2f}")
        
        return notes
    
    def _identify_priority_areas(
        self,
        state: SequentialSupervisorState,
        target_agent: str,
        agent_config: Dict[str, Any]
    ) -> List[str]:
        """Identify priority areas for the target agent.
        
        Args:
            state: Current supervisor state
            target_agent: Name of target agent
            agent_config: Configuration of target agent
            
        Returns:
            List of priority areas
        """
        priorities = []
        
        # Get agent expertise areas
        expertise_areas = agent_config.get("expertise_areas", [])
        
        # Priority based on agent type and sequence position
        position = state.sequence_position
        total_agents = len(state.planned_sequence)
        
        if position == 0:  # First agent
            priorities.extend([
                "Establish foundational understanding",
                "Identify key areas for investigation",
                "Set research direction"
            ])
        elif position == total_agents - 1:  # Last agent
            priorities.extend([
                "Synthesize all findings",
                "Provide recommendations",
                "Create implementation roadmap"
            ])
        else:  # Middle agents
            priorities.extend([
                "Build upon previous insights",
                "Explore specific expertise areas",
                "Prepare context for next agents"
            ])
        
        # Priority based on previous findings
        if state.agent_insights:
            all_insights_text = " ".join([
                insight for insights in state.agent_insights.values()
                for insight in insights
            ]).lower()
            
            # Check for gaps or areas needing attention
            agent_keywords = self.agent_interests.get(target_agent, [])
            missing_areas = [
                keyword for keyword in agent_keywords
                if keyword not in all_insights_text
            ]
            
            if missing_areas:
                priorities.append(f"Address gaps in: {', '.join(missing_areas[:3])}")
        
        # Priority based on research topic
        topic_lower = state.research_topic.lower()
        if any(keyword in topic_lower for keyword in ["implementation", "technical"]):
            if "technical" in target_agent.lower():
                priorities.append("Focus on implementation feasibility")
        
        if any(keyword in topic_lower for keyword in ["market", "business"]):
            if "market" in target_agent.lower():
                priorities.append("Analyze commercial viability")
        
        return priorities[:5]  # Limit priorities
    
    def _create_context_summary(
        self,
        state: SequentialSupervisorState,
        target_agent: str,
        relevant_insights: List[str],
        priority_areas: List[str]
    ) -> str:
        """Create a context summary for the agent.
        
        Args:
            state: Current supervisor state
            target_agent: Name of target agent
            relevant_insights: Relevant insights for agent
            priority_areas: Priority areas for agent
            
        Returns:
            Context summary string
        """
        summary_parts = [
            f"Research Topic: {state.research_topic}",
            f"Your Role: {target_agent.replace('-', ' ').title()}",
            f"Sequence Position: {state.sequence_position + 1} of {len(state.planned_sequence)}",
        ]
        
        if relevant_insights:
            summary_parts.extend([
                "",
                "Key Insights from Previous Agents:",
                *[f"• {insight}" for insight in relevant_insights[:3]]
            ])
        
        if priority_areas:
            summary_parts.extend([
                "",
                "Priority Areas for Your Analysis:",
                *[f"• {priority}" for priority in priority_areas[:3]]
            ])
        
        summary_parts.extend([
            "",
            "Your task is to build upon the previous insights while focusing on your areas of expertise.",
            "Provide specific, actionable insights that will be valuable for subsequent agents."
        ])
        
        return "\n".join(summary_parts)
    
    def create_agent_state(
        self,
        context_package: ContextPackage,
        agent_config: Dict[str, Any]
    ) -> SequentialAgentState:
        """Create initial state for agent execution.
        
        Args:
            context_package: Prepared context package
            agent_config: Agent configuration
            
        Returns:
            SequentialAgentState for agent execution
        """
        agent_state = SequentialAgentState(
            agent_name=context_package.target_agent,
            agent_type=agent_config.get("description", ""),
            sequence_position=0,  # Will be updated by supervisor
            research_topic=context_package.research_topic,
            assigned_questions=context_package.assigned_questions,
            previous_agent_insights=context_package.previous_insights,
            previous_agent_context=context_package.relevant_context,
            execution_start_time=datetime.utcnow(),
            tool_calls_made=0,
            completion_detected=False,
            completion_confidence=0.0,
            generated_insights=[],
            research_findings="",
            handoff_context={},
            next_agent_questions=[],
            insight_quality_scores=[],
            research_depth_score=0.0,
            agent_messages=[]
        )
        
        return agent_state
    
    def extract_handoff_context(
        self,
        agent_state: SequentialAgentState,
        completion_analysis: Any  # CompletionAnalysis type
    ) -> Dict[str, Any]:
        """Extract context from completed agent for handoff.
        
        Args:
            agent_state: Completed agent state
            completion_analysis: Analysis of agent completion
            
        Returns:
            Handoff context dictionary
        """
        handoff_context = {
            "source_agent": agent_state.agent_name,
            "execution_duration": (
                datetime.utcnow() - agent_state.execution_start_time
            ).total_seconds() if agent_state.execution_start_time else 0.0,
            "insights_generated": agent_state.generated_insights,
            "research_findings": agent_state.research_findings,
            "questions_addressed": agent_state.assigned_questions,
            "tool_calls_made": agent_state.tool_calls_made,
            "completion_confidence": completion_analysis.confidence,
            "quality_score": completion_analysis.quality_score,
            "research_depth": completion_analysis.research_depth,
            "next_agent_questions": completion_analysis.next_questions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return handoff_context