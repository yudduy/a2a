"""Research Director for dynamic question generation and insight tracking.

The SupervisorResearchDirector manages the sequential execution of specialized agents,
dynamically generating questions based on accumulated insights and tracking 
productivity transitions to optimize the research process.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from open_deep_research.configuration import Configuration
from open_deep_research.utils import get_api_key_for_model, get_model_config_for_provider, clean_reasoning_model_output
from open_deep_research.sequencing.models import (
    AdaptiveLearningState,
    AgentType,
    InsightTransition,
    InsightType,
    SequencePattern,
    ToolProductivityMetrics
)

logger = logging.getLogger(__name__)

# Initialize configurable model for research director
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "base_url"),
)


class QuestionGenerationResult(BaseModel):
    """Result from dynamic question generation process."""
    
    questions: List[str]
    insight_context: str
    expected_research_direction: str
    quality_score: float


class InsightAnalysisResult(BaseModel):
    """Result from analyzing insights for productivity tracking."""
    
    key_insights: List[str]
    insight_types: List[InsightType]
    novelty_score: float
    relevance_score: float
    actionability_score: float
    transition_quality: float


class SupervisorResearchDirector:
    """Manages sequential agent execution with dynamic question generation and adaptive learning."""
    
    def __init__(self, config: RunnableConfig):
        """Initialize the research director with configuration."""
        self.config = config
        self.configurable = Configuration.from_runnable_config(config)
        self.adaptive_state = AdaptiveLearningState()
        
        # Question generation prompts
        self.question_generation_prompt = self._get_question_generation_prompt()
        self.insight_analysis_prompt = self._get_insight_analysis_prompt()
        
        # Tracking state
        self.execution_history: List[Dict] = []
        self.current_sequence_insights: List[str] = []
        
    def _get_question_generation_prompt(self) -> str:
        """Get the system prompt for dynamic question generation."""
        return """You are a Research Director responsible for generating targeted questions for the next research agent based on previous insights.

Your role is to:
1. Analyze the insights from the previous agent
2. Generate 3-5 specific, actionable research questions that build on these insights
3. Ensure questions prevent cognitive offloading by requiring internal reasoning
4. Focus on areas that the next agent type can uniquely address

Guidelines for question generation:
- Questions should be specific and require deep domain expertise
- Avoid questions that simply ask for summarization of existing insights
- Focus on gaps, implications, or extensions of current findings
- Ensure questions require original analysis, not just information retrieval
- Target the unique strengths of the next agent type

Previous agent insights: {previous_insights}
Next agent type: {next_agent_type}
Research context: {research_context}

Generate questions that will lead to productive, non-redundant research."""

    def _get_insight_analysis_prompt(self) -> str:
        """Get the system prompt for insight quality analysis."""
        return """You are analyzing research insights for quality and productivity metrics.

Evaluate the following insights on these dimensions:
1. Novelty (0-1): How new or unexpected are these insights?
2. Relevance (0-1): How relevant are they to the research topic?
3. Actionability (0-1): How actionable are these insights for next steps?
4. Transition Quality (0-1): How well do they set up the next research phase?

Research insights to analyze: {insights}
Research context: {research_context}
Position in sequence: {sequence_position}

Provide scores and identify the key insight types present."""

    async def direct_next_investigation(
        self,
        previous_agent_insights: List[str],
        next_agent_type: AgentType,
        research_context: str,
        sequence_position: int
    ) -> QuestionGenerationResult:
        """Dynamically generate questions for the next agent based on accumulated insights.
        
        Args:
            previous_agent_insights: Insights from the previous agent
            next_agent_type: Type of the next agent to receive questions
            research_context: Overall research topic and context
            sequence_position: Position in sequence (1, 2, or 3)
            
        Returns:
            QuestionGenerationResult with generated questions and context
        """
        try:
            # Configure planner model for question generation
            model_config = get_model_config_for_provider(
                model_name=self.configurable.planner_model,
                api_key=get_api_key_for_model(self.configurable.planner_model, self.config),
                max_tokens=self.configurable.planner_model_max_tokens,
                tags=["langsmith:nostream"]
            )
            
            question_model = configurable_model.with_config(model_config)
            
            # Format insights for analysis
            insights_text = "\n".join([f"- {insight}" for insight in previous_agent_insights])
            
            # Generate questions based on agent type specialization
            agent_specific_prompt = self._get_agent_specific_question_prompt(
                next_agent_type, sequence_position
            )
            
            prompt_content = (
                self.question_generation_prompt + "\n\n" + agent_specific_prompt
            ).format(
                previous_insights=insights_text,
                next_agent_type=next_agent_type.value,
                research_context=research_context
            )
            
            response = await question_model.ainvoke([
                SystemMessage(content=prompt_content),
                HumanMessage(content="Generate targeted research questions based on the previous insights.")
            ])
            
            # Clean reasoning model output to remove thinking tags
            if hasattr(response, 'content') and response.content:
                response.content = clean_reasoning_model_output(response.content)
            
            # Parse response to extract questions
            questions = self._parse_questions_from_response(response.content)
            
            # Calculate quality score based on question specificity and relevance
            quality_score = self._assess_question_quality(questions, previous_agent_insights)
            
            result = QuestionGenerationResult(
                questions=questions,
                insight_context=insights_text,
                expected_research_direction=self._predict_research_direction(
                    questions, next_agent_type
                ),
                quality_score=quality_score
            )
            
            logger.info(f"Generated {len(questions)} questions for {next_agent_type.value} agent")
            return result
            
        except Exception as e:
            logger.error(f"Error in question generation: {e}")
            # Fallback to default questions
            return self._generate_fallback_questions(next_agent_type, research_context)
    
    def _get_agent_specific_question_prompt(self, agent_type: AgentType, position: int) -> str:
        """Get agent-specific question generation guidance."""
        base_prompts = {
            AgentType.ACADEMIC: """
Focus on generating questions that leverage academic research capabilities:
- Theoretical frameworks and foundational concepts
- Peer-reviewed research and scholarly analysis
- Research methodologies and validation approaches
- Gap analysis in current academic literature
- Theoretical implications and extensions
""",
            AgentType.INDUSTRY: """
Focus on generating questions that leverage industry analysis capabilities:
- Market dynamics and competitive landscape
- Business model implications and commercial viability
- Industry trends and adoption patterns
- Stakeholder analysis and value chain considerations
- Implementation challenges and market barriers
""",
            AgentType.TECHNICAL_TRENDS: """
Focus on generating questions that leverage technical trend analysis:
- Emerging technologies and implementation feasibility
- Technical architecture and scalability considerations
- Future technology roadmaps and convergence patterns
- Innovation cycles and adoption timelines
- Technical risks and implementation pathways
"""
        }
        
        position_guidance = {
            1: "As the first agent, focus on establishing foundational understanding.",
            2: "As the middle agent, focus on building bridges between insights and extending analysis.",
            3: "As the final agent, focus on synthesis, implementation, and future implications."
        }
        
        return base_prompts[agent_type] + "\n" + position_guidance.get(position, "")
    
    def _parse_questions_from_response(self, response_text: str) -> List[str]:
        """Parse questions from model response."""
        lines = response_text.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            # Look for numbered or bulleted questions
            if (line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')) and 
                '?' in line):
                # Clean up the question
                question = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                if question and len(question) > 10:  # Ensure substantial questions
                    questions.append(question)
        
        # Fallback: split by question marks
        if not questions:
            potential_questions = response_text.split('?')
            for q in potential_questions[:-1]:  # Last split will be empty
                q = q.strip()
                if len(q) > 20:  # Minimum question length
                    questions.append(q + '?')
        
        return questions[:5]  # Limit to 5 questions max
    
    def _assess_question_quality(self, questions: List[str], insights: List[str]) -> float:
        """Assess the quality of generated questions."""
        if not questions:
            return 0.0
        
        quality_scores = []
        
        for question in questions:
            score = 0.0
            
            # Length and specificity (0-0.3)
            if len(question) > 50:
                score += 0.1
            if len(question.split()) > 10:
                score += 0.1
            if any(word in question.lower() for word in ['how', 'why', 'what', 'which']):
                score += 0.1
            
            # Complexity and depth (0-0.4)
            complex_words = ['implications', 'framework', 'methodology', 'analysis', 
                           'assessment', 'evaluation', 'comparison', 'integration']
            if any(word in question.lower() for word in complex_words):
                score += 0.2
            
            if '?' in question and question.count('?') == 1:
                score += 0.1
            
            if any(word in question.lower() for word in ['and', 'or', 'relationship', 'between']):
                score += 0.1
            
            # Relevance to insights (0-0.3)
            insight_words = set()
            for insight in insights:
                insight_words.update(insight.lower().split())
            
            question_words = set(question.lower().split())
            overlap = len(insight_words.intersection(question_words))
            if overlap > 2:
                score += 0.15
            if overlap > 5:
                score += 0.15
            
            quality_scores.append(min(score, 1.0))
        
        return sum(quality_scores) / len(quality_scores)
    
    def _predict_research_direction(self, questions: List[str], agent_type: AgentType) -> str:
        """Predict the research direction based on generated questions."""
        direction_keywords = {
            AgentType.ACADEMIC: ["theoretical", "research", "scholarly", "academic"],
            AgentType.INDUSTRY: ["market", "business", "commercial", "industry"],
            AgentType.TECHNICAL_TRENDS: ["technical", "technology", "implementation", "trends"]
        }
        
        all_text = " ".join(questions).lower()
        directions = []
        
        for keyword in direction_keywords[agent_type]:
            if keyword in all_text:
                directions.append(keyword)
        
        if directions:
            return f"Research will focus on {', '.join(directions)} aspects"
        else:
            return f"Research will follow {agent_type.value} specialization"
    
    def _generate_fallback_questions(self, agent_type: AgentType, context: str) -> QuestionGenerationResult:
        """Generate fallback questions when dynamic generation fails."""
        fallback_questions = {
            AgentType.ACADEMIC: [
                "What theoretical frameworks are most relevant to this research area?",
                "What gaps exist in current academic literature on this topic?",
                "How do established research methodologies apply to this domain?",
                "What are the key scholarly debates surrounding this area?"
            ],
            AgentType.INDUSTRY: [
                "What market opportunities exist in this space?",
                "Who are the key stakeholders and competitors?", 
                "What business models are most viable for this application?",
                "What are the main commercial barriers to adoption?"
            ],
            AgentType.TECHNICAL_TRENDS: [
                "What emerging technologies are relevant to this area?",
                "What technical challenges need to be overcome?",
                "How do current technology trends align with this research?",
                "What implementation pathways are most feasible?"
            ]
        }
        
        return QuestionGenerationResult(
            questions=fallback_questions[agent_type],
            insight_context="Fallback questions due to generation error",
            expected_research_direction=f"General {agent_type.value} research",
            quality_score=0.6
        )
    
    async def track_insight_productivity(
        self,
        from_agent: AgentType,
        to_agent: AgentType,
        insights: List[str],
        research_context: str,
        execution_time: float
    ) -> InsightAnalysisResult:
        """Analyze insights for productivity and quality metrics.
        
        Args:
            from_agent: Source agent type
            to_agent: Destination agent type
            insights: Generated insights to analyze
            research_context: Research topic context
            execution_time: Time taken to generate insights
            
        Returns:
            InsightAnalysisResult with quality metrics and analysis
        """
        try:
            # Configure planner model for insight analysis
            model_config = get_model_config_for_provider(
                model_name=self.configurable.planner_model,
                api_key=get_api_key_for_model(self.configurable.planner_model, self.config),
                max_tokens=self.configurable.planner_model_max_tokens,
                tags=["langsmith:nostream"]
            )
            
            analysis_model = configurable_model.with_config(model_config)
            
            # Analyze insights for quality metrics
            insights_text = "\n".join([f"- {insight}" for insight in insights])
            
            prompt_content = self.insight_analysis_prompt.format(
                insights=insights_text,
                research_context=research_context,
                sequence_position=f"{from_agent.value} → {to_agent.value}"
            )
            
            response = await analysis_model.ainvoke([
                SystemMessage(content=prompt_content),
                HumanMessage(content="Analyze these insights and provide quality scores.")
            ])
            
            # Clean reasoning model output to remove thinking tags
            if hasattr(response, 'content') and response.content:
                response.content = clean_reasoning_model_output(response.content)
            
            # Parse quality metrics from response
            quality_metrics = self._parse_quality_metrics(response.content)
            
            # Identify insight types
            insight_types = self._classify_insight_types(insights)
            
            # Calculate transition quality
            transition_quality = self._calculate_transition_quality(
                from_agent, to_agent, insights, execution_time
            )
            
            result = InsightAnalysisResult(
                key_insights=insights,
                insight_types=insight_types,
                novelty_score=quality_metrics.get('novelty', 0.5),
                relevance_score=quality_metrics.get('relevance', 0.5),
                actionability_score=quality_metrics.get('actionability', 0.5),
                transition_quality=transition_quality
            )
            
            # Update adaptive learning state
            await self._update_adaptive_learning(from_agent, to_agent, result)
            
            logger.info(f"Tracked productivity for {from_agent.value} → {to_agent.value} transition")
            return result
            
        except Exception as e:
            logger.error(f"Error in insight productivity tracking: {e}")
            # Return fallback analysis
            return InsightAnalysisResult(
                key_insights=insights,
                insight_types=[InsightType.RESEARCH_GAP],
                novelty_score=0.5,
                relevance_score=0.5,
                actionability_score=0.5,
                transition_quality=0.5
            )
    
    def _parse_quality_metrics(self, response_text: str) -> Dict[str, float]:
        """Parse quality metrics from analysis response."""
        metrics = {}
        
        # Look for numeric scores in the response
        import re
        
        patterns = {
            'novelty': r'novelty.*?(\d+\.?\d*)',
            'relevance': r'relevance.*?(\d+\.?\d*)',
            'actionability': r'actionability.*?(\d+\.?\d*)',
            'transition': r'transition.*?quality.*?(\d+\.?\d*)'
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, response_text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    if score > 1.0:  # Handle scores on 0-10 scale
                        score = score / 10.0
                    metrics[metric] = min(max(score, 0.0), 1.0)
                except ValueError:
                    metrics[metric] = 0.5
            else:
                metrics[metric] = 0.5
        
        return metrics
    
    def _classify_insight_types(self, insights: List[str]) -> List[InsightType]:
        """Classify insights into types based on content."""
        insight_types = []
        
        type_keywords = {
            InsightType.THEORETICAL_FOUNDATION: ['theory', 'framework', 'concept', 'principle'],
            InsightType.MARKET_OPPORTUNITY: ['market', 'opportunity', 'demand', 'commercial'],
            InsightType.TECHNICAL_FEASIBILITY: ['technical', 'implementation', 'feasible', 'technology'],
            InsightType.IMPLEMENTATION_BARRIER: ['barrier', 'challenge', 'obstacle', 'limitation'],
            InsightType.RESEARCH_GAP: ['gap', 'lacking', 'missing', 'unexplored'],
            InsightType.COMPETITIVE_ADVANTAGE: ['advantage', 'competitive', 'differentiation', 'unique'],
            InsightType.FUTURE_TREND: ['future', 'trend', 'emerging', 'evolution'],
            InsightType.VALIDATION_CRITERIA: ['validation', 'criteria', 'measure', 'evaluation']
        }
        
        for insight in insights:
            insight_lower = insight.lower()
            matched_types = []
            
            for insight_type, keywords in type_keywords.items():
                if any(keyword in insight_lower for keyword in keywords):
                    matched_types.append(insight_type)
            
            if matched_types:
                insight_types.extend(matched_types)
            else:
                insight_types.append(InsightType.RESEARCH_GAP)  # Default
        
        return list(set(insight_types))  # Remove duplicates
    
    def _calculate_transition_quality(
        self,
        from_agent: AgentType,
        to_agent: AgentType,
        insights: List[str],
        execution_time: float
    ) -> float:
        """Calculate the quality of insight transition between agents."""
        quality_score = 0.0
        
        # Base score for having insights
        if insights:
            quality_score += 0.2
        
        # Quantity factor (0-0.2)
        insight_count = len(insights)
        if insight_count >= 3:
            quality_score += 0.1
        if insight_count >= 5:
            quality_score += 0.1
        
        # Quality factor (0-0.3)
        avg_insight_length = sum(len(insight) for insight in insights) / len(insights) if insights else 0
        if avg_insight_length > 100:
            quality_score += 0.15
        if avg_insight_length > 200:
            quality_score += 0.15
        
        # Efficiency factor (0-0.3)
        if execution_time < 60:  # Fast execution
            quality_score += 0.15
        elif execution_time < 120:  # Reasonable execution
            quality_score += 0.1
        
        if insight_count > 0:
            insights_per_minute = insight_count / (execution_time / 60)
            if insights_per_minute > 2:
                quality_score += 0.15
        
        return min(quality_score, 1.0)
    
    async def _update_adaptive_learning(
        self,
        from_agent: AgentType,
        to_agent: AgentType,
        analysis_result: InsightAnalysisResult
    ):
        """Update adaptive learning state based on transition analysis."""
        transition_key = f"{from_agent.value}→{to_agent.value}"
        
        # Update transition success rates
        current_rate = self.adaptive_state.transition_success_rates.get(transition_key, 0.5)
        new_rate = (
            current_rate * (1 - self.adaptive_state.adaptation_learning_rate) +
            analysis_result.transition_quality * self.adaptive_state.adaptation_learning_rate
        )
        self.adaptive_state.transition_success_rates[transition_key] = new_rate
        
        # Check if transition is productive
        is_productive = analysis_result.transition_quality >= self.adaptive_state.productive_transition_threshold
        
        # Create insight transition record
        transition = InsightTransition(
            from_agent=from_agent,
            to_agent=to_agent,
            source_insight="\n".join(analysis_result.key_insights),
            insight_type=analysis_result.insight_types[0] if analysis_result.insight_types else InsightType.RESEARCH_GAP,
            generated_questions=[],  # Will be filled by question generation
            question_quality_score=analysis_result.actionability_score,
            research_depth_achieved=analysis_result.relevance_score,
            novel_findings_discovered=len(analysis_result.key_insights),
            time_to_productive_research=0.0,  # Will be calculated in sequence execution
            productive_transition=is_productive,
            productivity_score=analysis_result.transition_quality
        )
        
        # Store transition pattern
        if transition_key not in self.adaptive_state.insight_transition_patterns:
            self.adaptive_state.insight_transition_patterns[transition_key] = []
        self.adaptive_state.insight_transition_patterns[transition_key].append(transition)
        
        # Update last updated timestamp
        self.adaptive_state.last_updated = datetime.utcnow()
        
        logger.debug(f"Updated adaptive learning for {transition_key}: rate={new_rate:.3f}")
    
    def get_optimal_sequence_for_topic(self, research_topic: str) -> Optional[str]:
        """Get the historically optimal sequence for a research topic type."""
        # Simple topic classification for demonstration
        topic_lower = research_topic.lower()
        
        if any(word in topic_lower for word in ['market', 'business', 'commercial', 'industry']):
            topic_type = "market_focused"
        elif any(word in topic_lower for word in ['technology', 'technical', 'implementation', 'future']):
            topic_type = "tech_focused"  
        elif any(word in topic_lower for word in ['theory', 'research', 'academic', 'scientific']):
            topic_type = "theory_focused"
        else:
            topic_type = "general"
        
        return self.adaptive_state.optimal_sequence_for_topic_types.get(topic_type)
    
    def prevent_cognitive_offloading(self, agent_type: AgentType, questions: List[str]) -> List[str]:
        """Modify questions to prevent cognitive offloading and ensure internal reasoning."""
        enhanced_questions = []
        
        for question in questions:
            # Add reasoning requirements
            if not any(phrase in question.lower() for phrase in ['analyze', 'evaluate', 'compare', 'synthesize']):
                if question.endswith('?'):
                    question = question[:-1] + " and provide your analytical reasoning?"
                else:
                    question += " - include your analytical reasoning."
            
            # Add domain-specific thinking requirements
            thinking_prompts = {
                AgentType.ACADEMIC: "Consider theoretical frameworks and research methodologies in your analysis.",
                AgentType.INDUSTRY: "Consider market dynamics and business implications in your analysis.",
                AgentType.TECHNICAL_TRENDS: "Consider technical feasibility and future trends in your analysis."
            }
            
            enhanced_question = f"{question}\n{thinking_prompts[agent_type]}"
            enhanced_questions.append(enhanced_question)
        
        return enhanced_questions