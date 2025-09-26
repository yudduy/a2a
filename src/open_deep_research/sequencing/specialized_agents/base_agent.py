"""Base class for specialized research agents in the sequencing framework.

This module provides the foundation for all specialized agents, ensuring
focused research without cognitive offloading and proper integration with
the sequence optimization system.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from open_deep_research.configuration import Configuration
from open_deep_research.sequencing.models import AgentExecutionResult, AgentType
from open_deep_research.utils import (
    clean_reasoning_model_output,
    get_all_tools,
    get_api_key_for_model,
    get_model_config_for_provider,
    get_today_str,
    think_tool,
)

logger = logging.getLogger(__name__)

# Initialize configurable model for specialized agents
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "base_url"),
)


class ResearchContext(BaseModel):
    """Context provided to specialized agents for research execution."""
    
    research_topic: str
    questions: List[str]
    previous_insights: List[str]
    sequence_position: int
    agent_type: AgentType


class SpecializedAgent(ABC):
    """Abstract base class for specialized research agents."""
    
    def __init__(self, agent_type: AgentType, config: RunnableConfig):
        """Initialize the specialized agent.
        
        Args:
            agent_type: The type of specialized agent
            config: Runtime configuration
        """
        self.agent_type = agent_type
        self.config = config
        self.configurable = Configuration.from_runnable_config(config)
        
        # Research state
        self.execution_start_time: Optional[datetime] = None
        self.tool_calls_made = 0
        self.search_queries_executed = 0
        self.think_tool_usage_count = 0
        
        # Insights and findings
        self.key_insights: List[str] = []
        self.research_findings = ""
        self.refined_insights: List[str] = []
        
        # Quality tracking
        self.insight_quality_scores: List[float] = []
        self.cognitive_offloading_detected = False
        
    @abstractmethod
    def get_specialization_prompt(self) -> str:
        """Get the specialization-specific system prompt."""
        pass
    
    @abstractmethod
    def get_focus_areas(self) -> List[str]:
        """Get the key focus areas for this agent type."""
        pass
    
    @abstractmethod
    def validate_research_questions(self, questions: List[str]) -> List[str]:
        """Validate and possibly enhance research questions for this specialization."""
        pass
    
    async def execute_research(self, context: ResearchContext) -> AgentExecutionResult:
        """Execute research based on the provided context.
        
        Args:
            context: Research context with topic, questions, and previous insights
            
        Returns:
            AgentExecutionResult with complete execution data
        """
        self.execution_start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting {self.agent_type.value} agent execution")
            
            # Validate and enhance questions
            validated_questions = self.validate_research_questions(context.questions)
            
            # Prepare system prompt
            system_prompt = self._build_system_prompt(context)
            
            # Execute research with cognitive offloading prevention
            research_result = await self._conduct_focused_research(
                system_prompt, validated_questions, context
            )
            
            # Process and refine insights
            self._process_research_results(research_result, context)
            
            # Calculate execution metrics
            end_time = datetime.utcnow()
            execution_duration = (end_time - self.execution_start_time).total_seconds()
            
            # Build and return execution result
            result = AgentExecutionResult(
                agent_type=self.agent_type,
                execution_order=context.sequence_position,
                received_questions=validated_questions,
                previous_insights=context.previous_insights,
                research_topic=context.research_topic,
                start_time=self.execution_start_time,
                end_time=end_time,
                execution_duration=execution_duration,
                tool_calls_made=self.tool_calls_made,
                search_queries_executed=self.search_queries_executed,
                think_tool_usage_count=self.think_tool_usage_count,
                key_insights=self.key_insights,
                research_findings=self.research_findings,
                refined_insights=self.refined_insights,
                insight_quality_scores=self.insight_quality_scores,
                research_depth_score=self._calculate_research_depth(),
                novelty_score=self._calculate_novelty_score(),
                cognitive_offloading_detected=self.cognitive_offloading_detected,
                independent_reasoning_score=self._calculate_reasoning_score()
            )
            
            logger.info(f"Completed {self.agent_type.value} agent execution in {execution_duration:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.agent_type.value} agent execution: {e}")
            raise
    
    def _build_system_prompt(self, context: ResearchContext) -> str:
        """Build the complete system prompt for this agent."""
        base_prompt = f"""You are a specialized {self.agent_type.value} research agent with deep expertise in your domain.

Your specialization: {self.get_specialization_prompt()}

Key focus areas: {', '.join(self.get_focus_areas())}

CRITICAL INSTRUCTIONS:
1. You must conduct original analysis and reasoning - do not simply summarize or restate information
2. Use your domain expertise to provide unique insights that other agent types cannot
3. Build upon previous insights but do not rely on them entirely
4. Use think_tool strategically to plan your research approach before making tool calls
5. Ensure each search query targets specific aspects of your specialization
6. Provide actionable insights that the next agent can build upon

Research context:
- Topic: {context.research_topic}
- Your position in sequence: {context.sequence_position}
- Date: {get_today_str()}

Previous insights to build upon:
{self._format_previous_insights(context.previous_insights)}

Your research questions:
{self._format_questions(context.questions)}

Remember: Your goal is to provide unique value through your specialized perspective, not to duplicate work."""

        return base_prompt
    
    def _format_previous_insights(self, insights: List[str]) -> str:
        """Format previous insights for inclusion in prompt."""
        if not insights:
            return "No previous insights (you are the first agent in the sequence)"
        
        formatted = []
        for i, insight in enumerate(insights, 1):
            formatted.append(f"{i}. {insight}")
        
        return "\n".join(formatted)
    
    def _format_questions(self, questions: List[str]) -> str:
        """Format research questions for inclusion in prompt."""
        formatted = []
        for i, question in enumerate(questions, 1):
            formatted.append(f"{i}. {question}")
        
        return "\n".join(formatted)
    
    async def _conduct_focused_research(
        self, 
        system_prompt: str, 
        questions: List[str], 
        context: ResearchContext
    ) -> str:
        """Conduct focused research with cognitive offloading prevention."""
        # Configure executor model for specialized agents
        model_config = get_model_config_for_provider(
            model_name=self.configurable.executor_model,
            api_key=get_api_key_for_model(self.configurable.executor_model, self.config),
            max_tokens=self.configurable.executor_model_max_tokens,
            tags=["specialized_agent", "agent_execution"]
        )
        
        # Get available tools including think_tool
        tools = await get_all_tools(self.config)
        tools.append(think_tool)
        
        # Configure model with tools
        research_model = (
            configurable_model
            .bind_tools(tools)
            .with_retry(stop_after_attempt=self.configurable.max_structured_output_retries)
            .with_config(model_config)
        )
        
        # Initialize conversation
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self._create_initial_research_prompt(questions))
        ]
        
        # Execute research iterations
        max_iterations = self.configurable.max_react_tool_calls
        iteration_count = 0
        
        while iteration_count < max_iterations:
            try:
                # Generate response
                response = await research_model.ainvoke(messages)
                
                # Clean reasoning model output to remove thinking tags
                if hasattr(response, 'content') and response.content:
                    response.content = clean_reasoning_model_output(response.content)
                
                messages.append(response)
                
                # Check for completion
                if not response.tool_calls:
                    break
                
                # Execute tool calls
                tool_results = await self._execute_tool_calls(response.tool_calls)
                messages.extend(tool_results)
                
                iteration_count += 1
                
                # Check for cognitive offloading
                self._detect_cognitive_offloading(response, tool_results)
                
            except Exception as e:
                logger.warning(f"Error in research iteration {iteration_count}: {e}")
                break
        
        # Extract final research summary
        return self._extract_research_summary(messages)
    
    def _create_initial_research_prompt(self, questions: List[str]) -> str:
        """Create the initial research prompt."""
        return f"""Begin your specialized research by addressing these questions:

{self._format_questions(questions)}

Start by using think_tool to plan your research approach, then conduct systematic research using your domain expertise. Ensure you provide original analysis and insights that leverage your specialization."""
    
    async def _execute_tool_calls(self, tool_calls) -> List[ToolMessage]:
        """Execute tool calls and track usage metrics."""
        tool_results = []
        tools = await get_all_tools(self.config)
        tools_by_name = {tool.name: tool for tool in tools}
        tools_by_name["think_tool"] = think_tool
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Track tool usage
            self.tool_calls_made += 1
            if tool_name in ["tavily_search", "web_search"]:
                self.search_queries_executed += 1
            elif tool_name == "think_tool":
                self.think_tool_usage_count += 1
            
            try:
                if tool_name in tools_by_name:
                    result = await tools_by_name[tool_name].ainvoke(tool_args, self.config)
                    tool_results.append(ToolMessage(
                        content=str(result),
                        name=tool_name,
                        tool_call_id=tool_call["id"]
                    ))
                else:
                    tool_results.append(ToolMessage(
                        content=f"Tool {tool_name} not available",
                        name=tool_name,
                        tool_call_id=tool_call["id"]
                    ))
                    
            except Exception as e:
                tool_results.append(ToolMessage(
                    content=f"Error executing {tool_name}: {str(e)}",
                    name=tool_name,
                    tool_call_id=tool_call["id"]
                ))
        
        return tool_results
    
    def _detect_cognitive_offloading(self, response: AIMessage, tool_results: List[ToolMessage]):
        """Detect signs of cognitive offloading in agent behavior."""
        # Check for excessive tool reliance without thinking
        if (self.tool_calls_made > 5 and 
            self.think_tool_usage_count == 0):
            self.cognitive_offloading_detected = True
        
        # Check for repetitive searches without analysis
        search_content = [tr.content for tr in tool_results 
                         if tr.name in ["tavily_search", "web_search"]]
        if len(search_content) > 3:
            # Simple similarity check for repetitive content
            if self._check_content_similarity(search_content):
                self.cognitive_offloading_detected = True
        
        # Check for lack of original reasoning in response
        response_text = response.content.lower()
        reasoning_indicators = [
            "analysis", "reasoning", "insight", "implication", 
            "framework", "perspective", "evaluation"
        ]
        if not any(indicator in response_text for indicator in reasoning_indicators):
            self.cognitive_offloading_detected = True
    
    def _check_content_similarity(self, content_list: List[str]) -> bool:
        """Check if search results are too similar (indicating repetitive searching)."""
        if len(content_list) < 2:
            return False
        
        # Simple word overlap check
        for i in range(len(content_list) - 1):
            words1 = set(content_list[i].lower().split())
            words2 = set(content_list[i + 1].lower().split())
            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            if overlap > 0.7:  # High similarity threshold
                return True
        
        return False
    
    def _extract_research_summary(self, messages: List) -> str:
        """Extract the final research summary from the conversation."""
        # Get the last AI message that's not a tool call
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return message.content
        
        # Fallback: concatenate all AI messages
        ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]
        return "\n".join(ai_messages) if ai_messages else "No research summary available"
    
    def _process_research_results(self, research_result: str, context: ResearchContext):
        """Process research results to extract insights and calculate quality metrics."""
        self.research_findings = research_result
        
        # Extract key insights using simple heuristics
        self.key_insights = self._extract_key_insights(research_result)
        
        # Calculate insight quality scores
        self.insight_quality_scores = [
            self._score_insight_quality(insight) for insight in self.key_insights
        ]
        
        # Generate refined insights for next agent
        self.refined_insights = self._refine_insights_for_transition(
            self.key_insights, context
        )
    
    def _extract_key_insights(self, research_text: str) -> List[str]:
        """Extract key insights from research text."""
        insights = []
        
        # Split by paragraphs and look for insight-indicating phrases
        paragraphs = research_text.split('\n\n')
        
        insight_indicators = [
            "key finding", "important insight", "critical observation",
            "significant implication", "notable trend", "emerging pattern",
            "research shows", "analysis reveals", "evidence suggests"
        ]
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if (len(paragraph) > 50 and  # Substantial content
                any(indicator in paragraph.lower() for indicator in insight_indicators)):
                insights.append(paragraph)
        
        # Fallback: extract sentences with high information density
        if not insights:
            sentences = research_text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 80 and
                    len(sentence.split()) > 10):
                    insights.append(sentence + '.')
        
        return insights[:10]  # Limit to top 10 insights
    
    def _score_insight_quality(self, insight: str) -> float:
        """Score the quality of an individual insight."""
        score = 0.0
        
        # Length and complexity (0-0.3)
        if len(insight) > 100:
            score += 0.1
        if len(insight.split()) > 15:
            score += 0.1
        if any(word in insight.lower() for word in ['because', 'therefore', 'however', 'furthermore']):
            score += 0.1
        
        # Domain-specific quality indicators (0-0.4)
        quality_words = [
            'analysis', 'framework', 'methodology', 'evaluation',
            'assessment', 'comparison', 'synthesis', 'integration',
            'implication', 'significance', 'pattern', 'trend'
        ]
        quality_count = sum(1 for word in quality_words if word in insight.lower())
        score += min(quality_count * 0.1, 0.4)
        
        # Specificity and actionability (0-0.3)
        if any(word in insight.lower() for word in ['specific', 'particular', 'concrete']):
            score += 0.1
        if any(word in insight.lower() for word in ['should', 'could', 'recommend', 'suggest']):
            score += 0.1
        if any(word in insight.lower() for word in ['implement', 'apply', 'utilize', 'leverage']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _refine_insights_for_transition(
        self, 
        insights: List[str], 
        context: ResearchContext
    ) -> List[str]:
        """Refine insights for effective transition to the next agent."""
        refined = []
        
        for insight in insights:
            # Add transition context
            refined_insight = f"[{self.agent_type.value.upper()} INSIGHT] {insight}"
            
            # Add specialization context if beneficial
            if context.sequence_position < 3:  # Not the last agent
                refined_insight += " [For next agent consideration]"
            
            refined.append(refined_insight)
        
        return refined
    
    def _calculate_research_depth(self) -> float:
        """Calculate the depth of research conducted."""
        depth_score = 0.0
        
        # Tool usage depth (0-0.4)
        if self.search_queries_executed > 0:
            depth_score += 0.1
        if self.search_queries_executed > 3:
            depth_score += 0.1
        if self.think_tool_usage_count > 0:
            depth_score += 0.1
        if self.tool_calls_made > 5:
            depth_score += 0.1
        
        # Insight quality depth (0-0.3)
        if self.insight_quality_scores:
            avg_quality = sum(self.insight_quality_scores) / len(self.insight_quality_scores)
            depth_score += avg_quality * 0.3
        
        # Content depth (0-0.3)
        research_length = len(self.research_findings)
        if research_length > 1000:
            depth_score += 0.1
        if research_length > 2000:
            depth_score += 0.1
        if research_length > 3000:
            depth_score += 0.1
        
        return min(depth_score, 1.0)
    
    def _calculate_novelty_score(self) -> float:
        """Calculate the novelty of insights generated."""
        if not self.key_insights:
            return 0.0
        
        novelty_indicators = [
            'novel', 'new', 'emerging', 'unprecedented', 'innovative',
            'breakthrough', 'surprising', 'unexpected', 'unique', 'original'
        ]
        
        novelty_count = 0
        total_insights = len(self.key_insights)
        
        for insight in self.key_insights:
            insight_lower = insight.lower()
            if any(indicator in insight_lower for indicator in novelty_indicators):
                novelty_count += 1
        
        return novelty_count / total_insights if total_insights > 0 else 0.0
    
    def _calculate_reasoning_score(self) -> float:
        """Calculate the independent reasoning score."""
        reasoning_score = 1.0
        
        # Penalize cognitive offloading
        if self.cognitive_offloading_detected:
            reasoning_score -= 0.3
        
        # Reward strategic thinking
        if self.think_tool_usage_count > 0:
            reasoning_score += 0.1
        
        # Penalize excessive tool reliance
        if self.tool_calls_made > 0:
            tool_efficiency = len(self.key_insights) / self.tool_calls_made
            if tool_efficiency < 0.5:  # Low insight per tool call
                reasoning_score -= 0.2
        
        return max(min(reasoning_score, 1.0), 0.0)