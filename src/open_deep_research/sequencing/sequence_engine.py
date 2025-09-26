"""Sequence Optimization Engine for executing and comparing agent orderings.

This engine executes different sequence patterns, tracks performance metrics,
and provides comparative analysis to prove that sequential ordering affects
productivity outcomes in research tasks.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.runnables import RunnableConfig

from open_deep_research.sequencing.metrics import MetricsCalculator
from open_deep_research.sequencing.models import (
    SEQUENCE_PATTERNS,
    AgentExecutionResult,
    AgentType,
    DynamicSequencePattern,
    QueryType,
    ResearchDomain,
    ScopeBreadth,
    SequenceAnalysis,
    SequenceComparison,
    SequencePattern,
    SequenceResult,
)
from open_deep_research.sequencing.research_director import SupervisorResearchDirector
from open_deep_research.sequencing.sequence_selector import SequenceAnalyzer
from open_deep_research.sequencing.specialized_agents import (
    AcademicAgent,
    IndustryAgent,
    ResearchContext,
    TechnicalTrendsAgent,
)

logger = logging.getLogger(__name__)

# Import metrics aggregator for real-time integration
try:
    from open_deep_research.sequencing.metrics_aggregator import MetricsAggregator
except ImportError:
    MetricsAggregator = None


class SequenceOptimizationEngine:
    """Engine for executing and optimizing agent sequence patterns."""
    
    def __init__(
        self, 
        config: RunnableConfig,
        metrics_aggregator: Optional['MetricsAggregator'] = None,
        enable_real_time_metrics: bool = True
    ):
        """Initialize the sequence optimization engine.
        
        Args:
            config: Runtime configuration for agents and models
            metrics_aggregator: Optional metrics aggregator for real-time tracking
            enable_real_time_metrics: Enable real-time metrics collection
        """
        self.config = config
        self.research_director = SupervisorResearchDirector(config)
        self.metrics_calculator = MetricsCalculator(enable_streaming=enable_real_time_metrics)
        self.sequence_analyzer = SequenceAnalyzer()
        
        # Metrics integration
        self.metrics_aggregator = metrics_aggregator
        self.enable_real_time_metrics = enable_real_time_metrics
        self._current_execution_id: Optional[str] = None
        self._current_strategy: Optional[str] = None
        
        # Dynamic agent creation - agents are created on demand
        self._agent_cache: Dict[AgentType, Any] = {}
        self._config = config
        
        # Execution history
        self.execution_history: List[SequenceResult] = []
        self.comparison_history: List[SequenceComparison] = []
        self.analysis_history: List[SequenceAnalysis] = []
    
    def _get_agent(self, agent_type: AgentType) -> Any:
        """Get or create an agent instance dynamically.
        
        Args:
            agent_type: The type of agent to create/retrieve
            
        Returns:
            The agent instance for the specified type
        """
        if agent_type not in self._agent_cache:
            if agent_type == AgentType.ACADEMIC:
                self._agent_cache[agent_type] = AcademicAgent(self._config)
            elif agent_type == AgentType.INDUSTRY:
                self._agent_cache[agent_type] = IndustryAgent(self._config)
            elif agent_type == AgentType.TECHNICAL_TRENDS:
                self._agent_cache[agent_type] = TechnicalTrendsAgent(self._config)
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")
        
        return self._agent_cache[agent_type]
    
    async def execute_sequence(
        self, 
        sequence_pattern: Union[SequencePattern, DynamicSequencePattern], 
        research_topic: str,
        execution_id: Optional[str] = None
    ) -> SequenceResult:
        """Execute a complete sequence pattern for a research topic.
        
        Args:
            sequence_pattern: The sequence pattern to execute (SequencePattern or DynamicSequencePattern)
            research_topic: The research topic to investigate
            execution_id: Optional execution ID for metrics tracking
            
        Returns:
            SequenceResult with complete execution data and metrics
        """
        # Handle both pattern types
        if isinstance(sequence_pattern, DynamicSequencePattern):
            pattern_description = f"dynamic pattern ({sequence_pattern.sequence_id})"
            agent_order = sequence_pattern.agent_order
            strategy = None  # Dynamic patterns don't have strategy enum
        else:
            pattern_description = sequence_pattern.strategy
            agent_order = sequence_pattern.agent_order
            strategy = sequence_pattern.strategy
        
        logger.info(f"Starting sequence execution: {pattern_description} for '{research_topic}'")
        
        # Set up metrics tracking
        self._current_execution_id = execution_id
        self._current_strategy = strategy
        
        start_time = datetime.utcnow()
        agent_results: List[AgentExecutionResult] = []
        insight_transitions = []
        
        # Track cumulative insights for linear context passing
        previous_insights: List[str] = []
        
        # Initialize metrics collection if aggregator is available
        if self.metrics_aggregator and execution_id and strategy:
            self.metrics_aggregator.collect_sequence_metrics(
                execution_id, 
                strategy,
                status_update="running"
            )
        
        try:
            # Execute each agent in sequence
            for position, agent_type in enumerate(agent_order, 1):
                logger.info(f"Executing agent {position}/{len(agent_order)}: {agent_type.value}")
                
                # Generate questions for this agent (except first)
                if position == 1:
                    # First agent gets initial research questions
                    questions = self._generate_initial_questions(research_topic, agent_type)
                else:
                    # Subsequent agents get dynamically generated questions
                    question_result = await self.research_director.direct_next_investigation(
                        previous_agent_insights=previous_insights,
                        next_agent_type=agent_type,
                        research_context=research_topic,
                        sequence_position=position
                    )
                    questions = question_result.questions
                
                # Create research context
                context = ResearchContext(
                    research_topic=research_topic,
                    questions=questions,
                    previous_insights=previous_insights.copy(),  # Linear context
                    sequence_position=position,
                    agent_type=agent_type
                )
                
                # Execute agent research using dynamic agent creation
                agent_instance = self._get_agent(agent_type)
                agent_result = await agent_instance.execute_research(context)
                agent_results.append(agent_result)
                
                # Real-time metrics collection
                if self.enable_real_time_metrics and self.metrics_aggregator and execution_id and strategy:
                    try:
                        # Collect metrics for this agent completion
                        self.metrics_aggregator.collect_sequence_metrics(
                            execution_id,
                            strategy,
                            agent_result=agent_result
                        )
                        
                        # Calculate real-time productivity
                        await self.metrics_calculator.calculate_real_time_productivity(
                            sequence_id=f"{execution_id}_{strategy if strategy else 'dynamic'}",
                            agent_results=agent_results,
                            insight_transitions=insight_transitions,
                            partial=position < len(agent_order)
                        )
                        
                        logger.debug(f"Real-time metrics updated for agent {position}/{len(agent_order)}")
                        
                    except Exception as e:
                        logger.warning(f"Error updating real-time metrics: {e}")
                
                # Track insight productivity if not first agent
                if position > 1:
                    insight_analysis = await self.research_director.track_insight_productivity(
                        from_agent=agent_order[position - 2],
                        to_agent=agent_type,
                        insights=agent_result.key_insights,
                        research_context=research_topic,
                        execution_time=agent_result.execution_duration
                    )
                    
                    # Create insight transition record
                    from open_deep_research.sequencing.models import (
                        InsightTransition,
                        InsightType,
                    )
                    transition = InsightTransition(
                        from_agent=agent_order[position - 2],
                        to_agent=agent_type,
                        source_insight="\n".join(previous_insights),
                        insight_type=insight_analysis.insight_types[0] if insight_analysis.insight_types else InsightType.RESEARCH_GAP,
                        generated_questions=questions,
                        question_quality_score=len([q for q in questions if len(q) > 50]) / len(questions) if questions else 0.0,
                        research_depth_achieved=agent_result.research_depth_score,
                        novel_findings_discovered=len(agent_result.key_insights),
                        time_to_productive_research=agent_result.execution_duration,
                        productive_transition=insight_analysis.transition_quality > 0.7,
                        productivity_score=insight_analysis.transition_quality
                    )
                    insight_transitions.append(transition)
                
                # Update previous insights for next agent (linear context)
                previous_insights = agent_result.refined_insights
                
                logger.info(f"Completed agent {position}: {len(agent_result.key_insights)} insights generated")
            
            # Generate final synthesis
            final_synthesis = await self._synthesize_research_findings(
                agent_results, research_topic, sequence_pattern
            )
            
            # Calculate overall metrics
            end_time = datetime.utcnow()
            total_duration = (end_time - start_time).total_seconds()
            
            overall_metrics = self.metrics_calculator.calculate_sequence_productivity(
                agent_results, insight_transitions, total_duration
            )
            
            # Build sequence result
            sequence_result = SequenceResult(
                sequence_pattern=sequence_pattern,
                research_topic=research_topic,
                start_time=start_time,
                end_time=end_time,
                total_duration=total_duration,
                agent_results=agent_results,
                insight_transitions=insight_transitions,
                final_research_synthesis=final_synthesis,
                comprehensive_findings=self._extract_comprehensive_findings(agent_results),
                overall_productivity_metrics=overall_metrics,
                sequence_efficiency_score=self._calculate_sequence_efficiency(agent_results),
                unique_insights_generated=len(set(
                    insight for result in agent_results for insight in result.key_insights
                )),
                research_breadth_score=self._calculate_research_breadth(agent_results),
                research_depth_score=sum(r.research_depth_score for r in agent_results) / max(len(agent_results), 1),
                final_quality_score=self._calculate_final_quality(agent_results),
                completeness_score=self._calculate_completeness_score(agent_results),
                actionability_score=self._calculate_actionability_score(agent_results)
            )
            
            # Final metrics collection
            if self.enable_real_time_metrics and self.metrics_aggregator and execution_id and strategy:
                try:
                    # Mark sequence as completed
                    self.metrics_aggregator.collect_sequence_metrics(
                        execution_id,
                        strategy,
                        status_update="completed"
                    )
                    
                    # Final real-time productivity calculation
                    await self.metrics_calculator.calculate_real_time_productivity(
                        sequence_id=f"{execution_id}_{strategy if strategy else 'dynamic'}",
                        agent_results=agent_results,
                        insight_transitions=insight_transitions,
                        partial=False
                    )
                    
                    logger.info(f"Final metrics collected for sequence {strategy if strategy else pattern_description}")
                    
                except Exception as e:
                    logger.warning(f"Error collecting final metrics: {e}")
            
            # Store in execution history
            self.execution_history.append(sequence_result)
            
            logger.info(f"Sequence execution completed: {total_duration:.1f}s, "
                       f"Tool Productivity: {overall_metrics.tool_productivity:.3f}")
            
            return sequence_result
            
        except Exception as e:
            # Report error to metrics aggregator
            if self.metrics_aggregator and execution_id and strategy:
                try:
                    self.metrics_aggregator.collect_sequence_metrics(
                        execution_id,
                        strategy,
                        status_update="failed",
                        error=str(e)
                    )
                except Exception as metrics_error:
                    logger.warning(f"Error reporting execution failure to metrics: {metrics_error}")
            
            logger.error(f"Error in sequence execution: {e}")
            raise
    
    def _generate_initial_questions(self, research_topic: str, agent_type: AgentType) -> List[str]:
        """Generate initial research questions for the first agent."""
        base_questions = [
            f"What are the key aspects of {research_topic} that require investigation?",
            f"What foundational understanding is needed about {research_topic}?",
            f"What are the primary considerations when analyzing {research_topic}?"
        ]
        
        # Add agent-specific initial questions
        agent_specific = {
            AgentType.ACADEMIC: [
                f"What does current academic research reveal about {research_topic}?",
                f"What theoretical frameworks apply to {research_topic}?",
                f"What research gaps exist in the study of {research_topic}?"
            ],
            AgentType.INDUSTRY: [
                f"What market opportunities exist related to {research_topic}?",
                f"What business models could be applied to {research_topic}?",
                f"What competitive landscape exists around {research_topic}?"
            ],
            AgentType.TECHNICAL_TRENDS: [
                f"What technical trends are relevant to {research_topic}?",
                f"What implementation challenges exist for {research_topic}?",
                f"What future technology developments could impact {research_topic}?"
            ]
        }
        
        return base_questions + agent_specific.get(agent_type, [])
    
    async def _synthesize_research_findings(
        self, 
        agent_results: List[AgentExecutionResult], 
        research_topic: str,
        sequence_pattern: Union[SequencePattern, DynamicSequencePattern]
    ) -> str:
        """Synthesize findings from all agents into a comprehensive summary."""
        # Handle both pattern types
        if isinstance(sequence_pattern, DynamicSequencePattern):
            strategy_description = f"Dynamic Pattern (ID: {sequence_pattern.sequence_id})"
            strategy_value = sequence_pattern.description
        else:
            strategy_description = f"Sequence Strategy: {sequence_pattern.strategy}"
            strategy_value = sequence_pattern.strategy
        
        synthesis_parts = [
            f"# Comprehensive Research Synthesis: {research_topic}",
            f"## {strategy_description}",
            f"## Agent Execution Order: {' → '.join([a.value for a in sequence_pattern.agent_order])}",
            "",
            "## Executive Summary",
            f"This research synthesis presents findings from a {len(agent_results)}-agent sequential analysis "
            f"following the {strategy_value} approach. Each agent contributed specialized "
            f"insights building upon previous findings to create comprehensive understanding.",
            ""
        ]
        
        # Add findings from each agent
        for i, result in enumerate(agent_results, 1):
            agent_name = result.agent_type.value.replace('_', ' ').title()
            synthesis_parts.extend([
                f"## {i}. {agent_name} Analysis",
                f"**Research Duration:** {result.execution_duration:.1f} seconds",
                f"**Tool Calls:** {result.tool_calls_made}",
                f"**Key Insights Generated:** {len(result.key_insights)}",
                "",
                "### Key Findings:",
            ])
            
            for j, insight in enumerate(result.key_insights, 1):
                synthesis_parts.append(f"{j}. {insight}")
            
            synthesis_parts.extend(["", "### Research Summary:", result.research_findings[:500] + "..." if len(result.research_findings) > 500 else result.research_findings, ""])
        
        # Add cross-agent insights
        all_insights = [insight for result in agent_results for insight in result.key_insights]
        unique_insights = list(set(all_insights))
        
        synthesis_parts.extend([
            "## Cross-Agent Analysis",
            f"**Total Insights Generated:** {len(all_insights)}",
            f"**Unique Insights:** {len(unique_insights)}",
            f"**Insight Redundancy:** {((len(all_insights) - len(unique_insights)) / max(len(all_insights), 1) * 100):.1f}%",
            "",
            "## Sequence Effectiveness",
            f"The {strategy_value} sequence demonstrated effective knowledge building "
            f"with each agent contributing specialized perspectives that enhanced overall research quality.",
            ""
        ])
        
        return "\n".join(synthesis_parts)
    
    def _extract_comprehensive_findings(self, agent_results: List[AgentExecutionResult]) -> List[str]:
        """Extract comprehensive findings from all agent results."""
        findings = []
        
        for result in agent_results:
            agent_prefix = f"[{result.agent_type.value.upper()}]"
            for insight in result.key_insights:
                findings.append(f"{agent_prefix} {insight}")
        
        return findings
    
    def _calculate_sequence_efficiency(self, agent_results: List[AgentExecutionResult]) -> float:
        """Calculate overall sequence efficiency."""
        if not agent_results:
            return 0.0
        
        total_insights = sum(len(r.key_insights) for r in agent_results)
        total_tool_calls = sum(r.tool_calls_made for r in agent_results)
        total_time = sum(r.execution_duration for r in agent_results)
        
        # Efficiency = (Insights / Tool_Calls) * (Insights / Time)
        if total_tool_calls == 0 or total_time == 0:
            return 0.0
        
        insight_per_tool = total_insights / total_tool_calls
        insight_per_time = total_insights / (total_time / 60)  # per minute
        
        return min((insight_per_tool * insight_per_time) / 10, 1.0)  # Normalize to 0-1
    
    def _calculate_research_breadth(self, agent_results: List[AgentExecutionResult]) -> float:
        """Calculate research breadth across different domains."""
        if not agent_results:
            return 0.0
        
        # Count different research domains covered
        domain_indicators = {
            'academic': ['research', 'study', 'theory', 'academic', 'scholarly'],
            'market': ['market', 'business', 'commercial', 'industry', 'customer'],
            'technical': ['technical', 'technology', 'implementation', 'architecture', 'system']
        }
        
        domains_covered = set()
        all_text = " ".join([r.research_findings for r in agent_results]).lower()
        
        for domain, indicators in domain_indicators.items():
            if any(indicator in all_text for indicator in indicators):
                domains_covered.add(domain)
        
        return len(domains_covered) / len(domain_indicators)
    
    def _calculate_final_quality(self, agent_results: List[AgentExecutionResult]) -> float:
        """Calculate final quality score."""
        if not agent_results:
            return 0.0
        
        quality_factors = []
        
        for result in agent_results:
            # Agent-specific quality metrics
            insight_quality = sum(result.insight_quality_scores) / len(result.insight_quality_scores) if result.insight_quality_scores else 0
            quality_factors.extend([
                result.research_depth_score,
                result.novelty_score,
                result.independent_reasoning_score,
                insight_quality
            ])
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
    
    def _calculate_completeness_score(self, agent_results: List[AgentExecutionResult]) -> float:
        """Calculate research completeness."""
        if not agent_results:
            return 0.0
        
        # Completeness based on having substantial findings from each agent
        completeness_scores = []
        
        for result in agent_results:
            agent_completeness = 0.0
            
            # Has substantial insights
            if len(result.key_insights) >= 3:
                agent_completeness += 0.3
            
            # Has substantial research content
            if len(result.research_findings) >= 500:
                agent_completeness += 0.3
            
            # Made adequate tool calls
            if result.tool_calls_made >= 3:
                agent_completeness += 0.2
            
            # Has good research depth
            if result.research_depth_score >= 0.6:
                agent_completeness += 0.2
            
            completeness_scores.append(agent_completeness)
        
        return sum(completeness_scores) / len(completeness_scores)
    
    def _calculate_actionability_score(self, agent_results: List[AgentExecutionResult]) -> float:
        """Calculate actionability of research findings."""
        if not agent_results:
            return 0.0
        
        actionable_indicators = [
            'recommend', 'suggest', 'should', 'could', 'implement',
            'apply', 'strategy', 'approach', 'method', 'framework',
            'next steps', 'action', 'solution', 'opportunity'
        ]
        
        actionability_scores = []
        
        for result in agent_results:
            all_text = (result.research_findings + " " + " ".join(result.key_insights)).lower()
            actionable_count = sum(1 for indicator in actionable_indicators if indicator in all_text)
            
            # Normalize by text length
            text_length = len(all_text.split())
            if text_length > 0:
                actionability_scores.append(min(actionable_count / (text_length / 100), 1.0))
            else:
                actionability_scores.append(0.0)
        
        return sum(actionability_scores) / len(actionability_scores)
    
    def analyze_research_query(self, research_topic: str) -> SequenceAnalysis:
        """Analyze a research query and recommend optimal sequence strategies.
        
        Args:
            research_topic: The research query/topic to analyze
            
        Returns:
            SequenceAnalysis with recommendations and detailed reasoning
        """
        logger.info(f"Analyzing research query for sequence recommendation: '{research_topic[:100]}...'")
        
        # Use the sequence analyzer to analyze the query
        analysis = self.sequence_analyzer.analyze_query(research_topic)
        
        # Convert string enums to proper enum types for consistency
        analysis_dict = {
            "analysis_id": analysis.analysis_id,
            "query_type": QueryType(analysis.query_type),
            "research_domain": ResearchDomain(analysis.research_domain),
            "complexity_score": analysis.complexity_score,
            "scope_breadth": ScopeBreadth(analysis.scope_breadth),
            "recommended_sequences": analysis.recommended_sequences,
            "primary_recommendation": analysis.primary_recommendation,
            "confidence": analysis.confidence,
            "explanation": analysis.explanation,
            "reasoning": analysis.reasoning,
            "query_characteristics": analysis.query_characteristics,
            "original_query": research_topic,
            "analysis_timestamp": analysis.analysis_timestamp
        }
        
        # Create proper SequenceAnalysis model instance
        sequence_analysis = SequenceAnalysis(**analysis_dict)
        
        # Store in analysis history
        self.analysis_history.append(sequence_analysis)
        
        logger.info(f"Query analysis complete: {sequence_analysis.primary_recommendation.value} "
                   f"(confidence: {sequence_analysis.confidence:.2f})")
        
        return sequence_analysis
    
    def explain_sequence_choice(self, analysis: SequenceAnalysis) -> str:
        """Generate detailed explanation for sequence choice.
        
        Args:
            analysis: The sequence analysis results
            
        Returns:
            Detailed explanation of why the sequence was chosen
        """
        return analysis.explanation
    
    def get_all_sequence_explanations(self, analysis: SequenceAnalysis) -> Dict[str, str]:
        """Get explanations for all sequence strategies.
        
        Args:
            analysis: The sequence analysis results
            
        Returns:
            Dictionary mapping each strategy to its explanation
        """
        return self.sequence_analyzer.explain_all_sequences(analysis)
    
    async def execute_recommended_sequence(self, research_topic: str) -> Tuple[SequenceResult, SequenceAnalysis]:
        """Analyze query and execute the recommended sequence.
        
        Args:
            research_topic: The research topic to investigate
            
        Returns:
            Tuple of (SequenceResult, SequenceAnalysis)
        """
        # Analyze the query first
        analysis = self.analyze_research_query(research_topic)
        
        # Get the recommended sequence pattern (maintains backward compatibility)
        if hasattr(analysis, 'primary_recommendation') and analysis.primary_recommendation in SEQUENCE_PATTERNS:
            recommended_pattern = SEQUENCE_PATTERNS[analysis.primary_recommendation]
            logger.info(f"Executing recommended sequence: {analysis.primary_recommendation.value} "
                       f"for query: '{research_topic}'")
        else:
            # Fallback to a default pattern if the recommendation isn't found
            recommended_pattern = SEQUENCE_PATTERNS["theory_first"]
            logger.warning(f"Using fallback sequence pattern for query: '{research_topic}'")
        
        # Execute the recommended sequence
        result = await self.execute_sequence(recommended_pattern, research_topic)
        
        return result, analysis
    
    async def execute_dynamic_sequence(
        self, 
        agent_order: List[AgentType], 
        research_topic: str,
        description: str = "Custom dynamic sequence",
        reasoning: str = "User-defined agent ordering",
        confidence_score: float = 0.8,
        expected_advantages: Optional[List[str]] = None,
        execution_id: Optional[str] = None
    ) -> SequenceResult:
        """Execute a custom dynamic sequence pattern.
        
        Args:
            agent_order: The ordered list of agents to execute
            research_topic: The research topic to investigate
            description: Description of the sequence pattern
            reasoning: Reasoning for this agent ordering
            confidence_score: Confidence in this sequence choice (0.0-1.0)
            expected_advantages: List of expected advantages of this ordering
            execution_id: Optional execution ID for metrics tracking
            
        Returns:
            SequenceResult with complete execution data and metrics
        """
        if expected_advantages is None:
            expected_advantages = ["Custom agent ordering", "Flexible research approach"]
        
        # Create dynamic sequence pattern
        dynamic_pattern = DynamicSequencePattern(
            agent_order=agent_order,
            description=description,
            reasoning=reasoning,
            confidence_score=confidence_score,
            expected_advantages=expected_advantages,
            topic_alignment_score=confidence_score  # Use confidence as alignment score
        )
        
        logger.info(f"Executing custom dynamic sequence: {' → '.join([a.value for a in agent_order])} "
                   f"for query: '{research_topic}'")
        
        # Execute the dynamic sequence
        result = await self.execute_sequence(dynamic_pattern, research_topic, execution_id)
        
        return result
    
    async def intelligent_sequence_comparison(
        self, 
        research_topic: str,
        include_analysis: bool = True
    ) -> Tuple[SequenceComparison, Optional[SequenceAnalysis]]:
        """Perform intelligent sequence comparison with analysis explanation.
        
        Args:
            research_topic: The research topic to investigate
            include_analysis: Whether to include query analysis
            
        Returns:
            Tuple of (SequenceComparison, Optional[SequenceAnalysis])
        """
        analysis = None
        if include_analysis:
            # Analyze the query first to provide context
            analysis = self.analyze_research_query(research_topic)
            logger.info(f"Query analysis recommends {analysis.primary_recommendation.value} "
                       f"(confidence: {analysis.confidence:.2f})")
        
        # Execute comparison of all sequences
        comparison = await self.compare_sequences(research_topic)
        
        # Log analysis vs actual results if analysis was performed
        if analysis:
            actual_best = comparison.highest_productivity_sequence
            predicted_best = analysis.primary_recommendation
            
            logger.info(f"Analysis predicted: {predicted_best.value}, "
                       f"Actual best performer: {actual_best.value}, "
                       f"Match: {predicted_best == actual_best}")
        
        return comparison, analysis
    
    def get_analysis_summary(self) -> Dict:
        """Get a summary of all query analyses performed."""
        if not self.analysis_history:
            return {"message": "No query analyses recorded"}
        
        # Aggregate analysis results
        query_types = {}
        domains = {}
        recommendations = {}
        avg_confidence = 0.0
        
        for analysis in self.analysis_history:
            # Count query types
            query_type = analysis.query_type.value
            query_types[query_type] = query_types.get(query_type, 0) + 1
            
            # Count domains
            domain = analysis.research_domain.value
            domains[domain] = domains.get(domain, 0) + 1
            
            # Count recommendations
            rec = analysis.primary_recommendation.value
            recommendations[rec] = recommendations.get(rec, 0) + 1
            
            # Sum confidence
            avg_confidence += analysis.confidence
        
        avg_confidence /= len(self.analysis_history)
        
        return {
            "total_analyses": len(self.analysis_history),
            "query_types_distribution": query_types,
            "research_domains_distribution": domains,
            "recommendations_distribution": recommendations,
            "average_confidence": avg_confidence,
            "most_common_query_type": max(query_types.items(), key=lambda x: x[1])[0] if query_types else None,
            "most_recommended_sequence": max(recommendations.items(), key=lambda x: x[1])[0] if recommendations else None
        }
    
    async def compare_sequences(
        self, 
        research_topic: str, 
        strategies: Optional[List[str]] = None
    ) -> SequenceComparison:
        """Execute and compare multiple sequence strategies for the same research topic.
        
        Args:
            research_topic: The research topic to investigate
            strategies: List of strategies to compare (default: all three)
            
        Returns:
            SequenceComparison with detailed comparative analysis
        """
        if strategies is None:
            strategies = ["theory_first", "market_first", "future_back"]
        
        logger.info(f"Starting sequence comparison for '{research_topic}' with {len(strategies)} strategies")
        
        # Execute all sequences
        sequence_results = []
        for strategy in strategies:
            if strategy in SEQUENCE_PATTERNS:
                pattern = SEQUENCE_PATTERNS[strategy]
                result = await self.execute_sequence(pattern, research_topic)
                sequence_results.append(result)
            else:
                logger.warning(f"Strategy {strategy} not found in SEQUENCE_PATTERNS, skipping")
        
        if not sequence_results:
            raise ValueError("No valid sequence patterns found for comparison")
        
        # Calculate comparative metrics
        comparison = self.metrics_calculator.compare_sequence_results(sequence_results)
        
        # Store in comparison history
        self.comparison_history.append(comparison)
        
        logger.info(f"Sequence comparison completed. Productivity variance: {comparison.productivity_variance:.3f}")
        
        return comparison
    
    async def batch_sequence_analysis(
        self, 
        research_topics: List[str],
        strategies: Optional[List[str]] = None
    ) -> List[SequenceComparison]:
        """Run sequence comparison across multiple research topics.
        
        Args:
            research_topics: List of research topics to analyze
            strategies: List of strategies to compare for each topic
            
        Returns:
            List of SequenceComparison results for each topic
        """
        logger.info(f"Starting batch analysis for {len(research_topics)} topics")
        
        comparisons = []
        for i, topic in enumerate(research_topics, 1):
            logger.info(f"Analyzing topic {i}/{len(research_topics)}: {topic}")
            try:
                comparison = await self.compare_sequences(topic, strategies)
                comparisons.append(comparison)
            except Exception as e:
                logger.error(f"Error analyzing topic '{topic}': {e}")
        
        logger.info(f"Batch analysis completed: {len(comparisons)} comparisons")
        return comparisons
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of all sequence performance data."""
        if not self.execution_history:
            return {"message": "No sequence executions recorded"}
        
        # Aggregate performance by strategy/pattern
        strategy_performance = {}
        for result in self.execution_history:
            # Handle both SequencePattern and DynamicSequencePattern
            if hasattr(result.sequence_pattern, 'strategy') and result.sequence_pattern.strategy:
                strategy_key = result.sequence_pattern.strategy
            elif hasattr(result.sequence_pattern, 'sequence_id'):
                strategy_key = f"dynamic_{result.sequence_pattern.sequence_id[:8]}"
            else:
                strategy_key = "unknown"
            
            if strategy_key not in strategy_performance:
                strategy_performance[strategy_key] = []
            strategy_performance[strategy_key].append(result.overall_productivity_metrics.tool_productivity)
        
        # Calculate averages
        strategy_averages = {
            str(strategy): sum(scores) / len(scores)
            for strategy, scores in strategy_performance.items()
        }
        
        # Find variance
        all_scores = [score for scores in strategy_performance.values() for score in scores]
        variance = max(all_scores) - min(all_scores) if all_scores else 0
        
        return {
            "total_executions": len(self.execution_history),
            "strategies_tested": list(strategy_averages.keys()),
            "average_productivity_by_strategy": strategy_averages,
            "productivity_variance": variance,
            "significant_difference": variance > 0.2,
            "comparisons_conducted": len(self.comparison_history)
        }