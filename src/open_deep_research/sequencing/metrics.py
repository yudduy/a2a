"""Comprehensive metrics system for tool productivity tracking and sequence comparison.

This module implements the core Tool Productivity metrics (Quality/Agent_Calls) and
provides detailed analysis capabilities for comparing sequence effectiveness and
identifying >20% variance in productivity outcomes.
"""

import logging
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from open_deep_research.sequencing.models import (
    AgentExecutionResult,
    InsightTransition,
    SequenceComparison,
    SequenceResult,
    SequenceStrategy,
    ToolProductivityMetrics
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for tool productivity and sequence comparison metrics."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.historical_baselines = {}
        self.variance_threshold = 0.2  # 20% variance threshold
    
    def calculate_sequence_productivity(
        self,
        agent_results: List[AgentExecutionResult],
        insight_transitions: List[InsightTransition],
        total_duration: float
    ) -> ToolProductivityMetrics:
        """Calculate comprehensive tool productivity metrics for a sequence.
        
        Args:
            agent_results: Results from all agents in the sequence
            insight_transitions: Transitions between agents
            total_duration: Total sequence execution time
            
        Returns:
            ToolProductivityMetrics with comprehensive analysis
        """
        if not agent_results:
            return self._create_empty_metrics()
        
        # Calculate core Tool Productivity metric
        research_quality = self._calculate_research_quality(agent_results)
        total_agent_calls = sum(r.tool_calls_made for r in agent_results)
        
        if total_agent_calls == 0:
            tool_productivity = 0.0
        else:
            tool_productivity = research_quality / total_agent_calls
        
        # Calculate efficiency metrics
        agent_efficiency = self._calculate_agent_efficiency(agent_results)
        context_efficiency = self._calculate_context_efficiency(agent_results, insight_transitions)
        time_to_value = self._calculate_time_to_value(agent_results)
        
        # Calculate quality breakdown
        quality_breakdown = self._calculate_quality_breakdown(agent_results)
        
        # Calculate agent performance metrics
        performance_metrics = self._calculate_agent_performance(agent_results)
        
        # Calculate context usage
        context_metrics = self._calculate_context_usage(agent_results, insight_transitions)
        
        return ToolProductivityMetrics(
            tool_productivity=tool_productivity,
            research_quality_score=research_quality,
            total_agent_calls=total_agent_calls,
            agent_efficiency=agent_efficiency,
            context_efficiency=context_efficiency,
            time_to_value=time_to_value,
            insight_novelty=quality_breakdown['novelty'],
            insight_relevance=quality_breakdown['relevance'], 
            insight_actionability=quality_breakdown['actionability'],
            research_completeness=quality_breakdown['completeness'],
            useful_insights_count=performance_metrics['useful_insights'],
            redundant_research_count=performance_metrics['redundant_research'],
            cognitive_offloading_incidents=performance_metrics['cognitive_offloading'],
            relevant_context_used=context_metrics['relevant_used'],
            total_context_available=context_metrics['total_available']
        )
    
    def _create_empty_metrics(self) -> ToolProductivityMetrics:
        """Create empty metrics for error cases."""
        return ToolProductivityMetrics(
            tool_productivity=0.0,
            research_quality_score=0.0,
            total_agent_calls=0,
            agent_efficiency=0.0,
            context_efficiency=0.0,
            time_to_value=0.0,
            insight_novelty=0.0,
            insight_relevance=0.0,
            insight_actionability=0.0,
            research_completeness=0.0,
            useful_insights_count=0,
            redundant_research_count=0,
            cognitive_offloading_incidents=0,
            relevant_context_used=0.0,
            total_context_available=0.0
        )
    
    def _calculate_research_quality(self, agent_results: List[AgentExecutionResult]) -> float:
        """Calculate overall research quality score."""
        quality_factors = []
        
        for result in agent_results:
            # Individual agent quality components
            depth_score = result.research_depth_score
            novelty_score = result.novelty_score
            reasoning_score = result.independent_reasoning_score
            
            # Insight quality
            if result.insight_quality_scores:
                avg_insight_quality = sum(result.insight_quality_scores) / len(result.insight_quality_scores)
            else:
                avg_insight_quality = 0.0
            
            # Agent quality score (weighted average)
            agent_quality = (
                depth_score * 0.3 +
                novelty_score * 0.25 +
                reasoning_score * 0.25 +
                avg_insight_quality * 0.2
            )
            
            quality_factors.append(agent_quality)
        
        # Overall quality is average across agents
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
    
    def _calculate_agent_efficiency(self, agent_results: List[AgentExecutionResult]) -> float:
        """Calculate agent efficiency as useful insights per agent call."""
        total_useful_insights = 0
        total_agent_calls = 0
        
        for result in agent_results:
            # Count useful insights (quality score > 0.5)
            useful_insights = sum(1 for score in result.insight_quality_scores if score > 0.5)
            total_useful_insights += useful_insights
            total_agent_calls += result.tool_calls_made
        
        if total_agent_calls == 0:
            return 0.0
        
        return total_useful_insights / total_agent_calls
    
    def _calculate_context_efficiency(
        self, 
        agent_results: List[AgentExecutionResult],
        insight_transitions: List[InsightTransition]
    ) -> float:
        """Calculate how efficiently context is used across agent transitions."""
        if not insight_transitions:
            return 1.0  # Perfect efficiency for single agent
        
        efficiency_scores = []
        
        for transition in insight_transitions:
            # Calculate if the transition was productive
            productivity_score = transition.productivity_score
            efficiency_scores.append(productivity_score)
        
        return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0
    
    def _calculate_time_to_value(self, agent_results: List[AgentExecutionResult]) -> float:
        """Calculate time to first significant insight (in seconds)."""
        for result in agent_results:
            if result.key_insights and any(score > 0.7 for score in result.insight_quality_scores):
                return result.execution_duration
        
        # If no high-quality insights, return total time
        return sum(r.execution_duration for r in agent_results)
    
    def _calculate_quality_breakdown(self, agent_results: List[AgentExecutionResult]) -> Dict[str, float]:
        """Calculate detailed quality breakdown metrics."""
        novelty_scores = []
        relevance_scores = []
        actionability_scores = []
        completeness_scores = []
        
        for result in agent_results:
            # Novelty from agent novelty score
            novelty_scores.append(result.novelty_score)
            
            # Relevance based on insight quality and research depth
            relevance = (result.research_depth_score + 
                        (sum(result.insight_quality_scores) / len(result.insight_quality_scores) 
                         if result.insight_quality_scores else 0)) / 2
            relevance_scores.append(relevance)
            
            # Actionability based on insight content analysis
            actionability = self._analyze_actionability(result.key_insights)
            actionability_scores.append(actionability)
            
            # Completeness based on research findings depth
            completeness = min(len(result.research_findings) / 1000, 1.0)  # Normalize by 1000 chars
            completeness_scores.append(completeness)
        
        return {
            'novelty': sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0,
            'relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
            'actionability': sum(actionability_scores) / len(actionability_scores) if actionability_scores else 0.0,
            'completeness': sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        }
    
    def _analyze_actionability(self, insights: List[str]) -> float:
        """Analyze actionability of insights."""
        if not insights:
            return 0.0
        
        actionable_indicators = [
            'recommend', 'suggest', 'should', 'could', 'implement',
            'apply', 'strategy', 'approach', 'method', 'framework',
            'action', 'solution', 'opportunity', 'develop', 'create'
        ]
        
        actionable_count = 0
        for insight in insights:
            insight_lower = insight.lower()
            if any(indicator in insight_lower for indicator in actionable_indicators):
                actionable_count += 1
        
        return actionable_count / len(insights)
    
    def _calculate_agent_performance(self, agent_results: List[AgentExecutionResult]) -> Dict[str, int]:
        """Calculate agent performance metrics."""
        useful_insights = 0
        redundant_research = 0
        cognitive_offloading = 0
        
        for result in agent_results:
            # Count useful insights (quality > 0.6)
            useful_insights += sum(1 for score in result.insight_quality_scores if score > 0.6)
            
            # Estimate redundant research (high tool calls but low insights)
            if result.tool_calls_made > 5 and len(result.key_insights) < 3:
                redundant_research += 1
            
            # Count cognitive offloading incidents
            if result.cognitive_offloading_detected:
                cognitive_offloading += 1
        
        return {
            'useful_insights': useful_insights,
            'redundant_research': redundant_research,
            'cognitive_offloading': cognitive_offloading
        }
    
    def _calculate_context_usage(
        self, 
        agent_results: List[AgentExecutionResult],
        insight_transitions: List[InsightTransition]
    ) -> Dict[str, float]:
        """Calculate context usage metrics."""
        if not insight_transitions:
            return {'relevant_used': 1.0, 'total_available': 1.0}
        
        relevant_context_used = 0.0
        total_context_available = 0.0
        
        for transition in insight_transitions:
            # Estimate context utilization based on transition productivity
            relevant_context_used += transition.productivity_score
            total_context_available += 1.0
        
        return {
            'relevant_used': relevant_context_used / total_context_available if total_context_available > 0 else 0.0,
            'total_available': total_context_available
        }
    
    def compare_sequence_results(self, sequence_results: List[SequenceResult]) -> SequenceComparison:
        """Compare multiple sequence results and calculate variance metrics.
        
        Args:
            sequence_results: List of sequence results to compare
            
        Returns:
            SequenceComparison with detailed comparative analysis
        """
        if len(sequence_results) < 2:
            raise ValueError("Need at least 2 sequence results for comparison")
        
        research_topic = sequence_results[0].research_topic
        
        # Extract productivity metrics
        productivity_scores = {
            result.sequence_pattern.strategy: result.overall_productivity_metrics.tool_productivity
            for result in sequence_results
        }
        
        quality_scores = {
            result.sequence_pattern.strategy: result.overall_productivity_metrics.research_quality_score
            for result in sequence_results
        }
        
        efficiency_scores = {
            result.sequence_pattern.strategy: result.overall_productivity_metrics.agent_efficiency
            for result in sequence_results
        }
        
        # Calculate variance
        productivity_values = list(productivity_scores.values())
        productivity_variance = statistics.stdev(productivity_values) if len(productivity_values) > 1 else 0.0
        
        # Determine significance
        significant_difference = productivity_variance > self.variance_threshold
        
        # Find best performing sequence
        best_strategy = max(productivity_scores.keys(), key=lambda k: productivity_scores[k])
        best_score = productivity_scores[best_strategy]
        worst_score = min(productivity_scores.values())
        
        productivity_advantage = ((best_score - worst_score) / worst_score * 100) if worst_score > 0 else 0
        
        # Analyze unique insights
        insight_analysis = self._analyze_sequence_insights(sequence_results)
        
        # Calculate statistical measures
        statistical_measures = self._calculate_statistical_significance(productivity_values)
        
        comparison = SequenceComparison(
            research_topic=research_topic,
            compared_sequences=sequence_results,
            productivity_variance=productivity_variance,
            significant_difference_detected=significant_difference,
            highest_productivity_sequence=best_strategy,
            productivity_advantage=productivity_advantage,
            productivity_rankings=productivity_scores,
            quality_rankings=quality_scores,
            efficiency_rankings=efficiency_scores,
            unique_insights_by_sequence=insight_analysis['unique_by_sequence'],
            shared_insights_across_sequences=insight_analysis['shared_insights'],
            sequence_specific_advantages=insight_analysis['sequence_advantages'],
            statistical_significance=statistical_measures['significance'],
            confidence_level=statistical_measures['confidence']
        )
        
        logger.info(f"Sequence comparison: variance={productivity_variance:.3f}, "
                   f"significant={significant_difference}, best={best_strategy.value}")
        
        return comparison
    
    def _analyze_sequence_insights(self, sequence_results: List[SequenceResult]) -> Dict:
        """Analyze insights across different sequences."""
        all_insights_by_sequence = {}
        all_insights = set()
        
        # Collect insights by sequence
        for result in sequence_results:
            strategy = result.sequence_pattern.strategy
            insights = set(result.comprehensive_findings)
            all_insights_by_sequence[strategy] = insights
            all_insights.update(insights)
        
        # Find shared insights
        shared_insights = set.intersection(*all_insights_by_sequence.values()) if all_insights_by_sequence else set()
        
        # Find unique insights per sequence
        unique_by_sequence = {}
        for strategy, insights in all_insights_by_sequence.items():
            other_insights = set()
            for other_strategy, other_insight_set in all_insights_by_sequence.items():
                if other_strategy != strategy:
                    other_insights.update(other_insight_set)
            unique_by_sequence[strategy] = list(insights - other_insights)
        
        # Identify sequence-specific advantages
        sequence_advantages = {}
        for strategy, unique_insights in unique_by_sequence.items():
            advantages = []
            
            # Analyze unique insights for advantages
            for insight in unique_insights:
                if any(word in insight.lower() for word in ['advantage', 'benefit', 'opportunity', 'strength']):
                    advantages.append(insight)
            
            # Add meta-advantages based on sequence characteristics
            if strategy == SequenceStrategy.THEORY_FIRST:
                advantages.append("Strong theoretical foundation enables rigorous analysis")
            elif strategy == SequenceStrategy.MARKET_FIRST:
                advantages.append("Market-driven focus ensures commercial relevance")
            elif strategy == SequenceStrategy.FUTURE_BACK:
                advantages.append("Future-oriented perspective identifies emerging opportunities")
            
            sequence_advantages[strategy] = advantages
        
        return {
            'unique_by_sequence': unique_by_sequence,
            'shared_insights': list(shared_insights),
            'sequence_advantages': sequence_advantages
        }
    
    def _calculate_statistical_significance(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical significance of variance."""
        if len(values) < 2:
            return {'significance': 0.0, 'confidence': 0.0}
        
        # Simple statistical measures
        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Calculate coefficient of variation as a measure of significance
        if mean_val > 0:
            coefficient_of_variation = std_dev / mean_val
            # Convert to significance score (higher CV = higher significance)
            significance = min(coefficient_of_variation * 2, 1.0)
        else:
            significance = 0.0
        
        # Estimate confidence based on variance
        confidence = max(0.5, 1.0 - (std_dev / max(values)) if max(values) > 0 else 0.5)
        
        return {
            'significance': significance,
            'confidence': confidence
        }
    
    def generate_productivity_report(self, comparison: SequenceComparison) -> str:
        """Generate a comprehensive productivity analysis report."""
        report_sections = [
            f"# Tool Productivity Analysis Report",
            f"## Research Topic: {comparison.research_topic}",
            f"## Analysis Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"This analysis compared {len(comparison.compared_sequences)} different agent sequence strategies "
            f"to measure productivity variance in research outcomes. The analysis {'**DID**' if comparison.significant_difference_detected else '**DID NOT**'} "
            f"detect significant differences (>20% variance) in tool productivity metrics.",
            "",
            f"**Key Finding:** {comparison.productivity_variance:.1%} variance detected between sequences",
            f"**Best Performing Sequence:** {comparison.highest_productivity_sequence.value}",
            f"**Productivity Advantage:** {comparison.productivity_advantage:.1f}%",
            "",
            "## Productivity Rankings",
            ""
        ]
        
        # Add productivity rankings
        sorted_productivity = sorted(
            comparison.productivity_rankings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (strategy, score) in enumerate(sorted_productivity, 1):
            report_sections.append(f"{i}. **{strategy.value}**: {score:.3f} (Quality/Agent_Calls)")
        
        report_sections.extend([
            "",
            "## Quality Analysis",
            ""
        ])
        
        # Add quality rankings
        sorted_quality = sorted(
            comparison.quality_rankings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (strategy, score) in enumerate(sorted_quality, 1):
            report_sections.append(f"{i}. **{strategy.value}**: {score:.3f} (Research Quality)")
        
        # Add unique insights analysis
        report_sections.extend([
            "",
            "## Sequence-Specific Insights",
            ""
        ])
        
        for strategy, insights in comparison.unique_insights_by_sequence.items():
            report_sections.extend([
                f"### {strategy.value} Unique Insights ({len(insights)} insights)",
                ""
            ])
            
            for insight in insights[:5]:  # Show top 5
                report_sections.append(f"- {insight}")
            
            if len(insights) > 5:
                report_sections.append(f"- ... and {len(insights) - 5} more insights")
            
            report_sections.append("")
        
        # Add statistical analysis
        report_sections.extend([
            "## Statistical Analysis",
            f"**Variance:** {comparison.productivity_variance:.3f}",
            f"**Statistical Significance:** {comparison.statistical_significance:.3f}",
            f"**Confidence Level:** {comparison.confidence_level:.3f}",
            "",
            "## Conclusion",
            f"The analysis {'provides strong evidence' if comparison.significant_difference_detected else 'suggests limited evidence'} "
            f"that different sequential agent orderings produce measurably different productivity outcomes. "
            f"The {comparison.highest_productivity_sequence.value} sequence demonstrated optimal performance "
            f"for this research topic with a {comparison.productivity_advantage:.1f}% productivity advantage."
        ])
        
        return "\n".join(report_sections)
    
    def update_historical_baselines(self, sequence_result: SequenceResult):
        """Update historical performance baselines for future comparisons."""
        strategy = sequence_result.sequence_pattern.strategy
        productivity = sequence_result.overall_productivity_metrics.tool_productivity
        
        if strategy not in self.historical_baselines:
            self.historical_baselines[strategy] = []
        
        self.historical_baselines[strategy].append(productivity)
        
        # Keep only recent results (last 20)
        if len(self.historical_baselines[strategy]) > 20:
            self.historical_baselines[strategy] = self.historical_baselines[strategy][-20:]
    
    def get_performance_trends(self) -> Dict[SequenceStrategy, Dict[str, float]]:
        """Get performance trends for each sequence strategy."""
        trends = {}
        
        for strategy, scores in self.historical_baselines.items():
            if len(scores) >= 2:
                trends[strategy] = {
                    'average': statistics.mean(scores),
                    'std_dev': statistics.stdev(scores),
                    'trend': 'improving' if scores[-1] > scores[0] else 'declining',
                    'sample_size': len(scores)
                }
        
        return trends