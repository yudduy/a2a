"""
Orchestration Optimization Framework

This module implements systematic testing and comparison of different orchestration patterns
to identify the most optimal strategy for different types of research tasks.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

try:
    from ..utils.research_types import ResearchState
except ImportError:
    # For running as standalone module
    from utils.research_types import ResearchState

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


@dataclass
class ResearchResult:
    """Result from research execution."""
    synthesis: str
    papers: List[Dict[str, Any]]
    trace_id: Optional[str] = None
    total_tokens: Optional[int] = None
    total_cost: Optional[float] = None


class EvaluationCriteria(BaseModel):
    """Individual evaluation criterion with score and reasoning."""

    name: str = Field(description="Name of the evaluation criterion")
    score: float = Field(description="Score from 0-10 for this criterion")
    max_score: float = Field(default=10.0, description="Maximum possible score for this criterion")
    reasoning: str = Field(description="Detailed reasoning for the assigned score")
    strengths: List[str] = Field(description="Specific strengths identified in this area")
    weaknesses: List[str] = Field(description="Specific weaknesses identified in this area")
    evidence_examples: List[str] = Field(description="Specific examples from the report supporting the score")


class EnhancedQualityMetrics(BaseModel):
    """Enhanced quality metrics for research synthesis."""

    # Individual criteria scores (0-10 scale)
    completeness: float = Field(0.0, description="Coverage of research topic (0-10)")
    depth: float = Field(0.0, description="Depth of analysis (0-10)")
    coherence: float = Field(0.0, description="Logical flow and structure (0-10)")
    innovation: float = Field(0.0, description="Novel insights (0-10)")
    actionability: float = Field(0.0, description="Practical recommendations (0-10)")

    # Weighted overall score (0-100)
    overall_score: float = Field(0.0, description="Weighted overall score (0-100)")

    # Meta-evaluation
    confidence_level: float = Field(0.0, description="Confidence in evaluation (0-1)")
    evaluation_notes: Optional[str] = Field(None, description="Additional notes about evaluation")

    # Evaluation metadata
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    evaluation_model: str = Field(default="default", description="Model used for evaluation")

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate overall score if not provided
        if self.overall_score == 0.0 and any([self.completeness, self.depth, self.coherence, self.innovation, self.actionability]):
            self.overall_score = self._calculate_weighted_score()

    def _calculate_weighted_score(self) -> float:
        """Calculate weighted overall score from individual criteria."""
        weights = {
            'completeness': 0.25,
            'depth': 0.25,
            'coherence': 0.20,
            'innovation': 0.15,
            'actionability': 0.15
        }

        return (
            self.completeness * weights['completeness'] +
            self.depth * weights['depth'] +
            self.coherence * weights['coherence'] +
            self.innovation * weights['innovation'] +
            self.actionability * weights['actionability']
        )


class CostEfficiencyMetrics(BaseModel):
    """Cost efficiency metrics for orchestration strategies."""

    quality_per_dollar: float = Field(0.0, description="Quality score per dollar spent")
    quality_per_token: float = Field(0.0, description="Quality score per token used")
    tokens_per_quality_unit: float = Field(0.0, description="Tokens needed per quality unit")
    cost_per_quality_unit: float = Field(0.0, description="Cost needed per quality unit")
    total_tokens: int = Field(0, description="Total tokens consumed")
    total_cost: float = Field(0.0, description="Total cost incurred")


class ContentMetrics(BaseModel):
    """Content analysis metrics for research synthesis."""

    synthesis_length: int = Field(0, description="Character count of synthesis")
    coherence_score: float = Field(0.0, description="Text coherence (0-1)")
    readability_score: float = Field(0.0, description="Readability score (0-1)")
    completeness_score: float = Field(0.0, description="Content completeness (0-1)")
    insight_density: float = Field(0.0, description="Insights per 1000 characters")
    structure_score: float = Field(0.0, description="Overall structure quality (0-1)")

    def __init__(self, **data):
        super().__init__(**data)
        # Clamp values to valid ranges
        self.coherence_score = max(0.0, min(1.0, self.coherence_score))
        self.readability_score = max(0.0, min(1.0, self.readability_score))
        self.completeness_score = max(0.0, min(1.0, self.completeness_score))
        self.insight_density = max(0.0, min(1.0, self.insight_density))
        self.structure_score = max(0.0, min(1.0, self.structure_score))


class OrchestrationStrategy(Enum):
    """Different orchestration strategies to test."""

    THEORY_FIRST = "theory_first"
    MARKET_FIRST = "market_first"
    TECHNICAL_FIRST = "technical_first"
    PARALLEL_ALL = "parallel_all"
    ADAPTIVE = "adaptive"
    SEQUENTIAL_SINGLE = "sequential_single"


class QueryType(Enum):
    """Classification of research query types."""

    ACADEMIC_THEORETICAL = "academic_theoretical"
    BUSINESS_COMMERCIAL = "business_commercial"
    TECHNICAL_IMPLEMENTATION = "technical_implementation"
    MULTI_DOMAIN = "multi_domain"
    UNKNOWN = "unknown"


@dataclass
class StrategyMetrics:
    """Performance metrics for an orchestration strategy."""

    strategy_name: str
    query_type: str
    completion_time: float
    total_tokens: int
    total_cost: float
    quality_score: float
    consistency_score: float  # Lower variance = higher consistency
    success_rate: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced metrics
    enhanced_quality: Optional[EnhancedQualityMetrics] = None
    cost_efficiency: Optional[CostEfficiencyMetrics] = None
    content_metrics: Optional[ContentMetrics] = None


@dataclass
class OrchestrationExperiment:
    """Results from a single orchestration experiment."""

    query: str
    query_type: QueryType
    strategy: OrchestrationStrategy
    result: ResearchResult
    metrics: StrategyMetrics
    experiment_id: str
    timestamp: str


class QualityEvaluator:
    """Enhanced quality evaluator that integrates with existing LLM Judge system."""

    def __init__(self, evaluation_model: str = "claude-3-5-sonnet"):
        self.logger = logging.getLogger(__name__)
        self.evaluation_model = evaluation_model
        self._llm_judge = None

    async def initialize(self):
        """Initialize the LLM judge system."""
        try:
            # Import the existing LLM Judge system
            try:
                from ...src.open_deep_research.evaluation.llm_judge import LLMJudge
                from ...src.open_deep_research.evaluation.models import ReportEvaluation
                from ...src.open_deep_research.evaluation.prompts import EvaluationPrompts
                from langchain_core.runnables import RunnableConfig

                # Initialize with basic config
                config = RunnableConfig(configurable={
                    "evaluation_model": f"anthropic:{self.evaluation_model}",
                    "max_retries": 3
                })

                self._llm_judge = LLMJudge(config=config)
                self.logger.info(f"QualityEvaluator initialized with model: {self.evaluation_model}")

            except ImportError as e:
                self.logger.warning(f"Could not import full LLM Judge system: {e}. Using fallback evaluation.")
                self._llm_judge = None

        except Exception as e:
            self.logger.error(f"Failed to initialize QualityEvaluator: {e}")
            raise

    async def evaluate_research_quality(
        self,
        synthesis: Optional[str],
        query: str,
        total_tokens: int = 0,
        total_cost: float = 0.0
    ) -> Tuple[EnhancedQualityMetrics, CostEfficiencyMetrics, ContentMetrics]:
        """
        Evaluate research synthesis quality using enhanced metrics.

        Args:
            synthesis: Research synthesis text
            query: Original research query
            total_tokens: Total tokens consumed
            total_cost: Total cost incurred

        Returns:
            Tuple of (enhanced_quality, cost_efficiency, content_metrics)
        """
        # Basic content metrics (always available)
        content_metrics = self._calculate_content_metrics(synthesis)

        # Cost efficiency metrics
        cost_efficiency = self._calculate_cost_efficiency_metrics(
            synthesis, query, total_tokens, total_cost
        )

        # Enhanced quality metrics
        if self._llm_judge:
            enhanced_quality = await self._evaluate_with_llm_judge(synthesis, query)
        else:
            enhanced_quality = self._evaluate_with_fallback(synthesis, query)

        return enhanced_quality, cost_efficiency, content_metrics

    async def _evaluate_with_llm_judge(self, synthesis: str, query: str) -> EnhancedQualityMetrics:
        """Evaluate using the full LLM Judge system."""
        try:
            # Use the existing LLM Judge to evaluate the synthesis
            evaluation = await self._llm_judge.evaluate_single_report(
                report=synthesis,
                research_topic=query,
                sequence_name="orchestration_experiment"
            )

            if evaluation:
                # Convert ReportEvaluation to EnhancedQualityMetrics
                return EnhancedQualityMetrics(
                    completeness=evaluation.completeness.score,
                    depth=evaluation.depth.score,
                    coherence=evaluation.coherence.score,
                    innovation=evaluation.innovation.score,
                    actionability=evaluation.actionability.score,
                    overall_score=evaluation.overall_score,
                    confidence_level=evaluation.confidence_level,
                    evaluation_notes=f"Full LLM Judge evaluation: {evaluation.executive_summary}",
                    evaluation_model=self.evaluation_model
                )
            else:
                # Fallback if LLM Judge fails
                return self._evaluate_with_fallback(synthesis, query)

        except Exception as e:
            self.logger.error(f"LLM Judge evaluation failed: {e}")
            return self._evaluate_with_fallback(synthesis, query)

    def _evaluate_with_fallback(self, synthesis: Optional[str], query: str) -> EnhancedQualityMetrics:
        """Fallback evaluation using content analysis."""
        # Simple heuristic-based evaluation
        synthesis_text = synthesis or ""
        synthesis_length = len(synthesis_text)

        # Basic scoring based on content characteristics
        completeness = min(10.0, synthesis_length / 100)  # Longer = more complete
        depth = min(10.0, synthesis_length / 150)  # Depth based on length
        coherence = 7.0 if synthesis_length > 500 else 5.0  # Basic coherence score
        innovation = 6.0  # Default innovation score
        actionability = 6.0  # Default actionability score

        # Weighted overall score
        weights = {'completeness': 0.25, 'depth': 0.25, 'coherence': 0.20, 'innovation': 0.15, 'actionability': 0.15}
        overall_score = (
            completeness * weights['completeness'] +
            depth * weights['depth'] +
            coherence * weights['coherence'] +
            innovation * weights['innovation'] +
            actionability * weights['actionability']
        )

        return EnhancedQualityMetrics(
            completeness=completeness,
            depth=depth,
            coherence=coherence,
            innovation=innovation,
            actionability=actionability,
            overall_score=overall_score,
            confidence_level=0.5,  # Lower confidence for fallback
            evaluation_notes="Fallback evaluation - basic content analysis",
            evaluation_model="fallback"
        )

    def _calculate_content_metrics(self, synthesis: Optional[str]) -> ContentMetrics:
        """Calculate content analysis metrics."""
        synthesis_text = synthesis or ""
        synthesis_length = len(synthesis_text)

        # Basic content analysis
        coherence_score = self._calculate_coherence(synthesis_text)
        readability_score = self._calculate_readability(synthesis_text)
        completeness_score = min(1.0, synthesis_length / 2000)  # Normalize to 0-1
        insight_density = self._calculate_insight_density(synthesis_text)
        structure_score = self._calculate_structure(synthesis_text)

        return ContentMetrics(
            synthesis_length=synthesis_length,
            coherence_score=coherence_score,
            readability_score=readability_score,
            completeness_score=completeness_score,
            insight_density=insight_density,
            structure_score=structure_score
        )

    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score (simple heuristic)."""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5

        # Simple coherence measure based on sentence length consistency
        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        if not sentence_lengths:
            return 0.5

        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        coherence = max(0.0, min(1.0, 1.0 - (variance / (avg_length ** 2))))

        return coherence

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simple heuristic)."""
        sentences = text.split('.')
        words = text.split()

        if len(sentences) == 0 or len(words) == 0:
            return 0.5

        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Simple readability score (lower is more readable)
        complexity = (avg_sentence_length * avg_word_length) / 100
        readability = max(0.0, min(1.0, 1.0 - complexity))

        return readability

    def _calculate_insight_density(self, text: str) -> float:
        """Calculate insight density (insights per 1000 characters)."""
        # Simple heuristic: count potential insight markers
        insight_markers = [
            "however", "therefore", "significantly", "importantly", "notably",
            "furthermore", "consequently", "specifically", "particularly"
        ]

        marker_count = sum(text.lower().count(marker) for marker in insight_markers)
        density = (marker_count * 1000) / max(len(text), 1)

        return min(1.0, density / 10)  # Cap at 1.0

    def _calculate_structure(self, text: str) -> float:
        """Calculate structure quality score."""
        # Look for structural elements
        sections = text.count("## ") + text.count("### ") + text.count("#### ")
        paragraphs = text.count("\n\n")

        if sections > 0 or paragraphs > 3:
            return 0.8  # Well structured
        else:
            return 0.5  # Basic structure

    def _calculate_cost_efficiency_metrics(
        self,
        synthesis: str,
        query: str,
        total_tokens: int,
        total_cost: float
    ) -> CostEfficiencyMetrics:
        """Calculate cost efficiency metrics."""
        if total_cost == 0:
            return CostEfficiencyMetrics(
                quality_per_dollar=0.0,
                quality_per_token=0.0,
                tokens_per_quality_unit=0.0,
                cost_per_quality_unit=0.0,
                total_tokens=total_tokens,
                total_cost=total_cost
            )

        # For now, use basic quality score (will be enhanced later)
        basic_quality = len(synthesis) / 1000  # Simple quality proxy
        quality_per_dollar = basic_quality / total_cost if total_cost > 0 else 0.0
        quality_per_token = basic_quality / total_tokens if total_tokens > 0 else 0.0
        tokens_per_quality_unit = total_tokens / basic_quality if basic_quality > 0 else 0.0
        cost_per_quality_unit = total_cost / basic_quality if basic_quality > 0 else 0.0

        return CostEfficiencyMetrics(
            quality_per_dollar=quality_per_dollar,
            quality_per_token=quality_per_token,
            tokens_per_quality_unit=tokens_per_quality_unit,
            cost_per_quality_unit=cost_per_quality_unit,
            total_tokens=total_tokens,
            total_cost=total_cost
        )


class OrchestrationOptimizer:
    """
    Systematic testing and optimization of orchestration patterns.

    This class implements A/B testing of different orchestration strategies
    to identify which patterns work best for different types of queries.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.experiments: List[OrchestrationExperiment] = []
        self.strategy_performance: Dict[str, List[StrategyMetrics]] = {}
        self.quality_evaluator: Optional[QualityEvaluator] = None

        # Strategy selection weights (updated based on performance)
        self.strategy_weights = {
            OrchestrationStrategy.THEORY_FIRST: 1.0,
            OrchestrationStrategy.MARKET_FIRST: 1.0,
            OrchestrationStrategy.TECHNICAL_FIRST: 1.0,
            OrchestrationStrategy.PARALLEL_ALL: 1.0,
            OrchestrationStrategy.ADAPTIVE: 1.0,
            OrchestrationStrategy.SEQUENTIAL_SINGLE: 1.0,
        }

        # Initialize quality evaluator
        self._initialize_quality_evaluator()

    def _initialize_quality_evaluator(self):
        """Initialize the quality evaluator."""
        try:
            self.quality_evaluator = QualityEvaluator()
            # Note: We'll initialize it properly when running experiments
            self.logger.info("QualityEvaluator initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize QualityEvaluator: {e}. Using basic metrics.")
            self.quality_evaluator = None

    def classify_query(self, query: str) -> QueryType:
        """
        Classify the type of research query.

        Args:
            query: Research query string

        Returns:
            QueryType classification
        """
        query_lower = query.lower()

        # Academic/theoretical indicators
        academic_keywords = [
            "theory", "literature", "academic", "research", "study",
            "analysis", "framework", "methodology", "philosophy",
            "concept", "principle", "hypothesis"
        ]

        # Business/commercial indicators
        business_keywords = [
            "market", "business", "commercial", "revenue", "profit",
            "sales", "strategy", "competition", "brand", "customer",
            "pricing", "ROI", "investment"
        ]

        # Technical/implementation indicators
        technical_keywords = [
            "implementation", "technical", "architecture", "code",
            "system", "algorithm", "engineering", "development",
            "infrastructure", "performance", "scalability"
        ]

        academic_score = sum(1 for word in academic_keywords if word in query_lower)
        business_score = sum(1 for word in business_keywords if word in query_lower)
        technical_score = sum(1 for word in technical_keywords if word in query_lower)

        if academic_score >= business_score and academic_score >= technical_score:
            return QueryType.ACADEMIC_THEORETICAL
        elif business_score >= academic_score and business_score >= technical_score:
            return QueryType.BUSINESS_COMMERCIAL
        elif technical_score >= academic_score and technical_score >= business_score:
            return QueryType.TECHNICAL_IMPLEMENTATION
        else:
            return QueryType.MULTI_DOMAIN

    def get_optimal_strategy(self, query_type: QueryType) -> OrchestrationStrategy:
        """
        Select the optimal strategy for a given query type based on historical performance.

        Args:
            query_type: Type of research query

        Returns:
            Recommended orchestration strategy
        """
        # Simple rule-based selection (can be enhanced with ML)
        strategy_mapping = {
            QueryType.ACADEMIC_THEORETICAL: OrchestrationStrategy.THEORY_FIRST,
            QueryType.BUSINESS_COMMERCIAL: OrchestrationStrategy.MARKET_FIRST,
            QueryType.TECHNICAL_IMPLEMENTATION: OrchestrationStrategy.TECHNICAL_FIRST,
            QueryType.MULTI_DOMAIN: OrchestrationStrategy.PARALLEL_ALL,
            QueryType.UNKNOWN: OrchestrationStrategy.ADAPTIVE,
        }

        return strategy_mapping.get(query_type, OrchestrationStrategy.ADAPTIVE)

    def get_strategy_weights(self) -> Dict[OrchestrationStrategy, float]:
        """Get current strategy selection weights."""
        return self.strategy_weights.copy()

    def select_strategy_for_query(self, query: str) -> Tuple[OrchestrationStrategy, QueryType]:
        """
        Select orchestration strategy for a given query.

        Args:
            query: Research query

        Returns:
            Tuple of (selected_strategy, query_type)
        """
        query_type = self.classify_query(query)
        optimal_strategy = self.get_optimal_strategy(query_type)

        # For exploration, occasionally try other strategies
        if random.random() < 0.2:  # 20% exploration rate
            available_strategies = list(OrchestrationStrategy)
            optimal_strategy = random.choice(available_strategies)

        return optimal_strategy, query_type

    async def run_experiment(
        self,
        query: str,
        orchestrator: Any,
        strategy: OrchestrationStrategy
    ) -> OrchestrationExperiment:
        """
        Run a single orchestration experiment.

        Args:
            query: Research query
            orchestrator: Orchestration engine
            strategy: Strategy to test

        Returns:
            Experiment results
        """
        start_time = time.time()

        try:
            # Execute research with specified strategy
            result = await orchestrator.execute_research_with_strategy(query, strategy)

            # Calculate metrics
            completion_time = time.time() - start_time

            # Initialize quality evaluator if not already done
            if self.quality_evaluator and not hasattr(self.quality_evaluator, '_llm_judge'):
                try:
                    await self.quality_evaluator.initialize()
                except Exception as e:
                    self.logger.warning(f"Failed to initialize quality evaluator: {e}")

            # Enhanced quality evaluation
            synthesis_text = result.synthesis or ""
            total_tokens = getattr(result, 'total_tokens', 0)
            total_cost = getattr(result, 'total_cost', 0.0)

            enhanced_quality = None
            cost_efficiency = None
            content_metrics = None

            if self.quality_evaluator:
                try:
                    enhanced_quality, cost_efficiency, content_metrics = await self.quality_evaluator.evaluate_research_quality(
                        synthesis=synthesis_text,
                        query=query,
                        total_tokens=total_tokens,
                        total_cost=total_cost
                    )
                except Exception as e:
                    self.logger.warning(f"Enhanced quality evaluation failed: {e}")

            # Calculate basic quality score (backward compatibility)
            basic_quality_score = 0.5  # Default fallback
            if enhanced_quality:
                basic_quality_score = enhanced_quality.overall_score / 100.0  # Convert from 0-100 to 0-1 scale

            # Extract metrics from result
            metrics = StrategyMetrics(
                strategy_name=strategy.value,
                query_type=str(self.classify_query(query)),
                completion_time=completion_time,
                total_tokens=total_tokens,
                total_cost=total_cost,
                quality_score=basic_quality_score,
                consistency_score=1.0,  # TODO: Calculate from multiple runs
                success_rate=1.0 if result.synthesis else 0.0,
                timestamp=datetime.utcnow().isoformat(),
                metadata={
                    "query_length": len(query),
                    "synthesis_length": len(synthesis_text),
                    "papers_found": len(result.papers or [])
                },
                enhanced_quality=enhanced_quality,
                cost_efficiency=cost_efficiency,
                content_metrics=content_metrics
            )

            experiment = OrchestrationExperiment(
                query=query,
                query_type=self.classify_query(query),
                strategy=strategy,
                result=result,
                metrics=metrics,
                experiment_id=f"exp_{int(time.time())}_{strategy.value}",
                timestamp=datetime.utcnow().isoformat()
            )

            # Store experiment
            self.experiments.append(experiment)

            # Update strategy performance
            if strategy.value not in self.strategy_performance:
                self.strategy_performance[strategy.value] = []
            self.strategy_performance[strategy.value].append(metrics)

            return experiment

        except Exception as e:
            self.logger.error(f"Experiment failed for strategy {strategy}: {e}")
            raise

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze performance across all experiments.

        Returns:
            Performance analysis results
        """
        if not self.experiments:
            return {"error": "No experiments completed"}

        # Convert to DataFrame for analysis
        data = []
        for exp in self.experiments:
            data.append({
                "strategy": exp.strategy.value,
                "query_type": exp.query_type.value,
                "completion_time": exp.metrics.completion_time,
                "total_tokens": exp.metrics.total_tokens,
                "total_cost": exp.metrics.total_cost,
                "quality_score": exp.metrics.quality_score,
                "success_rate": exp.metrics.success_rate,
                "synthesis_length": len(exp.result.synthesis or ""),
                "papers_found": len(exp.result.papers or [])
            })

        df = pd.DataFrame(data)

        # Overall statistics
        overall_stats = {
            "total_experiments": len(self.experiments),
            "strategies_tested": len(df["strategy"].unique()),
            "query_types_tested": len(df["query_type"].unique()),
            "average_completion_time": df["completion_time"].mean(),
            "average_quality_score": df["quality_score"].mean(),
            "average_success_rate": df["success_rate"].mean(),
        }

        # Strategy comparison
        strategy_comparison = {}
        for strategy in df["strategy"].unique():
            strategy_data = df[df["strategy"] == strategy]
            strategy_comparison[strategy] = {
                "experiment_count": len(strategy_data),
                "avg_completion_time": strategy_data["completion_time"].mean(),
                "avg_quality_score": strategy_data["quality_score"].mean(),
                "avg_success_rate": strategy_data["success_rate"].mean(),
                "std_completion_time": strategy_data["completion_time"].std(),
                "confidence_interval": self._calculate_confidence_interval(
                    strategy_data["quality_score"].tolist()
                )
            }

        # Query type analysis
        query_type_analysis = {}
        for query_type in df["query_type"].unique():
            type_data = df[df["query_type"] == query_type]
            query_type_analysis[query_type] = {
                "experiment_count": len(type_data),
                "best_strategy": type_data.loc[type_data["quality_score"].idxmax()]["strategy"],
                "avg_quality_score": type_data["quality_score"].mean(),
            }

        # Statistical significance testing
        significance_tests = self._perform_significance_tests(df)

        return {
            "overall_stats": overall_stats,
            "strategy_comparison": strategy_comparison,
            "query_type_analysis": query_type_analysis,
            "significance_tests": significance_tests,
            "recommendations": self._generate_recommendations(strategy_comparison, query_type_analysis)
        }

    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values."""
        if len(data) < 2:
            return (0.0, 0.0)

        mean = np.mean(data)
        std = np.std(data)
        n = len(data)
        z_score = stats.norm.ppf((1 + confidence) / 2)

        margin = z_score * (std / np.sqrt(n))
        return (mean - margin, mean + margin)

    def _perform_significance_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests between strategies."""
        strategies = df["strategy"].unique()
        results = {}

        for i, strategy1 in enumerate(strategies):
            for strategy2 in strategies[i+1:]:
                data1 = df[df["strategy"] == strategy1]["quality_score"]
                data2 = df[df["strategy"] == strategy2]["quality_score"]

                if len(data1) >= 2 and len(data2) >= 2:
                    t_stat, p_value = stats.ttest_ind(data1, data2)

                    results[f"{strategy1}_vs_{strategy2}"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "better_strategy": strategy1 if data1.mean() > data2.mean() else strategy2
                    }

        return results

    def _generate_recommendations(
        self,
        strategy_comparison: Dict,
        query_type_analysis: Dict
    ) -> List[str]:
        """Generate recommendations based on performance analysis."""
        recommendations = []

        # Find best overall strategy
        best_strategy = max(
            strategy_comparison.items(),
            key=lambda x: x[1]["avg_quality_score"]
        )

        recommendations.append(
            f"🎯 **Overall Best Strategy**: {best_strategy[0]} "
            f"(avg quality: {best_strategy[1]['avg_quality_score']:.3f})"
        )

        # Query-type specific recommendations
        for query_type, analysis in query_type_analysis.items():
            if analysis["experiment_count"] > 0:
                recommendations.append(
                    f"📚 **{query_type.title()} Queries**: Use {analysis['best_strategy']} "
                    f"(avg quality: {analysis['avg_quality_score']:.3f})"
                )

        # Performance insights
        fast_strategies = [
            name for name, metrics in strategy_comparison.items()
            if metrics["avg_completion_time"] < 60  # Less than 1 minute
        ]

        if fast_strategies:
            recommendations.append(
                f"⚡ **Fast Strategies** (for time-sensitive queries): {', '.join(fast_strategies)}"
            )

        return recommendations

    def export_results(self, filename: str):
        """Export experiment results to JSON file."""
        export_data = {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "query": exp.query,
                    "query_type": exp.query_type.value,
                    "strategy": exp.strategy.value,
                    "metrics": {
                        "completion_time": exp.metrics.completion_time,
                        "total_tokens": exp.metrics.total_tokens,
                        "total_cost": exp.metrics.total_cost,
                        "quality_score": exp.metrics.quality_score,
                        "success_rate": exp.metrics.success_rate,
                        "synthesis_length": len(exp.result.synthesis or ""),
                        "papers_found": len(exp.result.papers or [])
                    },
                    "timestamp": exp.timestamp
                }
                for exp in self.experiments
            ]
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Exported {len(self.experiments)} experiments to {filename}")

    def get_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        if not self.experiments:
            return "No experiments completed yet."

        analysis = self.analyze_performance()

        report = f"""
# Orchestration Optimization Report

## 📊 Overall Statistics
- **Total Experiments**: {analysis['overall_stats']['total_experiments']}
- **Strategies Tested**: {analysis['overall_stats']['strategies_tested']}
- **Query Types Tested**: {analysis['overall_stats']['query_types_tested']}
- **Average Completion Time**: {analysis['overall_stats']['average_completion_time']:.2f}s
- **Average Quality Score**: {analysis['overall_stats']['average_quality_score']:.3f}
- **Average Success Rate**: {analysis['overall_stats']['average_success_rate']:.3f}

## 🏆 Strategy Performance Comparison

"""

        for strategy, metrics in analysis['strategy_comparison'].items():
            report += f"""
### {strategy.replace('_', ' ').title()}
- **Experiments**: {metrics['experiment_count']}
- **Avg Quality Score**: {metrics['avg_quality_score']:.3f} ± {metrics.get('confidence_interval', (0, 0))[1] - metrics.get('confidence_interval', (0, 0))[0]:.3f}
- **Avg Completion Time**: {metrics['avg_completion_time']:.2f}s
- **Success Rate**: {metrics['avg_success_rate']:.3f}

"""

        report += """
## 🎯 Recommendations

"""
        for rec in analysis['recommendations']:
            report += f"- {rec}\n"

        return report
