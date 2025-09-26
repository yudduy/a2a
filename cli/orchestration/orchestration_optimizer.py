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

    def __init__(self, orchestrator: Optional[Any] = None):
        self.logger = logging.getLogger(__name__)
        self.experiments: List[OrchestrationExperiment] = []
        self.strategy_performance: Dict[str, List[StrategyMetrics]] = {}
        self.quality_evaluator: Optional[QualityEvaluator] = None
        self.significance_tests: Dict[str, Any] = {}
        self.cost_benefit_analysis: Dict[str, Any] = {}
        self.orchestrator = orchestrator
        self.cache = {}  # Simple cache for research results
        self.cache_hits = 0
        self.cache_misses = 0

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

    def get_cache_key(self, query: str, strategy: OrchestrationStrategy) -> str:
        """Generate a cache key for a query and strategy combination."""
        return f"{strategy.value}_{hash(query.lower().strip())}"

    def check_cache(self, query: str, strategy: OrchestrationStrategy) -> Optional[ResearchResult]:
        """Check if result is cached for the given query and strategy."""
        cache_key = self.get_cache_key(query, strategy)
        if cache_key in self.cache:
            self.cache_hits += 1
            self.logger.info(f"Cache hit for {strategy.value}: {query[:50]}...")
            return self.cache[cache_key]
        self.cache_misses += 1
        return None

    def cache_result(self, query: str, strategy: OrchestrationStrategy, result: ResearchResult):
        """Cache a research result."""
        cache_key = self.get_cache_key(query, strategy)
        self.cache[cache_key] = result
        self.logger.info(f"Cached result for {strategy.value}: {query[:50]}...")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

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
            # Check cache first
            cached_result = self.check_cache(query, strategy)
            if cached_result:
                # Use cached result
                result = cached_result
                completion_time = 0.1  # Minimal time for cached results
            else:
                # Execute research with specified strategy
                result = await orchestrator.execute_research_with_strategy(query, strategy)
                # Cache the result
                self.cache_result(query, strategy, result)
                completion_time = time.time() - start_time

            # Initialize quality evaluator if not already done
            if self.quality_evaluator and not hasattr(self.quality_evaluator, '_llm_judge'):
                try:
                    await self.quality_evaluator.initialize()
                except Exception as e:
                    self.logger.warning(f"Failed to initialize quality evaluator: {e}")

            # Enhanced quality evaluation
            synthesis_text = getattr(result, 'synthesis', "") or ""
            total_tokens = getattr(result, 'total_tokens', 0)
            total_cost = getattr(result, 'total_cost', 0.0)

            # Debug logging
            self.logger.info(f"Quality evaluation - synthesis: {len(synthesis_text)} chars, tokens: {total_tokens}, cost: {total_cost}")

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
                success_rate=1.0 if synthesis_text else 0.0,
                timestamp=datetime.utcnow().isoformat(),
                metadata={
                    "query_length": len(query),
                    "synthesis_length": len(synthesis_text),
                    "papers_found": len(getattr(result, 'papers', []) or [])
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
        Analyze performance across all experiments with statistical rigor.

        Returns:
            Performance analysis results with confidence intervals and significance testing
        """
        if not self.experiments:
            return {"error": "No experiments completed"}

        # Convert to DataFrame for analysis
        data = []
        for exp in self.experiments:
            # Extract enhanced quality metrics if available
            enhanced_quality = None
            cost_efficiency = None
            content_metrics = None

            if exp.metrics.enhanced_quality:
                enhanced_quality = {
                    "completeness": exp.metrics.enhanced_quality.completeness,
                    "depth": exp.metrics.enhanced_quality.depth,
                    "coherence": exp.metrics.enhanced_quality.coherence,
                    "innovation": exp.metrics.enhanced_quality.innovation,
                    "actionability": exp.metrics.enhanced_quality.actionability,
                    "overall_score": exp.metrics.enhanced_quality.overall_score
                }

            if exp.metrics.cost_efficiency:
                cost_efficiency = {
                    "quality_per_dollar": exp.metrics.cost_efficiency.quality_per_dollar,
                    "quality_per_token": exp.metrics.cost_efficiency.quality_per_token,
                    "tokens_per_quality_unit": exp.metrics.cost_efficiency.tokens_per_quality_unit,
                    "cost_per_quality_unit": exp.metrics.cost_efficiency.cost_per_quality_unit
                }

            if exp.metrics.content_metrics:
                content_metrics = {
                    "synthesis_length": exp.metrics.content_metrics.synthesis_length,
                    "coherence_score": exp.metrics.content_metrics.coherence_score,
                    "readability_score": exp.metrics.content_metrics.readability_score,
                    "completeness_score": exp.metrics.content_metrics.completeness_score,
                    "insight_density": exp.metrics.content_metrics.insight_density,
                    "structure_score": exp.metrics.content_metrics.structure_score
                }

            data.append({
                "strategy": exp.strategy.value,
                "query_type": exp.query_type.value,
                "completion_time": exp.metrics.completion_time,
                "total_tokens": exp.metrics.total_tokens,
                "total_cost": exp.metrics.total_cost,
                "quality_score": exp.metrics.quality_score,
                "success_rate": exp.metrics.success_rate,
                "synthesis_length": len(exp.result.synthesis or ""),
                "papers_found": len(exp.result.papers or []),
                "enhanced_quality": enhanced_quality,
                "cost_efficiency": cost_efficiency,
                "content_metrics": content_metrics
            })

        df = pd.DataFrame(data)

        # Overall statistics with confidence intervals
        overall_stats = {
            "total_experiments": len(self.experiments),
            "strategies_tested": len(df["strategy"].unique()),
            "query_types_tested": len(df["query_type"].unique()),
            "average_completion_time": df["completion_time"].mean(),
            "average_quality_score": df["quality_score"].mean(),
            "average_success_rate": df["success_rate"].mean(),
        }

        # Add confidence intervals for key metrics
        confidence_level = 0.95
        for metric in ["completion_time", "quality_score", "total_cost"]:
            if metric in df.columns and len(df[metric].dropna()) > 1:
                ci = self._calculate_confidence_interval(df[metric].dropna().tolist(), confidence_level)
                overall_stats[f"{metric}_confidence_interval"] = ci
                overall_stats[f"{metric}_margin_of_error"] = (ci[1] - ci[0]) / 2

        # Strategy comparison with statistical rigor
        strategy_comparison = {}
        for strategy in df["strategy"].unique():
            strategy_data = df[df["strategy"] == strategy]

            # Calculate confidence intervals for each metric
            quality_scores = strategy_data["quality_score"].dropna().tolist()
            completion_times = strategy_data["completion_time"].dropna().tolist()
            costs = strategy_data["total_cost"].dropna().tolist()

            strategy_comparison[strategy] = {
                "experiment_count": len(strategy_data),
                "avg_completion_time": strategy_data["completion_time"].mean(),
                "avg_quality_score": strategy_data["quality_score"].mean(),
                "avg_success_rate": strategy_data["success_rate"].mean(),
                "avg_cost": strategy_data["total_cost"].mean(),

                # Standard deviations
                "std_completion_time": strategy_data["completion_time"].std(),
                "std_quality_score": strategy_data["quality_score"].std(),
                "std_cost": strategy_data["total_cost"].std(),

                # Confidence intervals (95% confidence)
                "quality_confidence_interval": self._calculate_confidence_interval(quality_scores, 0.95) if quality_scores else (0, 0),
                "time_confidence_interval": self._calculate_confidence_interval(completion_times, 0.95) if completion_times else (0, 0),
                "cost_confidence_interval": self._calculate_confidence_interval(costs, 0.95) if costs else (0, 0),

                # Coefficient of variation (consistency measure)
                "quality_consistency": (strategy_data["quality_score"].std() / strategy_data["quality_score"].mean()) if strategy_data["quality_score"].mean() > 0 else float('inf'),
                "time_consistency": (strategy_data["completion_time"].std() / strategy_data["completion_time"].mean()) if strategy_data["completion_time"].mean() > 0 else float('inf'),
            }

        # Query type analysis
        query_type_analysis = {}
        for query_type in df["query_type"].unique():
            type_data = df[df["query_type"] == query_type]

            if len(type_data) > 0:
                best_row = type_data.loc[type_data["quality_score"].idxmax()]
                quality_scores = type_data["quality_score"].dropna().tolist()

                query_type_analysis[query_type] = {
                    "experiment_count": len(type_data),
                    "best_strategy": best_row["strategy"],
                    "best_strategy_score": best_row["quality_score"],
                    "avg_quality_score": type_data["quality_score"].mean(),
                    "quality_confidence_interval": self._calculate_confidence_interval(quality_scores, 0.95) if quality_scores else (0, 0),

                    # Strategy distribution for this query type
                    "strategy_distribution": type_data["strategy"].value_counts().to_dict(),
                    "avg_completion_time": type_data["completion_time"].mean(),
                    "avg_cost": type_data["total_cost"].mean()
                }

        # Statistical significance testing between strategies
        significance_tests = self._perform_significance_tests(df)
        self.significance_tests = significance_tests

        # Cost-benefit analysis
        cost_benefit_analysis = self._perform_cost_benefit_analysis(df)
        self.cost_benefit_analysis = cost_benefit_analysis

        return {
            "overall_stats": overall_stats,
            "strategy_comparison": strategy_comparison,
            "query_type_analysis": query_type_analysis,
            "significance_tests": significance_tests,
            "cost_benefit_analysis": cost_benefit_analysis,
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
                # Test multiple metrics
                metrics_to_test = ["quality_score", "completion_time", "total_cost"]

                for metric in metrics_to_test:
                    data1 = df[df["strategy"] == strategy1][metric].dropna()
                    data2 = df[df["strategy"] == strategy2][metric].dropna()

                    if len(data1) >= 2 and len(data2) >= 2:
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        mean1, mean2 = data1.mean(), data2.mean()

                        results[f"{strategy1}_vs_{strategy2}_{metric}"] = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "better_strategy": strategy1 if mean1 > mean2 else strategy2,
                            "mean_difference": mean1 - mean2,
                            "metric": metric,
                            "sample_size_1": len(data1),
                            "sample_size_2": len(data2),
                            "confidence_level": 0.95
                        }

        return results

    def _perform_cost_benefit_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform cost-benefit analysis of strategies."""
        cost_benefit = {}

        # Filter out strategies with no cost data
        valid_strategies = []
        for strategy in df["strategy"].unique():
            strategy_data = df[df["strategy"] == strategy]
            if strategy_data["total_cost"].mean() > 0:
                valid_strategies.append(strategy)

        for strategy in valid_strategies:
            strategy_data = df[df["strategy"] == strategy]

            quality_scores = strategy_data["quality_score"].dropna()
            costs = strategy_data["total_cost"].dropna()
            tokens = strategy_data["total_tokens"].dropna()

            if len(quality_scores) > 0 and len(costs) > 0:
                avg_quality = quality_scores.mean()
                avg_cost = costs.mean()
                avg_tokens = tokens.mean() if len(tokens) > 0 else 0

                # Calculate correlation between quality and cost if we have enough data
                quality_cost_correlation = 0
                if len(quality_scores) > 1 and len(costs) > 1:
                    try:
                        quality_cost_correlation = quality_scores.corr(costs)
                    except:
                        quality_cost_correlation = 0

                cost_benefit[strategy] = {
                    "quality_per_dollar": avg_quality / avg_cost if avg_cost > 0 else 0,
                    "quality_per_token": avg_quality / avg_tokens if avg_tokens > 0 else 0,
                    "cost_per_quality_unit": avg_cost / avg_quality if avg_quality > 0 else 0,
                    "tokens_per_quality_unit": avg_tokens / avg_quality if avg_quality > 0 else 0,

                    # Efficiency rankings
                    "quality_efficiency_rank": 0,  # Will be calculated
                    "cost_efficiency_rank": 0,     # Will be calculated

                    # Statistical measures
                    "quality_cost_correlation": quality_cost_correlation,
                    "quality_variability": quality_scores.std() if len(quality_scores) > 1 else 0,
                    "cost_variability": costs.std() if len(costs) > 1 else 0,

                    # Enhanced metrics from quality evaluation
                    "avg_synthesis_length": strategy_data["synthesis_length"].mean(),
                    "quality_consistency": (quality_scores.std() / avg_quality) if avg_quality > 0 else float('inf'),
                }

        # Calculate rankings
        if cost_benefit:
            # Quality efficiency ranking (higher quality per dollar is better)
            quality_efficiency = {s: data["quality_per_dollar"] for s, data in cost_benefit.items()}
            quality_efficiency_sorted = sorted(quality_efficiency.items(), key=lambda x: x[1], reverse=True)

            # Cost efficiency ranking (lower cost per quality is better)
            cost_efficiency = {s: data["cost_per_quality_unit"] for s, data in cost_benefit.items()}
            cost_efficiency_sorted = sorted(cost_efficiency.items(), key=lambda x: x[1], reverse=False)

            for i, (strategy, _) in enumerate(quality_efficiency_sorted):
                cost_benefit[strategy]["quality_efficiency_rank"] = i + 1

            for i, (strategy, _) in enumerate(cost_efficiency_sorted):
                cost_benefit[strategy]["cost_efficiency_rank"] = i + 1

        return cost_benefit

    async def run_comprehensive_test(self, test_queries: Dict[str, List[str]], orchestrator: Any, max_experiments_per_query: int = 3) -> Dict[str, Any]:
        """
        Run comprehensive testing across different query types.

        Args:
            test_queries: Dictionary mapping query types to lists of test queries
            max_experiments_per_query: Maximum number of experiments per query

        Returns:
            Comprehensive test results and analysis
        """
        self.logger.info(f"Starting comprehensive testing with {len(test_queries)} query types")

        all_results = []
        query_type_results = {}

        for query_type, queries in test_queries.items():
            self.logger.info(f"Testing query type: {query_type} with {len(queries)} queries")

            type_results = []

            for i, query in enumerate(queries):
                if i >= max_experiments_per_query:
                    break

                self.logger.info(f"Running experiment {i+1}/{min(len(queries), max_experiments_per_query)} for {query_type}: {query[:50]}...")

                try:
                    # Test a few strategies for this query
                    strategies_to_test = [
                        OrchestrationStrategy.THEORY_FIRST,
                        OrchestrationStrategy.MARKET_FIRST,
                        OrchestrationStrategy.TECHNICAL_FIRST
                    ]

                    for strategy in strategies_to_test:
                        try:
                            experiment = await self.run_experiment(query, orchestrator, strategy)
                            type_results.append(experiment)
                            all_results.append(experiment)
                            self.logger.info(f"✓ Completed {strategy.value} for {query_type}")
                        except Exception as e:
                            self.logger.error(f"✗ Failed {strategy.value} for {query_type}: {e}")

                except Exception as e:
                    self.logger.error(f"Failed to process query {query}: {e}")

            query_type_results[query_type] = type_results

        # Analyze comprehensive results
        analysis = self.analyze_performance()

        return {
            "comprehensive_analysis": analysis,
            "query_type_results": query_type_results,
            "total_experiments": len(all_results),
            "query_types_tested": list(test_queries.keys()),
            "strategies_tested": len(set(str(exp.strategy.value) for exp in all_results)),
            "cache_stats": self.get_cache_stats(),
            "performance_summary": {
                "average_quality": sum(exp.metrics.quality_score for exp in all_results) / len(all_results) if all_results else 0,
                "average_time": sum(exp.metrics.completion_time for exp in all_results) / len(all_results) if all_results else 0,
                "average_cost": sum(exp.metrics.total_cost for exp in all_results) / len(all_results) if all_results else 0,
                "success_rate": sum(1 for exp in all_results if exp.metrics.success_rate > 0) / len(all_results) if all_results else 0
            }
        }

    def get_test_queries_library(self) -> Dict[str, List[str]]:
        """Get a comprehensive library of test queries by type."""
        return {
            QueryType.ACADEMIC_THEORETICAL.value: [
                "quantum computing applications in healthcare",
                "machine learning ethics and governance frameworks",
                "blockchain for scientific research collaboration",
                "artificial intelligence in climate modeling",
                "neural networks for medical diagnosis",
                "cryptography in quantum computing",
                "explainable AI methodologies and frameworks",
                "natural language processing in education",
                "computer vision applications in autonomous vehicles",
                "reinforcement learning for robotics control"
            ],
            QueryType.BUSINESS_COMMERCIAL.value: [
                "market analysis of renewable energy trends",
                "AI startup investment opportunities 2024",
                "fintech disruption in traditional banking",
                "e-commerce personalization strategies",
                "blockchain applications in supply chain management",
                "cloud computing market analysis",
                "digital transformation in manufacturing",
                "subscription economy business models",
                "SaaS pricing strategies and optimization",
                "venture capital trends in deep tech"
            ],
            QueryType.TECHNICAL_IMPLEMENTATION.value: [
                "microservices architecture for ML systems",
                "distributed systems consensus algorithms",
                "edge computing optimization strategies",
                "container orchestration with Kubernetes",
                "serverless computing best practices",
                "database sharding and partitioning strategies",
                "API design patterns and best practices",
                "continuous integration and deployment pipelines",
                "monitoring and observability in cloud systems",
                "performance optimization for high-load applications"
            ],
            QueryType.MULTI_DOMAIN.value: [
                "AI in climate change mitigation strategies",
                "blockchain for healthcare data management",
                "IoT applications in smart city infrastructure",
                "augmented reality in medical training",
                "autonomous vehicles regulatory frameworks",
                "5G network infrastructure development",
                "renewable energy grid integration challenges",
                "cybersecurity in critical infrastructure",
                "digital twins in manufacturing optimization",
                "biotechnology and AI convergence trends"
            ]
        }

    def _generate_recommendations(
        self,
        strategy_comparison: Dict,
        query_type_analysis: Dict
    ) -> List[str]:
        """Generate recommendations based on performance analysis with statistical rigor."""
        recommendations = []

        # Find best overall strategy with confidence
        best_strategy = max(
            strategy_comparison.items(),
            key=lambda x: x[1]["avg_quality_score"]
        )

        ci = best_strategy[1].get("quality_confidence_interval", (0, 0))
        recommendations.append(
            f"🎯 **Overall Best Strategy**: {best_strategy[0]} "
            f"(avg quality: {best_strategy[1]['avg_quality_score']:.3f}, "
            f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])"
        )

        # Statistical significance insights (passed as parameter)
        significant_differences = []
        # Note: significance_tests would be passed as a parameter to this method
        # For now, we'll generate basic recommendations

        # Query-type specific recommendations
        for query_type, analysis in query_type_analysis.items():
            if analysis["experiment_count"] > 0:
                ci = analysis.get("quality_confidence_interval", (0, 0))
                recommendations.append(
                    f"📚 **{query_type.title()} Queries**: Use {analysis['best_strategy']} "
                    f"(avg quality: {analysis['avg_quality_score']:.3f}, "
                    f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])"
                )

        # Performance insights with consistency measures
        consistent_strategies = [
            name for name, metrics in strategy_comparison.items()
            if metrics.get("quality_consistency", float('inf')) < 0.2  # Low variance
        ]

        if consistent_strategies:
            recommendations.append(
                f"🎯 **Most Consistent Strategies** (low quality variance): {', '.join(consistent_strategies)}"
            )

        # Time-sensitive recommendations
        fast_strategies = [
            name for name, metrics in strategy_comparison.items()
            if metrics["avg_completion_time"] < 120  # Less than 2 minutes
        ]

        if fast_strategies:
            recommendations.append(
                f"⚡ **Fast Strategies** (for time-sensitive queries): {', '.join(fast_strategies)}"
            )

        # Cost efficiency recommendations
        if hasattr(self, 'cost_benefit_analysis') and self.cost_benefit_analysis:
            best_quality_efficiency = min(
                self.cost_benefit_analysis.items(),
                key=lambda x: x[1].get("cost_per_quality_unit", float('inf'))
            )
            recommendations.append(
                f"💰 **Most Cost-Efficient**: {best_quality_efficiency[0]} "
                f"(cost per quality unit: {best_quality_efficiency[1].get('cost_per_quality_unit', 0):.4f})"
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
