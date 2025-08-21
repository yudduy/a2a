"""Intelligent Sequence Selector and Analysis Agent.

This module provides intelligent analysis of research queries to recommend optimal
sequence strategies. The analysis agent examines query characteristics, research
domain, complexity, and scope to suggest appropriate sequence patterns with
detailed reasoning.

Core Functions:
- Query domain detection (academic, market, technical, hybrid)
- Complexity and scope analysis
- Sequence strategy recommendation
- Human-readable explanation generation
- Confidence scoring for recommendations

The selector integrates with the SequenceOptimizationEngine to provide dynamic
sequence selection while maintaining proven Theory First, Market First, and
Future Back patterns for comprehensive comparison.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from open_deep_research.sequencing.models import (
    QueryType,
    ResearchDomain,
    ScopeBreadth,
    SequenceStrategy,
    SEQUENCE_PATTERNS
)

logger = logging.getLogger(__name__)


class SequenceAnalysisResult:
    """Simple analysis results class for internal use."""
    
    def __init__(
        self,
        query_type: QueryType,
        research_domain: ResearchDomain,
        complexity_score: float,
        scope_breadth: ScopeBreadth,
        recommended_sequences: List[Tuple[SequenceStrategy, float]],
        primary_recommendation: SequenceStrategy,
        confidence: float,
        explanation: str,
        reasoning: Dict[str, str],
        query_characteristics: Dict[str, any],
        analysis_timestamp: Optional[datetime] = None
    ):
        self.analysis_id = str(uuid4())
        self.query_type = query_type
        self.research_domain = research_domain
        self.complexity_score = complexity_score  # 0.0 - 1.0
        self.scope_breadth = scope_breadth
        self.recommended_sequences = recommended_sequences  # [(strategy, confidence), ...]
        self.primary_recommendation = primary_recommendation
        self.confidence = confidence  # 0.0 - 1.0
        self.explanation = explanation
        self.reasoning = reasoning
        self.query_characteristics = query_characteristics
        self.analysis_timestamp = analysis_timestamp or datetime.utcnow()


class SequenceAnalyzer:
    """Intelligent analysis agent for sequence selection and explanation."""
    
    def __init__(self):
        """Initialize the sequence analyzer with domain patterns and keywords."""
        
        # Domain detection patterns
        self.domain_patterns = {
            "academic": {
                "keywords": [
                    "research", "study", "academic", "scientific", "theoretical",
                    "peer-reviewed", "publication", "journal", "methodology",
                    "hypothesis", "experiment", "analysis", "literature review",
                    "empirical", "scholarly", "university", "institution",
                    "framework", "model", "theory", "principle", "findings"
                ],
                "indicators": [
                    "what does research show", "academic studies", "theoretical foundation",
                    "research gap", "literature suggests", "peer reviewed",
                    "scientific evidence", "research methodology", "academic consensus"
                ]
            },
            "market": {
                "keywords": [
                    "market", "business", "commercial", "industry", "revenue",
                    "profit", "customers", "competitors", "sales", "marketing",
                    "strategy", "opportunity", "investment", "roi", "growth",
                    "demand", "supply", "pricing", "value proposition",
                    "business model", "monetization", "market share"
                ],
                "indicators": [
                    "market opportunity", "business potential", "commercial viability",
                    "revenue streams", "customer demand", "competitive landscape",
                    "market size", "business strategy", "monetization"
                ]
            },
            "technical": {
                "keywords": [
                    "technology", "technical", "implementation", "architecture",
                    "system", "software", "hardware", "platform", "infrastructure",
                    "development", "engineering", "programming", "algorithm",
                    "protocol", "standard", "specification", "api", "framework",
                    "tool", "solution", "scalability", "performance"
                ],
                "indicators": [
                    "technical feasibility", "implementation challenges",
                    "technical requirements", "system architecture", "technology stack",
                    "technical trends", "engineering approach", "technical innovation"
                ]
            }
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            "high": [
                "comprehensive analysis", "multi-faceted", "complex", "interdisciplinary",
                "multiple stakeholders", "various perspectives", "extensive research",
                "thorough investigation", "in-depth analysis", "holistic view",
                "systematic review", "broad scope", "complete understanding"
            ],
            "medium": [
                "detailed analysis", "specific aspects", "key factors", 
                "important considerations", "main elements", "core components",
                "primary factors", "significant issues", "relevant aspects"
            ],
            "low": [
                "brief overview", "quick analysis", "basic understanding",
                "simple explanation", "straightforward", "direct approach",
                "focused question", "specific query", "narrow scope"
            ]
        }
        
        # Innovation and trend indicators
        self.innovation_indicators = [
            "future", "emerging", "next generation", "breakthrough", "innovation",
            "disruptive", "cutting-edge", "state-of-the-art", "advanced",
            "novel", "revolutionary", "transformative", "pioneering",
            "trends", "forecast", "prediction", "outlook", "evolution"
        ]
        
        # Competitive intelligence indicators
        self.competitive_indicators = [
            "competitors", "competitive", "comparison", "benchmark",
            "market position", "competitive advantage", "industry leaders",
            "market share", "positioning", "differentiation", "competitive analysis"
        ]
    
    def analyze_query(self, research_topic: str) -> SequenceAnalysisResult:
        """Analyze a research query and recommend optimal sequence strategies.
        
        Args:
            research_topic: The research query/topic to analyze
            
        Returns:
            SequenceAnalysisResult with recommendations and detailed reasoning
        """
        logger.info(f"Analyzing research query: '{research_topic[:100]}...'")
        
        # Normalize query for analysis
        query_lower = research_topic.lower()
        
        # Detect research domain
        domain_scores = self._detect_research_domain(query_lower)
        primary_domain_str = max(domain_scores.items(), key=lambda x: x[1])[0]
        primary_domain = ResearchDomain(primary_domain_str)
        
        # Classify query type
        query_type_str = self._classify_query_type(query_lower, domain_scores)
        query_type = QueryType(query_type_str)
        
        # Assess complexity and scope
        complexity_score = self._assess_complexity(query_lower)
        scope_breadth_str = self._assess_scope_breadth(query_lower, complexity_score)
        scope_breadth = ScopeBreadth(scope_breadth_str)
        
        # Extract query characteristics
        characteristics = self._extract_query_characteristics(query_lower)
        
        # Generate sequence recommendations
        sequence_recommendations = self._recommend_sequences(
            query_type_str, primary_domain_str, complexity_score, characteristics
        )
        
        # Select primary recommendation
        primary_recommendation = sequence_recommendations[0][0]
        primary_confidence = sequence_recommendations[0][1]
        
        # Generate explanation and reasoning
        explanation = self._generate_explanation(
            research_topic, query_type_str, primary_domain_str, 
            primary_recommendation, complexity_score, characteristics
        )
        
        reasoning = self._generate_detailed_reasoning(
            query_type_str, primary_domain_str, primary_recommendation, 
            complexity_score, characteristics
        )
        
        analysis = SequenceAnalysisResult(
            query_type=query_type,
            research_domain=primary_domain,
            complexity_score=complexity_score,
            scope_breadth=scope_breadth,
            recommended_sequences=sequence_recommendations,
            primary_recommendation=primary_recommendation,
            confidence=primary_confidence,
            explanation=explanation,
            reasoning=reasoning,
            query_characteristics=characteristics
        )
        
        logger.info(f"Analysis complete: {primary_recommendation.value} (confidence: {primary_confidence:.2f})")
        return analysis
    
    def _detect_research_domain(self, query_lower: str) -> Dict[str, float]:
        """Detect the primary research domain(s) of the query."""
        domain_scores = {domain: 0.0 for domain in self.domain_patterns.keys()}
        
        # Score based on keyword presence
        for domain, patterns in self.domain_patterns.items():
            keyword_score = 0
            for keyword in patterns["keywords"]:
                if keyword in query_lower:
                    keyword_score += 1
            
            # Score based on indicator phrases
            indicator_score = 0
            for indicator in patterns["indicators"]:
                if indicator in query_lower:
                    indicator_score += 2  # Indicators are more specific
            
            # Normalize scores
            total_keywords = len(patterns["keywords"])
            total_indicators = len(patterns["indicators"])
            
            keyword_normalized = keyword_score / total_keywords if total_keywords > 0 else 0
            indicator_normalized = (indicator_score / 2) / total_indicators if total_indicators > 0 else 0
            
            # Weighted combination (indicators are more important)
            domain_scores[domain] = (keyword_normalized * 0.4) + (indicator_normalized * 0.6)
        
        return domain_scores
    
    def _classify_query_type(self, query_lower: str, domain_scores: Dict[str, float]) -> str:
        """Classify the specific type of research query."""
        
        # Check for competitive intelligence
        if any(indicator in query_lower for indicator in self.competitive_indicators):
            return "competitive_intelligence"
        
        # Check for innovation/trend focus
        if any(indicator in query_lower for indicator in self.innovation_indicators):
            return "innovation_exploration"
        
        # Check for trend analysis
        trend_keywords = ["trend", "forecast", "future", "outlook", "evolution", "direction"]
        if any(keyword in query_lower for keyword in trend_keywords):
            return "trend_analysis"
        
        # Determine based on domain dominance
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        primary_domain = sorted_domains[0][0]
        secondary_score = sorted_domains[1][1] if len(sorted_domains) > 1 else 0
        
        # Check for multi-domain queries
        if sorted_domains[0][1] > 0 and secondary_score > 0.3:
            return "hybrid_multi_domain"
        
        # Single domain classification
        domain_to_type = {
            "academic": "academic_research",
            "market": "market_analysis",
            "technical": "technical_feasibility"
        }
        
        return domain_to_type.get(primary_domain, "hybrid_multi_domain")
    
    def _assess_complexity(self, query_lower: str) -> float:
        """Assess the complexity of the research query."""
        complexity_score = 0.5  # Default medium complexity
        
        # Check for high complexity indicators
        high_count = sum(1 for indicator in self.complexity_indicators["high"] 
                        if indicator in query_lower)
        
        # Check for medium complexity indicators
        medium_count = sum(1 for indicator in self.complexity_indicators["medium"] 
                          if indicator in query_lower)
        
        # Check for low complexity indicators
        low_count = sum(1 for indicator in self.complexity_indicators["low"] 
                       if indicator in query_lower)
        
        # Adjust complexity based on indicators
        if high_count > 0:
            complexity_score += min(high_count * 0.2, 0.4)
        if low_count > 0:
            complexity_score -= min(low_count * 0.2, 0.4)
        
        # Consider query length (longer queries tend to be more complex)
        word_count = len(query_lower.split())
        if word_count > 20:
            complexity_score += 0.1
        elif word_count < 8:
            complexity_score -= 0.1
        
        return max(0.0, min(1.0, complexity_score))
    
    def _assess_scope_breadth(self, query_lower: str, complexity_score: float) -> str:
        """Assess the breadth of scope for the research query."""
        
        broad_indicators = [
            "comprehensive", "complete", "thorough", "extensive", "broad",
            "all aspects", "overall", "general", "wide", "full scope"
        ]
        
        narrow_indicators = [
            "specific", "particular", "focused", "targeted", "precise",
            "exact", "detailed", "narrow", "specialized", "certain"
        ]
        
        broad_count = sum(1 for indicator in broad_indicators if indicator in query_lower)
        narrow_count = sum(1 for indicator in narrow_indicators if indicator in query_lower)
        
        if broad_count > narrow_count and complexity_score > 0.6:
            return "broad"
        elif narrow_count > broad_count or complexity_score < 0.4:
            return "narrow"
        else:
            return "medium"
    
    def _extract_query_characteristics(self, query_lower: str) -> Dict[str, any]:
        """Extract detailed characteristics from the query."""
        characteristics = {
            "word_count": len(query_lower.split()),
            "has_question_mark": "?" in query_lower,
            "has_multiple_questions": query_lower.count("?") > 1,
            "mentions_timeframe": False,
            "mentions_comparison": False,
            "mentions_implementation": False,
            "innovation_focused": False,
            "research_focused": False,
            "business_focused": False
        }
        
        # Check for timeframe mentions
        timeframe_patterns = [
            r"\b\d{4}\b",  # Years like 2024
            r"\bnext \w+ years?\b",  # Next X years
            r"\bfuture\b", r"\bcurrent\b", r"\bpast\b",
            r"\brecent\b", r"\bupcoming\b", r"\bemerging\b"
        ]
        characteristics["mentions_timeframe"] = any(
            re.search(pattern, query_lower) for pattern in timeframe_patterns
        )
        
        # Check for comparison indicators
        comparison_words = ["compare", "versus", "vs", "difference", "compared to", "contrast"]
        characteristics["mentions_comparison"] = any(word in query_lower for word in comparison_words)
        
        # Check for implementation focus
        implementation_words = ["implement", "deployment", "adoption", "integration", "rollout"]
        characteristics["mentions_implementation"] = any(word in query_lower for word in implementation_words)
        
        # Check focus areas
        characteristics["innovation_focused"] = any(word in query_lower for word in self.innovation_indicators)
        characteristics["research_focused"] = any(word in query_lower for word in self.domain_patterns["academic"]["keywords"][:10])
        characteristics["business_focused"] = any(word in query_lower for word in self.domain_patterns["market"]["keywords"][:10])
        
        return characteristics
    
    def _recommend_sequences(
        self, 
        query_type: str, 
        primary_domain: str, 
        complexity_score: float,
        characteristics: Dict[str, any]
    ) -> List[Tuple[SequenceStrategy, float]]:
        """Recommend sequence strategies with confidence scores."""
        
        # Base recommendations by query type and domain
        recommendations = {}
        
        if query_type == "academic_research":
            recommendations[SequenceStrategy.THEORY_FIRST] = 0.85
            recommendations[SequenceStrategy.MARKET_FIRST] = 0.45
            recommendations[SequenceStrategy.FUTURE_BACK] = 0.35
            
        elif query_type == "market_analysis":
            recommendations[SequenceStrategy.MARKET_FIRST] = 0.85
            recommendations[SequenceStrategy.THEORY_FIRST] = 0.55
            recommendations[SequenceStrategy.FUTURE_BACK] = 0.40
            
        elif query_type == "technical_feasibility":
            recommendations[SequenceStrategy.THEORY_FIRST] = 0.70
            recommendations[SequenceStrategy.FUTURE_BACK] = 0.65
            recommendations[SequenceStrategy.MARKET_FIRST] = 0.50
            
        elif query_type == "innovation_exploration":
            recommendations[SequenceStrategy.FUTURE_BACK] = 0.80
            recommendations[SequenceStrategy.THEORY_FIRST] = 0.60
            recommendations[SequenceStrategy.MARKET_FIRST] = 0.55
            
        elif query_type == "trend_analysis":
            recommendations[SequenceStrategy.FUTURE_BACK] = 0.85
            recommendations[SequenceStrategy.MARKET_FIRST] = 0.60
            recommendations[SequenceStrategy.THEORY_FIRST] = 0.45
            
        elif query_type == "competitive_intelligence":
            recommendations[SequenceStrategy.MARKET_FIRST] = 0.80
            recommendations[SequenceStrategy.FUTURE_BACK] = 0.60
            recommendations[SequenceStrategy.THEORY_FIRST] = 0.40
            
        else:  # hybrid_multi_domain
            recommendations[SequenceStrategy.THEORY_FIRST] = 0.70
            recommendations[SequenceStrategy.MARKET_FIRST] = 0.70
            recommendations[SequenceStrategy.FUTURE_BACK] = 0.65
        
        # Adjust based on complexity
        if complexity_score > 0.7:
            # High complexity benefits from theoretical foundation
            recommendations[SequenceStrategy.THEORY_FIRST] += 0.10
        elif complexity_score < 0.4:
            # Low complexity can start with practical approaches
            recommendations[SequenceStrategy.MARKET_FIRST] += 0.10
        
        # Adjust based on characteristics
        if characteristics.get("innovation_focused", False):
            recommendations[SequenceStrategy.FUTURE_BACK] += 0.10
            
        if characteristics.get("business_focused", False):
            recommendations[SequenceStrategy.MARKET_FIRST] += 0.10
            
        if characteristics.get("research_focused", False):
            recommendations[SequenceStrategy.THEORY_FIRST] += 0.10
        
        # Normalize and sort
        max_score = max(recommendations.values())
        if max_score > 1.0:
            recommendations = {k: min(v, 1.0) for k, v in recommendations.items()}
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations
    
    def _generate_explanation(
        self, 
        research_topic: str, 
        query_type: str, 
        primary_domain: str,
        recommended_strategy: SequenceStrategy, 
        complexity_score: float,
        characteristics: Dict[str, any]
    ) -> str:
        """Generate human-readable explanation for sequence selection."""
        
        strategy_descriptions = {
            SequenceStrategy.THEORY_FIRST: "Theory First approach (Academic → Industry → Technical)",
            SequenceStrategy.MARKET_FIRST: "Market First approach (Industry → Academic → Technical)",
            SequenceStrategy.FUTURE_BACK: "Future Back approach (Technical → Academic → Industry)"
        }
        
        # Strategy-specific rationales
        strategy_rationales = {
            SequenceStrategy.THEORY_FIRST: {
                "strengths": [
                    "establishes strong theoretical foundation",
                    "ensures evidence-based analysis",
                    "provides scientific rigor to subsequent market and technical analysis",
                    "identifies fundamental principles before practical applications"
                ],
                "best_for": "research requiring solid academic grounding and theoretical understanding"
            },
            SequenceStrategy.MARKET_FIRST: {
                "strengths": [
                    "prioritizes commercial viability and market opportunities",
                    "focuses on practical business applications",
                    "identifies customer needs and market demand early",
                    "grounds theoretical research in real-world contexts"
                ],
                "best_for": "business-oriented research and commercial opportunity assessment"
            },
            SequenceStrategy.FUTURE_BACK: {
                "strengths": [
                    "starts with cutting-edge technical trends and innovations",
                    "identifies emerging opportunities before they become mainstream",
                    "ensures forward-looking perspective",
                    "connects future possibilities with current academic understanding and market reality"
                ],
                "best_for": "innovation exploration and technology trend analysis"
            }
        }
        
        explanation_parts = []
        
        # Opening context
        explanation_parts.append(
            f"For the research topic '{research_topic}', I recommend the "
            f"{strategy_descriptions[recommended_strategy]}."
        )
        
        # Domain and type analysis
        explanation_parts.append(
            f"This query is classified as {query_type.replace('_', ' ')} with a primary focus on "
            f"{primary_domain} research (complexity score: {complexity_score:.2f})."
        )
        
        # Strategy rationale
        rationale = strategy_rationales[recommended_strategy]
        explanation_parts.append(
            f"The {recommended_strategy.value.replace('_', ' ')} sequence is optimal because it {rationale['strengths'][0]} "
            f"and {rationale['strengths'][1]}. This approach is particularly effective for {rationale['best_for']}."
        )
        
        # Specific reasoning based on characteristics
        if characteristics.get("innovation_focused", False):
            explanation_parts.append(
                "The innovation focus in your query suggests that exploring future technical trends first "
                "will provide valuable forward-looking insights."
            )
        
        if characteristics.get("business_focused", False):
            explanation_parts.append(
                "The business orientation of your query indicates that understanding market dynamics "
                "and commercial opportunities should guide the research direction."
            )
        
        if characteristics.get("research_focused", False):
            explanation_parts.append(
                "The academic nature of your query suggests that establishing theoretical foundations "
                "will provide the most rigorous analytical framework."
            )
        
        if complexity_score > 0.7:
            explanation_parts.append(
                "Given the high complexity of this research area, the selected sequence ensures "
                "systematic knowledge building across all domains."
            )
        
        # Closing note
        explanation_parts.append(
            "While this sequence is recommended as optimal, all three sequence strategies will be "
            "executed for comprehensive comparison to validate the most productive approach for your specific research question."
        )
        
        return " ".join(explanation_parts)
    
    def _generate_detailed_reasoning(
        self, 
        query_type: str, 
        primary_domain: str,
        recommended_strategy: SequenceStrategy, 
        complexity_score: float,
        characteristics: Dict[str, any]
    ) -> Dict[str, str]:
        """Generate detailed reasoning breakdown."""
        
        reasoning = {
            "query_classification": f"Classified as {query_type} based on keyword analysis and domain indicators",
            "domain_analysis": f"Primary domain identified as {primary_domain} with supporting evidence from query content",
            "complexity_assessment": f"Complexity score of {complexity_score:.2f} based on scope indicators and query structure",
            "sequence_selection": f"Selected {recommended_strategy.value} based on domain focus and research objectives"
        }
        
        # Add characteristic-specific reasoning
        if characteristics.get("mentions_timeframe", False):
            reasoning["temporal_focus"] = "Query includes temporal elements, suggesting trend or evolution analysis"
        
        if characteristics.get("mentions_comparison", False):
            reasoning["comparative_analysis"] = "Comparison elements detected, indicating need for multi-perspective analysis"
        
        if characteristics.get("innovation_focused", False):
            reasoning["innovation_orientation"] = "Innovation focus identified, supporting future-oriented research approach"
        
        return reasoning
    
    def explain_all_sequences(self, analysis: SequenceAnalysisResult) -> Dict[SequenceStrategy, str]:
        """Generate explanations for why each sequence strategy could be valuable.
        
        Args:
            analysis: The sequence analysis results
            
        Returns:
            Dictionary mapping each strategy to its explanation
        """
        explanations = {}
        
        # Get confidence scores for each strategy
        strategy_confidences = {strategy: confidence for strategy, confidence in analysis.recommended_sequences}
        
        base_explanations = {
            SequenceStrategy.THEORY_FIRST: (
                "The Theory First approach would establish a strong academic foundation by starting with "
                "scholarly research and theoretical frameworks. This sequence is valuable when solid "
                "evidence-based understanding is crucial before exploring practical applications. "
                "It ensures that market analysis and technical implementation are grounded in "
                "rigorous academic knowledge."
            ),
            SequenceStrategy.MARKET_FIRST: (
                "The Market First approach would prioritize commercial viability and business opportunities "
                "by beginning with industry analysis. This sequence is valuable when understanding "
                "market demand, competitive landscape, and business models is essential. "
                "It ensures that academic research and technical development are guided by "
                "real-world market needs and commercial potential."
            ),
            SequenceStrategy.FUTURE_BACK: (
                "The Future Back approach would start with emerging technical trends and innovations "
                "to identify cutting-edge opportunities. This sequence is valuable when exploring "
                "forward-looking possibilities and technological disruptions. "
                "It ensures that academic validation and market assessment consider "
                "future-oriented developments and next-generation capabilities."
            )
        }
        
        for strategy, base_explanation in base_explanations.items():
            confidence = strategy_confidences.get(strategy, 0.5)
            
            if strategy == analysis.primary_recommendation:
                explanation = f"[RECOMMENDED - {confidence:.1%} confidence] {base_explanation}"
            else:
                explanation = f"[Alternative - {confidence:.1%} confidence] {base_explanation}"
            
            # Add specific reasoning based on query characteristics
            if strategy == SequenceStrategy.THEORY_FIRST and analysis.query_characteristics.get("research_focused", False):
                explanation += " This aligns particularly well with the academic focus of your query."
                
            elif strategy == SequenceStrategy.MARKET_FIRST and analysis.query_characteristics.get("business_focused", False):
                explanation += " This matches the business orientation evident in your research question."
                
            elif strategy == SequenceStrategy.FUTURE_BACK and analysis.query_characteristics.get("innovation_focused", False):
                explanation += " This complements the innovation and future-trends focus of your query."
            
            explanations[strategy] = explanation
        
        return explanations