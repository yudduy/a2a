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
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from open_deep_research.sequencing.models import (
    AgentType,
    DynamicSequencePattern,
    QueryType,
    ResearchDomain,
    ScopeBreadth,
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
        recommended_sequences: List[Tuple[str, float]],
        primary_recommendation: str,
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
        
        logger.info(f"Analysis complete: {primary_recommendation} (confidence: {primary_confidence:.2f})")
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
        sum(1 for indicator in self.complexity_indicators["medium"] 
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
    ) -> List[Tuple[str, float]]:
        """Recommend sequence strategies with confidence scores."""
        # Base recommendations by query type and domain
        recommendations = {}
        
        if query_type == "academic_research":
            recommendations["theory_first"] = 0.85
            recommendations["market_first"] = 0.45
            recommendations["future_back"] = 0.35
            
        elif query_type == "market_analysis":
            recommendations["market_first"] = 0.85
            recommendations["theory_first"] = 0.55
            recommendations["future_back"] = 0.40
            
        elif query_type == "technical_feasibility":
            recommendations["theory_first"] = 0.70
            recommendations["future_back"] = 0.65
            recommendations["market_first"] = 0.50
            
        elif query_type == "innovation_exploration":
            recommendations["future_back"] = 0.80
            recommendations["theory_first"] = 0.60
            recommendations["market_first"] = 0.55
            
        elif query_type == "trend_analysis":
            recommendations["future_back"] = 0.85
            recommendations["market_first"] = 0.60
            recommendations["theory_first"] = 0.45
            
        elif query_type == "competitive_intelligence":
            recommendations["market_first"] = 0.80
            recommendations["future_back"] = 0.60
            recommendations["theory_first"] = 0.40
            
        else:  # hybrid_multi_domain
            recommendations["theory_first"] = 0.70
            recommendations["market_first"] = 0.70
            recommendations["future_back"] = 0.65
        
        # Adjust based on complexity
        if complexity_score > 0.7:
            # High complexity benefits from theoretical foundation
            recommendations["theory_first"] += 0.10
        elif complexity_score < 0.4:
            # Low complexity can start with practical approaches
            recommendations["market_first"] += 0.10
        
        # Adjust based on characteristics
        if characteristics.get("innovation_focused", False):
            recommendations["future_back"] += 0.10
            
        if characteristics.get("business_focused", False):
            recommendations["market_first"] += 0.10
            
        if characteristics.get("research_focused", False):
            recommendations["theory_first"] += 0.10
        
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
        recommended_strategy: str, 
        complexity_score: float,
        characteristics: Dict[str, any]
    ) -> str:
        """Generate human-readable explanation for sequence selection."""
        strategy_descriptions = {
            "theory_first": "Theory First approach (Academic → Industry → Technical)",
            "market_first": "Market First approach (Industry → Academic → Technical)",
            "future_back": "Future Back approach (Technical → Academic → Industry)"
        }
        
        # Strategy-specific rationales
        strategy_rationales = {
            "theory_first": {
                "strengths": [
                    "establishes strong theoretical foundation",
                    "ensures evidence-based analysis",
                    "provides scientific rigor to subsequent market and technical analysis",
                    "identifies fundamental principles before practical applications"
                ],
                "best_for": "research requiring solid academic grounding and theoretical understanding"
            },
            "market_first": {
                "strengths": [
                    "prioritizes commercial viability and market opportunities",
                    "focuses on practical business applications",
                    "identifies customer needs and market demand early",
                    "grounds theoretical research in real-world contexts"
                ],
                "best_for": "business-oriented research and commercial opportunity assessment"
            },
            "future_back": {
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
            f"The {recommended_strategy.replace('_', ' ')} sequence is optimal because it {rationale['strengths'][0]} "
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
        recommended_strategy: str, 
        complexity_score: float,
        characteristics: Dict[str, any]
    ) -> Dict[str, str]:
        """Generate detailed reasoning breakdown."""
        reasoning = {
            "query_classification": f"Classified as {query_type} based on keyword analysis and domain indicators",
            "domain_analysis": f"Primary domain identified as {primary_domain} with supporting evidence from query content",
            "complexity_assessment": f"Complexity score of {complexity_score:.2f} based on scope indicators and query structure",
            "sequence_selection": f"Selected {recommended_strategy} based on domain focus and research objectives"
        }
        
        # Add characteristic-specific reasoning
        if characteristics.get("mentions_timeframe", False):
            reasoning["temporal_focus"] = "Query includes temporal elements, suggesting trend or evolution analysis"
        
        if characteristics.get("mentions_comparison", False):
            reasoning["comparative_analysis"] = "Comparison elements detected, indicating need for multi-perspective analysis"
        
        if characteristics.get("innovation_focused", False):
            reasoning["innovation_orientation"] = "Innovation focus identified, supporting future-oriented research approach"
        
        return reasoning
    
    def explain_all_sequences(self, analysis: SequenceAnalysisResult) -> Dict[str, str]:
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
            "theory_first": (
                "The Theory First approach would establish a strong academic foundation by starting with "
                "scholarly research and theoretical frameworks. This sequence is valuable when solid "
                "evidence-based understanding is crucial before exploring practical applications. "
                "It ensures that market analysis and technical implementation are grounded in "
                "rigorous academic knowledge."
            ),
            "market_first": (
                "The Market First approach would prioritize commercial viability and business opportunities "
                "by beginning with industry analysis. This sequence is valuable when understanding "
                "market demand, competitive landscape, and business models is essential. "
                "It ensures that academic research and technical development are guided by "
                "real-world market needs and commercial potential."
            ),
            "future_back": (
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
            if strategy == "theory_first" and analysis.query_characteristics.get("research_focused", False):
                explanation += " This aligns particularly well with the academic focus of your query."
                
            elif strategy == "market_first" and analysis.query_characteristics.get("business_focused", False):
                explanation += " This matches the business orientation evident in your research question."
                
            elif strategy == "future_back" and analysis.query_characteristics.get("innovation_focused", False):
                explanation += " This complements the innovation and future-trends focus of your query."
            
            explanations[strategy] = explanation
        
        return explanations
    
    def generate_dynamic_sequences(self, topic: str, num_sequences: int = 3) -> List[DynamicSequencePattern]:
        """Generate dynamic sequence patterns based on sophisticated topic analysis.
        
        This method leverages the existing sophisticated analysis infrastructure to generate
        multiple DynamicSequencePattern instances with optimal agent orderings based on
        the research topic characteristics.
        
        Args:
            topic: The research topic to analyze
            num_sequences: Number of dynamic sequences to generate (default: 3)
            
        Returns:
            List of DynamicSequencePattern instances with confidence scores and reasoning
        """
        logger.info(f"Generating {num_sequences} dynamic sequences for topic analysis")
        
        # Leverage existing sophisticated analysis
        analysis = self.analyze_query(topic)
        
        # Extract analysis components for sequence generation
        characteristics = analysis.query_characteristics
        
        # Generate diverse sequence patterns using analysis insights
        dynamic_sequences = []
        
        # 1. Generate primary sequence based on analysis recommendation
        primary_sequence = self._generate_primary_dynamic_sequence(
            analysis, topic, characteristics
        )
        dynamic_sequences.append(primary_sequence)
        
        # 2. Generate complementary sequences with different approaches
        complementary_sequences = self._generate_complementary_sequences(
            analysis, topic, characteristics, num_sequences - 1
        )
        dynamic_sequences.extend(complementary_sequences)
        
        # Sort by confidence score (highest first)
        dynamic_sequences.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Limit to requested number
        dynamic_sequences = dynamic_sequences[:num_sequences]
        
        logger.info(f"Generated {len(dynamic_sequences)} dynamic sequences with confidence scores: {[seq.confidence_score for seq in dynamic_sequences]}")
        
        return dynamic_sequences
    
    def _generate_primary_dynamic_sequence(
        self, 
        analysis: SequenceAnalysisResult, 
        topic: str,
        characteristics: Dict[str, any]
    ) -> DynamicSequencePattern:
        """Generate the primary dynamic sequence based on analysis recommendation."""
        # Map primary recommendation to agent order
        strategy_to_agents = {
            "theory_first": [AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
            "market_first": [AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS],
            "future_back": [AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY]
        }
        
        agent_order = strategy_to_agents[analysis.primary_recommendation]
        
        # Generate reasoning based on analysis
        reasoning = self._generate_dynamic_reasoning(
            analysis.primary_recommendation,
            analysis.query_type.value,
            analysis.research_domain.value,
            analysis.complexity_score,
            characteristics,
            is_primary=True
        )
        
        # Calculate topic alignment score based on analysis confidence
        topic_alignment_score = min(analysis.confidence * 1.1, 1.0)  # Boost primary recommendation
        
        # Generate expected advantages from analysis
        advantages = self._extract_sequence_advantages(
            analysis.primary_recommendation,
            characteristics,
            analysis.complexity_score
        )
        
        return DynamicSequencePattern(
            agent_order=agent_order,
            description=f"Primary dynamic sequence optimized for {analysis.query_type.value.replace('_', ' ')} research",
            reasoning=reasoning,
            confidence_score=analysis.confidence,
            expected_advantages=advantages,
            topic_alignment_score=topic_alignment_score,
            strategy=analysis.primary_recommendation
        )
    
    def _generate_complementary_sequences(
        self, 
        analysis: SequenceAnalysisResult, 
        topic: str,
        characteristics: Dict[str, any],
        num_complementary: int
    ) -> List[DynamicSequencePattern]:
        """Generate complementary dynamic sequences with different approaches."""
        complementary_sequences = []
        
        # Get alternative strategies from analysis recommendations
        alternative_strategies = [
            (strategy, confidence) for strategy, confidence in analysis.recommended_sequences
            if strategy != analysis.primary_recommendation
        ]
        
        # Strategy to agent mappings
        strategy_to_agents = {
            "theory_first": [AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS],
            "market_first": [AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS],
            "future_back": [AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY]
        }
        
        # Generate sequences for alternative strategies
        for i, (strategy, confidence) in enumerate(alternative_strategies[:num_complementary]):
            agent_order = strategy_to_agents[strategy]
            
            reasoning = self._generate_dynamic_reasoning(
                strategy,
                analysis.query_type.value,
                analysis.research_domain.value,
                analysis.complexity_score,
                characteristics,
                is_primary=False
            )
            
            # Topic alignment based on strategy fit
            topic_alignment_score = confidence * 0.9  # Slightly lower than primary
            
            advantages = self._extract_sequence_advantages(
                strategy,
                characteristics,
                analysis.complexity_score
            )
            
            sequence = DynamicSequencePattern(
                agent_order=agent_order,
                description=f"Alternative dynamic sequence emphasizing {strategy.replace('_', ' ')} approach",
                reasoning=reasoning,
                confidence_score=confidence,
                expected_advantages=advantages,
                topic_alignment_score=topic_alignment_score,
                strategy=strategy
            )
            
            complementary_sequences.append(sequence)
        
        # If we need more sequences and have fewer alternatives, generate hybrid approaches
        if len(complementary_sequences) < num_complementary:
            remaining_needed = num_complementary - len(complementary_sequences)
            hybrid_sequences = self._generate_hybrid_sequences(
                analysis, characteristics, remaining_needed
            )
            complementary_sequences.extend(hybrid_sequences)
        
        return complementary_sequences
    
    def _generate_hybrid_sequences(
        self,
        analysis: SequenceAnalysisResult,
        characteristics: Dict[str, any],
        num_needed: int
    ) -> List[DynamicSequencePattern]:
        """Generate hybrid sequences with novel agent orderings."""
        hybrid_sequences = []
        
        # Define some alternative orderings based on characteristics
        alternative_orderings = []
        
        # If innovation focused, try starting with technical but different follow-up
        if characteristics.get("innovation_focused", False):
            alternative_orderings.append({
                "order": [AgentType.TECHNICAL_TRENDS, AgentType.INDUSTRY, AgentType.ACADEMIC],
                "description": "Innovation-first sequence with market validation before academic grounding",
                "focus": "innovation_market_validation"
            })
        
        # If high complexity, try academic-heavy approach
        if analysis.complexity_score > 0.7:
            alternative_orderings.append({
                "order": [AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS, AgentType.INDUSTRY],
                "description": "Research-heavy sequence for complex topics with technical feasibility check",
                "focus": "research_technical_focus"
            })
        
        # If business focused, try industry-technical-academic
        if characteristics.get("business_focused", False):
            alternative_orderings.append({
                "order": [AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC],
                "description": "Business-first sequence with technical validation and academic support",
                "focus": "business_technical_validation"
            })
        
        # Generate sequences from alternative orderings
        for i, ordering in enumerate(alternative_orderings[:num_needed]):
            confidence_score = max(0.4, analysis.confidence * 0.7)  # Lower but reasonable confidence
            
            reasoning = (
                f"Hybrid sequence designed for {ordering['focus'].replace('_', ' ')} based on "
                f"query characteristics. This ordering provides {ordering['description'].lower()}."
            )
            
            advantages = [
                f"Optimized for {ordering['focus'].replace('_', ' ')}",
                "Novel approach not covered by standard strategies",
                "Balances multiple research perspectives"
            ]
            
            # Add specific advantages based on focus
            if "innovation" in ordering['focus']:
                advantages.append("Early identification of cutting-edge developments")
            if "research" in ordering['focus']:
                advantages.append("Strong theoretical foundation with practical validation")
            if "business" in ordering['focus']:
                advantages.append("Commercial viability assessed early in process")
            
            sequence = DynamicSequencePattern(
                agent_order=ordering["order"],
                description=ordering["description"],
                reasoning=reasoning,
                confidence_score=confidence_score,
                expected_advantages=advantages,
                topic_alignment_score=confidence_score * 0.8,
                strategy=None  # No direct strategy mapping for hybrids
            )
            
            hybrid_sequences.append(sequence)
        
        return hybrid_sequences
    
    def _generate_dynamic_reasoning(
        self,
        strategy: str,
        query_type: str,
        research_domain: str,
        complexity_score: float,
        characteristics: Dict[str, any],
        is_primary: bool
    ) -> str:
        """Generate detailed reasoning for dynamic sequence selection."""
        strategy_rationales = {
            "theory_first": {
                "core": "establishes rigorous academic foundation before practical applications",
                "strength": "evidence-based analysis grounded in scholarly research",
                "benefit": "ensures theoretical validity guides market and technical exploration"
            },
            "market_first": {
                "core": "prioritizes commercial viability and real-world market dynamics",
                "strength": "practical business focus with immediate applicability",
                "benefit": "grounds theoretical research in proven market needs and opportunities"
            },
            "future_back": {
                "core": "starts with emerging trends and cutting-edge technical developments",
                "strength": "forward-looking perspective identifying future opportunities",
                "benefit": "connects innovation potential with academic validation and market reality"
            }
        }
        
        rationale = strategy_rationales[strategy]
        
        reasoning_parts = [
            f"This dynamic sequence {rationale['core']}.",
            f"The approach provides {rationale['strength']}, which {rationale['benefit']}."
        ]
        
        # Add query-specific reasoning
        reasoning_parts.append(
            f"For {query_type.replace('_', ' ')} research in the {research_domain} domain, "
            f"this sequence leverages the optimal information flow pattern."
        )
        
        # Add complexity-based reasoning
        if complexity_score > 0.7:
            reasoning_parts.append(
                "The high complexity of this research topic benefits from the systematic "
                "knowledge building approach provided by this agent ordering."
            )
        elif complexity_score < 0.4:
            reasoning_parts.append(
                "The focused nature of this research topic allows for efficient progression "
                "through specialized agent perspectives."
            )
        
        # Add characteristic-specific insights
        if characteristics.get("innovation_focused", False):
            reasoning_parts.append(
                "The innovation focus in the query aligns well with this sequence's ability "
                "to identify and explore emerging opportunities."
            )
        
        if characteristics.get("business_focused", False):
            reasoning_parts.append(
                "The commercial orientation of the query is well-served by this sequence's "
                "attention to market dynamics and business applications."
            )
        
        if is_primary:
            reasoning_parts.append(
                "This sequence is selected as primary based on optimal alignment with "
                "query characteristics and research objectives."
            )
        else:
            reasoning_parts.append(
                "This alternative sequence provides valuable complementary perspectives "
                "for comprehensive research coverage."
            )
        
        return " ".join(reasoning_parts)
    
    def _extract_sequence_advantages(
        self,
        strategy: str,
        characteristics: Dict[str, any],
        complexity_score: float
    ) -> List[str]:
        """Extract specific advantages for a sequence strategy."""
        base_advantages = {
            "theory_first": [
                "Strong academic foundation guides research direction",
                "Evidence-based analysis ensures scientific rigor",
                "Theoretical insights inform practical applications",
                "Systematic knowledge building from first principles"
            ],
            "market_first": [
                "Commercial viability assessed early in process",
                "Market demand drives research prioritization",
                "Real-world constraints inform theoretical exploration",
                "Business opportunities identified upfront"
            ],
            "future_back": [
                "Cutting-edge trends identified before mainstream adoption",
                "Innovation opportunities explored systematically",
                "Future-oriented perspective guides current research",
                "Emerging technologies evaluated for potential impact"
            ]
        }
        
        advantages = base_advantages[strategy].copy()
        
        # Add characteristic-specific advantages
        if characteristics.get("mentions_comparison", False):
            advantages.append("Optimized for comparative analysis across domains")
        
        if characteristics.get("mentions_timeframe", False):
            advantages.append("Temporal analysis integrated throughout research process")
        
        if complexity_score > 0.7:
            advantages.append("Structured approach handles complex multi-faceted research")
        
        if characteristics.get("innovation_focused", False) and strategy == "future_back":
            advantages.append("Innovation focus perfectly aligned with technical-first approach")
        
        if characteristics.get("business_focused", False) and strategy == "market_first":
            advantages.append("Business orientation maximizes commercial insight generation")
        
        if characteristics.get("research_focused", False) and strategy == "theory_first":
            advantages.append("Academic focus leverages theoretical depth and scholarly rigor")
        
        return advantages