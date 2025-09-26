"""Structured output models for LLM-based evaluation of research reports."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluationCriteria(BaseModel):
    """Individual evaluation criterion with score and reasoning."""
    
    name: str = Field(description="Name of the evaluation criterion")
    score: float = Field(description="Score from 0-10 for this criterion")
    max_score: float = Field(default=10.0, description="Maximum possible score for this criterion")
    reasoning: str = Field(description="Detailed reasoning for the assigned score")
    strengths: List[str] = Field(description="Specific strengths identified in this area")
    weaknesses: List[str] = Field(description="Specific weaknesses identified in this area")
    evidence_examples: List[str] = Field(description="Specific examples from the report supporting the score")


class ReportEvaluation(BaseModel):
    """Comprehensive evaluation of a single research report."""
    
    # Report metadata
    report_id: str = Field(description="Unique identifier for the report")
    sequence_name: str = Field(description="Name of the agent sequence used")
    research_topic: str = Field(description="The research topic being evaluated")
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Individual criteria scores
    completeness: EvaluationCriteria = Field(description="Coverage of the research topic")
    depth: EvaluationCriteria = Field(description="Depth of analysis and investigation") 
    coherence: EvaluationCriteria = Field(description="How well insights build on each other sequentially")
    innovation: EvaluationCriteria = Field(description="Novel insights and perspectives")
    actionability: EvaluationCriteria = Field(description="Practical recommendations and next steps")
    
    # Overall assessment
    overall_score: float = Field(description="Weighted overall score (0-100)")
    weighted_criteria_scores: Dict[str, float] = Field(description="Individual criteria contributions to overall score")
    
    # Summary assessment
    executive_summary: str = Field(description="High-level summary of the report's quality")
    key_strengths: List[str] = Field(description="Top 3-5 overall strengths of the report")
    key_weaknesses: List[str] = Field(description="Top 3-5 areas for improvement")
    recommendation_quality: str = Field(description="Assessment of practical value and actionability")
    
    # Meta-evaluation
    confidence_level: float = Field(description="Confidence in evaluation accuracy (0-1)")
    evaluation_notes: Optional[str] = Field(default=None, description="Additional notes about the evaluation process")


class SequenceComparison(BaseModel):
    """Pairwise comparison between two agent sequences."""
    
    sequence_a: str = Field(description="Name of the first sequence")
    sequence_b: str = Field(description="Name of the second sequence")
    winner: str = Field(description="Name of the winning sequence")
    margin: float = Field(description="Score difference between sequences (0-100)")
    
    # Comparative strengths
    sequence_a_advantages: List[str] = Field(description="Areas where sequence A excelled")
    sequence_b_advantages: List[str] = Field(description="Areas where sequence B excelled")
    
    # Detailed comparison
    criteria_comparison: Dict[str, Dict[str, float]] = Field(
        description="Breakdown of scores by criteria for both sequences"
    )
    
    comparative_reasoning: str = Field(description="Detailed reasoning for the comparison outcome")
    use_case_recommendations: List[str] = Field(description="Scenarios where each sequence would be preferred")


class ComparativeAnalysis(BaseModel):
    """Comprehensive comparative analysis of multiple research reports."""
    
    # Analysis metadata
    research_topic: str = Field(description="The research topic being compared")
    num_reports_compared: int = Field(description="Number of reports in the comparison")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Rankings and winners
    overall_ranking: List[Dict[str, Any]] = Field(description="Reports ranked by overall score")
    best_sequence: str = Field(description="Name of the best-performing sequence overall")
    best_sequence_reasoning: str = Field(description="Why this sequence performed best")
    
    # Criteria-specific analysis
    criteria_leaders: Dict[str, str] = Field(description="Best sequence for each evaluation criterion")
    criteria_analysis: Dict[str, str] = Field(description="Analysis of performance patterns by criteria")
    
    # Pairwise comparisons
    pairwise_comparisons: List[SequenceComparison] = Field(description="All pairwise sequence comparisons")
    
    # Insights and patterns
    sequence_strengths_patterns: Dict[str, List[str]] = Field(
        description="Common strengths patterns by sequence type"
    )
    sequence_weakness_patterns: Dict[str, List[str]] = Field(
        description="Common weakness patterns by sequence type" 
    )
    
    # Strategic recommendations
    sequence_selection_guide: List[Dict[str, str]] = Field(
        description="Guidelines for selecting sequences based on research context"
    )
    improvement_recommendations: Dict[str, List[str]] = Field(
        description="Specific improvement suggestions for each sequence"
    )
    
    # Meta-analysis
    evaluation_confidence: float = Field(description="Overall confidence in the comparative analysis")
    methodology_notes: str = Field(description="Notes about the evaluation methodology used")


class EvaluationResult(BaseModel):
    """Complete evaluation result containing individual evaluations and comparative analysis."""
    
    # Core data
    individual_evaluations: List[ReportEvaluation] = Field(description="Detailed evaluation for each report")
    comparative_analysis: ComparativeAnalysis = Field(description="Cross-report comparative analysis")
    
    # Summary metrics
    score_statistics: Dict[str, float] = Field(description="Statistical summary of scores across reports")
    performance_gaps: Dict[str, float] = Field(description="Performance gaps between sequences")
    
    # Actionable insights
    winning_sequence: str = Field(description="Overall best-performing sequence")
    winning_sequence_score: float = Field(description="Score of the winning sequence")
    key_differentiators: List[str] = Field(description="Factors that distinguish top performers")
    sequence_recommendations: Dict[str, str] = Field(description="When to use each sequence type")
    
    # Evaluation metadata
    evaluation_model: str = Field(description="Model used for evaluation")
    evaluation_version: str = Field(default="1.0", description="Version of the evaluation system")
    processing_time: Optional[float] = Field(default=None, description="Time taken for evaluation in seconds")
    
    def get_winner_details(self) -> Dict[str, Any]:
        """Get detailed information about the winning sequence."""
        winning_eval = next(
            eval for eval in self.individual_evaluations 
            if eval.sequence_name == self.winning_sequence
        )
        
        return {
            "sequence_name": self.winning_sequence,
            "overall_score": winning_eval.overall_score,
            "key_strengths": winning_eval.key_strengths,
            "criteria_scores": {
                "completeness": winning_eval.completeness.score,
                "depth": winning_eval.depth.score,
                "coherence": winning_eval.coherence.score,
                "innovation": winning_eval.innovation.score,
                "actionability": winning_eval.actionability.score
            },
            "executive_summary": winning_eval.executive_summary
        }
    
    def get_criteria_rankings(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get rankings for each evaluation criteria."""
        criteria_rankings = {}
        
        for criterion in ["completeness", "depth", "coherence", "innovation", "actionability"]:
            sorted_evals = sorted(
                self.individual_evaluations,
                key=lambda x: getattr(x, criterion).score,
                reverse=True
            )
            
            criteria_rankings[criterion] = [
                {
                    "sequence_name": eval.sequence_name,
                    "score": getattr(eval, criterion).score,
                    "reasoning": getattr(eval, criterion).reasoning[:200] + "..."
                }
                for eval in sorted_evals
            ]
        
        return criteria_rankings