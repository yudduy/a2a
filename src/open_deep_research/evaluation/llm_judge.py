"""LLM-based judge system for evaluating sequential multi-agent research reports."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Ensure List is available at module level - defensive programming against import issues
try:
    from typing import List as typing_List
    List = typing_List
except ImportError:
    # Fallback for older Python versions or import issues
    List = list

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from open_deep_research.configuration import Configuration
from open_deep_research.state import RunningReport
from open_deep_research.utils import (
    get_api_key_for_model,
    get_model_config_for_provider,
    clean_reasoning_model_output,
    is_token_limit_exceeded
)

from .models import (
    EvaluationCriteria,
    ReportEvaluation,
    ComparativeAnalysis,
    EvaluationResult,
    SequenceComparison
)
from .prompts import EvaluationPrompts

# Set up logging
logger = logging.getLogger(__name__)


class LLMJudge:
    """LLM-powered evaluation system for sequential multi-agent research reports.
    
    This class provides comprehensive evaluation capabilities including:
    - Individual report scoring across multiple criteria
    - Comparative analysis between different agent sequences
    - Winner selection and performance ranking
    - Detailed insights for sequence optimization
    """

    def __init__(self, config: Optional[RunnableConfig] = None, 
                 evaluation_model: Optional[str] = None,
                 max_retries: int = 3):
        """Initialize the LLM judge system.
        
        Args:
            config: Runtime configuration for model access and settings
            evaluation_model: Specific model to use for evaluation (overrides config)
            max_retries: Maximum number of retry attempts for failed evaluations
        """
        self.config = config or {}
        self.configuration = Configuration.from_runnable_config(config)
        self.max_retries = max_retries
        
        # Configure evaluation model with fallback hierarchy
        self.evaluation_model_name = (
            evaluation_model or 
            getattr(self.configuration, 'evaluation_model', None) or
            self.configuration.final_report_model or  # Fallback to report model
            "anthropic:claude-3-5-sonnet"  # Default fallback
        )
        
        # Initialize model configuration
        self._model: Optional[BaseChatModel] = None
        self._model_config = {}
        
        logger.info(f"Initialized LLMJudge with model: {self.evaluation_model_name}")

    async def _get_evaluation_model(self) -> BaseChatModel:
        """Get or initialize the evaluation model with configuration."""
        if self._model is None:
            try:
                # Get API key and model configuration
                api_key = get_api_key_for_model(self.evaluation_model_name, self.config)
                if not api_key:
                    raise ValueError(f"No API key found for model: {self.evaluation_model_name}")
                
                model_config = get_model_config_for_provider(
                    self.evaluation_model_name,
                    api_key,
                    max_tokens=getattr(self.configuration, 'evaluation_model_max_tokens', 8192),
                    tags=["evaluation", "llm_judge", "report_scoring"]
                )
                
                self._model = init_chat_model(**model_config)
                self._model_config = model_config
                
                logger.info(f"Successfully initialized evaluation model: {self.evaluation_model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize evaluation model {self.evaluation_model_name}: {e}")
                raise RuntimeError(f"Could not initialize evaluation model: {e}")
                
        return self._model

    async def evaluate_reports(self, 
                             reports: Dict[str, Union[str, RunningReport]], 
                             research_topic: str,
                             sequence_names: Optional[List[str]] = None) -> EvaluationResult:
        """Evaluate multiple research reports and provide comparative analysis.
        
        Args:
            reports: Dictionary mapping sequence names to report content or RunningReport objects
            research_topic: The research topic being evaluated
            sequence_names: Optional list of sequence names (inferred from reports keys if not provided)
            
        Returns:
            Complete evaluation result with individual scores and comparative analysis
            
        Raises:
            ValueError: If reports are empty or malformed
            RuntimeError: If evaluation fails after all retries
        """
        if not reports:
            raise ValueError("Reports dictionary cannot be empty")
        
        start_time = time.time()
        sequence_names = sequence_names or list(reports.keys())
        
        logger.info(f"Starting evaluation of {len(reports)} reports for topic: {research_topic}")
        
        try:
            # Step 1: Evaluate each report individually
            individual_evaluations = await self._evaluate_individual_reports(
                reports, research_topic, sequence_names
            )
            
            # Step 2: Perform comparative analysis
            comparative_analysis = await self._perform_comparative_analysis(
                individual_evaluations, research_topic, sequence_names
            )
            
            # Step 3: Calculate summary metrics and insights
            result = self._compile_evaluation_result(
                individual_evaluations, comparative_analysis, start_time
            )
            
            logger.info(
                f"Evaluation completed. Winner: {result.winning_sequence} "
                f"(Score: {result.winning_sequence_score:.1f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise RuntimeError(f"Failed to evaluate reports: {e}")

    async def _evaluate_individual_reports(self, 
                                         reports: Dict[str, Union[str, RunningReport]],
                                         research_topic: str,
                                         sequence_names: List[str]) -> List[ReportEvaluation]:
        """Evaluate each report individually across all criteria."""
        evaluation_tasks = []
        
        for sequence_name in sequence_names:
            if sequence_name not in reports:
                logger.warning(f"Sequence {sequence_name} not found in reports, skipping")
                continue
                
            task = self.evaluate_single_report(
                report=reports[sequence_name],
                research_topic=research_topic,
                sequence_name=sequence_name
            )
            evaluation_tasks.append(task)
        
        # Execute all evaluations concurrently
        individual_evaluations = await asyncio.gather(*evaluation_tasks)
        
        # Filter out None results (failed evaluations)
        valid_evaluations = [eval for eval in individual_evaluations if eval is not None]
        
        if not valid_evaluations:
            raise RuntimeError("All individual report evaluations failed")
        
        logger.info(f"Successfully evaluated {len(valid_evaluations)} individual reports")
        return valid_evaluations

    async def evaluate_single_report(self, 
                                   report: Union[str, RunningReport],
                                   research_topic: str,
                                   sequence_name: str) -> Optional[ReportEvaluation]:
        """Evaluate a single research report across all criteria.
        
        Args:
            report: Either a string report or RunningReport object
            research_topic: The research topic being evaluated
            sequence_name: Name of the agent sequence that generated the report
            
        Returns:
            Detailed evaluation of the report, or None if evaluation fails
        """
        try:
            # Extract report content
            if isinstance(report, RunningReport):
                report_content = self._extract_report_content_from_running_report(report)
                report_id = f"{sequence_name}_{report.start_time.isoformat()}"
            else:
                report_content = str(report)
                report_id = f"{sequence_name}_{datetime.now().isoformat()}"
            
            # Get evaluation model
            model = await self._get_evaluation_model()
            
            # Prepare evaluation prompt
            prompt = EvaluationPrompts.get_single_report_prompt(
                sequence_name=sequence_name,
                research_topic=research_topic,
                report_content=report_content
            )
            
            # Execute evaluation with retries
            evaluation = await self._execute_with_retry(
                self._call_evaluation_model,
                model,
                prompt,
                ReportEvaluation,
                context=f"single report evaluation for {sequence_name}"
            )
            
            if evaluation:
                # Set metadata
                evaluation.report_id = report_id
                evaluation.sequence_name = sequence_name
                evaluation.research_topic = research_topic
                evaluation.evaluation_timestamp = datetime.now()
                
                logger.info(f"Successfully evaluated report for {sequence_name} (Score: {evaluation.overall_score:.1f})")
                return evaluation
            else:
                logger.error(f"Failed to evaluate report for sequence: {sequence_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating report for {sequence_name}: {e}")
            return None

    async def _perform_comparative_analysis(self,
                                          individual_evaluations: List[ReportEvaluation],
                                          research_topic: str,
                                          sequence_names: List[str]) -> ComparativeAnalysis:
        """Perform comprehensive comparative analysis of evaluated reports."""
        try:
            # Prepare evaluation summaries for the prompt
            eval_summaries = []
            for eval in individual_evaluations:
                summary = f"""
**{eval.sequence_name} (Score: {eval.overall_score:.1f}/100)**
- Completeness: {eval.completeness.score:.1f}/10 - {eval.completeness.reasoning}
- Depth: {eval.depth.score:.1f}/10 - {eval.depth.reasoning}
- Coherence: {eval.coherence.score:.1f}/10 - {eval.coherence.reasoning}  
- Innovation: {eval.innovation.score:.1f}/10 - {eval.innovation.reasoning}
- Actionability: {eval.actionability.score:.1f}/10 - {eval.actionability.reasoning}

Key Strengths: {', '.join(eval.key_strengths[:3])}
Key Weaknesses: {', '.join(eval.key_weaknesses[:3])}
Executive Summary: {eval.executive_summary}
"""
                eval_summaries.append(summary)
            
            # Generate comparative analysis prompt
            prompt = EvaluationPrompts.get_comparative_analysis_prompt(
                research_topic=research_topic,
                sequence_names=sequence_names,
                individual_evaluations="\n\n".join(eval_summaries)
            )
            
            # Get model and execute analysis
            model = await self._get_evaluation_model()
            
            analysis = await self._execute_with_retry(
                self._call_evaluation_model,
                model,
                prompt,
                ComparativeAnalysis,
                context="comparative analysis"
            )
            
            if analysis:
                # Set metadata
                analysis.research_topic = research_topic
                analysis.num_reports_compared = len(individual_evaluations)
                analysis.analysis_timestamp = datetime.now()
                
                # Generate pairwise comparisons if needed
                analysis.pairwise_comparisons = await self._generate_pairwise_comparisons(
                    individual_evaluations, research_topic
                )
                
                logger.info(f"Comparative analysis completed. Best sequence: {analysis.best_sequence}")
                return analysis
            else:
                raise RuntimeError("Failed to generate comparative analysis")
                
        except Exception as e:
            logger.error(f"Error in comparative analysis: {e}")
            raise RuntimeError(f"Comparative analysis failed: {e}")

    async def _generate_pairwise_comparisons(self,
                                           evaluations: List[ReportEvaluation],
                                           research_topic: str) -> List[SequenceComparison]:
        """Generate pairwise comparisons between all sequence pairs."""
        if len(evaluations) < 2:
            return []
        
        comparisons = []
        comparison_tasks = []
        
        # Create all pairwise comparison tasks
        for i in range(len(evaluations)):
            for j in range(i + 1, len(evaluations)):
                task = self._compare_pair(
                    evaluations[i], evaluations[j], research_topic
                )
                comparison_tasks.append(task)
        
        # Execute comparisons concurrently
        pairwise_results = await asyncio.gather(*comparison_tasks, return_exceptions=True)
        
        # Process results and filter out failures
        for result in pairwise_results:
            if isinstance(result, SequenceComparison):
                comparisons.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Pairwise comparison failed: {result}")
        
        logger.info(f"Generated {len(comparisons)} pairwise comparisons")
        return comparisons

    async def _compare_pair(self,
                          eval_a: ReportEvaluation,
                          eval_b: ReportEvaluation,
                          research_topic: str) -> Optional[SequenceComparison]:
        """Compare two specific reports in detail."""
        try:
            # This is a simplified comparison based on existing evaluations
            # In a full implementation, you might want more detailed pairwise prompts
            
            winner = eval_a.sequence_name if eval_a.overall_score > eval_b.overall_score else eval_b.sequence_name
            margin = abs(eval_a.overall_score - eval_b.overall_score)
            
            # Identify advantages for each sequence
            a_advantages = []
            b_advantages = []
            
            criteria_map = {
                "completeness": (eval_a.completeness.score, eval_b.completeness.score),
                "depth": (eval_a.depth.score, eval_b.depth.score),
                "coherence": (eval_a.coherence.score, eval_b.coherence.score),
                "innovation": (eval_a.innovation.score, eval_b.innovation.score),
                "actionability": (eval_a.actionability.score, eval_b.actionability.score)
            }
            
            criteria_comparison = {}
            for criterion, (score_a, score_b) in criteria_map.items():
                criteria_comparison[criterion] = {
                    eval_a.sequence_name: score_a,
                    eval_b.sequence_name: score_b
                }
                
                if score_a > score_b:
                    a_advantages.append(f"Superior {criterion} ({score_a:.1f} vs {score_b:.1f})")
                elif score_b > score_a:
                    b_advantages.append(f"Superior {criterion} ({score_b:.1f} vs {score_a:.1f})")
            
            # Generate reasoning based on score differences and strengths
            comparative_reasoning = f"""
{winner} wins with a margin of {margin:.1f} points ({eval_a.overall_score:.1f} vs {eval_b.overall_score:.1f}).
            
Key factors in {winner}'s victory:
- {eval_a.key_strengths[0] if winner == eval_a.sequence_name else eval_b.key_strengths[0]}
- {eval_a.key_strengths[1] if winner == eval_a.sequence_name and len(eval_a.key_strengths) > 1 else eval_b.key_strengths[1] if len(eval_b.key_strengths) > 1 else 'Strong overall performance'}

Areas where the other sequence showed strength:
- {eval_b.key_strengths[0] if winner == eval_a.sequence_name else eval_a.key_strengths[0]}
""".strip()
            
            comparison = SequenceComparison(
                sequence_a=eval_a.sequence_name,
                sequence_b=eval_b.sequence_name,
                winner=winner,
                margin=margin,
                sequence_a_advantages=a_advantages,
                sequence_b_advantages=b_advantages,
                criteria_comparison=criteria_comparison,
                comparative_reasoning=comparative_reasoning,
                use_case_recommendations=[
                    f"Use {eval_a.sequence_name} when prioritizing its key strengths",
                    f"Use {eval_b.sequence_name} when its advantages are more relevant"
                ]
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing {eval_a.sequence_name} vs {eval_b.sequence_name}: {e}")
            return None

    def _compile_evaluation_result(self,
                                 individual_evaluations: List[ReportEvaluation],
                                 comparative_analysis: ComparativeAnalysis,
                                 start_time: float) -> EvaluationResult:
        """Compile the complete evaluation result from individual and comparative analyses."""
        
        # Calculate summary statistics
        scores = [eval.overall_score for eval in individual_evaluations]
        score_statistics = {
            "mean": sum(scores) / len(scores),
            "max": max(scores),
            "min": min(scores),
            "std_dev": (sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / len(scores)) ** 0.5
        }
        
        # Calculate performance gaps
        sorted_evals = sorted(individual_evaluations, key=lambda x: x.overall_score, reverse=True)
        performance_gaps = {}
        
        best_score = sorted_evals[0].overall_score
        for eval in sorted_evals[1:]:
            performance_gaps[eval.sequence_name] = best_score - eval.overall_score
        
        # Identify winning sequence
        winning_sequence = sorted_evals[0].sequence_name
        winning_sequence_score = sorted_evals[0].overall_score
        
        # Extract key differentiators from comparative analysis
        key_differentiators = getattr(comparative_analysis, 'key_differentiators', [
            "Research depth and analytical rigor",
            "Coherent narrative structure", 
            "Actionable recommendations",
            "Novel insights and perspectives"
        ])
        
        # Generate sequence recommendations
        sequence_recommendations = {}
        for eval in individual_evaluations:
            # Create recommendations based on strengths
            if eval.key_strengths:
                top_strength = eval.key_strengths[0]
                sequence_recommendations[eval.sequence_name] = f"Best for: {top_strength}"
            else:
                sequence_recommendations[eval.sequence_name] = "General research tasks"
        
        # Compile final result
        result = EvaluationResult(
            individual_evaluations=individual_evaluations,
            comparative_analysis=comparative_analysis,
            score_statistics=score_statistics,
            performance_gaps=performance_gaps,
            winning_sequence=winning_sequence,
            winning_sequence_score=winning_sequence_score,
            key_differentiators=key_differentiators,
            sequence_recommendations=sequence_recommendations,
            evaluation_model=self.evaluation_model_name,
            processing_time=time.time() - start_time
        )
        
        return result

    async def _execute_with_retry(self,
                                func,
                                *args,
                                context: str = "operation",
                                **kwargs) -> Optional[Any]:
        """Execute a function with retry logic and error handling."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if result is not None:
                    return result
                    
            except Exception as e:
                last_exception = e
                
                # Check if this is a token limit error and handle appropriately
                if is_token_limit_exceeded(e, self.evaluation_model_name):
                    logger.warning(f"Token limit exceeded in {context}, attempt {attempt + 1}")
                    # For token limit errors, you might want to truncate content
                    # This is a simplified approach - you could implement more sophisticated truncation
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                
                # For other errors, log and retry
                logger.warning(f"Attempt {attempt + 1} failed for {context}: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1 * (attempt + 1))  # Linear backoff for other errors
        
        logger.error(f"All {self.max_retries + 1} attempts failed for {context}. Last error: {last_exception}")
        return None

    async def _call_evaluation_model(self,
                                   model: BaseChatModel,
                                   prompt: str,
                                   output_class: type,
                                   **kwargs) -> Optional[Any]:
        """Call the evaluation model with structured output parsing."""
        try:
            # Create structured output model
            structured_model = model.with_structured_output(output_class)
            
            # Execute the model call
            messages = [HumanMessage(content=prompt)]
            response = await structured_model.ainvoke(messages)
            
            # Validate the response
            if isinstance(response, output_class):
                return response
            else:
                logger.warning(f"Received unexpected response type: {type(response)}")
                return None
                
        except ValidationError as e:
            logger.error(f"Validation error in model output: {e}")
            return None
        except Exception as e:
            logger.error(f"Error calling evaluation model: {e}")
            raise

    def _extract_report_content_from_running_report(self, running_report: RunningReport) -> str:
        """Extract readable content from a RunningReport object."""
        try:
            content_parts = []
            
            # Add executive summary if available
            if running_report.executive_summary:
                content_parts.append(f"## Executive Summary\n{running_report.executive_summary}\n")
            
            # Add detailed findings
            if running_report.detailed_findings:
                content_parts.append("## Detailed Findings")
                for i, finding in enumerate(running_report.detailed_findings, 1):
                    content_parts.append(f"### Finding {i}\n{finding}\n")
            
            # Add recommendations
            if running_report.recommendations:
                content_parts.append("## Recommendations")
                for i, rec in enumerate(running_report.recommendations, 1):
                    content_parts.append(f"{i}. {rec}")
                content_parts.append("")
            
            # Add insights from agents
            if running_report.all_insights:
                content_parts.append("## Key Insights")
                for insight in running_report.all_insights:
                    content_parts.append(f"- {insight}")
                content_parts.append("")
            
            # If no structured content, try to extract from agent reports
            if not content_parts and running_report.agent_reports:
                content_parts.append("## Research Content")
                for report in running_report.agent_reports:
                    content_parts.append(f"### {report.agent_name} ({report.agent_type})")
                    content_parts.append(report.research_content)
                    content_parts.append("")
            
            final_content = "\n".join(content_parts)
            
            # Fallback if no content could be extracted
            if not final_content.strip():
                final_content = f"Research Report for: {running_report.research_topic}\n\nStatus: {running_report.completion_status}\nAgents Executed: {running_report.total_agents_executed}"
            
            return final_content
            
        except Exception as e:
            logger.error(f"Error extracting content from RunningReport: {e}")
            return f"Error extracting report content: {e}"

    def compare_reports(self, evaluations: List[ReportEvaluation]) -> Dict[str, Any]:
        """Compare multiple report evaluations and return summary insights.
        
        Args:
            evaluations: List of report evaluations to compare
            
        Returns:
            Dictionary with comparison insights and rankings
        """
        if not evaluations:
            return {"error": "No evaluations provided"}
        
        # Sort by overall score
        sorted_evals = sorted(evaluations, key=lambda x: x.overall_score, reverse=True)
        
        # Create ranking
        ranking = []
        for i, eval in enumerate(sorted_evals, 1):
            ranking.append({
                "rank": i,
                "sequence_name": eval.sequence_name,
                "overall_score": eval.overall_score,
                "key_strengths": eval.key_strengths[:2],  # Top 2 strengths
                "executive_summary": eval.executive_summary[:200] + "..."  # Truncated summary
            })
        
        # Identify criteria leaders
        criteria_leaders = {}
        for criterion in ["completeness", "depth", "coherence", "innovation", "actionability"]:
            best_eval = max(evaluations, key=lambda x: getattr(x, criterion).score)
            criteria_leaders[criterion] = {
                "sequence": best_eval.sequence_name,
                "score": getattr(best_eval, criterion).score
            }
        
        # Calculate performance spread
        scores = [eval.overall_score for eval in evaluations]
        performance_spread = max(scores) - min(scores)
        
        return {
            "winner": sorted_evals[0].sequence_name,
            "winner_score": sorted_evals[0].overall_score,
            "ranking": ranking,
            "criteria_leaders": criteria_leaders,
            "performance_spread": performance_spread,
            "total_evaluated": len(evaluations)
        }

    def determine_winner(self, evaluations: List[ReportEvaluation]) -> Dict[str, Any]:
        """Determine the winning sequence based on evaluation scores.
        
        Args:
            evaluations: List of report evaluations
            
        Returns:
            Dictionary with winner information and justification
        """
        if not evaluations:
            return {"error": "No evaluations provided for winner determination"}
        
        # Find the highest scoring evaluation
        winner = max(evaluations, key=lambda x: x.overall_score)
        
        # Calculate margin over second place
        sorted_evals = sorted(evaluations, key=lambda x: x.overall_score, reverse=True)
        margin = 0.0
        if len(sorted_evals) > 1:
            margin = sorted_evals[0].overall_score - sorted_evals[1].overall_score
        
        # Identify what made this sequence win
        winning_factors = []
        for criterion in ["completeness", "depth", "coherence", "innovation", "actionability"]:
            criterion_score = getattr(winner, criterion).score
            # Check if this criterion score is significantly higher than others
            other_scores = [getattr(eval, criterion).score for eval in evaluations if eval != winner]
            if other_scores and criterion_score > max(other_scores):
                winning_factors.append(f"Superior {criterion} ({criterion_score:.1f})")
        
        return {
            "winner": winner.sequence_name,
            "winning_score": winner.overall_score,
            "margin_over_second": margin,
            "winning_factors": winning_factors,
            "key_strengths": winner.key_strengths,
            "recommendation": f"Use {winner.sequence_name} for similar research topics",
            "confidence": winner.confidence_level
        }