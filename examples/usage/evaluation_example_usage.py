"""Example usage and integration demonstrations for the LLM Judge system."""

from datetime import datetime
from typing import Dict

from langchain_core.runnables import RunnableConfig

from open_deep_research.evaluation import LLMJudge
from open_deep_research.state import AgentExecutionReport, RunningReport


class LLMJudgeExamples:
    """Collection of examples demonstrating LLM Judge system usage."""

    @staticmethod
    def create_sample_config() -> RunnableConfig:
        """Create a sample configuration for testing."""
        return RunnableConfig(
            configurable={
                "evaluation_model": "anthropic:claude-3-5-sonnet",
                "evaluation_model_max_tokens": 8192,
                "apiKeys": {
                    "ANTHROPIC_API_KEY": "your_api_key_here"
                }
            }
        )

    @staticmethod
    def create_sample_reports() -> Dict[str, str]:
        """Create sample research reports for testing."""
        return {
            "theory_first": """
# AI Safety Research Report

## Executive Summary
This comprehensive analysis examines the current state of AI safety research, focusing on alignment challenges, interpretability advances, and regulatory frameworks. The research reveals significant progress in technical safety measures while highlighting persistent challenges in value alignment and scalable oversight.

## Detailed Analysis

### Technical Safety Foundations
Current AI safety research has established several foundational approaches including constitutional AI, reward modeling, and interpretability techniques. These methods show promise but face scalability challenges as model capabilities increase.

### Alignment Challenges
The core challenge of ensuring AI systems pursue intended objectives remains partially unsolved. Recent advances in RLHF and constitutional training provide partial solutions but require significant refinement for AGI-level systems.

### Regulatory Landscape
Emerging regulatory frameworks show promise but lack technical depth required for effective governance of advanced AI systems.

## Recommendations
1. Increase investment in interpretability research
2. Develop better evaluation frameworks for alignment
3. Foster collaboration between technical researchers and policymakers
4. Prioritize research on scalable oversight mechanisms
5. Create standardized safety testing protocols
            """,
            
            "market_first": """
# AI Safety Market Analysis Report

## Executive Summary
The AI safety market is experiencing rapid growth driven by regulatory requirements and corporate risk management needs. Current market size is estimated at $2.1B with projected CAGR of 45% through 2028. Key drivers include regulatory compliance, reputational risk, and technical necessity.

## Market Dynamics

### Current Market Size and Growth
The global AI safety market has reached $2.1 billion in 2024, with enterprise adoption accelerating due to regulatory pressures and high-profile AI incidents. Major segments include safety tooling ($800M), consulting services ($600M), and compliance solutions ($700M).

### Key Players and Solutions
Leading companies include Anthropic (constitutional AI), OpenAI (safety research), and emerging startups focusing on interpretability and alignment solutions. Enterprise adoption is primarily driven by financial services and healthcare sectors.

### Investment Patterns
Venture capital investment in AI safety reached $450M in 2024, with significant interest in commercial interpretability tools and automated safety testing platforms.

## Technical Implementation Trends
Organizations are prioritizing practical safety measures including bias detection, output monitoring, and human oversight systems over theoretical alignment research.

## Recommendations
1. Focus on commercially viable safety solutions
2. Target high-regulatory industries for early adoption
3. Develop standardized safety metrics and benchmarks
4. Build integrated safety platforms rather than point solutions
5. Establish partnerships with major cloud providers
            """,
            
            "technical_first": """
# Technical Deep Dive: AI Safety Implementation

## Executive Summary
This technical analysis explores current AI safety implementation approaches, including mechanistic interpretability, constitutional training methods, and scalable oversight architectures. Key findings show significant progress in localized interpretability techniques while system-wide alignment remains challenging.

## Technical Architecture Analysis

### Mechanistic Interpretability Advances
Recent breakthroughs in activation patching, causal scrubbing, and circuit analysis have enabled deeper understanding of transformer internals. However, these techniques primarily work on smaller models and don't yet scale to production systems.

### Constitutional AI Implementation
Technical implementation of constitutional AI involves multi-stage training: supervised fine-tuning on constitutional principles, preference model training, and reinforcement learning from AI feedback (RLAIF). Current implementations show 40-60% improvement in harmlessness metrics.

### Scalable Oversight Systems
Implementation of scalable oversight requires automated red-teaming, continuous monitoring, and hierarchical safety checks. Current architectures achieve ~85% detection rates for problematic outputs but struggle with novel attack vectors.

### Safety Evaluation Frameworks
Technical evaluation requires comprehensive testing across capability, alignment, and robustness dimensions. Current frameworks include red-teaming protocols, adversarial testing, and capability evaluations.

## Implementation Challenges
- Computational overhead of safety measures (15-30% latency increase)
- Integration complexity with existing ML pipelines  
- Limited effectiveness against sophisticated attacks
- Difficulty measuring alignment in complex domains

## Recommendations
1. Implement staged safety architectures with multiple checkpoints
2. Deploy automated monitoring with human oversight triggers
3. Use ensemble methods for improved robustness
4. Integrate safety evaluation into CI/CD pipelines
5. Develop domain-specific safety protocols
            """
        }

    @staticmethod 
    def create_sample_running_reports() -> Dict[str, RunningReport]:
        """Create sample RunningReport objects for testing."""
        # Create sample agent execution reports
        theory_agent_report = AgentExecutionReport(
            agent_name="theory_researcher",
            agent_type="academic_theory",
            execution_start=datetime.now(),
            execution_end=datetime.now(),
            execution_duration=120.5,
            insights=["Constitutional AI shows promise for alignment", "Interpretability research is making steady progress"],
            research_content="Comprehensive analysis of AI safety theoretical foundations...",
            questions_addressed=["What are the core challenges in AI alignment?", "How effective are current interpretability techniques?"],
            completion_confidence=0.85,
            insight_quality_score=0.9,
            research_depth_score=0.88,
            handoff_context={"focus_areas": ["alignment", "interpretability"], "key_findings": ["constitutional_ai_effective"]},
            suggested_next_questions=["How can these theories be implemented at scale?"]
        )
        
        market_agent_report = AgentExecutionReport(
            agent_name="market_analyst",
            agent_type="market_research",
            execution_start=datetime.now(),
            execution_end=datetime.now(),
            execution_duration=95.0,
            insights=["AI safety market growing at 45% CAGR", "Enterprise adoption driven by compliance"],
            research_content="Analysis of AI safety market dynamics and commercial opportunities...",
            questions_addressed=["What is the current market size?", "Who are the key players?"],
            completion_confidence=0.92,
            insight_quality_score=0.87,
            research_depth_score=0.83,
            handoff_context={"market_size": "$2.1B", "growth_rate": "45%", "key_sectors": ["fintech", "healthcare"]},
            suggested_next_questions=["What technical solutions are most commercially viable?"]
        )
        
        reports = {
            "theory_first": RunningReport(
                research_topic="AI Safety Research Analysis",
                sequence_name="theory_first",
                start_time=datetime.now(),
                agent_reports=[theory_agent_report],
                all_insights=["Constitutional AI shows promise", "Market adoption accelerating"],
                insight_connections=[{"from": "theory", "to": "market", "connection": "theoretical advances enable commercial products"}],
                executive_summary="Comprehensive AI safety analysis covering theoretical foundations and practical implementation challenges.",
                detailed_findings=[
                    "Current AI safety research has established foundational approaches but faces scalability challenges",
                    "Constitutional AI and RLHF show promise for near-term alignment solutions",
                    "Interpretability techniques are advancing but remain limited to smaller models"
                ],
                recommendations=[
                    "Increase investment in scalable interpretability research",
                    "Develop standardized safety evaluation frameworks", 
                    "Foster collaboration between researchers and practitioners"
                ],
                total_agents_executed=1,
                total_execution_time=120.5,
                completion_status="completed"
            ),
            
            "market_first": RunningReport(
                research_topic="AI Safety Research Analysis",
                sequence_name="market_first", 
                start_time=datetime.now(),
                agent_reports=[market_agent_report],
                all_insights=["Market growing at 45% CAGR", "Compliance driving adoption"],
                insight_connections=[{"from": "market", "to": "technical", "connection": "market demand shapes technical priorities"}],
                executive_summary="Market-focused analysis of AI safety landscape with emphasis on commercial opportunities and adoption patterns.",
                detailed_findings=[
                    "AI safety market reached $2.1B in 2024 with strong growth trajectory",
                    "Enterprise adoption primarily driven by regulatory compliance needs",
                    "Key opportunities in safety tooling and automated monitoring solutions"
                ],
                recommendations=[
                    "Focus on commercially viable safety solutions",
                    "Target high-regulatory industries for early adoption",
                    "Develop integrated safety platforms"
                ],
                total_agents_executed=1,
                total_execution_time=95.0,
                completion_status="completed"
            )
        }
        
        return reports

    @staticmethod
    async def example_basic_evaluation():
        """Basic example of evaluating multiple research reports."""
        # Setup
        config = LLMJudgeExamples.create_sample_config()
        judge = LLMJudge(config=config)
        
        # Sample reports
        reports = LLMJudgeExamples.create_sample_reports()
        research_topic = "AI Safety Research and Implementation"
        
        try:
            # Evaluate reports
            result = await judge.evaluate_reports(
                reports=reports,
                research_topic=research_topic
            )
            
            # Display results
            
            for i, eval in enumerate(sorted(result.individual_evaluations, 
                                          key=lambda x: x.overall_score, reverse=True), 1):
                pass
            
            for differentiator in result.key_differentiators:
                pass
            
            for sequence, recommendation in result.sequence_recommendations.items():
                pass
            
            return result
            
        except Exception:
            return None

    @staticmethod
    async def example_single_report_evaluation():
        """Example of evaluating a single research report."""
        config = LLMJudgeExamples.create_sample_config()
        judge = LLMJudge(config=config)
        
        # Evaluate single report
        reports = LLMJudgeExamples.create_sample_reports()
        single_report = reports["theory_first"]
        
        try:
            result = await judge.evaluate_single_report(
                report=single_report,
                research_topic="AI Safety Research Analysis",
                sequence_name="theory_first"
            )
            
            if result:
                
                
                for strength in result.key_strengths[:3]:
                    pass
                
                for weakness in result.key_weaknesses[:3]:
                    pass
                
                
            return result
            
        except Exception:
            return None

    @staticmethod
    async def example_running_report_evaluation():
        """Example of evaluating RunningReport objects."""
        config = LLMJudgeExamples.create_sample_config()
        judge = LLMJudge(config=config)
        
        # Create RunningReport objects
        running_reports = LLMJudgeExamples.create_sample_running_reports()
        
        try:
            result = await judge.evaluate_reports(
                reports=running_reports,
                research_topic="AI Safety Research Analysis"
            )
            
            
            ca = result.comparative_analysis
            
            for criterion, leader in ca.criteria_leaders.items():
                pass
            
            return result
            
        except Exception:
            return None

    @staticmethod
    async def example_performance_comparison():
        """Example showing detailed performance comparison features."""
        config = LLMJudgeExamples.create_sample_config()
        judge = LLMJudge(config=config)
        
        reports = LLMJudgeExamples.create_sample_reports()
        
        try:
            result = await judge.evaluate_reports(
                reports=reports,
                research_topic="AI Safety Research and Implementation"
            )
            
            # Use helper methods
            result.get_winner_details()
            criteria_rankings = result.get_criteria_rankings()
            
            
            for criterion, ranking in criteria_rankings.items():
                ranking[0]
            
            
            return result
            
        except Exception:
            return None

    @staticmethod
    def example_configuration_options():
        """Show different configuration options for the LLM Judge."""
        # Basic configuration
        basic_config = RunnableConfig(
            configurable={
                "evaluation_model": "anthropic:claude-3-5-sonnet",
                "evaluation_model_max_tokens": 4096
            }
        )
        
        # Advanced configuration with custom settings
        RunnableConfig(
            configurable={
                "evaluation_model": "openai:gpt-4.1",
                "evaluation_model_max_tokens": 8192,
                "max_structured_output_retries": 5,
                "apiKeys": {
                    "OPENAI_API_KEY": "your_key_here",
                    "ANTHROPIC_API_KEY": "backup_key_here"
                }
            }
        )
        
        
        
        # Show different initialization options
        
        # Option 1: Use config model
        LLMJudge(config=basic_config)
        
        # Option 2: Override with specific model
        LLMJudge(config=basic_config, evaluation_model="openai:gpt-4o")
        
        # Option 3: Custom retry settings
        LLMJudge(config=basic_config, max_retries=5)


async def run_all_examples():
    """Run all example demonstrations."""
    examples = LLMJudgeExamples()
    
    # Run examples in sequence
    await examples.example_basic_evaluation()
    await examples.example_single_report_evaluation() 
    await examples.example_running_report_evaluation()
    await examples.example_performance_comparison()
    examples.example_configuration_options()
    


if __name__ == "__main__":
    """Run examples when script is executed directly."""
    
    # Uncomment to run examples (requires API keys)
    # asyncio.run(run_all_examples())