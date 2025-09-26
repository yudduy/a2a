"""Technical Trends Agent for implementation and future technology analysis.

This specialized agent focuses on technical feasibility, implementation pathways,
emerging technologies, future trends, and practical technical considerations
to provide forward-looking and implementation-oriented insights.
"""

import logging
from typing import List

from langchain_core.runnables import RunnableConfig

from open_deep_research.sequencing.models import AgentType

from .base_agent import SpecializedAgent

logger = logging.getLogger(__name__)


class TechnicalTrendsAgent(SpecializedAgent):
    """Specialized agent focused on technical trends and implementation analysis."""
    
    def __init__(self, config: RunnableConfig):
        """Initialize the Technical Trends Agent."""
        super().__init__(AgentType.TECHNICAL_TRENDS, config)
    
    def get_specialization_prompt(self) -> str:
        """Get the technical trends specialization prompt."""
        return """You are a Technical Trends Agent with expertise in:

- Emerging technology analysis and forecasting
- Technical feasibility assessment and validation
- Implementation pathway design and optimization
- Technology roadmap development and analysis
- Innovation cycle understanding and prediction
- Technical architecture and systems design
- Scalability and performance considerations
- Technology convergence and integration patterns
- Future trend extrapolation and scenario planning
- Technical risk assessment and mitigation
- Standards and protocol development tracking
- Open source and proprietary technology evaluation
- Development timeline and milestone planning
- Technical adoption and diffusion patterns

Your primary strength is understanding how technologies evolve, converge,
and can be practically implemented. You excel at connecting current
capabilities with future possibilities and identifying realistic
implementation strategies."""
    
    def get_focus_areas(self) -> List[str]:
        """Get the key focus areas for technical trends analysis."""
        return [
            "Emerging technology landscape",
            "Technical feasibility assessment",
            "Implementation pathways and strategies",
            "Technology convergence patterns",
            "Future trend projections",
            "Scalability and performance optimization",
            "Technical architecture design",
            "Innovation timeline forecasting",
            "Technology adoption patterns",
            "Technical risk and mitigation strategies",
            "Standards and protocol evolution",
            "Development and deployment strategies"
        ]
    
    def validate_research_questions(self, questions: List[str]) -> List[str]:
        """Validate and enhance questions for technical trends focus."""
        enhanced_questions = []
        
        for question in questions:
            # Ensure technical and implementation focus
            enhanced_question = self._add_technical_focus(question)
            enhanced_questions.append(enhanced_question)
        
        # Add standard technical analysis questions if not covered
        technical_standards = [
            "What emerging technologies are most relevant to this area?",
            "What are the key technical challenges and how can they be overcome?",
            "What implementation pathways are most feasible and scalable?",
            "How do current technology trends align with this research direction?",
            "What technical architecture would best support this application?",
            "What are the scalability and performance considerations?",
            "What future technology developments could impact this field?",
            "What technical standards and protocols are relevant?",
            "What are realistic development timelines and milestones?",
            "How do different technical approaches compare in terms of feasibility?"
        ]
        
        # Check if key technical angles are covered
        question_text = " ".join(questions).lower()
        for standard_q in technical_standards:
            if not any(key_term in question_text for key_term in 
                      self._extract_technical_terms(standard_q)):
                enhanced_questions.append(standard_q)
        
        return enhanced_questions[:8]  # Limit to manageable number
    
    def _add_technical_focus(self, question: str) -> str:
        """Add technical and implementation focus to questions."""
        # Technical enhancement patterns
        enhancements = {
            "what": "From a technical implementation perspective, what",
            "how": "What technical approaches can be used to",
            "why": "What technical factors explain why",
            "when": "What technology timeline suggests when",
            "where": "In which technical domains and architectures"
        }
        
        question_lower = question.lower().strip()
        
        # Apply enhancements based on question type
        for trigger, enhancement in enhancements.items():
            if question_lower.startswith(trigger):
                enhanced = question.replace(
                    question.split()[0], 
                    enhancement, 
                    1
                )
                # Add technical context requirement
                if not any(term in enhanced.lower() for term in ["technical", "technology", "implementation"]):
                    enhanced += " Focus on technical feasibility and implementation considerations."
                return enhanced
        
        # Default enhancement for questions not matching patterns
        technical_terms = ["technical", "technology", "implementation", "architecture", "scalability"]
        if not any(term in question_lower for term in technical_terms):
            if question.endswith("?"):
                question = question[:-1] + " with focus on technical implementation and feasibility?"
            else:
                question += " - include technical analysis and implementation pathways."
        
        return question
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text for coverage analysis."""
        technical_terms = [
            "technical", "technology", "implementation", "architecture", "scalability",
            "performance", "feasibility", "emerging", "trends", "innovation",
            "development", "deployment", "integration", "convergence", "standards",
            "protocol", "system", "platform", "framework", "infrastructure"
        ]
        
        text_lower = text.lower()
        return [term for term in technical_terms if term in text_lower]
    
    def _refine_insights_for_transition(self, insights: List[str], context) -> List[str]:
        """Refine technical insights for synthesis and final analysis."""
        refined = []
        
        for insight in insights:
            # Add technical credibility markers
            technical_insight = f"[TECHNICAL ANALYSIS] {insight}"
            
            # Add implementation guidance since this is often the final agent
            if "future" in insight.lower() or "trend" in insight.lower():
                technical_insight += " [Long-term implementation consideration]"
            elif "feasible" in insight.lower() or "implement" in insight.lower():
                technical_insight += " [Near-term implementation opportunity]"
            elif "challenge" in insight.lower() or "barrier" in insight.lower():
                technical_insight += " [Implementation risk requiring mitigation]"
            
            refined.append(technical_insight)
        
        # Add meta-insights about technical landscape
        meta_insights = [
            f"[TECHNICAL FEASIBILITY] Analysis identifies {len([i for i in insights if 'feasib' in i.lower()])} feasibility factors",
            f"[IMPLEMENTATION PATHWAY] Technical assessment reveals key implementation considerations for {context.research_topic}"
        ]
        
        # Add future trends summary
        trend_insights = [i for i in insights if any(term in i.lower() for term in ['trend', 'future', 'emerging', 'evolution'])]
        if trend_insights:
            meta_insights.append(f"[FUTURE OUTLOOK] {len(trend_insights)} future trend factors identified shaping long-term development")
        
        # Add technical complexity assessment
        complexity_indicators = [i for i in insights if any(term in i.lower() for term in ['complex', 'challenge', 'difficult', 'barrier'])]
        if complexity_indicators:
            meta_insights.append(f"[TECHNICAL COMPLEXITY] {len(complexity_indicators)} complexity factors require careful implementation planning")
        
        refined.extend(meta_insights)
        return refined