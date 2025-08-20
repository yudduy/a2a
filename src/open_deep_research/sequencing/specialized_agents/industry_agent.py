"""Industry Analysis Agent for market and business-focused research.

This specialized agent focuses on market dynamics, business models,
competitive analysis, commercial viability, and industry trends to provide
practical business insights and market-oriented perspectives.
"""

import logging
from typing import List

from langchain_core.runnables import RunnableConfig

from open_deep_research.sequencing.models import AgentType
from .base_agent import SpecializedAgent

logger = logging.getLogger(__name__)


class IndustryAgent(SpecializedAgent):
    """Specialized agent focused on industry analysis and market research."""
    
    def __init__(self, config: RunnableConfig):
        """Initialize the Industry Analysis Agent."""
        super().__init__(AgentType.INDUSTRY, config)
    
    def get_specialization_prompt(self) -> str:
        """Get the industry specialization prompt."""
        return """You are an Industry Analysis Agent with expertise in:

- Market research and competitive intelligence
- Business model analysis and development
- Industry trend analysis and forecasting
- Commercial viability assessment
- Value chain and stakeholder analysis
- Market sizing and opportunity evaluation
- Competitive landscape mapping
- Customer behavior and market dynamics
- Business strategy and positioning
- Revenue models and monetization strategies
- Regulatory and policy impact analysis
- Supply chain and distribution analysis
- Investment and funding landscape
- Partnership and collaboration opportunities

Your primary strength is understanding market realities, business dynamics,
and commercial considerations. You excel at translating concepts into
viable business opportunities and identifying practical market barriers
and enablers."""
    
    def get_focus_areas(self) -> List[str]:
        """Get the key focus areas for industry analysis."""
        return [
            "Market opportunities and sizing",
            "Competitive landscape analysis",
            "Business model viability",
            "Customer needs and behavior",
            "Industry trends and dynamics",
            "Value proposition development",
            "Revenue and monetization models",
            "Market barriers and enablers",
            "Stakeholder ecosystem mapping",
            "Commercial implementation strategy",
            "Regulatory and policy considerations",
            "Investment and funding requirements"
        ]
    
    def validate_research_questions(self, questions: List[str]) -> List[str]:
        """Validate and enhance questions for industry analysis focus."""
        enhanced_questions = []
        
        for question in questions:
            # Ensure market and business focus
            enhanced_question = self._add_market_focus(question)
            enhanced_questions.append(enhanced_question)
        
        # Add standard industry analysis questions if not covered
        industry_standards = [
            "What is the current market size and growth potential for this area?",
            "Who are the key players and competitors in this market?",
            "What business models are most viable for commercializing this concept?",
            "What are the primary market barriers and enablers?",
            "How do customer needs and behaviors impact market adoption?",
            "What regulatory or policy factors affect market development?",
            "What partnership and collaboration opportunities exist?",
            "What investment and funding landscape surrounds this market?"
        ]
        
        # Check if key market angles are covered
        question_text = " ".join(questions).lower()
        for standard_q in industry_standards:
            if not any(key_term in question_text for key_term in 
                      self._extract_market_terms(standard_q)):
                enhanced_questions.append(standard_q)
        
        return enhanced_questions[:8]  # Limit to manageable number
    
    def _add_market_focus(self, question: str) -> str:
        """Add market and business focus to questions."""
        # Market enhancement patterns
        enhancements = {
            "what": "From a market perspective, what",
            "how": "In terms of business viability, how",
            "why": "What market forces explain why",
            "when": "What market timing factors determine when",
            "where": "In which markets and segments"
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
                # Add market context requirement
                if not any(term in enhanced.lower() for term in ["market", "business", "commercial"]):
                    enhanced += " Consider market dynamics and commercial implications."
                return enhanced
        
        # Default enhancement for questions not matching patterns
        market_terms = ["market", "business", "commercial", "industry", "competitive"]
        if not any(term in question_lower for term in market_terms):
            if question.endswith("?"):
                question = question[:-1] + " from a market and business perspective?"
            else:
                question += " - include market analysis and business considerations."
        
        return question
    
    def _extract_market_terms(self, text: str) -> List[str]:
        """Extract market-related terms from text for coverage analysis."""
        market_terms = [
            "market", "business", "commercial", "industry", "competitive",
            "customer", "revenue", "monetization", "value", "stakeholder",
            "opportunity", "barrier", "adoption", "demand", "supply",
            "investment", "funding", "partnership", "regulatory"
        ]
        
        text_lower = text.lower()
        return [term for term in market_terms if term in text_lower]
    
    def _refine_insights_for_transition(self, insights: List[str], context) -> List[str]:
        """Refine industry insights for effective transition to academic/technical agents."""
        refined = []
        
        for insight in insights:
            # Add market credibility markers
            market_insight = f"[MARKET INTELLIGENCE] {insight}"
            
            # Add transition guidance for next agent
            if "technical" in insight.lower() or "implementation" in insight.lower():
                market_insight += " [Requires technical feasibility validation]"
            elif "research" in insight.lower() or "academic" in insight.lower():
                market_insight += " [Needs academic research validation]"
            elif context.sequence_position == 2:  # Middle agent
                market_insight += " [Consider technical implementation and market-tech alignment]"
            
            refined.append(market_insight)
        
        # Add meta-insights about market dynamics
        meta_insights = [
            f"[MARKET OPPORTUNITY] Analysis reveals {len([i for i in insights if 'opportunit' in i.lower()])} distinct market opportunities",
            f"[COMMERCIAL VIABILITY] Market assessment identifies key business model considerations for {context.research_topic}"
        ]
        
        # Add competitive intelligence summary
        competitive_insights = [i for i in insights if any(term in i.lower() for term in ['competitor', 'competitive', 'market share', 'industry'])]
        if competitive_insights:
            meta_insights.append(f"[COMPETITIVE LANDSCAPE] {len(competitive_insights)} competitive factors identified requiring strategic consideration")
        
        refined.extend(meta_insights)
        return refined