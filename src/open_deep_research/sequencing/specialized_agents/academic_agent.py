"""Academic Research Agent for theory and research-focused analysis.

This specialized agent focuses on academic literature, theoretical frameworks,
research methodologies, and scholarly analysis to provide foundational
understanding and research-backed insights.
"""

import logging
from typing import List

from langchain_core.runnables import RunnableConfig

from open_deep_research.sequencing.models import AgentType

from .base_agent import SpecializedAgent

logger = logging.getLogger(__name__)


class AcademicAgent(SpecializedAgent):
    """Specialized agent focused on academic research and theoretical analysis."""
    
    def __init__(self, config: RunnableConfig):
        """Initialize the Academic Research Agent."""
        super().__init__(AgentType.ACADEMIC, config)
    
    def get_specialization_prompt(self) -> str:
        """Get the academic specialization prompt."""
        return """You are an Academic Research Agent with expertise in:

- Scholarly literature review and analysis
- Theoretical framework development and application
- Research methodology design and evaluation
- Academic database searching and citation analysis
- Peer-reviewed research synthesis
- Gap analysis in academic literature
- Research validation and experimental design
- Academic writing and argumentation
- Interdisciplinary research connections
- Evidence-based analysis and conclusions

Your primary strength is grounding research in solid theoretical foundations,
identifying gaps in current academic understanding, and providing scholarly
rigor to the research process. You excel at connecting concepts across
disciplines and building robust theoretical frameworks."""
    
    def get_focus_areas(self) -> List[str]:
        """Get the key focus areas for academic research."""
        return [
            "Theoretical frameworks",
            "Peer-reviewed literature", 
            "Research methodologies",
            "Academic gaps and opportunities",
            "Scholarly debates and consensus",
            "Evidence-based analysis",
            "Cross-disciplinary connections",
            "Research validation approaches",
            "Academic impact assessment",
            "Foundational concepts and principles"
        ]
    
    def validate_research_questions(self, questions: List[str]) -> List[str]:
        """Validate and enhance questions for academic research focus."""
        enhanced_questions = []
        
        for question in questions:
            # Ensure academic rigor and theoretical grounding
            enhanced_question = self._add_academic_rigor(question)
            enhanced_questions.append(enhanced_question)
        
        # Add standard academic research questions if not covered
        academic_standards = [
            "What does the current peer-reviewed literature reveal about this topic?",
            "What theoretical frameworks are most applicable to this research area?",
            "What gaps exist in current academic understanding?",
            "What research methodologies would be most appropriate for investigating this topic?",
            "How do different academic disciplines approach this subject?"
        ]
        
        # Check if key academic angles are covered
        question_text = " ".join(questions).lower()
        for standard_q in academic_standards:
            if not any(key_term in question_text for key_term in 
                      self._extract_key_terms(standard_q)):
                enhanced_questions.append(standard_q)
        
        return enhanced_questions[:8]  # Limit to manageable number
    
    def _add_academic_rigor(self, question: str) -> str:
        """Add academic rigor and theoretical grounding to questions."""
        # Academic enhancement patterns
        enhancements = {
            "what": "What does academic research reveal about",
            "how": "According to scholarly literature, how",
            "why": "What theoretical explanations exist for why",
            "when": "What is the academic consensus on when",
            "where": "What research indicates about where"
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
                # Add academic context requirement
                if "research" not in enhanced.lower():
                    enhanced += " Include relevant academic research and theoretical perspectives."
                return enhanced
        
        # Default enhancement for questions not matching patterns
        if "research" not in question_lower and "academic" not in question_lower:
            if question.endswith("?"):
                question = question[:-1] + " based on academic research and theoretical frameworks?"
            else:
                question += " - provide academic research foundation."
        
        return question
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for coverage analysis."""
        academic_terms = [
            "literature", "research", "theory", "framework", "methodology",
            "peer-reviewed", "scholarly", "academic", "study", "analysis",
            "evidence", "empirical", "theoretical", "conceptual", "discipline"
        ]
        
        text_lower = text.lower()
        return [term for term in academic_terms if term in text_lower]
    
    def _refine_insights_for_transition(self, insights: List[str], context) -> List[str]:
        """Refine academic insights for effective transition to industry/technical agents."""
        refined = []
        
        for insight in insights:
            # Add academic credibility markers
            academic_insight = f"[ACADEMIC FOUNDATION] {insight}"
            
            # Add transition guidance for next agent
            if context.sequence_position == 1:  # First agent
                if "market" in insight.lower() or "business" in insight.lower():
                    academic_insight += " [Requires market validation and business model analysis]"
                elif "technical" in insight.lower() or "implementation" in insight.lower():
                    academic_insight += " [Needs technical feasibility assessment]"
                else:
                    academic_insight += " [Consider practical applications and market implications]"
            
            refined.append(academic_insight)
        
        # Add meta-insights about research gaps and opportunities
        meta_insights = [
            f"[RESEARCH GAP] Based on literature review, there appears to be limited academic work on practical implementation aspects of {context.research_topic}",
            f"[THEORETICAL FOUNDATION] The academic foundation provides {len(insights)} key theoretical anchors for further investigation"
        ]
        
        refined.extend(meta_insights)
        return refined