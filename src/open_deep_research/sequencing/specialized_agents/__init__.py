"""Specialized research agents for the sequential optimization framework.

This module provides specialized agents that focus on different aspects of research:
- AcademicAgent: Theory and research-focused analysis
- IndustryAgent: Market and business-focused analysis  
- TechnicalTrendsAgent: Implementation and future technology analysis

Each agent is designed to provide unique value through their specialized perspective
while preventing cognitive offloading and ensuring focused research execution.
"""

from .academic_agent import AcademicAgent
from .base_agent import ResearchContext, SpecializedAgent
from .industry_agent import IndustryAgent
from .technical_trends_agent import TechnicalTrendsAgent

__all__ = [
    "SpecializedAgent",
    "ResearchContext", 
    "AcademicAgent",
    "IndustryAgent",
    "TechnicalTrendsAgent"
]