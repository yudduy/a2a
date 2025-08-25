"""Agent registry and management system for sequential multi-agent workflows."""

from .registry import AgentRegistry
from .loader import AgentLoader
from .completion_detector import CompletionDetector

__all__ = [
    "AgentRegistry",
    "AgentLoader", 
    "CompletionDetector"
]