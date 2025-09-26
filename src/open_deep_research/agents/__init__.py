"""Agent registry and management system for sequential multi-agent workflows."""

from .completion_detector import CompletionDetector
from .loader import AgentLoader
from .registry import AgentRegistry

__all__ = [
    "AgentRegistry",
    "AgentLoader", 
    "CompletionDetector"
]