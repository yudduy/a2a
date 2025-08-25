"""Agent capability mapper for extracting and formatting agent descriptions.

This module provides functionality to analyze agent configurations and extract
capabilities in a format suitable for LLM reasoning about sequence generation.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from .sequence_models import AgentCapability
from ..agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


class AgentCapabilityMapper:
    """Maps agent configurations to structured capability descriptions."""
    
    def __init__(self, agent_registry: Optional[AgentRegistry] = None):
        """Initialize the capability mapper.
        
        Args:
            agent_registry: Optional agent registry. If None, creates a new one.
        """
        self.agent_registry = agent_registry or AgentRegistry()
        
        # Common research expertise keywords for categorization
        self.expertise_keywords = {
            "academic": ["academic", "scholarly", "research", "literature", "peer-reviewed", "citation"],
            "technical": ["technical", "engineering", "implementation", "architecture", "code", "system"],
            "market": ["market", "business", "commercial", "industry", "economic", "financial"],
            "data": ["data", "analytics", "statistics", "quantitative", "metrics", "analysis"],
            "regulatory": ["regulatory", "compliance", "legal", "policy", "governance", "standards"],
            "user": ["user", "customer", "experience", "interface", "usability", "human"],
            "security": ["security", "privacy", "risk", "threat", "vulnerability", "safety"],
            "innovation": ["innovation", "emerging", "future", "trends", "cutting-edge", "novel"]
        }
    
    def _extract_expertise_areas(self, agent_config: Dict[str, Any]) -> List[str]:
        """Extract expertise areas from agent configuration.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            List of expertise area strings
        """
        expertise_areas = []
        
        # Check for explicit expertise_areas field (from markdown frontmatter)
        if "expertise_areas" in agent_config:
            if isinstance(agent_config["expertise_areas"], list):
                expertise_areas.extend(agent_config["expertise_areas"])
            elif isinstance(agent_config["expertise_areas"], str):
                expertise_areas.append(agent_config["expertise_areas"])
        
        # Check for explicit expertise field (legacy support)
        if "expertise" in agent_config and not expertise_areas:
            if isinstance(agent_config["expertise"], list):
                expertise_areas.extend(agent_config["expertise"])
            elif isinstance(agent_config["expertise"], str):
                expertise_areas.append(agent_config["expertise"])
        
        # If we have explicit expertise areas, use them and skip inference
        if expertise_areas:
            return expertise_areas
        
        # Extract from description, system_prompt, and other text fields
        text_fields = []
        if "description" in agent_config:
            text_fields.append(agent_config["description"].lower())
        if "system_prompt" in agent_config:
            text_fields.append(agent_config["system_prompt"].lower())
        if "prompt" in agent_config:
            text_fields.append(agent_config["prompt"].lower())
        if "instructions" in agent_config:
            text_fields.append(agent_config["instructions"].lower())
        
        combined_text = " ".join(text_fields)
        
        # Categorize based on keywords
        for category, keywords in self.expertise_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                if category not in [area.lower() for area in expertise_areas]:
                    expertise_areas.append(category.title())
        
        # If no expertise found, use "General Research"
        if not expertise_areas:
            expertise_areas = ["General Research"]
        
        return expertise_areas
    
    def _extract_description(self, agent_config: Dict[str, Any]) -> str:
        """Extract agent description from configuration.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            Agent description string
        """
        if "description" in agent_config:
            return agent_config["description"]
        
        if "system_prompt" in agent_config:
            # Extract first sentence or first 200 chars of system prompt as description
            prompt = agent_config["system_prompt"]
            first_sentence = prompt.split('.')[0]
            if len(first_sentence) < 300:
                return first_sentence + "."
            else:
                return prompt[:200] + "..."
        
        if "prompt" in agent_config:
            # Extract first sentence or first 100 chars of prompt as description
            prompt = agent_config["prompt"]
            first_sentence = prompt.split('.')[0]
            if len(first_sentence) < 200:
                return first_sentence + "."
            else:
                return prompt[:100] + "..."
        
        return "Specialized research agent"
    
    def _extract_use_cases(self, agent_config: Dict[str, Any]) -> List[str]:
        """Extract typical use cases from agent configuration.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            List of use case strings
        """
        use_cases = []
        
        # Check for explicit use_cases field
        if "use_cases" in agent_config:
            if isinstance(agent_config["use_cases"], list):
                use_cases.extend(agent_config["use_cases"])
            elif isinstance(agent_config["use_cases"], str):
                use_cases.append(agent_config["use_cases"])
        
        # Check for explicit examples field (from markdown frontmatter)
        if "examples" in agent_config and isinstance(agent_config["examples"], list):
            use_cases.extend(agent_config["examples"])
        
        # Infer from expertise areas only if no explicit use cases or examples found
        if not use_cases:
            expertise_areas = self._extract_expertise_areas(agent_config)
            
            for area in expertise_areas:
                area_lower = area.lower()
                if "academic" in area_lower:
                    use_cases.append("Literature reviews and academic research")
                elif "technical" in area_lower:
                    use_cases.append("Technical analysis and implementation research")
                elif "market" in area_lower:
                    use_cases.append("Market analysis and business research")
                elif "data" in area_lower:
                    use_cases.append("Data analysis and quantitative research")
                elif "regulatory" in area_lower:
                    use_cases.append("Compliance and regulatory research")
                elif "user" in area_lower:
                    use_cases.append("User experience and customer research")
                elif "security" in area_lower:
                    use_cases.append("Security analysis and risk assessment")
                elif "innovation" in area_lower:
                    use_cases.append("Innovation and trend analysis")
        
        # Default use cases if none found
        if not use_cases:
            use_cases = ["General research and information gathering"]
        
        return list(set(use_cases))  # Remove duplicates
    
    def _extract_core_responsibilities(self, agent_config: Dict[str, Any]) -> List[str]:
        """Extract core responsibilities from agent system prompt.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            List of core responsibility strings
        """
        responsibilities = []
        
        # Extract from system_prompt if available
        system_prompt = agent_config.get("system_prompt", "")
        if system_prompt:
            # Look for common responsibility section markers
            responsibility_markers = [
                "## Core Responsibilities",
                "### Core Responsibilities", 
                "## Responsibilities",
                "### Responsibilities",
                "## Your Role",
                "### Your Role"
            ]
            
            for marker in responsibility_markers:
                if marker in system_prompt:
                    # Find the section and extract responsibilities
                    section_start = system_prompt.find(marker)
                    next_section_start = section_start + len(marker)
                    
                    # Find the next major section (##) that's NOT a subsection (###)
                    # We need to find a line that starts with exactly "##" followed by space
                    lines = system_prompt[next_section_start:].split('\n')
                    section_end = len(system_prompt)  # default to end of prompt
                    
                    for i, line in enumerate(lines):
                        # Look for major section headers (## but not ###)
                        if line.startswith('## ') and not line.startswith('### '):
                            section_end = next_section_start + sum(len(l) + 1 for l in lines[:i])
                            break
                    
                    section_text = system_prompt[section_start:section_end]
                    
                    # Extract responsibilities using multiple patterns
                    import re
                    
                    # Look for subsection headers (### Something) - these are major responsibilities
                    subsection_pattern = r'^###\s*([^#\n]+)'
                    subsections = re.findall(subsection_pattern, section_text, re.MULTILINE)
                    
                    # Also look for bullet points under each subsection
                    bullet_pattern = r'^-\s+([^\n]+)'
                    bullets = re.findall(bullet_pattern, section_text, re.MULTILINE)
                    
                    # Prefer subsection headers as they represent major responsibility areas
                    if subsections:
                        responsibilities.extend(subsections)
                    elif bullets:
                        responsibilities.extend(bullets[:8])  # Limit to avoid too many bullets
                    
                    break
        
        # Clean up responsibilities
        cleaned = []
        for resp in responsibilities:
            # Remove markdown formatting and extra whitespace
            cleaned_resp = resp.strip().replace("**", "").replace("*", "")
            if cleaned_resp and len(cleaned_resp) > 10:  # Only meaningful responsibilities
                cleaned.append(cleaned_resp)
        
        return cleaned[:5]  # Limit to top 5 responsibilities
    
    def _create_strength_summary(self, agent_config: Dict[str, Any]) -> str:
        """Create a one-line strength summary for the agent.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            One-line strength summary
        """
        expertise_areas = self._extract_expertise_areas(agent_config)
        agent_name = agent_config.get("name", "Agent")
        
        if len(expertise_areas) == 1:
            return f"Specializes in {expertise_areas[0].lower()} research and analysis"
        elif len(expertise_areas) == 2:
            return f"Expert in {expertise_areas[0].lower()} and {expertise_areas[1].lower()} research"
        elif len(expertise_areas) > 2:
            return f"Multi-disciplinary expert covering {', '.join(expertise_areas[:2]).lower()} and more"
        else:
            return "Versatile research agent for comprehensive analysis"
    
    def map_agent_to_capability(self, agent_name: str, agent_config: Dict[str, Any]) -> AgentCapability:
        """Map a single agent configuration to capability description.
        
        Args:
            agent_name: Name of the agent
            agent_config: Agent configuration dictionary
            
        Returns:
            AgentCapability object
        """
        return AgentCapability(
            name=agent_name,
            expertise_areas=self._extract_expertise_areas(agent_config),
            description=self._extract_description(agent_config),
            typical_use_cases=self._extract_use_cases(agent_config),
            strength_summary=self._create_strength_summary(agent_config),
            core_responsibilities=self._extract_core_responsibilities(agent_config),
            completion_indicators=agent_config.get("completion_indicators", [])
        )
    
    def get_all_agent_capabilities(self) -> List[AgentCapability]:
        """Get capability descriptions for all available agents.
        
        Returns:
            List of AgentCapability objects for all agents
        """
        capabilities = []
        agent_names = self.agent_registry.list_agents()
        
        for agent_name in agent_names:
            try:
                agent_config = self.agent_registry.get_agent(agent_name)
                if agent_config:
                    capability = self.map_agent_to_capability(agent_name, agent_config)
                    capabilities.append(capability)
                else:
                    logger.warning(f"Agent {agent_name} not found in registry")
                    # Create minimal capability for missing agents
                    capabilities.append(AgentCapability(
                        name=agent_name,
                        expertise_areas=["General Research"],
                        description="Research agent",
                        typical_use_cases=["General research tasks"],
                        strength_summary="General purpose research agent"
                    ))
            except Exception as e:
                logger.warning(f"Failed to map agent {agent_name} to capability: {e}")
                # Create minimal capability for problematic agents
                capabilities.append(AgentCapability(
                    name=agent_name,
                    expertise_areas=["General Research"],
                    description="Research agent",
                    typical_use_cases=["General research tasks"],
                    strength_summary="General purpose research agent"
                ))
        
        return capabilities
    
    def get_agent_capabilities_by_names(self, agent_names: List[str]) -> List[AgentCapability]:
        """Get capability descriptions for specific agents.
        
        Args:
            agent_names: List of agent names to get capabilities for
            
        Returns:
            List of AgentCapability objects for specified agents
        """
        capabilities = []
        
        for agent_name in agent_names:
            try:
                agent_config = self.agent_registry.get_agent(agent_name)
                if agent_config:
                    capability = self.map_agent_to_capability(agent_name, agent_config)
                    capabilities.append(capability)
                else:
                    logger.warning(f"Agent {agent_name} not found in registry")
                    # Create minimal capability for missing agents
                    capabilities.append(AgentCapability(
                        name=agent_name,
                        expertise_areas=["General Research"],
                        description=f"Research agent: {agent_name}",
                        typical_use_cases=["General research tasks"],
                        strength_summary="General purpose research agent"
                    ))
            except Exception as e:
                logger.warning(f"Failed to get capability for agent {agent_name}: {e}")
                capabilities.append(AgentCapability(
                    name=agent_name,
                    expertise_areas=["General Research"],
                    description=f"Research agent: {agent_name}",
                    typical_use_cases=["General research tasks"],
                    strength_summary="General purpose research agent"
                ))
        
        return capabilities
    
    def filter_capabilities_by_expertise(
        self, 
        capabilities: List[AgentCapability], 
        required_expertise: List[str]
    ) -> List[AgentCapability]:
        """Filter capabilities by required expertise areas.
        
        Args:
            capabilities: List of agent capabilities
            required_expertise: List of required expertise areas
            
        Returns:
            Filtered list of capabilities matching requirements
        """
        filtered = []
        
        for capability in capabilities:
            # Check if any of the agent's expertise areas match requirements
            agent_expertise_lower = [area.lower() for area in capability.expertise_areas]
            required_lower = [area.lower() for area in required_expertise]
            
            if any(req in agent_expertise_lower for req in required_lower):
                filtered.append(capability)
        
        return filtered
    
    def get_expertise_summary(self, capabilities: List[AgentCapability]) -> Dict[str, int]:
        """Get a summary of expertise areas across all capabilities.
        
        Args:
            capabilities: List of agent capabilities
            
        Returns:
            Dictionary mapping expertise areas to agent counts
        """
        expertise_counts = {}
        
        for capability in capabilities:
            for expertise in capability.expertise_areas:
                expertise_counts[expertise] = expertise_counts.get(expertise, 0) + 1
        
        return expertise_counts