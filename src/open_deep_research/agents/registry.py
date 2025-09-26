"""Dynamic agent registry following Claude Code pattern.

This module manages the loading and registration of agents from Markdown files,
supporting both project-specific and user-global agent definitions with
appropriate precedence handling.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .loader import AgentLoader

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Dynamic agent registry for managing agent definitions."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the agent registry.
        
        Args:
            project_root: Root directory of the project. If None, uses current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        # Agent directory paths (Claude Code pattern)
        self.project_agents_dir = self.project_root / "agents"
        self.user_agents_dir = Path.home() / ".open_deep_research/agents"
        
        # Agent storage
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._load_order: List[str] = []  # Track loading order for debugging
        
        # Load all agents
        self._load_all_agents()
    
    def _load_all_agents(self) -> None:
        """Load agents with project agents overriding user agents."""
        self._agents.clear()
        self._load_order.clear()
        
        # Load user agents first (lower priority)
        if self.user_agents_dir.exists():
            user_agents = AgentLoader.load_agents_from_directory(self.user_agents_dir)
            for name, config in user_agents.items():
                self._agents[name] = config
                config["_source"] = "user"
                self._load_order.append(f"user:{name}")
            
            logger.info(f"Loaded {len(user_agents)} user agents from {self.user_agents_dir}")
        
        # Load project agents (override user agents)
        if self.project_agents_dir.exists():
            project_agents = AgentLoader.load_agents_from_directory(self.project_agents_dir)
            overridden_count = 0
            
            for name, config in project_agents.items():
                if name in self._agents:
                    overridden_count += 1
                    logger.debug(f"Project agent '{name}' overriding user agent")
                
                self._agents[name] = config
                config["_source"] = "project"
                self._load_order.append(f"project:{name}")
            
            logger.info(f"Loaded {len(project_agents)} project agents from {self.project_agents_dir}")
            if overridden_count > 0:
                logger.info(f"Project agents overrode {overridden_count} user agents")
        
        # Log final state
        total_agents = len(self._agents)
        if total_agents == 0:
            logger.warning("No agents loaded. Consider creating agent definitions in agents/")
        else:
            logger.info(f"Agent registry initialized with {total_agents} agents")
    
    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent configuration dictionary or None if not found
        """
        return self._agents.get(name)
    
    def has_agent(self, name: str) -> bool:
        """Check if agent exists in registry.
        
        Args:
            name: Agent name
            
        Returns:
            True if agent exists, False otherwise
        """
        return name in self._agents
    
    def list_agents(self) -> List[str]:
        """Get list of all agent names.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
    
    def list_agents_detailed(self) -> List[Dict[str, Any]]:
        """Get detailed list of all agents with metadata.
        
        Returns:
            List of agent information dictionaries
        """
        agents = []
        for name, config in self._agents.items():
            agents.append({
                "name": name,
                "description": config.get("description", ""),
                "source": config.get("_source", "unknown"),
                "file_path": config.get("file_path", ""),
                "expertise_areas": config.get("expertise_areas", []),
                "tools": config.get("tools"),
                "completion_indicators": config.get("completion_indicators", [])
            })
        return agents
    
    def get_agents_by_expertise(self, expertise_area: str) -> List[str]:
        """Get agents that have specific expertise area.
        
        Args:
            expertise_area: Area of expertise to search for
            
        Returns:
            List of agent names that have the specified expertise
        """
        matching_agents = []
        expertise_lower = expertise_area.lower()
        
        for name, config in self._agents.items():
            expertise_areas = config.get("expertise_areas", [])
            if any(expertise_lower in area.lower() for area in expertise_areas):
                matching_agents.append(name)
        
        return matching_agents
    
    def search_agents(self, query: str) -> List[str]:
        """Search agents by name, description, or expertise.
        
        Args:
            query: Search query
            
        Returns:
            List of matching agent names
        """
        query_lower = query.lower()
        matching_agents = []
        
        for name, config in self._agents.items():
            # Search in name
            if query_lower in name.lower():
                matching_agents.append(name)
                continue
            
            # Search in description
            description = config.get("description", "").lower()
            if query_lower in description:
                matching_agents.append(name)
                continue
            
            # Search in expertise areas
            expertise_areas = config.get("expertise_areas", [])
            if any(query_lower in area.lower() for area in expertise_areas):
                matching_agents.append(name)
                continue
        
        return matching_agents
    
    def validate_all_agents(self) -> Dict[str, List[str]]:
        """Validate all loaded agents and return warnings.
        
        Returns:
            Dictionary mapping agent names to lists of warning messages
        """
        validation_results = {}
        
        for name, config in self._agents.items():
            warnings = AgentLoader.validate_agent_definition(config)
            if warnings:
                validation_results[name] = warnings
        
        return validation_results
    
    def reload_agents(self) -> None:
        """Reload all agents from disk."""
        logger.info("Reloading agents from disk")
        self._load_all_agents()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent registry.
        
        Returns:
            Dictionary with registry statistics
        """
        project_count = sum(1 for config in self._agents.values() if config.get("_source") == "project")
        user_count = sum(1 for config in self._agents.values() if config.get("_source") == "user")
        
        # Count agents with specific features
        with_tools = sum(1 for config in self._agents.values() if config.get("tools") is not None)
        with_expertise = sum(1 for config in self._agents.values() if config.get("expertise_areas"))
        with_completion_indicators = sum(1 for config in self._agents.values() if config.get("completion_indicators"))
        
        return {
            "total_agents": len(self._agents),
            "project_agents": project_count,
            "user_agents": user_count,
            "agents_with_custom_tools": with_tools,
            "agents_with_expertise_areas": with_expertise,
            "agents_with_completion_indicators": with_completion_indicators,
            "project_agents_dir": str(self.project_agents_dir),
            "user_agents_dir": str(self.user_agents_dir),
            "project_dir_exists": self.project_agents_dir.exists(),
            "user_dir_exists": self.user_agents_dir.exists()
        }
    
    def create_agent_directories(self) -> None:
        """Create agent directories if they don't exist."""
        # Create project agents directory
        self.project_agents_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created project agents directory: {self.project_agents_dir}")
        
        # Create user agents directory
        self.user_agents_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created user agents directory: {self.user_agents_dir}")
    
    def get_agent_conflicts(self) -> List[Dict[str, Any]]:
        """Get list of agents where project agents override user agents.
        
        Returns:
            List of conflict information dictionaries
        """
        conflicts = []
        
        # Track which agents were overridden
        project_agents = set()
        user_agents = set()
        
        for entry in self._load_order:
            source, name = entry.split(":", 1)
            if source == "project":
                if name in user_agents:
                    conflicts.append({
                        "agent_name": name,
                        "user_file": self.user_agents_dir / f"{name}.md",
                        "project_file": self.project_agents_dir / f"{name}.md"
                    })
                project_agents.add(name)
            else:
                user_agents.add(name)
        
        return conflicts