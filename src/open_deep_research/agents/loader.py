"""Agent loader for parsing Markdown files with YAML frontmatter.

This module implements the Claude Code pattern for agent definitions,
where agents are stored as Markdown files with YAML frontmatter containing
configuration and metadata.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

logger = logging.getLogger(__name__)


class AgentLoader:
    """Load agents from Markdown files with YAML frontmatter."""
    
    @staticmethod
    def parse_agent_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse Markdown file with YAML frontmatter.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            Dictionary containing agent configuration and system prompt
            
        Raises:
            ValueError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Agent file not found: {file_path}")
        
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Error reading agent file {file_path}: {e}")
        
        # Extract YAML frontmatter and content
        pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)'
        match = re.match(pattern, content, re.DOTALL)
        
        if not match:
            raise ValueError(f"Invalid agent file format: {file_path}. Must have YAML frontmatter.")
        
        try:
            # Parse YAML frontmatter
            yaml_content = match.group(1)
            frontmatter = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML frontmatter in {file_path}: {e}")
        
        # Extract system prompt
        system_prompt = match.group(2).strip()
        
        # Validate required fields
        AgentLoader._validate_agent_config(frontmatter, file_path)
        
        return {
            **frontmatter,
            "system_prompt": system_prompt,
            "file_path": str(file_path),
            "file_name": file_path.name
        }
    
    @staticmethod
    def _validate_agent_config(config: Dict[str, Any], file_path: Path) -> None:
        """Validate agent configuration has required fields.
        
        Args:
            config: Parsed configuration dictionary
            file_path: Path to the file being validated
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ["name", "description"]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in {file_path}")
        
        # Validate name format
        name = config["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Agent name must be a non-empty string in {file_path}")
        
        # Ensure name is valid identifier
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
            raise ValueError(f"Agent name '{name}' must be valid identifier (letters, numbers, hyphens, underscores) in {file_path}")
        
        # Validate description
        description = config["description"]
        if not isinstance(description, str) or len(description.strip()) < 10:
            raise ValueError(f"Agent description must be at least 10 characters in {file_path}")
        
        # Validate optional fields if present
        if "tools" in config and config["tools"] is not None:
            tools = config["tools"]
            if not isinstance(tools, list):
                raise ValueError(f"Agent tools must be a list or null in {file_path}")
        
        if "expertise_areas" in config:
            expertise = config["expertise_areas"]
            if not isinstance(expertise, list):
                raise ValueError(f"Expertise areas must be a list in {file_path}")
        
        if "completion_indicators" in config:
            indicators = config["completion_indicators"]
            if not isinstance(indicators, list):
                raise ValueError(f"Completion indicators must be a list in {file_path}")
    
    @staticmethod
    def load_agents_from_directory(directory: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
        """Load all agent files from a directory.
        
        Args:
            directory: Directory containing agent Markdown files
            
        Returns:
            Dictionary mapping agent names to their configurations
        """
        directory = Path(directory)
        agents = {}
        
        if not directory.exists():
            logger.debug(f"Agent directory does not exist: {directory}")
            return agents
        
        # Find all Markdown files
        for agent_file in directory.glob("*.md"):
            try:
                agent_config = AgentLoader.parse_agent_file(agent_file)
                agent_name = agent_config["name"]
                
                if agent_name in agents:
                    logger.warning(f"Duplicate agent name '{agent_name}' found in {agent_file}. "
                                  f"Overriding previous definition.")
                
                agents[agent_name] = agent_config
                logger.debug(f"Loaded agent '{agent_name}' from {agent_file}")
                
            except Exception as e:
                logger.error(f"Error loading agent from {agent_file}: {e}")
                # Continue loading other agents
        
        logger.info(f"Loaded {len(agents)} agents from {directory}")
        return agents
    
    @staticmethod
    def validate_agent_definition(agent_config: Dict[str, Any]) -> List[str]:
        """Validate an agent definition and return any warnings.
        
        Args:
            agent_config: Agent configuration to validate
            
        Returns:
            List of warning messages (empty if no warnings)
        """
        warnings = []
        
        # Check for recommended fields
        recommended_fields = {
            "expertise_areas": "Consider adding expertise_areas to help users understand the agent's specialization",
            "completion_indicators": "Consider adding completion_indicators for automatic handoff detection",
            "examples": "Consider adding examples to help users understand when to use this agent"
        }
        
        for field, message in recommended_fields.items():
            if field not in agent_config:
                warnings.append(message)
        
        # Check system prompt length
        system_prompt = agent_config.get("system_prompt", "")
        if len(system_prompt) < 100:
            warnings.append("System prompt is quite short. Consider adding more detailed instructions.")
        
        # Check for tool specification
        tools = agent_config.get("tools")
        if tools is None:
            warnings.append("Agent will inherit all tools by default. Consider specifying tools explicitly if you want to limit them.")
        
        return warnings