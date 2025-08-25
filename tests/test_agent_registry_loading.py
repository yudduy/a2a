"""Comprehensive tests for Agent Registry loading and validation.

This module tests:
- Agent loading from .open_deep_research/agents/ directory
- Markdown and YAML agent definition parsing
- Project vs user agent precedence handling
- Agent validation and error handling
- Configuration integration
- Registry performance and caching

Test Categories:
1. Agent file loading and parsing
2. Directory structure validation
3. Agent precedence (project vs user)
4. Configuration validation
5. Performance and caching
6. Error handling and recovery
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import yaml

from open_deep_research.agents.registry import AgentRegistry
from open_deep_research.agents.loader import AgentLoader
from open_deep_research.configuration import Configuration, AgentFileFormat


class TestAgentRegistryLoading:
    """Test agent registry loading functionality."""
    
    def setup_method(self):
        """Set up test directories and files."""
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_agents_dir = self.temp_dir / ".open_deep_research" / "agents"
        self.user_agents_dir = Path(tempfile.mkdtemp()) / "agents"
        
        # Create directories
        self.project_agents_dir.mkdir(parents=True, exist_ok=True)
        self.user_agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test agent files
        self._create_test_agents()
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        if self.user_agents_dir.parent.exists():
            shutil.rmtree(self.user_agents_dir.parent)
    
    def _create_test_agents(self):
        """Create comprehensive test agent definitions."""
        
        # Project agents (higher precedence)
        project_academic = """# Academic Research Agent

## Description
Advanced academic research specialist focusing on peer-reviewed literature, theoretical frameworks, and empirical studies.

## Expertise Areas
- Academic research methodologies
- Systematic literature reviews
- Statistical analysis and interpretation
- Theoretical framework development
- Peer review processes
- Grant writing and research proposals

## Tools
- search
- scraper
- pdf_reader
- citation_manager

## Completion Indicators
- Literature review completed
- Theoretical framework established
- Research methodology defined
- Academic analysis finished
- Scholarly findings documented
- Peer review standards met

## Focus Questions
- What does the current academic literature say about this topic?
- What theoretical frameworks are most applicable?
- What research methodologies should be employed?
- How does this relate to established academic theory?
- What are the gaps in current scholarly research?

## Additional Configuration
tools_timeout: 120
max_iterations: 10
quality_threshold: 0.8
"""
        
        project_industry = """# Industry Analysis Agent

## Description  
Comprehensive industry analysis specialist with expertise in market dynamics, competitive intelligence, and business strategy.

## Expertise Areas
- Market research and analysis
- Competitive intelligence
- Business model evaluation
- Industry trend analysis
- Regulatory compliance
- Stakeholder analysis

## Tools
- search
- scraper
- market_data_api
- financial_analyzer

## Completion Indicators
- Market analysis complete
- Competitive landscape mapped
- Industry trends identified
- Business impact assessed
- Strategic recommendations formulated
- Market opportunity quantified

## Focus Questions
- What is the current market size and growth potential?
- Who are the key players and what are their strategies?
- What regulatory factors impact this industry?
- What are the main market drivers and barriers?
- How is this industry expected to evolve?

## Additional Configuration
tools_timeout: 90
max_iterations: 8
quality_threshold: 0.75
"""
        
        # User agents (lower precedence)
        user_academic = """# Basic Academic Agent

## Description
Basic academic research agent.

## Expertise Areas
- Basic research
- Literature review

## Tools
- search

## Completion Indicators
- Research done

## Focus Questions
- What does research say?
"""
        
        user_technical = """# Technical Analysis Agent

## Description
Technical analysis and implementation specialist focusing on emerging technologies and technical feasibility.

## Expertise Areas
- Technical architecture assessment
- Technology stack evaluation
- Implementation feasibility analysis
- Performance optimization
- Scalability analysis
- Security assessment

## Tools
- search
- scraper
- code_analyzer
- performance_tester

## Completion Indicators
- Technical analysis complete
- Architecture evaluated
- Implementation roadmap created
- Performance benchmarks established
- Security assessment finished
- Scalability plan developed

## Focus Questions
- What are the technical requirements and constraints?
- How can this be implemented effectively?
- What are the performance implications?
- What security considerations must be addressed?
- How will this scale with growth?

## Additional Configuration
tools_timeout: 150
max_iterations: 12
quality_threshold: 0.85
"""
        
        # Write project agents
        (self.project_agents_dir / "academic_researcher.md").write_text(project_academic)
        (self.project_agents_dir / "industry_analyst.md").write_text(project_industry)
        
        # Write user agents (academic will be overridden, technical is unique)
        (self.user_agents_dir / "academic_researcher.md").write_text(user_academic)
        (self.user_agents_dir / "technical_specialist.md").write_text(user_technical)
    
    def test_basic_agent_loading(self):
        """Test basic agent loading from project directory."""
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        # Should load project agents
        agents = registry.list_agents()
        assert len(agents) >= 2
        assert "academic_researcher" in agents
        assert "industry_analyst" in agents
        
        # Verify agent details
        academic = registry.get_agent("academic_researcher")
        assert academic is not None
        assert academic.get("_source") == "project"
        assert "Advanced academic research specialist" in academic.get("description", "")
        assert "Academic research methodologies" in academic.get("expertise_areas", [])
    
    def test_user_vs_project_agent_precedence(self):
        """Test that project agents override user agents with same name."""
        # Temporarily modify user agents directory path
        original_user_dir = self.user_agents_dir
        
        # Create registry that can see both directories
        with patch.object(AgentRegistry, '__init__') as mock_init:
            def custom_init(self, project_root=None):
                self.project_root = Path(project_root) if project_root else Path.cwd()
                self.project_agents_dir = self.project_root / ".open_deep_research/agents"
                self.user_agents_dir = original_user_dir
                self._agents = {}
                self._load_order = []
                self._load_all_agents()
            
            mock_init.side_effect = custom_init
            
            registry = AgentRegistry(project_root=str(self.temp_dir))
            
            # Should have academic_researcher from project (not user)
            academic = registry.get_agent("academic_researcher")
            assert academic is not None
            assert academic.get("_source") == "project"
            assert "Advanced academic research specialist" in academic.get("description", "")
            
            # Should have technical_specialist from user
            technical = registry.get_agent("technical_specialist")
            assert technical is not None
            assert technical.get("_source") == "user"
            assert "Technical analysis and implementation specialist" in technical.get("description", "")
    
    def test_agent_configuration_parsing(self):
        """Test parsing of agent configuration fields."""
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        academic = registry.get_agent("academic_researcher")
        industry = registry.get_agent("industry_analyst")
        
        # Test academic agent configuration
        assert len(academic.get("expertise_areas", [])) == 6
        assert "Academic research methodologies" in academic.get("expertise_areas", [])
        assert "Systematic literature reviews" in academic.get("expertise_areas", [])
        
        assert len(academic.get("tools", [])) == 4
        assert "search" in academic.get("tools", [])
        assert "pdf_reader" in academic.get("tools", [])
        
        assert len(academic.get("completion_indicators", [])) == 6
        assert "Literature review completed" in academic.get("completion_indicators", [])
        
        assert len(academic.get("focus_questions", [])) == 5
        assert "What does the current academic literature say" in academic.get("focus_questions", [])[0]
        
        # Test industry agent configuration
        assert len(industry.get("expertise_areas", [])) == 6
        assert "Market research and analysis" in industry.get("expertise_areas", [])
        
        assert len(industry.get("tools", [])) == 4
        assert "market_data_api" in industry.get("tools", [])
    
    def test_agent_validation(self):
        """Test agent validation functionality."""
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        # Validate all agents
        validation_results = registry.validate_all_agents()
        
        # Should have no warnings for well-formed agents
        assert len(validation_results) == 0 or all(len(warnings) == 0 for warnings in validation_results.values())
        
        # Create invalid agent for testing
        invalid_agent = """# Invalid Agent
## Description
Missing required sections
"""
        (self.project_agents_dir / "invalid_agent.md").write_text(invalid_agent)
        
        # Reload and validate
        registry.reload_agents()
        validation_results = registry.validate_all_agents()
        
        # Should have warnings for invalid agent
        if "invalid_agent" in validation_results:
            assert len(validation_results["invalid_agent"]) > 0
    
    def test_agent_search_functionality(self):
        """Test agent search and filtering capabilities."""
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        # Search by name
        academic_results = registry.search_agents("academic")
        assert "academic_researcher" in academic_results
        
        # Search by expertise
        market_results = registry.get_agents_by_expertise("market")
        assert "industry_analyst" in market_results
        
        # Search by description
        analysis_results = registry.search_agents("analysis")
        assert "industry_analyst" in analysis_results
        
        # Test case-insensitive search
        upper_results = registry.search_agents("ACADEMIC")
        assert "academic_researcher" in upper_results
    
    def test_registry_statistics(self):
        """Test registry statistics and metadata."""
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        stats = registry.get_registry_stats()
        
        assert stats["total_agents"] >= 2
        assert stats["project_agents"] >= 2
        assert stats["agents_with_custom_tools"] >= 2
        assert stats["agents_with_expertise_areas"] >= 2
        assert stats["agents_with_completion_indicators"] >= 2
        assert stats["project_dir_exists"] is True
        assert str(self.project_agents_dir) in stats["project_agents_dir"]
    
    def test_detailed_agent_listing(self):
        """Test detailed agent listing with metadata."""
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        detailed_agents = registry.list_agents_detailed()
        
        assert len(detailed_agents) >= 2
        
        # Find academic researcher
        academic = next((agent for agent in detailed_agents if agent["name"] == "academic_researcher"), None)
        assert academic is not None
        assert academic["source"] == "project"
        assert len(academic["expertise_areas"]) > 0
        assert "Advanced academic research specialist" in academic["description"]
        assert academic["tools"] is not None
        assert len(academic["completion_indicators"]) > 0


class TestAgentFileFormatSupport:
    """Test support for different agent file formats (Markdown, YAML)."""
    
    def setup_method(self):
        """Set up test for file format testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / ".open_deep_research" / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_yaml_agent_loading(self):
        """Test loading agents from YAML files."""
        yaml_agent = {
            "name": "yaml_agent",
            "description": "Agent defined in YAML format",
            "expertise_areas": [
                "YAML processing",
                "Configuration management",
                "Data serialization"
            ],
            "tools": ["search", "yaml_parser"],
            "completion_indicators": [
                "YAML processing complete",
                "Configuration validated"
            ],
            "focus_questions": [
                "How should this be structured in YAML?",
                "What configuration options are available?"
            ],
            "additional_config": {
                "timeout": 60,
                "max_iterations": 5
            }
        }
        
        # Write YAML agent
        yaml_path = self.agents_dir / "yaml_agent.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_agent, f)
        
        # Test YAML loading directly
        agents = AgentLoader.load_agents_from_directory(self.agents_dir)
        assert "yaml_agent" in agents
        
        loaded_agent = agents["yaml_agent"]
        assert loaded_agent["description"] == "Agent defined in YAML format"
        assert "YAML processing" in loaded_agent["expertise_areas"]
        assert "yaml_parser" in loaded_agent["tools"]
    
    def test_mixed_format_loading(self):
        """Test loading agents from mixed Markdown and YAML files."""
        # Create Markdown agent
        md_content = """# Markdown Agent

## Description
Agent defined in Markdown format

## Expertise Areas
- Markdown processing
- Documentation generation

## Tools
- search
- markdown_parser

## Completion Indicators
- Markdown processing complete

## Focus Questions
- How should this be formatted in Markdown?
"""
        (self.agents_dir / "markdown_agent.md").write_text(md_content)
        
        # Create YAML agent
        yaml_agent = {
            "name": "yaml_agent",
            "description": "Agent defined in YAML format",
            "expertise_areas": ["YAML processing"],
            "tools": ["search"],
            "completion_indicators": ["YAML processing complete"]
        }
        yaml_path = self.agents_dir / "yaml_agent.yml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_agent, f)
        
        # Load both
        agents = AgentLoader.load_agents_from_directory(self.agents_dir)
        
        assert len(agents) >= 2
        assert "markdown_agent" in agents
        assert "yaml_agent" in agents
        
        # Verify content
        md_agent = agents["markdown_agent"]
        yaml_loaded = agents["yaml_agent"]
        
        assert "Markdown processing" in md_agent["expertise_areas"]
        assert "YAML processing" in yaml_loaded["expertise_areas"]


class TestAgentRegistryPerformance:
    """Test agent registry performance and caching."""
    
    def setup_method(self):
        """Set up performance testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / ".open_deep_research" / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create multiple agents for performance testing
        self._create_many_agents()
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_many_agents(self):
        """Create many agents for performance testing."""
        base_content = """# Agent {name}

## Description
Test agent number {number} for performance testing

## Expertise Areas
- Performance testing
- Load testing
- Agent {number} expertise

## Tools
- search
- agent_{number}_tool

## Completion Indicators
- Agent {number} task complete
- Performance test finished

## Focus Questions
- How does agent {number} perform?
- What are the performance characteristics?
"""
        
        for i in range(20):
            agent_name = f"performance_agent_{i:02d}"
            content = base_content.format(name=agent_name.replace("_", " ").title(), number=i, agent_number=i)
            (self.agents_dir / f"{agent_name}.md").write_text(content)
    
    def test_loading_performance(self):
        """Test agent loading performance with many agents."""
        import time
        
        # Measure loading time
        start_time = time.time()
        registry = AgentRegistry(project_root=str(self.temp_dir))
        loading_time = time.time() - start_time
        
        # Should load all agents
        agents = registry.list_agents()
        assert len(agents) == 20
        
        # Loading should be reasonable fast (< 2 seconds for 20 agents)
        assert loading_time < 2.0, f"Loading time {loading_time:.3f}s is too slow"
        
        print(f"Loaded {len(agents)} agents in {loading_time:.3f}s")
    
    def test_reload_performance(self):
        """Test agent reload performance."""
        import time
        
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        # Measure reload time
        start_time = time.time()
        registry.reload_agents()
        reload_time = time.time() - start_time
        
        # Reload should be fast
        assert reload_time < 1.0, f"Reload time {reload_time:.3f}s is too slow"
        
        print(f"Reloaded {len(registry.list_agents())} agents in {reload_time:.3f}s")
    
    def test_search_performance(self):
        """Test agent search performance."""
        import time
        
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        # Test multiple searches
        search_queries = ["performance", "agent", "test", "expertise", "tool"]
        
        for query in search_queries:
            start_time = time.time()
            results = registry.search_agents(query)
            search_time = time.time() - start_time
            
            # Search should be fast
            assert search_time < 0.1, f"Search for '{query}' took {search_time:.3f}s"
            assert len(results) > 0, f"No results for '{query}'"


class TestAgentRegistryErrorHandling:
    """Test error handling and edge cases in agent registry."""
    
    def setup_method(self):
        """Set up error handling tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.agents_dir = self.temp_dir / ".open_deep_research" / "agents"
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_missing_agents_directory(self):
        """Test handling when agents directory doesn't exist."""
        # Don't create the directory
        registry = AgentRegistry(project_root=str(self.temp_dir))
        
        # Should handle gracefully
        agents = registry.list_agents()
        assert len(agents) == 0
        
        stats = registry.get_registry_stats()
        assert stats["total_agents"] == 0
        assert stats["project_dir_exists"] is False
    
    def test_invalid_agent_files(self):
        """Test handling of invalid agent files."""
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create invalid files
        (self.agents_dir / "empty_file.md").write_text("")
        (self.agents_dir / "invalid_yaml.yaml").write_text("invalid: yaml: content:]")
        (self.agents_dir / "not_agent_file.txt").write_text("This is not an agent file")
        
        # Create one valid agent
        valid_agent = """# Valid Agent

## Description
This is a valid agent

## Expertise Areas
- Testing

## Tools
- search
"""
        (self.agents_dir / "valid_agent.md").write_text(valid_agent)
        
        # Should load only valid agent
        registry = AgentRegistry(project_root=str(self.temp_dir))
        agents = registry.list_agents()
        
        assert "valid_agent" in agents
        # Invalid files should be ignored
        assert "empty_file" not in agents
        assert "invalid_yaml" not in agents
        assert "not_agent_file" not in agents
    
    def test_agent_conflicts_detection(self):
        """Test detection of agent naming conflicts."""
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        user_agents_dir = Path(tempfile.mkdtemp()) / "agents"
        user_agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create conflicting agents
        project_agent = """# Project Agent
## Description
Project version
## Expertise Areas
- Project expertise
## Tools
- search
"""
        user_agent = """# User Agent
## Description  
User version
## Expertise Areas
- User expertise
## Tools
- search
"""
        
        (self.agents_dir / "conflict_agent.md").write_text(project_agent)
        (user_agents_dir / "conflict_agent.md").write_text(user_agent)
        
        # Mock registry to use both directories
        with patch.object(AgentRegistry, '__init__') as mock_init:
            def custom_init(self, project_root=None):
                self.project_root = Path(project_root) if project_root else Path.cwd()
                self.project_agents_dir = self.agents_dir
                self.user_agents_dir = user_agents_dir
                self._agents = {}
                self._load_order = []
                self._load_all_agents()
            
            mock_init.side_effect = custom_init
            
            registry = AgentRegistry(project_root=str(self.temp_dir))
            
            # Should detect conflicts
            conflicts = registry.get_agent_conflicts()
            assert len(conflicts) >= 1
            
            conflict_agent = next((c for c in conflicts if c["agent_name"] == "conflict_agent"), None)
            assert conflict_agent is not None
            assert "conflict_agent.md" in str(conflict_agent["project_file"])
        
        # Clean up
        shutil.rmtree(user_agents_dir.parent)
    
    def test_directory_creation(self):
        """Test automatic directory creation."""
        # Use non-existent directory
        new_temp_dir = Path(tempfile.mkdtemp())
        new_temp_dir.rmdir()  # Remove so we can test creation
        
        registry = AgentRegistry(project_root=str(new_temp_dir))
        
        # Create directories
        registry.create_agent_directories()
        
        # Verify directories were created
        project_dir = new_temp_dir / ".open_deep_research" / "agents"
        assert project_dir.exists()
        
        # Clean up
        if new_temp_dir.exists():
            shutil.rmtree(new_temp_dir)


# Import patch for mocking
from unittest.mock import patch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])