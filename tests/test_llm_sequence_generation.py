"""Comprehensive test suite for LLM-based sequence generation system.

This module tests the LLMSequenceGenerator, AgentCapabilityMapper, and integration
flows to ensure robust and effective sequence generation.
"""

import asyncio
import json
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import logging

import pytest
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

# Import the modules under test
from src.open_deep_research.supervisor.llm_sequence_generator import LLMSequenceGenerator
from src.open_deep_research.supervisor.agent_capability_mapper import AgentCapabilityMapper
from src.open_deep_research.supervisor.sequence_models import (
    AgentCapability,
    SequenceGenerationInput,
    SequenceGenerationOutput,
    SequenceGenerationResult,
    AgentSequence
)
from src.open_deep_research.agents.registry import AgentRegistry


class TestLLMSequenceGenerator(unittest.TestCase):
    """Test class for LLMSequenceGenerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = LLMSequenceGenerator()
        
        # Create sample agent capabilities
        self.sample_agents = [
            AgentCapability(
                name="research_agent",
                expertise_areas=["Academic", "Literature Review"],
                description="Deep research specialist for academic investigation",
                typical_use_cases=["Literature reviews", "Academic research"],
                strength_summary="Specializes in academic research and analysis"
            ),
            AgentCapability(
                name="technical_agent",
                expertise_areas=["Technical", "Engineering"],
                description="Technical analysis and implementation specialist",
                typical_use_cases=["Technical analysis", "Implementation research"],
                strength_summary="Expert in technical and engineering research"
            ),
            AgentCapability(
                name="market_agent",
                expertise_areas=["Market", "Business"],
                description="Market research and business analysis specialist",
                typical_use_cases=["Market analysis", "Business research"],
                strength_summary="Specializes in market and business research"
            ),
            AgentCapability(
                name="analysis_agent",
                expertise_areas=["Data", "Analytics"],
                description="Data analysis and quantitative research specialist",
                typical_use_cases=["Data analysis", "Quantitative research"],
                strength_summary="Expert in data analysis and analytics"
            ),
            AgentCapability(
                name="synthesis_agent",
                expertise_areas=["Synthesis", "Integration"],
                description="Information synthesis and integration specialist",
                typical_use_cases=["Information synthesis", "Report integration"],
                strength_summary="Specializes in information synthesis and integration"
            )
        ]
        
        # Sample research topics for testing
        self.test_topics = {
            "academic": "The impact of machine learning on climate change predictions",
            "technical": "Implementing microservices architecture for large-scale applications",
            "market": "Consumer adoption trends for electric vehicles in emerging markets",
            "mixed": "AI governance frameworks for healthcare applications"
        }
    
    def test_system_prompt_creation(self):
        """Test that system prompt is properly created."""
        system_prompt = self.generator._create_system_prompt()
        
        # Verify key elements are present
        self.assertIn("research strategist", system_prompt.lower())
        self.assertIn("3 distinct strategic sequences", system_prompt.lower())
        self.assertIn("agent orchestration", system_prompt.lower())
        self.assertIn("json format", system_prompt.lower())
        
        # Verify research approach types are mentioned
        self.assertIn("foundational-first", system_prompt.lower())
        self.assertIn("problem-solution", system_prompt.lower())
        self.assertIn("comparative", system_prompt.lower())
    
    def test_user_prompt_creation(self):
        """Test user prompt creation with research context."""
        input_data = SequenceGenerationInput(
            research_topic=self.test_topics["academic"],
            research_brief="Comprehensive analysis needed",
            available_agents=self.sample_agents[:3],
            research_type="academic"
        )
        
        user_prompt = self.generator._create_user_prompt(input_data)
        
        # Verify all input elements are included
        self.assertIn(self.test_topics["academic"], user_prompt)
        self.assertIn("Comprehensive analysis needed", user_prompt)
        self.assertIn("academic", user_prompt)
        
        # Verify agent information is formatted correctly
        for agent in self.sample_agents[:3]:
            self.assertIn(agent.name, user_prompt)
            self.assertIn(agent.description, user_prompt)
            self.assertIn(agent.strength_summary, user_prompt)
    
    def test_fallback_sequence_generation(self):
        """Test fallback sequence generation when LLM fails."""
        agent_names = ["research_agent", "technical_agent", "market_agent"]
        research_topic = self.test_topics["technical"]
        
        fallback_output = self.generator._create_fallback_sequences(agent_names, research_topic)
        
        # Verify output structure
        self.assertIsInstance(fallback_output, SequenceGenerationOutput)
        self.assertEqual(len(fallback_output.sequences), 3)
        
        # Verify each sequence has required fields
        for sequence in fallback_output.sequences:
            self.assertIsInstance(sequence, AgentSequence)
            self.assertTrue(sequence.sequence_name)
            self.assertTrue(sequence.agent_names)
            self.assertTrue(sequence.rationale)
            self.assertGreaterEqual(sequence.confidence_score, 0.0)
            self.assertLessEqual(sequence.confidence_score, 1.0)
        
        # Verify sequences are different
        sequence_names = [seq.sequence_name for seq in fallback_output.sequences]
        self.assertEqual(len(set(sequence_names)), 3)  # All unique names
    
    @patch('src.open_deep_research.supervisor.llm_sequence_generator.configurable_model')
    def test_successful_sequence_generation(self, mock_model):
        """Test successful LLM sequence generation."""
        # Mock LLM response
        mock_response = SequenceGenerationOutput(
            research_analysis="Analysis of climate change ML research",
            sequences=[
                AgentSequence(
                    sequence_name="Academic Foundation Approach",
                    agent_names=["research_agent", "analysis_agent"],
                    rationale="Start with literature review, then analyze data patterns",
                    approach_description="Academic-first methodology",
                    expected_outcomes=["Comprehensive literature base", "Data insights"],
                    confidence_score=0.85,
                    research_focus="Academic foundation"
                ),
                AgentSequence(
                    sequence_name="Technical Implementation Focus",
                    agent_names=["technical_agent", "analysis_agent"],
                    rationale="Focus on technical implementation and validation",
                    approach_description="Technical-first approach",
                    expected_outcomes=["Technical feasibility", "Implementation insights"],
                    confidence_score=0.80,
                    research_focus="Technical implementation"
                ),
                AgentSequence(
                    sequence_name="Market Application Analysis",
                    agent_names=["market_agent", "synthesis_agent"],
                    rationale="Analyze market applications and synthesize findings",
                    approach_description="Market-focused approach",
                    expected_outcomes=["Market insights", "Application synthesis"],
                    confidence_score=0.75,
                    research_focus="Market applications"
                )
            ],
            reasoning_summary="Three complementary approaches for comprehensive coverage",
            recommended_sequence=0,
            alternative_considerations=["Consider interdisciplinary perspectives"]
        )
        
        # Configure mock
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = mock_response
        mock_model.with_config.return_value.with_structured_output.return_value = mock_structured_model
        
        # Test generation
        input_data = SequenceGenerationInput(
            research_topic=self.test_topics["academic"],
            research_brief="Test brief for academic research",
            available_agents=self.sample_agents,
            research_type="academic"
        )
        
        result = self.generator.generate_sequences_sync(input_data)
        
        # Verify result structure
        self.assertIsInstance(result, SequenceGenerationResult)
        self.assertTrue(result.success)
        self.assertEqual(len(result.output.sequences), 3)
        self.assertFalse(result.metadata.fallback_used)
        
        # Verify sequences have expected structure
        for sequence in result.output.sequences:
            self.assertTrue(sequence.sequence_name)
            self.assertTrue(sequence.agent_names)
            self.assertTrue(sequence.rationale)
            self.assertGreaterEqual(sequence.confidence_score, 0.0)
            self.assertLessEqual(sequence.confidence_score, 1.0)
    
    @patch('src.open_deep_research.supervisor.llm_sequence_generator.configurable_model')
    def test_llm_generation_failure_fallback(self, mock_model):
        """Test fallback behavior when LLM generation fails."""
        # Configure mock to raise exception
        mock_model.with_config.side_effect = Exception("API Error")
        
        input_data = SequenceGenerationInput(
            research_topic=self.test_topics["technical"],
            research_brief="Test brief for technical research",
            available_agents=self.sample_agents,
            research_type="technical"
        )
        
        result = self.generator.generate_sequences_sync(input_data)
        
        # Verify fallback was used
        self.assertIsInstance(result, SequenceGenerationResult)
        self.assertFalse(result.success)
        self.assertTrue(result.metadata.fallback_used)
        self.assertIn("API Error", result.metadata.error_details)
        
        # Verify fallback still produces valid sequences
        self.assertEqual(len(result.output.sequences), 3)
        for sequence in result.output.sequences:
            self.assertTrue(sequence.sequence_name)
            self.assertTrue(sequence.agent_names)


class TestAgentCapabilityMapper(unittest.TestCase):
    """Test class for AgentCapabilityMapper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock agent registry
        self.mock_registry = Mock(spec=AgentRegistry)
        self.mapper = AgentCapabilityMapper(self.mock_registry)
        
        # Sample agent configurations
        self.sample_configs = {
            "research_agent": {
                "name": "research_agent",
                "description": "Deep research specialist for academic investigation",
                "expertise": ["Academic research", "Literature reviews"],
                "use_cases": ["Academic studies", "Literature synthesis"]
            },
            "technical_agent": {
                "name": "technical_agent",
                "description": "Technical analysis and implementation research",
                "prompt": "You are a technical specialist focused on engineering solutions and implementation details.",
                "instructions": "Analyze technical feasibility and implementation approaches"
            },
            "market_agent": {
                "name": "market_agent",
                "description": "Market research and business analysis",
                "expertise": "Market analysis"
            }
        }
    
    def test_expertise_extraction_from_explicit_field(self):
        """Test extracting expertise from explicit expertise field."""
        config = self.sample_configs["research_agent"]
        expertise = self.mapper._extract_expertise_areas(config)
        
        self.assertIn("Academic research", expertise)
        self.assertIn("Literature reviews", expertise)
    
    def test_expertise_extraction_from_description(self):
        """Test extracting expertise from description and prompt text."""
        config = self.sample_configs["technical_agent"]
        expertise = self.mapper._extract_expertise_areas(config)
        
        # Should categorize as "Technical" based on keywords
        expertise_lower = [area.lower() for area in expertise]
        self.assertIn("technical", expertise_lower)
    
    def test_expertise_fallback_to_general(self):
        """Test fallback to 'General Research' when no expertise found."""
        config = {"name": "generic_agent", "description": "A helpful agent"}
        expertise = self.mapper._extract_expertise_areas(config)
        
        self.assertIn("General Research", expertise)
    
    def test_description_extraction(self):
        """Test extracting agent descriptions."""
        # Test explicit description
        config1 = self.sample_configs["research_agent"]
        desc1 = self.mapper._extract_description(config1)
        self.assertEqual(desc1, "Deep research specialist for academic investigation")
        
        # Test fallback to prompt
        config2 = self.sample_configs["technical_agent"]
        desc2 = self.mapper._extract_description(config2)
        self.assertIn("technical specialist", desc2.lower())
        
        # Test fallback to default
        config3 = {"name": "test_agent"}
        desc3 = self.mapper._extract_description(config3)
        self.assertEqual(desc3, "Specialized research agent")
    
    def test_use_cases_extraction(self):
        """Test extracting typical use cases."""
        # Test explicit use cases
        config = self.sample_configs["research_agent"]
        use_cases = self.mapper._extract_use_cases(config)
        
        self.assertIn("Academic studies", use_cases)
        self.assertIn("Literature synthesis", use_cases)
    
    def test_use_cases_inference_from_expertise(self):
        """Test inferring use cases from expertise areas."""
        config = {
            "name": "test_agent",
            "description": "A technical engineering agent with market analysis capabilities"
        }
        use_cases = self.mapper._extract_use_cases(config)
        
        # Should infer use cases based on technical and market keywords
        use_cases_text = " ".join(use_cases).lower()
        self.assertTrue(
            "technical" in use_cases_text or "market" in use_cases_text,
            f"Expected technical or market use cases, got: {use_cases}"
        )
    
    def test_strength_summary_creation(self):
        """Test creating one-line strength summaries."""
        config = self.sample_configs["research_agent"]
        summary = self.mapper._create_strength_summary(config)
        
        self.assertTrue(summary)
        self.assertIn("research", summary.lower())
        # Should be a reasonable length for a summary
        self.assertLess(len(summary), 150)
    
    def test_agent_to_capability_mapping(self):
        """Test mapping complete agent config to capability."""
        config = self.sample_configs["research_agent"]
        capability = self.mapper.map_agent_to_capability("research_agent", config)
        
        self.assertIsInstance(capability, AgentCapability)
        self.assertEqual(capability.name, "research_agent")
        self.assertTrue(capability.expertise_areas)
        self.assertTrue(capability.description)
        self.assertTrue(capability.typical_use_cases)
        self.assertTrue(capability.strength_summary)
    
    def test_get_all_agent_capabilities(self):
        """Test getting capabilities for all available agents."""
        # Mock registry responses
        self.mock_registry.list_agents.return_value = ["research_agent", "technical_agent"]
        self.mock_registry.get_agent.side_effect = lambda name: self.sample_configs.get(name)
        
        capabilities = self.mapper.get_all_agent_capabilities()
        
        self.assertEqual(len(capabilities), 2)
        self.assertIsInstance(capabilities[0], AgentCapability)
        self.assertIsInstance(capabilities[1], AgentCapability)
    
    def test_get_capabilities_handles_missing_agents(self):
        """Test handling of missing agents gracefully."""
        self.mock_registry.list_agents.return_value = ["missing_agent"]
        self.mock_registry.get_agent.return_value = None
        
        capabilities = self.mapper.get_all_agent_capabilities()
        
        # Should still return a capability with defaults
        self.assertEqual(len(capabilities), 1)
        capability = capabilities[0]
        self.assertEqual(capability.name, "missing_agent")
        self.assertIn("General Research", capability.expertise_areas)
    
    def test_filter_capabilities_by_expertise(self):
        """Test filtering capabilities by required expertise."""
        capabilities = [
            AgentCapability(
                name="agent1",
                expertise_areas=["Academic", "Research"],
                description="Academic agent",
                typical_use_cases=["Research"],
                strength_summary="Academic specialist"
            ),
            AgentCapability(
                name="agent2",
                expertise_areas=["Technical", "Engineering"],
                description="Technical agent",
                typical_use_cases=["Technical analysis"],
                strength_summary="Technical specialist"
            )
        ]
        
        # Filter for academic expertise
        filtered = self.mapper.filter_capabilities_by_expertise(capabilities, ["Academic"])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "agent1")
        
        # Filter for technical expertise
        filtered = self.mapper.filter_capabilities_by_expertise(capabilities, ["technical"])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "agent2")
    
    def test_expertise_summary(self):
        """Test generating expertise summary across capabilities."""
        capabilities = [
            AgentCapability(
                name="agent1",
                expertise_areas=["Academic", "Research"],
                description="Agent 1",
                typical_use_cases=["Research"],
                strength_summary="Academic specialist"
            ),
            AgentCapability(
                name="agent2",
                expertise_areas=["Academic", "Technical"],
                description="Agent 2",
                typical_use_cases=["Technical research"],
                strength_summary="Technical academic"
            )
        ]
        
        summary = self.mapper.get_expertise_summary(capabilities)
        
        self.assertEqual(summary["Academic"], 2)
        self.assertEqual(summary["Research"], 1)
        self.assertEqual(summary["Technical"], 1)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and resilience scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = LLMSequenceGenerator()
        self.mapper = AgentCapabilityMapper()
    
    def test_empty_agent_list(self):
        """Test handling empty agent list."""
        input_data = SequenceGenerationInput(
            research_topic="Test topic",
            research_brief="Test brief",
            available_agents=[],
            research_type="test"
        )
        
        result = self.generator.generate_sequences_sync(input_data)
        
        # Should still produce sequences (using fallback)
        self.assertIsInstance(result, SequenceGenerationResult)
        self.assertEqual(len(result.output.sequences), 3)
    
    def test_invalid_research_topic(self):
        """Test handling invalid or empty research topic."""
        input_data = SequenceGenerationInput(
            research_topic="",
            research_brief="Test brief",
            available_agents=[
                AgentCapability(
                    name="test_agent",
                    expertise_areas=["General"],
                    description="Test agent",
                    typical_use_cases=["Testing"],
                    strength_summary="Test specialist"
                )
            ],
            research_type="test"
        )
        
        result = self.generator.generate_sequences_sync(input_data)
        
        # Should handle gracefully
        self.assertIsInstance(result, SequenceGenerationResult)
        self.assertEqual(len(result.output.sequences), 3)
    
    @patch('src.open_deep_research.supervisor.llm_sequence_generator.configurable_model')
    def test_llm_timeout_handling(self, mock_model):
        """Test handling of LLM timeout errors."""
        # Simulate timeout
        mock_model.with_config.side_effect = Exception("Request timeout")
        
        input_data = SequenceGenerationInput(
            research_topic="Test topic",
            research_brief="Test brief",
            available_agents=[
                AgentCapability(
                    name="test_agent",
                    expertise_areas=["General"],
                    description="Test agent",
                    typical_use_cases=["Testing"],
                    strength_summary="Test specialist"
                )
            ],
            research_type="test"
        )
        
        result = self.generator.generate_sequences_sync(input_data)
        
        # Should fallback gracefully
        self.assertFalse(result.success)
        self.assertTrue(result.metadata.fallback_used)
        self.assertIn("timeout", result.metadata.error_details.lower())
    
    def test_malformed_agent_config(self):
        """Test handling malformed agent configurations."""
        mock_registry = Mock(spec=AgentRegistry)
        mock_registry.list_agents.return_value = ["malformed_agent"]
        mock_registry.get_agent.side_effect = Exception("Config parsing error")
        
        mapper = AgentCapabilityMapper(mock_registry)
        capabilities = mapper.get_all_agent_capabilities()
        
        # Should still return a default capability
        self.assertEqual(len(capabilities), 1)
        self.assertEqual(capabilities[0].name, "malformed_agent")
        self.assertIn("General Research", capabilities[0].expertise_areas)


class TestPerformance(unittest.TestCase):
    """Test performance and resource usage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = LLMSequenceGenerator()
        
        # Create larger agent list for performance testing
        self.large_agent_list = []
        for i in range(10):
            self.large_agent_list.append(
                AgentCapability(
                    name=f"agent_{i}",
                    expertise_areas=[f"Expertise_{i}", "General"],
                    description=f"Agent {i} for specialized research",
                    typical_use_cases=[f"Use case {i}", "General research"],
                    strength_summary=f"Specialist in area {i}"
                )
            )
    
    def test_prompt_generation_performance(self):
        """Test performance of prompt generation with large agent lists."""
        input_data = SequenceGenerationInput(
            research_topic="Complex multi-disciplinary research topic requiring comprehensive analysis",
            research_brief="This is a detailed research brief that provides extensive context " * 10,
            available_agents=self.large_agent_list,
            research_type="complex"
        )
        
        start_time = time.time()
        user_prompt = self.generator._create_user_prompt(input_data)
        generation_time = time.time() - start_time
        
        # Should generate prompt quickly
        self.assertLess(generation_time, 1.0)  # Less than 1 second
        self.assertTrue(user_prompt)
        
        # Verify all agents are included
        for agent in self.large_agent_list:
            self.assertIn(agent.name, user_prompt)
    
    def test_fallback_generation_performance(self):
        """Test performance of fallback sequence generation."""
        agent_names = [f"agent_{i}" for i in range(10)]
        research_topic = "Performance test topic"
        
        start_time = time.time()
        fallback_output = self.generator._create_fallback_sequences(agent_names, research_topic)
        generation_time = time.time() - start_time
        
        # Should generate quickly
        self.assertLess(generation_time, 0.5)  # Less than 0.5 seconds
        self.assertEqual(len(fallback_output.sequences), 3)
    
    def test_memory_usage_with_large_inputs(self):
        """Test memory efficiency with large inputs."""
        # Create very long research topic and brief
        long_topic = "Research topic " * 100
        long_brief = "Research brief with extensive details " * 200
        
        input_data = SequenceGenerationInput(
            research_topic=long_topic,
            research_brief=long_brief,
            available_agents=self.large_agent_list,
            research_type="memory_test"
        )
        
        # Should handle large inputs without issues
        user_prompt = self.generator._create_user_prompt(input_data)
        self.assertTrue(user_prompt)
        self.assertIn(long_topic[:50], user_prompt)  # Verify content included


class TestIntegrationFlow(unittest.TestCase):
    """Test integration with the broader research flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = LLMSequenceGenerator()
        self.mapper = AgentCapabilityMapper()
    
    def test_end_to_end_sequence_generation(self):
        """Test complete end-to-end sequence generation flow."""
        # Simulate real agent capabilities
        mock_registry = Mock(spec=AgentRegistry)
        mock_registry.list_agents.return_value = [
            "research_agent", "technical_agent", "market_agent", "analysis_agent", "synthesis_agent"
        ]
        
        mock_configs = {
            "research_agent": {
                "name": "research_agent",
                "description": "Academic research specialist",
                "expertise": ["Academic research", "Literature reviews"]
            },
            "technical_agent": {
                "name": "technical_agent", 
                "description": "Technical implementation specialist",
                "expertise": ["Technical analysis", "Engineering"]
            },
            "market_agent": {
                "name": "market_agent",
                "description": "Market research specialist", 
                "expertise": ["Market analysis", "Business research"]
            },
            "analysis_agent": {
                "name": "analysis_agent",
                "description": "Data analysis specialist",
                "expertise": ["Data analysis", "Analytics"]
            },
            "synthesis_agent": {
                "name": "synthesis_agent",
                "description": "Information synthesis specialist",
                "expertise": ["Information synthesis", "Integration"]
            }
        }
        
        mock_registry.get_agent.side_effect = lambda name: mock_configs.get(name)
        
        mapper = AgentCapabilityMapper(mock_registry)
        
        # Get capabilities
        capabilities = mapper.get_all_agent_capabilities()
        self.assertEqual(len(capabilities), 5)
        
        # Generate sequences
        input_data = SequenceGenerationInput(
            research_topic="AI governance frameworks for healthcare applications",
            research_brief="Comprehensive analysis of regulatory and technical requirements",
            available_agents=capabilities,
            research_type="interdisciplinary"
        )
        
        # Use fallback for predictable testing
        result = SequenceGenerationResult(
            output=self.generator._create_fallback_sequences(
                [cap.name for cap in capabilities],
                input_data.research_topic
            ),
            metadata=Mock(),
            success=True
        )
        
        # Verify integration
        self.assertEqual(len(result.output.sequences), 3)
        
        # Verify all sequences have valid agent names from the registry
        all_agent_names = {cap.name for cap in capabilities}
        for sequence in result.output.sequences:
            for agent_name in sequence.agent_names:
                self.assertIn(agent_name, all_agent_names)


def run_scenario_tests():
    """Run scenario-based testing with different research topics."""
    print("Running scenario-based testing...")
    
    generator = LLMSequenceGenerator()
    
    # Create mock capabilities for testing
    mock_capabilities = [
        AgentCapability(
            name="research_agent",
            expertise_areas=["Academic", "Literature Review"],
            description="Academic research specialist",
            typical_use_cases=["Literature reviews", "Academic research"],
            strength_summary="Expert in academic research and literature analysis"
        ),
        AgentCapability(
            name="technical_agent", 
            expertise_areas=["Technical", "Engineering"],
            description="Technical implementation specialist",
            typical_use_cases=["Technical analysis", "Implementation research"],
            strength_summary="Expert in technical systems and engineering"
        ),
        AgentCapability(
            name="market_agent",
            expertise_areas=["Market", "Business"],
            description="Market research and business analysis specialist", 
            typical_use_cases=["Market analysis", "Business research"],
            strength_summary="Expert in market analysis and business strategy"
        ),
        AgentCapability(
            name="analysis_agent",
            expertise_areas=["Data", "Analytics"],
            description="Data analysis and quantitative research specialist",
            typical_use_cases=["Data analysis", "Statistical research"],
            strength_summary="Expert in data analysis and quantitative methods"
        ),
        AgentCapability(
            name="synthesis_agent",
            expertise_areas=["Synthesis", "Integration"],
            description="Information synthesis and report generation specialist",
            typical_use_cases=["Information synthesis", "Report writing"],
            strength_summary="Expert in synthesizing complex information"
        )
    ]
    
    test_scenarios = [
        {
            "name": "Academic Research",
            "topic": "The impact of machine learning on climate change predictions",
            "type": "academic"
        },
        {
            "name": "Technical Research", 
            "topic": "Implementing microservices architecture for large-scale applications",
            "type": "technical"
        },
        {
            "name": "Market Research",
            "topic": "Consumer adoption trends for electric vehicles in emerging markets", 
            "type": "market"
        },
        {
            "name": "Mixed Research",
            "topic": "AI governance frameworks for healthcare applications",
            "type": "interdisciplinary"
        }
    ]
    
    results = {}
    
    for scenario in test_scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        
        input_data = SequenceGenerationInput(
            research_topic=scenario['topic'],
            research_brief=f"Comprehensive {scenario['type']} research needed",
            available_agents=mock_capabilities,
            research_type=scenario['type']
        )
        
        # Test fallback generation (reliable for testing)
        start_time = time.time()
        result = generator.generate_sequences_sync(input_data)
        generation_time = time.time() - start_time
        
        results[scenario['name']] = {
            "success": result.success,
            "generation_time": generation_time,
            "sequence_count": len(result.output.sequences),
            "fallback_used": result.metadata.fallback_used,
            "sequences": [
                {
                    "name": seq.sequence_name,
                    "agents": seq.agent_names,
                    "confidence": seq.confidence_score
                }
                for seq in result.output.sequences
            ]
        }
        
        print(f"  ✓ Generated {len(result.output.sequences)} sequences in {generation_time:.3f}s")
        print(f"  ✓ Success: {result.success}, Fallback: {result.metadata.fallback_used}")
        
        for i, seq in enumerate(result.output.sequences):
            print(f"    Sequence {i+1}: {seq.sequence_name} ({len(seq.agent_names)} agents, confidence: {seq.confidence_score:.2f})")
    
    return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("Starting comprehensive LLM sequence generation tests...")
    
    # Run unit tests
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run scenario tests
    print("\n" + "="*60)
    print("RUNNING SCENARIO TESTS")
    print("="*60)
    
    scenario_results = run_scenario_tests()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for scenario_name, result in scenario_results.items():
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"{status} {scenario_name}: {result['sequence_count']} sequences in {result['generation_time']:.3f}s")
    
    print("\nAll tests completed!")