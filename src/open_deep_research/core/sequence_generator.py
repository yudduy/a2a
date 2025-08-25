"""Unified sequence generator consolidating orchestration and supervisor sequence generation.

This module provides the UnifiedSequenceGenerator that combines the rule-based
sequence generation from orchestration with the LLM-powered sequence generation
from supervisor, eliminating code duplication while preserving all functionality.
"""

import json
import re
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
from dataclasses import dataclass, field

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ValidationError

from open_deep_research.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

# Initialize configurable model for LLM-based generation
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "base_url"),
)


class TopicType(Enum):
    """Classification of research topic types."""
    ACADEMIC = "academic"           # Research-heavy, theory-focused
    MARKET = "market"              # Business/commercial focus
    TECHNICAL = "technical"        # Implementation/technology focus
    MIXED = "mixed"                # Balanced across multiple domains
    ANALYSIS = "analysis"          # Data analysis and interpretation focus
    SYNTHESIS = "synthesis"        # Integration and summary focus


class SequenceStrategy(Enum):
    """Unified sequence generation strategies."""
    THEORY_FIRST = "theory_first"          # Academic → Analysis → Market → Technical
    MARKET_FIRST = "market_first"          # Market → Technical → Research → Analysis
    TECHNICAL_FIRST = "technical_first"    # Technical → Research → Analysis → Market
    ANALYSIS_FIRST = "analysis_first"      # Analysis → Research → Market → Technical
    BALANCED = "balanced"                  # Mixed approach based on topic analysis
    CUSTOM = "custom"                      # User-defined sequence
    LLM_GENERATED = "llm_generated"        # LLM-generated strategic sequences


@dataclass
class TopicAnalysis:
    """Analysis results for a research topic."""
    
    topic_type: TopicType
    complexity_score: float  # 0.0 - 1.0
    scope_breadth: float     # 0.0 - 1.0 (narrow to broad)
    keywords: List[str]
    domain_indicators: Dict[str, float]  # Domain → confidence score
    estimated_agents_needed: int
    priority_areas: List[str]
    
    # Additional characteristics
    time_sensitivity: float = 0.5  # 0.0 - 1.0
    data_intensity: float = 0.5    # 0.0 - 1.0
    market_relevance: float = 0.5  # 0.0 - 1.0
    technical_complexity: float = 0.5  # 0.0 - 1.0


@dataclass
class GeneratedSequence:
    """A generated agent sequence with scoring metadata."""
    
    sequence_name: str
    strategy: SequenceStrategy
    agents: List[str]
    score: float  # 0.0 - 1.0 fitness score for the topic
    rationale: str
    estimated_duration: float  # in hours
    confidence: float  # 0.0 - 1.0 confidence in this sequence
    
    # Detailed scoring breakdown
    topic_fit_score: float = 0.0
    coverage_score: float = 0.0
    efficiency_score: float = 0.0
    expertise_match_score: float = 0.0


class AgentCapability(BaseModel):
    """Represents an agent's capabilities for LLM reasoning."""
    
    name: str = Field(description="Agent name/identifier")
    expertise_areas: List[str] = Field(description="Areas of expertise/specialization")
    description: str = Field(description="Brief description of agent capabilities")
    typical_use_cases: List[str] = Field(description="Common scenarios where this agent is useful")
    strength_summary: str = Field(description="One-line summary of agent's main strength")


class AgentSequence(BaseModel):
    """Represents a single strategic sequence of agents."""
    
    sequence_name: str = Field(description="Descriptive name for this sequence strategy")
    agent_names: List[str] = Field(description="Ordered list of agent names to execute")
    rationale: str = Field(description="Detailed reasoning for why this sequence is effective")
    approach_description: str = Field(description="High-level description of the research approach")
    expected_outcomes: List[str] = Field(description="Expected outcomes from this sequence")
    confidence_score: float = Field(
        ge=0.0, le=1.0, 
        description="Confidence in sequence effectiveness (0-1)"
    )
    research_focus: str = Field(description="Primary research focus of this sequence")


class SequenceGenerationInput(BaseModel):
    """Input data for unified sequence generation."""
    
    research_topic: str = Field(description="The research topic/question to address")
    research_brief: Optional[str] = Field(description="Additional research context or brief")
    available_agents: List[AgentCapability] = Field(description="Available agents with capabilities")
    research_type: Optional[str] = Field(description="Type of research (academic, technical, market, etc.)")
    constraints: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Any constraints on sequence generation"
    )
    
    # Generation mode selection
    generation_mode: str = Field(
        default="hybrid",
        description="Generation mode: 'rule_based', 'llm_based', or 'hybrid'"
    )
    num_sequences: int = Field(default=3, ge=1, le=5, description="Number of sequences to generate")
    strategies: Optional[List[SequenceStrategy]] = Field(default=None, description="Specific strategies to use")


class SequenceGenerationOutput(BaseModel):
    """Complete output from unified sequence generation."""
    
    research_analysis: str = Field(description="Analysis of the research requirements")
    sequences: List[AgentSequence] = Field(
        min_items=1, max_items=5,
        description="Generated strategic agent sequences"
    )
    reasoning_summary: str = Field(description="Summary of overall reasoning approach")
    recommended_sequence: int = Field(
        ge=0,
        description="Index of recommended sequence"
    )
    alternative_considerations: List[str] = Field(
        description="Alternative approaches or considerations"
    )
    
    # Unified metadata
    generation_mode: str = Field(description="Mode used for generation")
    topic_analysis: Optional[TopicAnalysis] = Field(default=None, description="Topic analysis results")


class SequenceGenerationMetadata(BaseModel):
    """Metadata about the sequence generation process."""
    
    generation_timestamp: str = Field(description="When sequences were generated")
    model_used: str = Field(description="Method/model used for generation")
    input_token_count: Optional[int] = Field(description="Tokens in input")
    output_token_count: Optional[int] = Field(description="Tokens in output")
    generation_time_seconds: Optional[float] = Field(description="Time taken for generation")
    fallback_used: bool = Field(default=False, description="Whether fallback logic was used")
    error_details: Optional[str] = Field(default=None, description="Any error details if fallback used")
    generation_mode: str = Field(description="Mode used for generation")


class SequenceGenerationResult(BaseModel):
    """Complete result including output and metadata."""
    
    output: SequenceGenerationOutput
    metadata: SequenceGenerationMetadata
    success: bool = Field(description="Whether generation was successful")


class UnifiedSequenceGenerator:
    """Unified sequence generator combining rule-based and LLM-powered approaches.
    
    This class consolidates functionality from both orchestration.sequence_generator
    and supervisor.llm_sequence_generator, eliminating code duplication while
    preserving all features and providing a unified interface.
    """
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        model_config: Optional[RunnableConfig] = None,
        debug_mode: bool = False
    ):
        """Initialize the unified sequence generator.
        
        Args:
            agent_registry: Registry containing available agents
            model_config: Optional configuration for the LLM model
            debug_mode: Enable detailed logging for debugging
        """
        self.agent_registry = agent_registry
        self.model_config = model_config or RunnableConfig()
        self.debug_mode = debug_mode
        
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        
        # Domain classification patterns (from orchestration)
        self._domain_patterns = self._initialize_domain_patterns()
        
        # Agent type mappings (from orchestration)
        self._agent_type_mappings = self._initialize_agent_mappings()
        
        # LLM system prompt (from supervisor)
        self._llm_system_prompt = self._create_llm_system_prompt()
        
        logger.info("UnifiedSequenceGenerator initialized with debug_mode=%s", debug_mode)
    
    # =====================================================================
    # PUBLIC API - Unified Interface
    # =====================================================================
    
    async def generate_sequences(
        self,
        input_data: SequenceGenerationInput
    ) -> SequenceGenerationResult:
        """Generate optimal agent sequences using unified approach.
        
        Args:
            input_data: Input containing research topic and generation parameters
            
        Returns:
            SequenceGenerationResult with sequences and metadata
        """
        start_time = time.time()
        generation_timestamp = datetime.now().isoformat()
        
        # Determine generation mode
        mode = input_data.generation_mode.lower()
        
        try:
            if mode == "rule_based":
                result = await self._generate_rule_based(input_data, start_time, generation_timestamp)
            elif mode == "llm_based":
                result = await self._generate_llm_based(input_data, start_time, generation_timestamp)
            elif mode == "hybrid":
                result = await self._generate_hybrid(input_data, start_time, generation_timestamp)
            else:
                raise ValueError(f"Unsupported generation mode: {mode}")
            
            logger.info(f"Generated {len(result.output.sequences)} sequences using {mode} mode in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Sequence generation failed: {e}")
            # Fallback to rule-based generation
            return await self._generate_fallback(input_data, start_time, generation_timestamp, str(e))
    
    def generate_sequences_sync(
        self,
        input_data: SequenceGenerationInput
    ) -> SequenceGenerationResult:
        """Synchronous version of sequence generation.
        
        Args:
            input_data: Input containing research topic and generation parameters
            
        Returns:
            SequenceGenerationResult with sequences and metadata
        """
        start_time = time.time()
        generation_timestamp = datetime.now().isoformat()
        
        # Determine generation mode
        mode = input_data.generation_mode.lower()
        
        try:
            if mode == "rule_based":
                result = self._generate_rule_based_sync(input_data, start_time, generation_timestamp)
            elif mode == "llm_based":
                result = self._generate_llm_based_sync(input_data, start_time, generation_timestamp)
            elif mode == "hybrid":
                result = self._generate_hybrid_sync(input_data, start_time, generation_timestamp)
            else:
                raise ValueError(f"Unsupported generation mode: {mode}")
            
            logger.info(f"Generated {len(result.output.sequences)} sequences using {mode} mode in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Sequence generation failed: {e}")
            # Fallback to rule-based generation
            return self._generate_fallback_sync(input_data, start_time, generation_timestamp, str(e))
    
    # =====================================================================
    # RULE-BASED GENERATION (from orchestration.sequence_generator)
    # =====================================================================
    
    async def _generate_rule_based(
        self,
        input_data: SequenceGenerationInput,
        start_time: float,
        generation_timestamp: str
    ) -> SequenceGenerationResult:
        """Generate sequences using rule-based approach."""
        return self._generate_rule_based_sync(input_data, start_time, generation_timestamp)
    
    def _generate_rule_based_sync(
        self,
        input_data: SequenceGenerationInput,
        start_time: float,
        generation_timestamp: str
    ) -> SequenceGenerationResult:
        """Generate sequences using rule-based approach (synchronous)."""
        
        # Convert AgentCapability objects to agent names
        available_agents = [agent.name for agent in input_data.available_agents]
        
        # Validate agent availability
        registry_agents = set(self.agent_registry.list_agents())
        invalid_agents = [agent for agent in available_agents if agent not in registry_agents]
        if invalid_agents:
            raise ValueError(f"Agents not found in registry: {invalid_agents}")
        
        # Step 1: Analyze topic characteristics
        topic_analysis = self.analyze_topic_characteristics(input_data.research_topic)
        logger.debug("Topic analysis completed: %s", topic_analysis.topic_type)
        
        # Step 2: Determine strategies to use
        strategies = input_data.strategies
        if strategies is None:
            strategies = self._select_strategies_for_topic(topic_analysis, input_data.num_sequences)
        
        # Step 3: Generate sequences for each strategy
        generated_sequences = []
        
        for strategy in strategies[:input_data.num_sequences]:
            try:
                sequence = self._generate_sequence_for_strategy(
                    strategy, available_agents, topic_analysis
                )
                
                # Convert GeneratedSequence to AgentSequence
                agent_sequence = AgentSequence(
                    sequence_name=sequence.sequence_name,
                    agent_names=sequence.agents,
                    rationale=sequence.rationale,
                    approach_description=f"{strategy.value} strategy: {sequence.rationale}",
                    expected_outcomes=self._get_expected_outcomes_for_strategy(strategy, topic_analysis),
                    confidence_score=sequence.confidence,
                    research_focus=self._get_research_focus_for_strategy(strategy, topic_analysis)
                )
                generated_sequences.append(agent_sequence)
                
            except Exception as e:
                logger.warning("Failed to generate sequence for strategy %s: %s", strategy, e)
                continue
        
        # Step 4: Rank sequences by topic fit
        ranked_sequences = self._rank_agent_sequences_by_topic(generated_sequences, topic_analysis)
        
        # Create output
        output = SequenceGenerationOutput(
            research_analysis=self._create_research_analysis(topic_analysis),
            sequences=ranked_sequences,
            reasoning_summary=f"Rule-based sequence generation using {len(strategies)} strategies based on topic analysis",
            recommended_sequence=0 if ranked_sequences else 0,
            alternative_considerations=self._get_alternative_considerations(topic_analysis),
            generation_mode="rule_based",
            topic_analysis=topic_analysis
        )
        
        generation_time = time.time() - start_time
        
        metadata = SequenceGenerationMetadata(
            generation_timestamp=generation_timestamp,
            model_used="rule_based_analyzer",
            input_token_count=None,
            output_token_count=None,
            generation_time_seconds=generation_time,
            fallback_used=False,
            generation_mode="rule_based"
        )
        
        return SequenceGenerationResult(
            output=output,
            metadata=metadata,
            success=True
        )
    
    def analyze_topic_characteristics(self, research_topic: str) -> TopicAnalysis:
        """Analyze research topic to determine optimal sequence characteristics.
        
        This method is consolidated from the orchestration sequence generator
        with improved accuracy and additional characteristics.
        """
        logger.debug("Analyzing topic characteristics for: %s", research_topic[:50])
        
        topic_lower = research_topic.lower()
        
        # Extract keywords
        keywords = self._extract_keywords(research_topic)
        
        # Calculate domain indicators
        domain_indicators = self._calculate_domain_indicators(topic_lower)
        
        # Determine primary topic type
        topic_type = self._classify_topic_type(domain_indicators, topic_lower)
        
        # Calculate complexity metrics
        complexity_score = self._calculate_complexity_score(research_topic, keywords)
        scope_breadth = self._calculate_scope_breadth(research_topic, keywords)
        
        # Estimate agents needed
        estimated_agents = self._estimate_agents_needed(
            complexity_score, scope_breadth, len(keywords)
        )
        
        # Identify priority areas
        priority_areas = self._identify_priority_areas(domain_indicators, keywords)
        
        # Calculate additional characteristics
        time_sensitivity = self._calculate_time_sensitivity(topic_lower)
        data_intensity = self._calculate_data_intensity(topic_lower)
        market_relevance = self._calculate_market_relevance(topic_lower)
        technical_complexity = self._calculate_technical_complexity(topic_lower)
        
        analysis = TopicAnalysis(
            topic_type=topic_type,
            complexity_score=complexity_score,
            scope_breadth=scope_breadth,
            keywords=keywords,
            domain_indicators=domain_indicators,
            estimated_agents_needed=estimated_agents,
            priority_areas=priority_areas,
            time_sensitivity=time_sensitivity,
            data_intensity=data_intensity,
            market_relevance=market_relevance,
            technical_complexity=technical_complexity
        )
        
        logger.debug("Topic analysis complete: type=%s, complexity=%.2f, agents=%d",
                    analysis.topic_type, analysis.complexity_score, analysis.estimated_agents_needed)
        
        return analysis
    
    # =====================================================================
    # LLM-BASED GENERATION (from supervisor.llm_sequence_generator)
    # =====================================================================
    
    async def _generate_llm_based(
        self,
        input_data: SequenceGenerationInput,
        start_time: float,
        generation_timestamp: str
    ) -> SequenceGenerationResult:
        """Generate sequences using LLM-based approach."""
        
        try:
            # Prepare messages
            system_message = SystemMessage(content=self._llm_system_prompt)
            user_message = HumanMessage(content=self._create_llm_user_prompt(input_data))
            
            messages = [system_message, user_message]
            
            # Calculate input tokens (approximate)
            input_text = self._llm_system_prompt + self._create_llm_user_prompt(input_data)
            input_token_count = len(input_text.split()) * 1.3  # Rough approximation
            
            # Generate response using configurable model
            model = configurable_model.with_config(self.model_config)
            
            logger.info(f"Generating sequences for research topic: {input_data.research_topic}")
            raw_response = await model.ainvoke(messages)
            
            # Extract and parse JSON response
            response = self._parse_llm_response(raw_response.content)
            
            generation_time = time.time() - start_time
            
            # Calculate output tokens (approximate)
            output_text = json.dumps(response.dict() if hasattr(response, 'dict') else str(response))
            output_token_count = len(output_text.split()) * 1.3
            
            # Create metadata
            metadata = SequenceGenerationMetadata(
                generation_timestamp=generation_timestamp,
                model_used=str(model),
                input_token_count=int(input_token_count),
                output_token_count=int(output_token_count),
                generation_time_seconds=generation_time,
                fallback_used=False,
                generation_mode="llm_based"
            )
            
            logger.info(f"Successfully generated {len(response.sequences)} sequences in {generation_time:.2f}s")
            
            return SequenceGenerationResult(
                output=response,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"LLM sequence generation failed: {e}")
            # Use fallback generation
            return await self._generate_fallback(input_data, start_time, generation_timestamp, str(e))
    
    def _generate_llm_based_sync(
        self,
        input_data: SequenceGenerationInput,
        start_time: float,
        generation_timestamp: str
    ) -> SequenceGenerationResult:
        """Generate sequences using LLM-based approach (synchronous)."""
        
        try:
            # Prepare messages
            system_message = SystemMessage(content=self._llm_system_prompt)
            user_message = HumanMessage(content=self._create_llm_user_prompt(input_data))
            
            messages = [system_message, user_message]
            
            # Calculate input tokens (approximate)
            input_text = self._llm_system_prompt + self._create_llm_user_prompt(input_data)
            input_token_count = len(input_text.split()) * 1.3  # Rough approximation
            
            # Generate response using configurable model
            model = configurable_model.with_config(self.model_config)
            
            logger.info(f"Generating sequences for research topic: {input_data.research_topic}")
            raw_response = model.invoke(messages)
            
            # Extract and parse JSON response
            response = self._parse_llm_response(raw_response.content)
            
            generation_time = time.time() - start_time
            
            # Calculate output tokens (approximate)
            output_text = json.dumps(response.dict() if hasattr(response, 'dict') else str(response))
            output_token_count = len(output_text.split()) * 1.3
            
            # Create metadata
            metadata = SequenceGenerationMetadata(
                generation_timestamp=generation_timestamp,
                model_used=str(model),
                input_token_count=int(input_token_count),
                output_token_count=int(output_token_count),
                generation_time_seconds=generation_time,
                fallback_used=False,
                generation_mode="llm_based"
            )
            
            logger.info(f"Successfully generated {len(response.sequences)} sequences in {generation_time:.2f}s")
            
            return SequenceGenerationResult(
                output=response,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"LLM sequence generation failed: {e}")
            # Use fallback generation
            return self._generate_fallback_sync(input_data, start_time, generation_timestamp, str(e))
    
    # =====================================================================
    # HYBRID GENERATION (Best of Both Worlds)
    # =====================================================================
    
    async def _generate_hybrid(
        self,
        input_data: SequenceGenerationInput,
        start_time: float,
        generation_timestamp: str
    ) -> SequenceGenerationResult:
        """Generate sequences using hybrid approach combining rule-based and LLM methods."""
        
        try:
            # Step 1: Get topic analysis from rule-based approach
            topic_analysis = self.analyze_topic_characteristics(input_data.research_topic)
            
            # Step 2: Generate some sequences using rule-based approach
            rule_based_input = SequenceGenerationInput(
                research_topic=input_data.research_topic,
                research_brief=input_data.research_brief,
                available_agents=input_data.available_agents,
                research_type=input_data.research_type,
                constraints=input_data.constraints,
                generation_mode="rule_based",
                num_sequences=max(1, input_data.num_sequences // 2),  # Half from rule-based
                strategies=input_data.strategies
            )
            
            rule_result = await self._generate_rule_based(rule_based_input, start_time, generation_timestamp)
            rule_sequences = rule_result.output.sequences
            
            # Step 3: Generate remaining sequences using LLM approach
            llm_based_input = SequenceGenerationInput(
                research_topic=input_data.research_topic,
                research_brief=input_data.research_brief,
                available_agents=input_data.available_agents,
                research_type=input_data.research_type,
                constraints=input_data.constraints,
                generation_mode="llm_based",
                num_sequences=input_data.num_sequences - len(rule_sequences),
                strategies=None
            )
            
            if llm_based_input.num_sequences > 0:
                try:
                    llm_result = await self._generate_llm_based(llm_based_input, start_time, generation_timestamp)
                    llm_sequences = llm_result.output.sequences
                except Exception as e:
                    logger.warning(f"LLM generation failed in hybrid mode: {e}")
                    llm_sequences = []
            else:
                llm_sequences = []
            
            # Step 4: Combine and rank all sequences
            all_sequences = rule_sequences + llm_sequences
            
            # Limit to requested number and rank by confidence
            all_sequences = sorted(all_sequences, key=lambda s: s.confidence_score, reverse=True)
            final_sequences = all_sequences[:input_data.num_sequences]
            
            # Create combined output
            output = SequenceGenerationOutput(
                research_analysis=self._create_research_analysis(topic_analysis),
                sequences=final_sequences,
                reasoning_summary=f"Hybrid generation combining rule-based ({len(rule_sequences)} sequences) and LLM-based ({len(llm_sequences)} sequences) approaches",
                recommended_sequence=0 if final_sequences else 0,
                alternative_considerations=self._get_alternative_considerations(topic_analysis),
                generation_mode="hybrid",
                topic_analysis=topic_analysis
            )
            
            generation_time = time.time() - start_time
            
            metadata = SequenceGenerationMetadata(
                generation_timestamp=generation_timestamp,
                model_used="hybrid_rule_and_llm",
                input_token_count=None,  # Mixed tokens from both approaches
                output_token_count=None,
                generation_time_seconds=generation_time,
                fallback_used=False,
                generation_mode="hybrid"
            )
            
            return SequenceGenerationResult(
                output=output,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Hybrid sequence generation failed: {e}")
            # Fallback to rule-based only
            return await self._generate_fallback(input_data, start_time, generation_timestamp, str(e))
    
    def _generate_hybrid_sync(
        self,
        input_data: SequenceGenerationInput,
        start_time: float,
        generation_timestamp: str
    ) -> SequenceGenerationResult:
        """Generate sequences using hybrid approach (synchronous)."""
        
        try:
            # Step 1: Get topic analysis from rule-based approach
            topic_analysis = self.analyze_topic_characteristics(input_data.research_topic)
            
            # Step 2: Generate some sequences using rule-based approach
            rule_based_input = SequenceGenerationInput(
                research_topic=input_data.research_topic,
                research_brief=input_data.research_brief,
                available_agents=input_data.available_agents,
                research_type=input_data.research_type,
                constraints=input_data.constraints,
                generation_mode="rule_based",
                num_sequences=max(1, input_data.num_sequences // 2),  # Half from rule-based
                strategies=input_data.strategies
            )
            
            rule_result = self._generate_rule_based_sync(rule_based_input, start_time, generation_timestamp)
            rule_sequences = rule_result.output.sequences
            
            # Step 3: Generate remaining sequences using LLM approach
            llm_based_input = SequenceGenerationInput(
                research_topic=input_data.research_topic,
                research_brief=input_data.research_brief,
                available_agents=input_data.available_agents,
                research_type=input_data.research_type,
                constraints=input_data.constraints,
                generation_mode="llm_based",
                num_sequences=input_data.num_sequences - len(rule_sequences),
                strategies=None
            )
            
            if llm_based_input.num_sequences > 0:
                try:
                    llm_result = self._generate_llm_based_sync(llm_based_input, start_time, generation_timestamp)
                    llm_sequences = llm_result.output.sequences
                except Exception as e:
                    logger.warning(f"LLM generation failed in hybrid mode: {e}")
                    llm_sequences = []
            else:
                llm_sequences = []
            
            # Step 4: Combine and rank all sequences
            all_sequences = rule_sequences + llm_sequences
            
            # Limit to requested number and rank by confidence
            all_sequences = sorted(all_sequences, key=lambda s: s.confidence_score, reverse=True)
            final_sequences = all_sequences[:input_data.num_sequences]
            
            # Create combined output
            output = SequenceGenerationOutput(
                research_analysis=self._create_research_analysis(topic_analysis),
                sequences=final_sequences,
                reasoning_summary=f"Hybrid generation combining rule-based ({len(rule_sequences)} sequences) and LLM-based ({len(llm_sequences)} sequences) approaches",
                recommended_sequence=0 if final_sequences else 0,
                alternative_considerations=self._get_alternative_considerations(topic_analysis),
                generation_mode="hybrid",
                topic_analysis=topic_analysis
            )
            
            generation_time = time.time() - start_time
            
            metadata = SequenceGenerationMetadata(
                generation_timestamp=generation_timestamp,
                model_used="hybrid_rule_and_llm",
                input_token_count=None,  # Mixed tokens from both approaches
                output_token_count=None,
                generation_time_seconds=generation_time,
                fallback_used=False,
                generation_mode="hybrid"
            )
            
            return SequenceGenerationResult(
                output=output,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Hybrid sequence generation failed: {e}")
            # Fallback to rule-based only
            return self._generate_fallback_sync(input_data, start_time, generation_timestamp, str(e))
    
    # =====================================================================
    # FALLBACK GENERATION
    # =====================================================================
    
    async def _generate_fallback(
        self,
        input_data: SequenceGenerationInput,
        start_time: float,
        generation_timestamp: str,
        error: str
    ) -> SequenceGenerationResult:
        """Generate fallback sequences when primary generation fails."""
        return self._generate_fallback_sync(input_data, start_time, generation_timestamp, error)
    
    def _generate_fallback_sync(
        self,
        input_data: SequenceGenerationInput,
        start_time: float,
        generation_timestamp: str,
        error: str
    ) -> SequenceGenerationResult:
        """Generate fallback sequences when primary generation fails (synchronous)."""
        logger.warning("Using fallback sequence generation")
        
        agent_names = [agent.name for agent in input_data.available_agents]
        
        # Simple fallback strategy: create basic sequences
        sequences = []
        
        # Sequence 1: Linear approach with first available agents
        if len(agent_names) >= 1:
            seq1_agents = agent_names[:min(3, len(agent_names))]
            sequences.append(AgentSequence(
                sequence_name="Linear Research Approach",
                agent_names=seq1_agents,
                rationale="Sequential investigation using available research agents",
                approach_description="Systematic research approach with available agents",
                expected_outcomes=["Comprehensive research coverage", "Detailed findings"],
                confidence_score=0.6,
                research_focus="General research investigation"
            ))
        
        # Sequence 2: Alternative combination if enough agents
        if len(agent_names) >= 2:
            seq2_agents = agent_names[1:min(4, len(agent_names))]
            if not seq2_agents:
                seq2_agents = agent_names[:1]
            sequences.append(AgentSequence(
                sequence_name="Alternative Research Path",
                agent_names=seq2_agents,
                rationale="Alternative research sequence for different perspective",
                approach_description="Alternative approach to research investigation",
                expected_outcomes=["Different research angles", "Complementary insights"],
                confidence_score=0.5,
                research_focus="Alternative research perspective"
            ))
        
        # Sequence 3: Focused approach
        if len(agent_names) >= 1:
            seq3_agents = [agent_names[0]] if agent_names else []
            if len(agent_names) >= 3:
                seq3_agents.append(agent_names[-1])
            sequences.append(AgentSequence(
                sequence_name="Focused Research Strategy",
                agent_names=seq3_agents,
                rationale="Focused research with selected specialized agents",
                approach_description="Targeted research approach with key agents",
                expected_outcomes=["Focused insights", "Targeted analysis"],
                confidence_score=0.4,
                research_focus="Focused research investigation"
            ))
        
        # Ensure we have at least the requested number of sequences
        while len(sequences) < input_data.num_sequences:
            sequences.append(AgentSequence(
                sequence_name=f"Basic Research {len(sequences) + 1}",
                agent_names=agent_names[:1] if agent_names else [],
                rationale="Basic research approach with available resources",
                approach_description="Fundamental research investigation",
                expected_outcomes=["Basic research coverage"],
                confidence_score=0.3,
                research_focus="Basic research"
            ))
        
        output = SequenceGenerationOutput(
            research_analysis=f"Fallback analysis for research topic: {input_data.research_topic}",
            sequences=sequences[:input_data.num_sequences],
            reasoning_summary="Generated using fallback logic due to primary generation failure",
            recommended_sequence=0,
            alternative_considerations=["Consider manual sequence customization", "Retry with different generation mode"],
            generation_mode="fallback"
        )
        
        generation_time = time.time() - start_time
        
        metadata = SequenceGenerationMetadata(
            generation_timestamp=generation_timestamp,
            model_used="fallback",
            input_token_count=None,
            output_token_count=None,
            generation_time_seconds=generation_time,
            fallback_used=True,
            error_details=error,
            generation_mode="fallback"
        )
        
        return SequenceGenerationResult(
            output=output,
            metadata=metadata,
            success=False
        )
    
    # =====================================================================
    # PRIVATE HELPER METHODS - Consolidated from Both Modules
    # =====================================================================
    
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize domain classification patterns."""
        return {
            "academic": [
                r"\b(?:research|study|analysis|literature|academic|scholarly|peer.?review|journal|publication)\b",
                r"\b(?:hypothesis|methodology|findings|results|conclusion|abstract|citation)\b",
                r"\b(?:experiment|survey|interview|questionnaire|data.?collection|sample|population)\b"
            ],
            "market": [
                r"\b(?:market|business|commercial|revenue|profit|sales|customer|client|consumer)\b",
                r"\b(?:competition|competitor|pricing|strategy|growth|opportunity|roi|investment)\b",
                r"\b(?:brand|marketing|advertising|promotion|market.?share|target.?audience)\b"
            ],
            "technical": [
                r"\b(?:technology|technical|software|hardware|system|platform|architecture|design)\b",
                r"\b(?:programming|coding|development|implementation|deployment|infrastructure)\b",
                r"\b(?:algorithm|framework|api|database|server|cloud|security|performance)\b"
            ],
            "analysis": [
                r"\b(?:analysis|analyze|statistical|quantitative|qualitative|metrics|measurement)\b",
                r"\b(?:data|dataset|visualization|pattern|trend|correlation|regression)\b",
                r"\b(?:comparison|benchmark|evaluation|assessment|score|rating|performance)\b"
            ]
        }
    
    def _initialize_agent_mappings(self) -> Dict[str, List[str]]:
        """Initialize agent type mappings based on agent names and expertise."""
        return {
            "research": ["research_agent", "academic_agent", "literature_agent"],
            "analysis": ["analysis_agent", "data_agent", "statistical_agent"],
            "market": ["market_agent", "business_agent", "commercial_agent"],
            "technical": ["technical_agent", "engineering_agent", "implementation_agent"],
            "synthesis": ["synthesis_agent", "summary_agent", "integration_agent"]
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from research topic."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after",
            "above", "below", "between", "among", "an", "a", "as", "are", "was", "were",
            "is", "be", "been", "being", "have", "has", "had", "do", "does", "did"
        }
        
        # Extract words (3+ characters, alphanumeric)
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:20]  # Limit to top 20 keywords
    
    def _calculate_domain_indicators(self, text: str) -> Dict[str, float]:
        """Calculate confidence scores for different domains."""
        domain_scores = {}
        
        for domain, patterns in self._domain_patterns.items():
            score = 0.0
            total_matches = 0
            
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
                total_matches += matches
            
            # Normalize by text length and pattern complexity
            text_length_factor = min(1.0, len(text) / 500)  # Normalize for text length
            domain_scores[domain] = min(1.0, (score * text_length_factor) / max(1, len(patterns)))
        
        return domain_scores
    
    def _classify_topic_type(self, domain_indicators: Dict[str, float], text: str) -> TopicType:
        """Classify the primary topic type based on domain indicators."""
        max_score = max(domain_indicators.values()) if domain_indicators else 0
        
        if max_score < 0.2:
            return TopicType.MIXED
        
        # Find dominant domain
        dominant_domain = max(domain_indicators.items(), key=lambda x: x[1])[0]
        
        # Check for mixed topics (multiple high scores)
        high_score_domains = [d for d, s in domain_indicators.items() if s > 0.3]
        
        if len(high_score_domains) > 2:
            return TopicType.MIXED
        
        # Map domain to topic type
        domain_to_topic = {
            "academic": TopicType.ACADEMIC,
            "market": TopicType.MARKET,
            "technical": TopicType.TECHNICAL,
            "analysis": TopicType.ANALYSIS
        }
        
        return domain_to_topic.get(dominant_domain, TopicType.MIXED)
    
    def _calculate_complexity_score(self, text: str, keywords: List[str]) -> float:
        """Calculate complexity score based on text characteristics."""
        factors = [
            len(text) > 200,                    # Substantial description
            len(keywords) > 10,                 # Rich keyword set
            len(text.split('. ')) > 3,         # Multiple sentences
            any(word in text.lower() for word in ['complex', 'comprehensive', 'detailed', 'in-depth']),
            len(set(keywords)) / max(1, len(keywords)) > 0.8,  # Keyword diversity
        ]
        
        return sum(factors) / len(factors)
    
    def _calculate_scope_breadth(self, text: str, keywords: List[str]) -> float:
        """Calculate scope breadth (narrow vs. broad research)."""
        breadth_indicators = [
            any(word in text.lower() for word in ['comprehensive', 'overview', 'survey', 'review']),
            any(word in text.lower() for word in ['multiple', 'various', 'different', 'across']),
            len(keywords) > 15,  # Many different concepts
            len(text.split(' and ')) > 3,  # Multiple connected topics
        ]
        
        return sum(breadth_indicators) / len(breadth_indicators)
    
    def _estimate_agents_needed(self, complexity: float, breadth: float, keyword_count: int) -> int:
        """Estimate number of agents needed for the research."""
        base_agents = 2  # Minimum: research + synthesis
        
        # Add agents based on complexity
        if complexity > 0.7:
            base_agents += 2
        elif complexity > 0.4:
            base_agents += 1
        
        # Add agents based on breadth
        if breadth > 0.6:
            base_agents += 1
        
        # Add agents based on keyword richness
        if keyword_count > 15:
            base_agents += 1
        
        return min(5, max(2, base_agents))  # Cap between 2-5 agents
    
    def _identify_priority_areas(self, domain_indicators: Dict[str, float], keywords: List[str]) -> List[str]:
        """Identify priority research areas."""
        # Get top domains by score
        sorted_domains = sorted(domain_indicators.items(), key=lambda x: x[1], reverse=True)
        priority_areas = [domain for domain, score in sorted_domains if score > 0.2]
        
        # Add keyword-based priorities
        if any(word in ' '.join(keywords) for word in ['data', 'analysis', 'statistical']):
            if 'analysis' not in priority_areas:
                priority_areas.append('analysis')
        
        return priority_areas[:4]  # Limit to top 4 priorities
    
    def _calculate_time_sensitivity(self, text: str) -> float:
        """Calculate time sensitivity of the research topic."""
        time_sensitive_terms = ['urgent', 'immediate', 'current', 'latest', 'recent', 'now', 'today', '2024', '2025']
        sensitive_count = sum(1 for term in time_sensitive_terms if term in text)
        return min(1.0, sensitive_count / 3)
    
    def _calculate_data_intensity(self, text: str) -> float:
        """Calculate data intensity requirements."""
        data_terms = ['data', 'dataset', 'analysis', 'statistical', 'quantitative', 'metrics', 'measurement']
        data_count = sum(1 for term in data_terms if term in text)
        return min(1.0, data_count / 4)
    
    def _calculate_market_relevance(self, text: str) -> float:
        """Calculate market relevance of the topic."""
        market_terms = ['market', 'business', 'commercial', 'revenue', 'profit', 'customer', 'competitor']
        market_count = sum(1 for term in market_terms if term in text)
        return min(1.0, market_count / 3)
    
    def _calculate_technical_complexity(self, text: str) -> float:
        """Calculate technical complexity requirements."""
        tech_terms = ['technical', 'technology', 'implementation', 'architecture', 'system', 'software', 'platform']
        tech_count = sum(1 for term in tech_terms if term in text)
        return min(1.0, tech_count / 3)
    
    def _select_strategies_for_topic(self, topic_analysis: TopicAnalysis, num_sequences: int) -> List[SequenceStrategy]:
        """Select appropriate strategies based on topic analysis."""
        strategies = []
        
        # Primary strategy based on topic type
        if topic_analysis.topic_type == TopicType.ACADEMIC:
            strategies.append(SequenceStrategy.THEORY_FIRST)
        elif topic_analysis.topic_type == TopicType.MARKET:
            strategies.append(SequenceStrategy.MARKET_FIRST)
        elif topic_analysis.topic_type == TopicType.TECHNICAL:
            strategies.append(SequenceStrategy.TECHNICAL_FIRST)
        elif topic_analysis.topic_type == TopicType.ANALYSIS:
            strategies.append(SequenceStrategy.ANALYSIS_FIRST)
        else:
            strategies.append(SequenceStrategy.BALANCED)
        
        # Add complementary strategies based on secondary characteristics
        remaining_strategies = [s for s in SequenceStrategy if s not in strategies and s not in [SequenceStrategy.CUSTOM, SequenceStrategy.LLM_GENERATED]]
        
        # Prioritize based on topic characteristics
        if topic_analysis.market_relevance > 0.5 and SequenceStrategy.MARKET_FIRST not in strategies:
            strategies.append(SequenceStrategy.MARKET_FIRST)
        
        if topic_analysis.technical_complexity > 0.5 and SequenceStrategy.TECHNICAL_FIRST not in strategies:
            strategies.append(SequenceStrategy.TECHNICAL_FIRST)
        
        if topic_analysis.data_intensity > 0.5 and SequenceStrategy.ANALYSIS_FIRST not in strategies:
            strategies.append(SequenceStrategy.ANALYSIS_FIRST)
        
        # Fill remaining slots
        for strategy in remaining_strategies:
            if len(strategies) >= num_sequences:
                break
            if strategy not in strategies:
                strategies.append(strategy)
        
        return strategies[:num_sequences]
    
    def _generate_sequence_for_strategy(
        self,
        strategy: SequenceStrategy,
        available_agents: List[str],
        topic_analysis: TopicAnalysis
    ) -> GeneratedSequence:
        """Generate a sequence for a specific strategy."""
        
        # Generate agent sequence based on strategy
        if strategy == SequenceStrategy.THEORY_FIRST:
            agents = self._create_theory_first_sequence(available_agents, topic_analysis)
            rationale = "Academic foundation → analytical insights → market context → technical implementation"
        elif strategy == SequenceStrategy.MARKET_FIRST:
            agents = self._create_market_first_sequence(available_agents, topic_analysis)
            rationale = "Market opportunity → technical feasibility → research validation → analytical synthesis"
        elif strategy == SequenceStrategy.TECHNICAL_FIRST:
            agents = self._create_technical_first_sequence(available_agents, topic_analysis)
            rationale = "Technical constraints → research foundation → analytical insights → market validation"
        elif strategy == SequenceStrategy.ANALYSIS_FIRST:
            agents = self._create_analysis_first_sequence(available_agents, topic_analysis)
            rationale = "Data analysis → research foundation → market context → technical implementation"
        elif strategy == SequenceStrategy.BALANCED:
            agents = self._create_balanced_sequence(available_agents, topic_analysis)
            rationale = "Balanced approach considering all domain requirements equally"
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        # Ensure we have at least some agents
        if not agents:
            agents = available_agents[:min(3, len(available_agents))]
        
        # Estimate duration (rough heuristic)
        estimated_duration = len(agents) * 0.5 + topic_analysis.complexity_score * 2
        
        # Create sequence name
        sequence_name = f"{strategy.value}_{len(agents)}_agents"
        
        return GeneratedSequence(
            sequence_name=sequence_name,
            strategy=strategy,
            agents=agents,
            score=0.0,  # Will be calculated in ranking
            rationale=rationale,
            estimated_duration=estimated_duration,
            confidence=0.0  # Will be calculated in ranking
        )
    
    def _create_theory_first_sequence(self, agents: List[str], topic_analysis: TopicAnalysis) -> List[str]:
        """Create theory-first sequence: Academic → Analysis → Market → Technical."""
        sequence = []
        used_agents = set()
        
        # Step 1: Research/Academic agents first
        research_agents = self._get_agents_by_type("research", agents)
        if research_agents:
            best_research = self._select_best_agent(research_agents, "research", topic_analysis)
            if best_research:
                sequence.append(best_research)
                used_agents.add(best_research)
        
        # Step 2: Analysis agents
        analysis_agents = [a for a in self._get_agents_by_type("analysis", agents) if a not in used_agents]
        if analysis_agents:
            best_analysis = self._select_best_agent(analysis_agents, "analysis", topic_analysis)
            if best_analysis:
                sequence.append(best_analysis)
                used_agents.add(best_analysis)
        
        # Step 3: Market agents (if relevant)
        if topic_analysis.market_relevance > 0.3:
            market_agents = [a for a in self._get_agents_by_type("market", agents) if a not in used_agents]
            if market_agents:
                best_market = self._select_best_agent(market_agents, "market", topic_analysis)
                if best_market:
                    sequence.append(best_market)
                    used_agents.add(best_market)
        
        # Step 4: Technical agents (if relevant)
        if topic_analysis.technical_complexity > 0.4:
            technical_agents = [a for a in self._get_agents_by_type("technical", agents) if a not in used_agents]
            if technical_agents:
                best_technical = self._select_best_agent(technical_agents, "technical", topic_analysis)
                if best_technical:
                    sequence.append(best_technical)
                    used_agents.add(best_technical)
        
        # Step 5: Synthesis agent if available
        synthesis_agents = [a for a in self._get_agents_by_type("synthesis", agents) if a not in used_agents]
        if synthesis_agents and len(sequence) >= 2:
            best_synthesis = self._select_best_agent(synthesis_agents, "synthesis", topic_analysis)
            if best_synthesis:
                sequence.append(best_synthesis)
        
        return sequence
    
    def _create_market_first_sequence(self, agents: List[str], topic_analysis: TopicAnalysis) -> List[str]:
        """Create market-first sequence: Market → Technical → Research → Analysis."""
        sequence = []
        used_agents = set()
        
        # Step 1: Market agents first
        market_agents = self._get_agents_by_type("market", agents)
        if market_agents:
            best_market = self._select_best_agent(market_agents, "market", topic_analysis)
            if best_market:
                sequence.append(best_market)
                used_agents.add(best_market)
        
        # Step 2: Technical agents (for implementation context)
        technical_agents = [a for a in self._get_agents_by_type("technical", agents) if a not in used_agents]
        if technical_agents:
            best_technical = self._select_best_agent(technical_agents, "technical", topic_analysis)
            if best_technical:
                sequence.append(best_technical)
                used_agents.add(best_technical)
        
        # Step 3: Research agents (to validate market assumptions)
        research_agents = [a for a in self._get_agents_by_type("research", agents) if a not in used_agents]
        if research_agents:
            best_research = self._select_best_agent(research_agents, "research", topic_analysis)
            if best_research:
                sequence.append(best_research)
                used_agents.add(best_research)
        
        # Step 4: Analysis agents
        analysis_agents = [a for a in self._get_agents_by_type("analysis", agents) if a not in used_agents]
        if analysis_agents:
            best_analysis = self._select_best_agent(analysis_agents, "analysis", topic_analysis)
            if best_analysis:
                sequence.append(best_analysis)
                used_agents.add(best_analysis)
        
        # Step 5: Synthesis agent if needed
        synthesis_agents = [a for a in self._get_agents_by_type("synthesis", agents) if a not in used_agents]
        if synthesis_agents and len(sequence) >= 3:
            best_synthesis = self._select_best_agent(synthesis_agents, "synthesis", topic_analysis)
            if best_synthesis:
                sequence.append(best_synthesis)
        
        return sequence
    
    def _create_technical_first_sequence(self, agents: List[str], topic_analysis: TopicAnalysis) -> List[str]:
        """Create technical-first sequence: Technical → Research → Analysis → Market."""
        sequence = []
        used_agents = set()
        
        # Step 1: Technical agents first
        technical_agents = self._get_agents_by_type("technical", agents)
        if technical_agents:
            best_technical = self._select_best_agent(technical_agents, "technical", topic_analysis)
            if best_technical:
                sequence.append(best_technical)
                used_agents.add(best_technical)
        
        # Step 2: Research agents (for technical validation)
        research_agents = [a for a in self._get_agents_by_type("research", agents) if a not in used_agents]
        if research_agents:
            best_research = self._select_best_agent(research_agents, "research", topic_analysis)
            if best_research:
                sequence.append(best_research)
                used_agents.add(best_research)
        
        # Step 3: Analysis agents
        analysis_agents = [a for a in self._get_agents_by_type("analysis", agents) if a not in used_agents]
        if analysis_agents:
            best_analysis = self._select_best_agent(analysis_agents, "analysis", topic_analysis)
            if best_analysis:
                sequence.append(best_analysis)
                used_agents.add(best_analysis)
        
        # Step 4: Market agents (if relevant)
        if topic_analysis.market_relevance > 0.2:
            market_agents = [a for a in self._get_agents_by_type("market", agents) if a not in used_agents]
            if market_agents:
                best_market = self._select_best_agent(market_agents, "market", topic_analysis)
                if best_market:
                    sequence.append(best_market)
                    used_agents.add(best_market)
        
        # Step 5: Synthesis agent if needed
        synthesis_agents = [a for a in self._get_agents_by_type("synthesis", agents) if a not in used_agents]
        if synthesis_agents and len(sequence) >= 2:
            best_synthesis = self._select_best_agent(synthesis_agents, "synthesis", topic_analysis)
            if best_synthesis:
                sequence.append(best_synthesis)
        
        return sequence
    
    def _create_analysis_first_sequence(self, agents: List[str], topic_analysis: TopicAnalysis) -> List[str]:
        """Create analysis-first sequence: Analysis → Research → Market → Technical."""
        sequence = []
        used_agents = set()
        
        # Start with analysis agents
        analysis_agents = self._get_agents_by_type("analysis", agents)
        if analysis_agents:
            best_analysis = self._select_best_agent(analysis_agents, "analysis", topic_analysis)
            if best_analysis:
                sequence.append(best_analysis)
                used_agents.add(best_analysis)
        
        # Add other agents in order
        for agent_type in ["research", "market", "technical", "synthesis"]:
            type_agents = [a for a in self._get_agents_by_type(agent_type, agents) if a not in used_agents]
            if type_agents:
                if agent_type == "market" and topic_analysis.market_relevance < 0.3:
                    continue  # Skip if not market-relevant
                if agent_type == "technical" and topic_analysis.technical_complexity < 0.3:
                    continue  # Skip if not technically complex
                
                best_agent = self._select_best_agent(type_agents, agent_type, topic_analysis)
                if best_agent:
                    sequence.append(best_agent)
                    used_agents.add(best_agent)
        
        return sequence
    
    def _create_balanced_sequence(self, agents: List[str], topic_analysis: TopicAnalysis) -> List[str]:
        """Create balanced sequence based on topic priority areas."""
        sequence = []
        used_agents = set()
        
        # Order by priority areas from topic analysis
        priority_order = topic_analysis.priority_areas.copy()
        
        # Ensure we have a logical flow
        if "research" in priority_order and "analysis" in priority_order:
            # Put research before analysis
            if priority_order.index("research") > priority_order.index("analysis"):
                priority_order.remove("research")
                priority_order.insert(priority_order.index("analysis"), "research")
        
        # Add agents based on priority order
        for priority in priority_order:
            type_agents = [a for a in self._get_agents_by_type(priority, agents) if a not in used_agents]
            if type_agents:
                best_agent = self._select_best_agent(type_agents, priority, topic_analysis)
                if best_agent:
                    sequence.append(best_agent)
                    used_agents.add(best_agent)
        
        # Add synthesis if we have multiple agents
        if len(sequence) >= 2:
            synthesis_agents = [a for a in self._get_agents_by_type("synthesis", agents) if a not in used_agents]
            if synthesis_agents:
                best_synthesis = self._select_best_agent(synthesis_agents, "synthesis", topic_analysis)
                if best_synthesis:
                    sequence.append(best_synthesis)
        
        return sequence
    
    def _get_agents_by_type(self, agent_type: str, available_agents: List[str]) -> List[str]:
        """Get agents of a specific type from available agents."""
        matching_agents = []
        
        # Check agent names for type indicators
        type_patterns = {
            "research": ["research", "academic", "literature"],
            "analysis": ["analysis", "data", "statistical", "analytic"],
            "market": ["market", "business", "commercial"],
            "technical": ["technical", "tech", "engineering", "implementation"],
            "synthesis": ["synthesis", "summary", "integration", "synth"]
        }
        
        patterns = type_patterns.get(agent_type, [agent_type])
        
        for agent in available_agents:
            agent_lower = agent.lower()
            if any(pattern in agent_lower for pattern in patterns):
                matching_agents.append(agent)
            else:
                # Check expertise areas from agent registry
                agent_config = self.agent_registry.get_agent(agent)
                if agent_config:
                    expertise = agent_config.get("expertise_areas", [])
                    expertise_text = " ".join(expertise).lower()
                    if any(pattern in expertise_text for pattern in patterns):
                        matching_agents.append(agent)
        
        return matching_agents
    
    def _select_best_agent(self, candidates: List[str], agent_type: str, topic_analysis: TopicAnalysis) -> Optional[str]:
        """Select the best agent from candidates based on topic analysis."""
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Score agents based on expertise match
        agent_scores = {}
        
        for agent in candidates:
            score = 0.0
            agent_config = self.agent_registry.get_agent(agent)
            
            if agent_config:
                expertise_areas = agent_config.get("expertise_areas", [])
                expertise_text = " ".join(expertise_areas).lower()
                
                # Score based on keyword overlap
                for keyword in topic_analysis.keywords:
                    if keyword in expertise_text:
                        score += 1.0
                
                # Normalize by number of expertise areas
                if expertise_areas:
                    score /= len(expertise_areas)
            
            agent_scores[agent] = score
        
        # Return agent with highest score
        return max(agent_scores.items(), key=lambda x: x[1])[0]
    
    def _rank_agent_sequences_by_topic(
        self,
        sequences: List[AgentSequence],
        topic_analysis: TopicAnalysis
    ) -> List[AgentSequence]:
        """Rank agent sequences by fitness for the specific topic."""
        logger.debug("Ranking %d sequences by topic fitness", len(sequences))
        
        # Calculate fitness scores based on agent capabilities and topic fit
        for sequence in sequences:
            # Calculate comprehensive fitness score
            topic_fit = self._calculate_agent_sequence_topic_fit(sequence, topic_analysis)
            coverage = self._calculate_agent_sequence_coverage(sequence, topic_analysis)
            
            # Update confidence score based on fitness
            sequence.confidence_score = min(1.0, (topic_fit + coverage) / 2)
            
            logger.debug("Sequence '%s' scored %.3f confidence (fit=%.2f, coverage=%.2f)",
                        sequence.sequence_name, sequence.confidence_score, topic_fit, coverage)
        
        # Sort by confidence score (descending)
        ranked_sequences = sorted(sequences, key=lambda s: s.confidence_score, reverse=True)
        
        logger.info("Ranked sequences: %s",
                   [f"{s.sequence_name}({s.confidence_score:.2f})" for s in ranked_sequences])
        
        return ranked_sequences
    
    def _calculate_agent_sequence_topic_fit(self, sequence: AgentSequence, topic_analysis: TopicAnalysis) -> float:
        """Calculate how well the sequence fits the topic."""
        # Simple heuristic based on agent types and topic requirements
        agent_types = []
        for agent_name in sequence.agent_names:
            for agent_type, patterns in self._agent_type_mappings.items():
                if any(pattern in agent_name.lower() for pattern in patterns):
                    agent_types.append(agent_type)
                    break
        
        # Check coverage of priority areas
        covered_areas = 0
        for priority_area in topic_analysis.priority_areas:
            if any(priority_area in agent_type for agent_type in agent_types):
                covered_areas += 1
        
        return covered_areas / max(1, len(topic_analysis.priority_areas))
    
    def _calculate_agent_sequence_coverage(self, sequence: AgentSequence, topic_analysis: TopicAnalysis) -> float:
        """Calculate domain coverage of the sequence."""
        # Check if sequence covers key domain requirements
        coverage_score = 0.0
        
        if topic_analysis.market_relevance > 0.5:
            if any("market" in agent.lower() or "business" in agent.lower() for agent in sequence.agent_names):
                coverage_score += 0.25
        
        if topic_analysis.technical_complexity > 0.5:
            if any("technical" in agent.lower() or "engineering" in agent.lower() for agent in sequence.agent_names):
                coverage_score += 0.25
        
        if topic_analysis.data_intensity > 0.5:
            if any("analysis" in agent.lower() or "data" in agent.lower() for agent in sequence.agent_names):
                coverage_score += 0.25
        
        # Bonus for having research/academic coverage
        if any("research" in agent.lower() or "academic" in agent.lower() for agent in sequence.agent_names):
            coverage_score += 0.25
        
        return coverage_score
    
    def _create_research_analysis(self, topic_analysis: TopicAnalysis) -> str:
        """Create research analysis summary from topic analysis."""
        return f"""Research Topic Analysis:
- Topic Type: {topic_analysis.topic_type.value}
- Complexity Score: {topic_analysis.complexity_score:.2f}
- Scope Breadth: {topic_analysis.scope_breadth:.2f}
- Estimated Agents Needed: {topic_analysis.estimated_agents_needed}
- Priority Areas: {', '.join(topic_analysis.priority_areas)}
- Market Relevance: {topic_analysis.market_relevance:.2f}
- Technical Complexity: {topic_analysis.technical_complexity:.2f}
- Data Intensity: {topic_analysis.data_intensity:.2f}

The research topic shows {topic_analysis.topic_type.value} characteristics with {topic_analysis.complexity_score:.1f} complexity level. 
Key focus areas include {', '.join(topic_analysis.priority_areas[:3])} requiring approximately {topic_analysis.estimated_agents_needed} specialized agents."""
    
    def _get_expected_outcomes_for_strategy(self, strategy: SequenceStrategy, topic_analysis: TopicAnalysis) -> List[str]:
        """Get expected outcomes for a given strategy."""
        base_outcomes = {
            SequenceStrategy.THEORY_FIRST: [
                "Solid theoretical foundation",
                "Evidence-based insights",
                "Academic rigor"
            ],
            SequenceStrategy.MARKET_FIRST: [
                "Market viability assessment",
                "Commercial opportunities",
                "Business model validation"
            ],
            SequenceStrategy.TECHNICAL_FIRST: [
                "Technical feasibility analysis",
                "Implementation roadmap",
                "Architecture recommendations"
            ],
            SequenceStrategy.ANALYSIS_FIRST: [
                "Data-driven insights",
                "Pattern recognition",
                "Quantitative analysis"
            ],
            SequenceStrategy.BALANCED: [
                "Comprehensive coverage",
                "Multi-perspective analysis",
                "Balanced insights"
            ]
        }
        
        outcomes = base_outcomes.get(strategy, ["Comprehensive research results"])
        
        # Add topic-specific outcomes
        if topic_analysis.market_relevance > 0.5:
            outcomes.append("Market insights")
        if topic_analysis.technical_complexity > 0.5:
            outcomes.append("Technical recommendations")
        if topic_analysis.data_intensity > 0.5:
            outcomes.append("Data analysis")
        
        return outcomes
    
    def _get_research_focus_for_strategy(self, strategy: SequenceStrategy, topic_analysis: TopicAnalysis) -> str:
        """Get research focus for a given strategy."""
        focus_map = {
            SequenceStrategy.THEORY_FIRST: "Academic research and theoretical foundation",
            SequenceStrategy.MARKET_FIRST: "Market analysis and commercial viability", 
            SequenceStrategy.TECHNICAL_FIRST: "Technical implementation and feasibility",
            SequenceStrategy.ANALYSIS_FIRST: "Data analysis and quantitative insights",
            SequenceStrategy.BALANCED: f"Balanced {topic_analysis.topic_type.value} research approach"
        }
        
        return focus_map.get(strategy, "Comprehensive research investigation")
    
    def _get_alternative_considerations(self, topic_analysis: TopicAnalysis) -> List[str]:
        """Get alternative considerations based on topic analysis."""
        considerations = []
        
        if topic_analysis.complexity_score > 0.7:
            considerations.append("Consider breaking down into sub-topics for deeper analysis")
        
        if topic_analysis.scope_breadth > 0.7:
            considerations.append("Consider focused approaches for specific domains")
        
        if topic_analysis.time_sensitivity > 0.5:
            considerations.append("Prioritize time-sensitive aspects and current data")
        
        if len(topic_analysis.priority_areas) > 3:
            considerations.append("Consider specialized sequences for each priority area")
        
        considerations.extend([
            "Manual sequence customization based on specific requirements",
            "Parallel execution of complementary sequences",
            "Iterative refinement based on initial findings"
        ])
        
        return considerations
    
    def _create_llm_system_prompt(self) -> str:
        """Create the system prompt for LLM-based sequence generation."""
        return """You are an expert research strategist and agent orchestration specialist. Your role is to analyze research topics and generate strategic sequences of specialized research agents.

## Your Task
Analyze the given research topic and available agents, then generate exactly 3 distinct strategic sequences for conducting comprehensive research.

## Key Principles
1. **Strategic Diversity**: Each sequence should represent a fundamentally different research approach
2. **Agent Synergy**: Agents should build on each other's work in logical progression  
3. **Comprehensive Coverage**: Sequences should collectively cover all important research angles
4. **Efficiency**: Avoid redundancy while ensuring thorough investigation

## Research Approach Types
- **Foundational-First**: Start with background/fundamentals, then specialized analysis
- **Problem-Solution**: Identify problems/gaps, then explore solutions and implementations
- **Comparative**: Compare different approaches, technologies, or methodologies
- **Stakeholder-Centric**: Analyze from different stakeholder perspectives
- **Timeline-Based**: Historical context → current state → future trends

## Agent Selection Guidelines
- Consider each agent's expertise areas and typical use cases
- Ensure logical information flow between agents
- Balance depth vs breadth based on research topic complexity
- Account for potential information dependencies

## Required JSON Output Structure
You must output a JSON object with exactly these fields:
```json
{
  "research_analysis": "string - Analysis of the research requirements",
  "sequences": [
    {
      "sequence_name": "string - Descriptive name for sequence strategy",
      "agent_names": ["string"] - Ordered list of agent names,
      "rationale": "string - Detailed reasoning for effectiveness",
      "approach_description": "string - High-level research approach",
      "expected_outcomes": ["string"] - Expected outcomes,
      "confidence_score": 0.0-1.0 - Confidence in effectiveness,
      "research_focus": "string - Primary research focus"
    }
    // exactly 3 sequence objects
  ],
  "reasoning_summary": "string - Summary of overall reasoning approach",
  "recommended_sequence": 0-2 - Index of recommended sequence,
  "alternative_considerations": ["string"] - Alternative approaches
}
```

## Output Requirements
- Exactly 3 sequences, each with 2-4 agents
- Clear rationale for each sequence's approach
- Confidence scores based on agent capabilities match to research needs
- Identify which sequence you recommend and why

CRITICAL: Respond with ONLY the valid JSON object structure shown above. Do not include any thinking, explanations, markdown code blocks, or text outside the JSON structure. Do not use <think> tags or any other XML tags. Start your response directly with the opening brace {."""
    
    def _create_llm_user_prompt(self, input_data: SequenceGenerationInput) -> str:
        """Create the user prompt with research context and available agents."""
        
        # Format available agents
        agent_descriptions = []
        for agent in input_data.available_agents:
            agent_desc = f"""
**{agent.name}**
- Expertise: {', '.join(agent.expertise_areas)}
- Description: {agent.description}
- Strength: {agent.strength_summary}
- Use Cases: {', '.join(agent.typical_use_cases)}
"""
            agent_descriptions.append(agent_desc)
        
        agents_text = "\n".join(agent_descriptions)
        
        # Format research context
        research_context = f"""
## Research Topic
{input_data.research_topic}
"""
        
        if input_data.research_brief:
            research_context += f"""
## Research Brief
{input_data.research_brief}
"""
        
        if input_data.research_type:
            research_context += f"""
## Research Type
{input_data.research_type}
"""
        
        # Format constraints if any
        constraints_text = ""
        if input_data.constraints:
            constraints_text = f"""
## Constraints
{json.dumps(input_data.constraints, indent=2)}
"""
        
        return f"""{research_context}

## Available Agents
{agents_text}
{constraints_text}

## Instructions
Analyze this research topic and generate exactly 3 strategic agent sequences. Each sequence should represent a different research approach and provide comprehensive coverage of the topic.

For each sequence:
1. Select 2-4 agents that work synergistically 
2. Order them logically for optimal information flow
3. Provide detailed rationale for the approach
4. Assign confidence score based on agent-topic fit

Output your response as valid JSON matching the SequenceGenerationOutput schema."""
    
    def _parse_llm_response(self, raw_content: str) -> SequenceGenerationOutput:
        """Parse the raw LLM response into a structured output."""
        logger.debug(f"Raw LLM response: {raw_content[:500]}...")  # Log first 500 chars
        
        # Try to extract JSON from the response
        try:
            # Look for JSON structure in the response
            # Handle cases where there might be code blocks or extra text
            
            # First try to find JSON code block
            json_start_markers = ['```json\n', '```\n{', '{']
            
            start_idx = -1
            
            # Try different JSON extraction strategies
            for start_marker in json_start_markers:
                start_pos = raw_content.find(start_marker)
                if start_pos != -1:
                    start_idx = start_pos + len(start_marker)
                    if start_marker == '{':
                        start_idx = start_pos  # Include the opening brace
                    break
            
            if start_idx == -1:
                raise ValueError("No JSON structure found in response")
            
            # Find the end of JSON
            brace_count = 0
            json_end_idx = -1
            
            for i, char in enumerate(raw_content[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end_idx = i + 1
                        break
            
            if json_end_idx == -1:
                # Fallback to rfind
                json_end_idx = raw_content.rfind('}') + 1
                if json_end_idx == 0:
                    raise ValueError("No closing brace found in JSON")
            
            json_content = raw_content[start_idx:json_end_idx].strip()
            logger.debug(f"Extracted JSON content: {json_content[:200]}...")
            
            parsed_json = json.loads(json_content)
            
            # Add generation mode to output if not present
            if "generation_mode" not in parsed_json:
                parsed_json["generation_mode"] = "llm_based"
            
            response = SequenceGenerationOutput(**parsed_json)
            return response
            
        except (json.JSONDecodeError, ValueError, ValidationError) as parse_error:
            logger.error(f"Failed to parse LLM response as JSON: {parse_error}")
            logger.debug(f"Problematic content: {raw_content}")
            raise Exception(f"JSON parsing failed: {parse_error}")