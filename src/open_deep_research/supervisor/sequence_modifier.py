"""Sequence modifier for dynamic agent sequence adjustments.

This module enables intelligent modification of agent sequences based on
discoveries and findings during execution, optimizing research workflows.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from open_deep_research.state import SequentialSupervisorState
from open_deep_research.supervisor.completion_analyzer import CompletionAnalysis

logger = logging.getLogger(__name__)


class SequenceModification:
    """Represents a modification to the agent sequence."""
    
    def __init__(
        self,
        modification_type: str,
        position: int,
        agent_name: Optional[str] = None,
        reason: str = "",
        confidence: float = 0.0,
        expected_benefits: List[str] = None
    ):
        self.modification_type = modification_type  # insert, remove, replace, reorder
        self.position = position
        self.agent_name = agent_name
        self.reason = reason
        self.confidence = confidence
        self.expected_benefits = expected_benefits or []
        self.timestamp = datetime.utcnow()


class SequenceModifier:
    """Dynamically modify agent sequences based on discoveries and findings."""
    
    def __init__(self, available_agents: List[str]):
        """Initialize the sequence modifier.
        
        Args:
            available_agents: List of available agent names from registry
        """
        self.available_agents = available_agents
        
        # Modification thresholds
        self.min_modification_confidence = 0.6
        self.max_sequence_length = 10
        self.min_quality_threshold = 0.4
        
        # Agent specialization mapping for intelligent insertion
        self.agent_specializations = {
            "research-agent": ["academic", "literature", "theory", "methodology"],
            "analysis-agent": ["data", "analysis", "pattern", "interpretation"],
            "market-agent": ["business", "market", "commercial", "competitive"],
            "technical-agent": ["implementation", "architecture", "technical", "system"],
            "synthesis-agent": ["integration", "summary", "strategic", "recommendations"]
        }
    
    def analyze_modification_opportunities(
        self,
        state: SequentialSupervisorState,
        completion_analysis: CompletionAnalysis,
        current_agent: str
    ) -> List[SequenceModification]:
        """Analyze opportunities for sequence modification.
        
        Args:
            state: Current supervisor state
            completion_analysis: Analysis from completed agent
            current_agent: Name of agent that just completed
            
        Returns:
            List of potential sequence modifications
        """
        logger.debug(f"Analyzing modification opportunities after '{current_agent}' completion")
        
        modifications = []
        
        # Analyze based on insights and findings
        insights = completion_analysis.insights_extracted
        handoff_context = completion_analysis.handoff_context
        
        # 1. Check for gap detection that requires specialist
        gap_modifications = self._detect_knowledge_gaps(
            state, insights, handoff_context
        )
        modifications.extend(gap_modifications)
        
        # 2. Check for complexity that requires additional agents
        complexity_modifications = self._analyze_complexity_requirements(
            state, insights, completion_analysis.quality_score
        )
        modifications.extend(complexity_modifications)
        
        # 3. Check for domain shifts that suggest reordering
        domain_modifications = self._analyze_domain_shifts(
            state, insights, handoff_context
        )
        modifications.extend(domain_modifications)
        
        # 4. Check for quality issues that suggest intervention
        quality_modifications = self._analyze_quality_issues(
            state, completion_analysis
        )
        modifications.extend(quality_modifications)
        
        # Filter and rank modifications
        viable_modifications = self._filter_viable_modifications(modifications, state)
        ranked_modifications = self._rank_modifications(viable_modifications)
        
        logger.info(f"Found {len(ranked_modifications)} viable sequence modifications")
        return ranked_modifications
    
    def _detect_knowledge_gaps(
        self,
        state: SequentialSupervisorState,
        insights: List[str],
        handoff_context: Dict[str, Any]
    ) -> List[SequenceModification]:
        """Detect knowledge gaps that require specialist agents.
        
        Args:
            state: Current supervisor state
            insights: Insights from completed agent
            handoff_context: Context from agent handoff
            
        Returns:
            List of modifications to address knowledge gaps
        """
        modifications = []
        
        # Analyze insights for gap indicators
        insight_text = " ".join(insights).lower()
        handoff_context.get("identified_topics", [])
        
        # Define gap patterns and required specialists
        gap_patterns = {
            "technical_gap": {
                "patterns": ["need technical", "implementation unclear", "architecture required", "how to build"],
                "agent": "technical-agent",
                "confidence": 0.8,
                "benefits": ["Technical feasibility analysis", "Implementation roadmap", "Architecture design"]
            },
            "market_gap": {
                "patterns": ["market unclear", "business model", "commercial viability", "monetization"],
                "agent": "market-agent", 
                "confidence": 0.7,
                "benefits": ["Market analysis", "Business model validation", "Commercial strategy"]
            },
            "research_gap": {
                "patterns": ["need more research", "literature review", "academic study", "validation needed"],
                "agent": "research-agent",
                "confidence": 0.75,
                "benefits": ["Academic validation", "Literature synthesis", "Research methodology"]
            },
            "analysis_gap": {
                "patterns": ["data analysis", "pattern analysis", "statistical", "interpretation needed"],
                "agent": "analysis-agent",
                "confidence": 0.7,
                "benefits": ["Data interpretation", "Pattern recognition", "Statistical analysis"]
            }
        }
        
        for gap_type, gap_info in gap_patterns.items():
            # Check if patterns are present in insights
            pattern_matches = sum(
                1 for pattern in gap_info["patterns"]
                if pattern in insight_text
            )
            
            if pattern_matches >= 1:  # At least one pattern match
                agent = gap_info["agent"]
                
                # Check if this agent is already in the sequence
                if agent not in state.planned_sequence and agent in self.available_agents:
                    position = state.sequence_position + 1  # Insert after current position
                    
                    modification = SequenceModification(
                        modification_type="insert",
                        position=position,
                        agent_name=agent,
                        reason=f"Knowledge gap detected: {gap_type} requires {agent}",
                        confidence=gap_info["confidence"],
                        expected_benefits=gap_info["benefits"]
                    )
                    modifications.append(modification)
                    
                    logger.debug(f"Gap detected: {gap_type} -> suggest inserting {agent}")
        
        return modifications
    
    def _analyze_complexity_requirements(
        self,
        state: SequentialSupervisorState,
        insights: List[str],
        quality_score: float
    ) -> List[SequenceModification]:
        """Analyze if complexity requires additional agents.
        
        Args:
            state: Current supervisor state
            insights: Insights from completed agent
            quality_score: Quality score of recent output
            
        Returns:
            List of modifications for complexity handling
        """
        modifications = []
        
        # High complexity indicators
        complexity_indicators = [
            "complex", "complicated", "multifaceted", "interdisciplinary",
            "requires multiple", "various aspects", "different perspectives"
        ]
        
        insight_text = " ".join(insights).lower()
        complexity_score = sum(
            1 for indicator in complexity_indicators
            if indicator in insight_text
        ) / max(len(complexity_indicators), 1)
        
        # If high complexity and current quality is good, consider adding synthesis
        if complexity_score > 0.3 and quality_score > 0.6:
            # Check if we need a synthesis agent at the end
            if "synthesis-agent" not in state.planned_sequence and "synthesis-agent" in self.available_agents:
                position = len(state.planned_sequence)  # Add at end
                
                modification = SequenceModification(
                    modification_type="insert",
                    position=position,
                    agent_name="synthesis-agent",
                    reason=f"High complexity (score: {complexity_score:.2f}) requires synthesis agent",
                    confidence=0.7,
                    expected_benefits=["Complex insight integration", "Multi-perspective synthesis", "Strategic recommendations"]
                )
                modifications.append(modification)
        
        return modifications
    
    def _analyze_domain_shifts(
        self,
        state: SequentialSupervisorState,
        insights: List[str],
        handoff_context: Dict[str, Any]
    ) -> List[SequenceModification]:
        """Analyze if domain shifts suggest sequence reordering.
        
        Args:
            state: Current supervisor state
            insights: Insights from completed agent
            handoff_context: Context from agent handoff
            
        Returns:
            List of modifications for domain optimization
        """
        modifications = []
        
        identified_topics = handoff_context.get("identified_topics", [])
        
        # If we discovered strong technical aspects but haven't used technical agent
        if ("technical_analysis" in identified_topics and 
            "technical-agent" in state.planned_sequence[state.sequence_position:] and
            state.sequence_position < len(state.planned_sequence) - 2):
            
            # Suggest moving technical agent earlier
            current_tech_pos = state.planned_sequence.index("technical-agent")
            new_position = state.sequence_position + 1
            
            if current_tech_pos > new_position:
                modification = SequenceModification(
                    modification_type="reorder",
                    position=new_position,
                    agent_name="technical-agent",
                    reason="Technical insights discovered - move technical agent earlier",
                    confidence=0.6,
                    expected_benefits=["Earlier technical validation", "Better technical context for subsequent agents"]
                )
                modifications.append(modification)
        
        return modifications
    
    def _analyze_quality_issues(
        self,
        state: SequentialSupervisorState,
        completion_analysis: CompletionAnalysis
    ) -> List[SequenceModification]:
        """Analyze quality issues that might require intervention.
        
        Args:
            state: Current supervisor state
            completion_analysis: Analysis from completed agent
            
        Returns:
            List of modifications to address quality issues
        """
        modifications = []
        
        # If quality is low and we have research capacity, consider adding research agent
        if (completion_analysis.quality_score < self.min_quality_threshold and
            completion_analysis.research_depth < 0.5):
            
            # Consider adding research agent for validation
            if ("research-agent" not in state.executed_agents and 
                "research-agent" in self.available_agents and
                len(state.planned_sequence) < self.max_sequence_length):
                
                position = state.sequence_position + 1
                
                modification = SequenceModification(
                    modification_type="insert",
                    position=position,
                    agent_name="research-agent",
                    reason=f"Low quality output (score: {completion_analysis.quality_score:.2f}) requires validation",
                    confidence=0.5,
                    expected_benefits=["Quality validation", "Additional research depth", "Fact verification"]
                )
                modifications.append(modification)
        
        return modifications
    
    def _filter_viable_modifications(
        self,
        modifications: List[SequenceModification],
        state: SequentialSupervisorState
    ) -> List[SequenceModification]:
        """Filter modifications to only viable ones.
        
        Args:
            modifications: List of potential modifications
            state: Current supervisor state
            
        Returns:
            List of viable modifications
        """
        viable = []
        
        for mod in modifications:
            # Check confidence threshold
            if mod.confidence < self.min_modification_confidence:
                continue
            
            # Check sequence length limits
            if mod.modification_type == "insert":
                if len(state.planned_sequence) >= self.max_sequence_length:
                    continue
                
                # Check if agent is available
                if mod.agent_name not in self.available_agents:
                    continue
                
                # Check if agent already executed
                if mod.agent_name in state.executed_agents:
                    continue
            
            # Check position validity
            if mod.position < state.sequence_position:
                continue  # Can't modify past positions
            
            viable.append(mod)
        
        return viable
    
    def _rank_modifications(
        self,
        modifications: List[SequenceModification]
    ) -> List[SequenceModification]:
        """Rank modifications by importance and confidence.
        
        Args:
            modifications: List of viable modifications
            
        Returns:
            List of ranked modifications (highest priority first)
        """
        # Sort by confidence score (descending)
        ranked = sorted(
            modifications,
            key=lambda m: m.confidence,
            reverse=True
        )
        
        return ranked
    
    def apply_modification(
        self,
        state: SequentialSupervisorState,
        modification: SequenceModification
    ) -> SequentialSupervisorState:
        """Apply a sequence modification to the state.
        
        Args:
            state: Current supervisor state
            modification: Modification to apply
            
        Returns:
            Updated supervisor state
        """
        logger.info(f"Applying sequence modification: {modification.modification_type} "
                   f"{modification.agent_name} at position {modification.position}")
        
        # Create new sequence
        new_sequence = state.planned_sequence.copy()
        
        if modification.modification_type == "insert":
            new_sequence.insert(modification.position, modification.agent_name)
        
        elif modification.modification_type == "remove":
            if modification.position < len(new_sequence):
                new_sequence.pop(modification.position)
        
        elif modification.modification_type == "replace":
            if modification.position < len(new_sequence):
                new_sequence[modification.position] = modification.agent_name
        
        elif modification.modification_type == "reorder":
            # Move agent to new position
            if modification.agent_name in new_sequence:
                current_pos = new_sequence.index(modification.agent_name)
                agent = new_sequence.pop(current_pos)
                new_sequence.insert(modification.position, agent)
        
        # Record the modification
        modification_record = {
            "timestamp": modification.timestamp.isoformat(),
            "type": modification.modification_type,
            "position": modification.position,
            "agent": modification.agent_name,
            "reason": modification.reason,
            "confidence": modification.confidence,
            "expected_benefits": modification.expected_benefits,
            "original_sequence": state.planned_sequence.copy(),
            "new_sequence": new_sequence.copy()
        }
        
        # Update state
        updated_state = state.copy()
        updated_state.planned_sequence = new_sequence
        updated_state.sequence_modifications.append(modification_record)
        
        logger.info(f"Sequence modified: {state.planned_sequence} -> {new_sequence}")
        
        return updated_state
    
    def should_modify_sequence(
        self,
        state: SequentialSupervisorState,
        completion_analysis: CompletionAnalysis
    ) -> Tuple[bool, Optional[SequenceModification]]:
        """Determine if sequence should be modified and return best modification.
        
        Args:
            state: Current supervisor state
            completion_analysis: Analysis from completed agent
            
        Returns:
            Tuple of (should_modify, best_modification)
        """
        # Don't modify if we're near the end
        if state.sequence_position >= len(state.planned_sequence) - 1:
            return False, None
        
        # Don't modify if we've already made too many modifications
        if len(state.sequence_modifications) >= 3:
            return False, None
        
        # Get potential modifications
        modifications = self.analyze_modification_opportunities(
            state, completion_analysis, state.current_agent or "unknown"
        )
        
        if not modifications:
            return False, None
        
        # Return best modification
        best_modification = modifications[0]
        
        # Apply additional filters for modification decision
        if best_modification.confidence < 0.7:
            return False, None
        
        return True, best_modification
    
    def get_modification_summary(
        self,
        state: SequentialSupervisorState
    ) -> Dict[str, Any]:
        """Get a summary of all sequence modifications made.
        
        Args:
            state: Current supervisor state
            
        Returns:
            Summary of modifications
        """
        modifications = state.sequence_modifications
        
        summary = {
            "total_modifications": len(modifications),
            "modification_types": {},
            "agents_added": [],
            "agents_removed": [],
            "sequence_evolution": [],
            "modification_reasons": []
        }
        
        for mod in modifications:
            mod_type = mod["type"]
            summary["modification_types"][mod_type] = summary["modification_types"].get(mod_type, 0) + 1
            
            if mod_type == "insert":
                summary["agents_added"].append(mod["agent"])
            elif mod_type == "remove":
                summary["agents_removed"].append(mod["agent"])
            
            summary["sequence_evolution"].append({
                "timestamp": mod["timestamp"],
                "from": mod["original_sequence"],
                "to": mod["new_sequence"],
                "reason": mod["reason"]
            })
            
            summary["modification_reasons"].append(mod["reason"])
        
        return summary