"""Langfuse trace collection and GRPO learning integration.

This module implements observability and reinforcement learning from human feedback
using Langfuse as the backbone for trace management and GRPO optimization.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import numpy as np

from pydantic import BaseModel, Field


class TraceCollector:
    """Langfuse-based trace collector for observability and learning."""

    def __init__(self, langfuse_config: Optional[Dict[str, str]] = None):
        """Initialize trace collector.

        Args:
            langfuse_config: Langfuse configuration (optional, uses env vars if not provided)
        """
        self.logger = logging.getLogger(__name__)
        self.langfuse = None
        self.session_id = self._generate_session_id()
        self.is_enabled = True
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.episodes: List[Dict[str, Any]] = []

        # Initialize Langfuse if available
        self._initialize_langfuse(langfuse_config)

    def _initialize_langfuse(self, config: Optional[Dict[str, str]]):
        """Initialize Langfuse client.

        Args:
            config: Configuration dictionary
        """
        try:
            import langfuse

            if config:
                self.langfuse = langfuse.Langfuse(
                    public_key=config.get("public_key"),
                    secret_key=config.get("secret_key"),
                    host=config.get("host", "https://api.langfuse.com")
                )
            else:
                # Try to initialize from environment variables
                try:
                    self.langfuse = langfuse.Langfuse()
                except Exception:
                    self.logger.warning("Langfuse not configured, running in local-only mode")
                    self.is_enabled = False

            if self.langfuse:
                self.logger.info("Langfuse initialized successfully")

        except ImportError:
            self.logger.warning("Langfuse not available, running in local-only mode")
            self.is_enabled = False

    def _generate_session_id(self) -> str:
        """Generate unique session identifier.

        Returns:
            Session ID string
        """
        return f"session_{int(datetime.utcnow().timestamp())}_{np.random.randint(1000, 9999)}"

    def trace_agent_execution(self, agent_id: str, task: Any) -> str:
        """Create trace for agent execution.

        Args:
            agent_id: Agent identifier
            task: Task being executed

        Returns:
            Trace ID
        """
        trace_id = f"{agent_id}_{int(datetime.utcnow().timestamp())}"

        if self.langfuse and self.is_enabled:
            trace = self.langfuse.trace(
                name=f"{agent_id}_execution",
                metadata={
                    "agent_id": agent_id,
                    "task_type": getattr(task, "type", "unknown"),
                    "session_id": self.session_id
                }
            )
            self.traces[trace_id] = {"langfuse_trace": trace, "agent_id": agent_id}
        else:
            self.traces[trace_id] = {"agent_id": agent_id, "local_trace": True}

        self.logger.debug(f"Created trace {trace_id} for agent {agent_id}")
        return trace_id

    def record_generation(self, trace_id: str, model: str, input_data: str, output_data: str,
                         usage: Dict[str, int], metadata: Optional[Dict[str, Any]] = None):
        """Record model generation in trace.

        Args:
            trace_id: Trace identifier
            model: Model name
            input_data: Input to model
            output_data: Model output
            usage: Token usage information
            metadata: Additional metadata
        """
        if trace_id not in self.traces:
            self.logger.warning(f"Trace {trace_id} not found for generation recording")
            return

        generation_data = {
            "model": model,
            "input": input_data,
            "output": output_data,
            "usage": usage,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }

        if self.langfuse and self.is_enabled:
            trace = self.traces[trace_id].get("langfuse_trace")
            if trace:
                generation = trace.generation(
                    name=f"{model}_generation",
                    model=model,
                    input=input_data,
                    metadata=metadata or {}
                )
                generation.end(
                    output=output_data,
                    usage=usage
                )
        else:
            # Store locally
            if "generations" not in self.traces[trace_id]:
                self.traces[trace_id]["generations"] = []
            self.traces[trace_id]["generations"].append(generation_data)

    def record_score(self, trace_id: str, score_name: str, score_value: float,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record score for trace (for GRPO learning).

        Args:
            trace_id: Trace identifier
            score_name: Name of the score (e.g., "quality", "relevance")
            score_value: Score value
            metadata: Additional metadata
        """
        if trace_id not in self.traces:
            self.logger.warning(f"Trace {trace_id} not found for score recording")
            return

        score_data = {
            "name": score_name,
            "value": score_value,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }

        if self.langfuse and self.is_enabled:
            trace = self.traces[trace_id].get("langfuse_trace")
            if trace:
                trace.score(
                    name=score_name,
                    value=score_value
                )
        else:
            # Store locally
            if "scores" not in self.traces[trace_id]:
                self.traces[trace_id]["scores"] = []
            self.traces[trace_id]["scores"].append(score_data)

    def record_episode(self, query: str, strategies: List[Dict[str, Any]], rewards: List[float]):
        """Record research episode for GRPO training.

        Args:
            query: Research query
            strategies: List of strategies used
            rewards: Rewards for each strategy
        """
        episode = {
            "query": query,
            "strategies": strategies,
            "rewards": rewards,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id
        }

        self.episodes.append(episode)

        if self.langfuse and self.is_enabled:
            # Store episode in Langfuse dataset
            self.langfuse.dataset_item(
                dataset_name="research_episodes",
                item={
                    "query": query,
                    "strategies": strategies,
                    "rewards": rewards,
                    "session_id": self.session_id
                }
            )

        self.logger.debug(f"Recorded episode for query: {query}")

    def get_trace_data(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace data.

        Args:
            trace_id: Trace identifier

        Returns:
            Trace data dictionary
        """
        return self.traces.get(trace_id)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session.

        Returns:
            Session summary
        """
        trace_count = len(self.traces)
        episode_count = len(self.episodes)

        # Calculate average scores if available
        all_scores = []
        for trace in self.traces.values():
            if "scores" in trace:
                all_scores.extend([s["value"] for s in trace["scores"]])

        avg_score = np.mean(all_scores) if all_scores else 0.0

        return {
            "session_id": self.session_id,
            "trace_count": trace_count,
            "episode_count": episode_count,
            "average_score": avg_score,
            "duration": (datetime.utcnow() - datetime.fromtimestamp(int(self.session_id.split('_')[1]))).total_seconds()
        }


class GRPOLearner:
    """GRPO-based learner for optimizing orchestration patterns."""

    def __init__(self, trace_collector: TraceCollector, policy_model_path: Optional[str] = None):
        """Initialize GRPO learner.

        Args:
            trace_collector: TraceCollector instance
            policy_model_path: Path to saved policy model (optional)
        """
        self.trace_collector = trace_collector
        self.logger = logging.getLogger(__name__)
        self.policy_model = None
        self.episodes: List[Dict[str, Any]] = []
        self.is_training = False

        # Initialize policy model (simplified for now)
        self._initialize_policy_model(policy_model_path)

    def _initialize_policy_model(self, model_path: Optional[str]):
        """Initialize policy model.

        Args:
            model_path: Path to saved model
        """
        # For now, use a simple rule-based policy
        # In a real implementation, this would load a learned model
        self.policy_model = {
            "type": "rule_based",
            "rules": {
                "academic_query": ["academic_agent", "analysis_agent"],
                "technical_query": ["technical_agent", "implementation_agent"],
                "market_query": ["market_agent", "business_agent"]
            },
            "default_sequence": ["research_agent", "synthesis_agent"]
        }

        if model_path:
            try:
                # Load saved model
                with open(model_path, 'r') as f:
                    self.policy_model = json.load(f)
                self.logger.info(f"Loaded policy model from {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load policy model: {e}")

    def classify_query(self, query: str) -> str:
        """Classify query to determine optimal strategy.

        Args:
            query: Research query

        Returns:
            Query classification
        """
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in ["academic", "research", "paper", "study", "literature"]):
            return "academic_query"
        elif any(keyword in query_lower for keyword in ["technical", "implementation", "code", "system", "architecture"]):
            return "technical_query"
        elif any(keyword in query_lower for keyword in ["market", "business", "industry", "commercial", "competitive"]):
            return "market_query"
        else:
            return "general_query"

    def sample_strategy(self, query: str) -> List[str]:
        """Sample orchestration strategy for query.

        Args:
            query: Research query

        Returns:
            List of agent names for strategy
        """
        query_type = self.classify_query(query)
        rules = self.policy_model["rules"]

        if query_type in rules:
            return rules[query_type]
        else:
            return self.policy_model["default_sequence"]

    def compute_reward(self, result: Dict[str, Any]) -> float:
        """Compute reward for research result.

        Args:
            result: Research result

        Returns:
            Reward value
        """
        # Multi-objective reward function
        quality_score = result.get("quality_score", 0.0)
        paper_count = len(result.get("papers", []))
        insight_count = len(result.get("insights", []))

        # Weighted combination
        reward = (
            0.5 * quality_score +           # Quality is most important
            0.3 * min(paper_count / 5.0, 1.0) +  # Paper count (up to 5)
            0.2 * min(insight_count / 10.0, 1.0)  # Insight count (up to 10)
        )

        return reward

    async def collect_episodes(self, n_episodes: int = 100) -> List[Dict[str, Any]]:
        """Collect research episodes for GRPO training.

        Args:
            n_episodes: Number of episodes to collect

        Returns:
            List of collected episodes
        """
        episodes = []

        for i in range(n_episodes):
            # Sample a research query (in real implementation, this would come from user)
            query = self._sample_research_query(i)

            # Generate K different orchestration strategies
            strategies = []
            for k in range(4):  # GRPO typically uses 4 samples
                strategy = self.sample_strategy(query)
                # Add some variation
                varied_strategy = self._vary_strategy(strategy)

                strategies.append({
                    "strategy": varied_strategy,
                    "query": query
                })

            episodes.append({
                "query": query,
                "strategies": strategies,
                "episode_id": f"ep_{i}_{int(datetime.utcnow().timestamp())}"
            })

        self.logger.info(f"Collected {len(episodes)} episodes for GRPO training")
        return episodes

    def _sample_research_query(self, episode_num: int) -> str:
        """Sample research query for testing.

        Args:
            episode_num: Episode number

        Returns:
            Research query
        """
        queries = [
            "Research quantum computing applications in healthcare",
            "Analyze machine learning trends in 2024",
            "Investigate blockchain implementation challenges",
            "Study renewable energy market trends",
            "Research AI ethics and governance frameworks",
            "Analyze cloud computing adoption patterns",
            "Investigate cybersecurity threats in IoT",
            "Study natural language processing advancements",
            "Research autonomous vehicle technology",
            "Analyze big data analytics in finance"
        ]

        return queries[episode_num % len(queries)]

    def _vary_strategy(self, base_strategy: List[str]) -> List[str]:
        """Add variation to strategy for exploration.

        Args:
            base_strategy: Base strategy

        Returns:
            Varied strategy
        """
        # Simple variation - in real implementation, this would be more sophisticated
        varied = base_strategy.copy()

        if len(varied) > 1:
            # Swap two random agents
            idx1, idx2 = np.random.choice(len(varied), 2, replace=False)
            varied[idx1], varied[idx2] = varied[idx2], varied[idx1]

        return varied

    def train_policy(self, episodes: List[Dict[str, Any]]):
        """GRPO training loop.

        Args:
            episodes: Training episodes
        """
        self.logger.info("Starting GRPO training")

        for episode in episodes:
            query = episode["query"]
            strategies = episode["strategies"]

            # In a real implementation, this would execute each strategy
            # and compute actual rewards. For now, we'll use simulated rewards.

            rewards = []
            for strategy in strategies:
                # Simulate reward based on strategy characteristics
                simulated_reward = self._simulate_reward(strategy, query)
                rewards.append(simulated_reward)

            # Compute group baseline (mean reward)
            baseline = np.mean(rewards)

            # Compute advantages
            advantages = [r - baseline for r in rewards]

            # Update policy (simplified)
            for strategy, advantage in zip(strategies, advantages):
                if advantage > 0:
                    self._reinforce_strategy(query, strategy["strategy"], advantage)

            # Record episode
            self.trace_collector.record_episode(query, strategies, rewards)

        self.logger.info("GRPO training completed")

    def _simulate_reward(self, strategy: Dict[str, Any], query: str) -> float:
        """Simulate reward for strategy (for testing).

        Args:
            strategy: Strategy to evaluate
            query: Research query

        Returns:
            Simulated reward
        """
        query_type = self.classify_query(query)
        strategy_agents = strategy["strategy"]

        # Simple reward simulation based on agent count and relevance
        base_reward = len(strategy_agents) * 0.1  # More agents = higher potential reward

        # Bonus for using appropriate agents
        if query_type == "academic_query" and "academic_agent" in strategy_agents:
            base_reward += 0.3
        elif query_type == "technical_query" and "technical_agent" in strategy_agents:
            base_reward += 0.3
        elif query_type == "market_query" and "market_agent" in strategy_agents:
            base_reward += 0.3

        return min(base_reward, 1.0)

    def _reinforce_strategy(self, query: str, strategy: List[str], advantage: float):
        """Reinforce successful strategies.

        Args:
            query: Research query
            strategy: Successful strategy
            advantage: Advantage value
        """
        # Simple reinforcement - increase probability of good strategies
        query_type = self.classify_query(query)

        if query_type in self.policy_model["rules"]:
            # Update rule with successful strategy
            current_rule = self.policy_model["rules"][query_type]

            # If this strategy performed better, update the rule
            if advantage > 0.1:  # Significant advantage
                self.policy_model["rules"][query_type] = strategy
                self.logger.debug(f"Updated rule for {query_type} with better strategy")

    def save_policy_model(self, path: str):
        """Save policy model to file.

        Args:
            path: File path to save model
        """
        try:
            with open(path, 'w') as f:
                json.dump(self.policy_model, f, indent=2)
            self.logger.info(f"Saved policy model to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save policy model: {e}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics.

        Returns:
            Training statistics
        """
        episodes = self.trace_collector.episodes

        if not episodes:
            return {"episode_count": 0, "average_reward": 0.0}

        rewards = []
        for episode in episodes:
            rewards.extend(episode.get("rewards", []))

        return {
            "episode_count": len(episodes),
            "average_reward": np.mean(rewards) if rewards else 0.0,
            "max_reward": np.max(rewards) if rewards else 0.0,
            "min_reward": np.min(rewards) if rewards else 0.0,
            "total_strategies_evaluated": sum(len(ep["strategies"]) for ep in episodes)
        }
