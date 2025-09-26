#!/usr/bin/env python3
"""Main entry point for the Research CLI system.

This module provides the command-line interface for the research system,
implementing the architecture blueprint with A2A protocol, LangGraph orchestration,
and Langfuse integration.
"""

import asyncio
import logging
import sys
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # dotenv not available, skip loading

from .agents.research_agent import create_agent_registry
from .core.cli_interface import ResearchCLI
from .orchestration.langgraph_orchestrator import OrchestrationEngine
from .orchestration.trace_collector import GRPOLearner, TraceCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('research_cli.log')
    ]
)

logger = logging.getLogger(__name__)


class ResearchCLIApp:
    """Main Research CLI application."""

    def __init__(self):
        """Initialize the CLI application."""
        self.cli: Optional[ResearchCLI] = None
        self.orchestrator: Optional[OrchestrationEngine] = None
        self.trace_collector: Optional[TraceCollector] = None
        self.grpo_learner: Optional[GRPOLearner] = None
        self.agent_registry = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize all CLI components."""
        if self.is_initialized:
            return

        try:
            logger.info("Initializing Research CLI system...")

            # Initialize core components
            self.cli = ResearchCLI()
            await self.cli.initialize()

            self.orchestrator = self.cli.orchestrator
            self.trace_collector = self.cli.trace_collector

            # Create agent registry with default agents
            self.agent_registry = create_agent_registry()

            # Register agents with orchestrator
            for agent in self.agent_registry.agents.values():
                self.orchestrator.register_agent(agent)

            # Initialize GRPO learner
            if self.trace_collector:
                self.grpo_learner = GRPOLearner(self.trace_collector)

            self.is_initialized = True
            logger.info("Research CLI system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Research CLI: {e}")
            raise

    async def close(self):
        """Close all CLI resources."""
        if self.cli:
            await self.cli.close()
        logger.info("Research CLI system closed")

    async def run_research(self, query: str, streaming: bool = True):
        """Run research with the given query.

        Args:
            query: Research query
            streaming: Whether to use streaming output
        """
        if not self.is_initialized:
            await self.initialize()

        return await self.cli.research(query, streaming=streaming)

    async def run_interactive(self):
        """Run interactive CLI mode."""
        if not self.is_initialized:
            await self.initialize()

        await self.cli.interactive_mode()

    async def run_training(self, episodes: int = 100):
        """Run GRPO training.

        Args:
            episodes: Number of training episodes
        """
        if not self.is_initialized or not self.grpo_learner:
            await self.initialize()

        logger.info(f"Starting GRPO training with {episodes} episodes...")

        # Collect episodes
        episodes_data = await self.grpo_learner.collect_episodes(episodes)

        # Run training
        self.grpo_learner.train_policy(episodes_data)

        # Display training statistics
        if self.trace_collector:
            self.trace_collector.get_session_summary()
            self.cli.console.print("\n[green]Training completed![/green]")
            self.cli.display_training_stats()

    async def show_stats(self):
        """Show system statistics."""
        if not self.is_initialized:
            await self.initialize()

        if self.trace_collector:
            self.cli.display_training_stats()

    async def show_config(self):
        """Show system configuration."""
        if not self.is_initialized:
            await self.initialize()

        self.cli.display_config()
        self.cli.display_agents()

    def parse_args(self, args: list) -> tuple:
        """Parse command line arguments.

        Args:
            args: Command line arguments

        Returns:
            Tuple of (command, args_list)
        """
        if not args:
            return ("interactive", [])

        command = args[0].lower()

        # Handle special cases
        if command in ["--help", "-h"]:
            return ("help", [])
        elif command in ["--version", "-v"]:
            return ("version", [])
        elif command == "--config":
            return ("config", [])
        elif command == "--train":
            episodes = 100
            if len(args) > 1:
                try:
                    episodes = int(args[1])
                except ValueError:
                    pass
            return ("train", [episodes])
        elif command.startswith("--"):
            # Remove leading dashes
            cmd = command[2:] if command.startswith("--") else command[1:]
            return (cmd, args[1:])

        return (command, args[1:])


async def main():
    """Main CLI application entry point."""
    app = ResearchCLIApp()

    try:
        # Parse command line arguments
        if len(sys.argv) < 2:
            # No arguments - run interactive mode
            await app.run_interactive()
        else:
            command, args = app.parse_args(sys.argv[1:])

            await app.initialize()

            if command == "research":
                if not args:
                    app.cli.console.print("[red]Research query required[/red]")
                    sys.exit(1)

                query = " ".join(args)
                await app.run_research(query, streaming=True)

            elif command == "train":
                episodes = 100
                if args:
                    try:
                        episodes = int(args[0])
                    except (ValueError, TypeError):
                        pass
                await app.run_training(episodes)

            elif command == "stats":
                await app.show_stats()

            elif command == "config":
                await app.show_config()

            elif command == "interactive":
                await app.run_interactive()

            elif command == "help":
                app.cli.display_help()

            elif command == "version":
                app.cli.console.print("[blue]Research CLI v0.1.0[/blue]")
                app.cli.console.print("Built with A2A protocol, LangGraph, and Langfuse integration")

            else:
                app.cli.console.print(f"[red]Unknown command: {command}[/red]")
                app.cli.display_help()
                sys.exit(1)

    except KeyboardInterrupt:
        app.cli.console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        logger.error(f"CLI error: {e}")
        app.cli.console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    finally:
        await app.close()


def cli_entry_point():
    """CLI entry point function."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_entry_point()
