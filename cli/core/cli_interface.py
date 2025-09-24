"""Rich-based CLI interface for research system.

Provides beautiful terminal output with streaming, progress tracking,
and interactive research execution.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import json

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.columns import Columns
from rich.spinner import Spinner
from rich.status import Status

from ..orchestration.langgraph_orchestrator import OrchestrationEngine, ResearchResult
from ..orchestration.trace_collector import TraceCollector
from ..utils.research_types import StreamMessage


class ResearchCLI:
    """Rich-based CLI interface for research execution."""

    def __init__(self):
        """Initialize CLI interface."""
        self.console = Console()
        self.orchestrator: Optional[OrchestrationEngine] = None
        self.trace_collector: Optional[TraceCollector] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize CLI components."""
        self.orchestrator = OrchestrationEngine()
        await self.orchestrator.initialize()

        self.trace_collector = TraceCollector()
        self.logger.info("CLI interface initialized")

    async def close(self):
        """Close CLI resources."""
        if self.orchestrator:
            await self.orchestrator.close()
        self.logger.info("CLI interface closed")

    async def research(self, query: str, streaming: bool = True) -> ResearchResult:
        """Execute research with CLI output.

        Args:
            query: Research query
            streaming: Whether to show streaming output

        Returns:
            ResearchResult
        """
        if not self.orchestrator:
            await self.initialize()

        # Create execution tree for visualization
        execution_tree = self._create_execution_tree(query)

        if streaming:
            return await self._research_with_streaming(query, execution_tree)
        else:
            return await self._research_synchronous(query)

    async def _research_with_streaming(self, query: str, execution_tree: Tree) -> ResearchResult:
        """Execute research with streaming visualization.

        Args:
            query: Research query
            execution_tree: Rich Tree for visualization

        Returns:
            ResearchResult
        """
        self.console.print(f"\n[bold blue]🔬 Starting Research:[/bold blue] {query}\n")

        with Live(execution_tree, console=self.console, refresh_per_second=4):
            # Start research execution
            async for event in self.orchestrator.execute_stream(query):
                await self._update_execution_tree(execution_tree, event)

        # Get final result
        result = await self.orchestrator.execute_research(query)

        # Display final result
        await self._display_final_result(result, query)

        return result

    async def _research_synchronous(self, query: str) -> ResearchResult:
        """Execute research synchronously.

        Args:
            query: Research query

        Returns:
            ResearchResult
        """
        self.console.print(f"\n[bold blue]🔬 Researching:[/bold blue] {query}\n")

        with Status("Executing research workflow...", console=self.console) as status:
            status.update("Analyzing query...")
            await asyncio.sleep(0.5)

            status.update("Generating research sequences...")
            await asyncio.sleep(0.5)

            status.update("Executing parallel research...")
            await asyncio.sleep(1.0)

            status.update("Synthesizing results...")
            await asyncio.sleep(0.5)

            # Execute research
            result = await self.orchestrator.execute_research(query)

        # Display result
        await self._display_final_result(result, query)

        return result

    def _create_execution_tree(self, query: str) -> Tree:
        """Create execution visualization tree.

        Args:
            query: Research query

        Returns:
            Rich Tree for execution visualization
        """
        tree = Tree(f"🔬 Research Query: {query}")

        # Add initial nodes
        analyze_node = tree.add("📋 Query Analysis")
        analyze_node.add("⏳ Pending...")

        sequences_node = tree.add("⚙️ Strategic Sequences")
        sequences_node.add("⏳ Pending...")

        execution_node = tree.add("🚀 Parallel Execution")
        execution_node.add("⏳ Pending...")

        synthesis_node = tree.add("📝 Synthesis")
        synthesis_node.add("⏳ Pending...")

        evaluation_node = tree.add("✅ Quality Evaluation")
        evaluation_node.add("⏳ Pending...")

        return tree

    async def _update_execution_tree(self, tree: Tree, event: StreamMessage):
        """Update execution tree with new event.

        Args:
            tree: Execution tree to update
            event: Stream event
        """
        content = event.content

        if isinstance(content, dict):
            if content.get("type") == "agent_spawn":
                agent_name = content.get("agent_id", "Unknown Agent")
                # Find agent node and update
                for branch in tree.children:
                    if "Strategic Sequences" in str(branch.label):
                        branch.add(f"🤖 {agent_name}")

            elif content.get("type") == "agent_progress":
                agent_name = content.get("agent_id", "Unknown Agent")
                status = content.get("status", "Progressing...")
                # Update existing agent node
                for branch in tree.children:
                    if "Strategic Sequences" in str(branch.label):
                        for agent_branch in branch.children:
                            if agent_name in str(agent_branch.label):
                                agent_branch.label = f"🤖 {agent_name}: {status}"

            elif content.get("type") == "agent_complete":
                agent_name = content.get("agent_id", "Unknown Agent")
                summary = content.get("summary", "Completed")
                # Mark as completed
                for branch in tree.children:
                    if "Strategic Sequences" in str(branch.label):
                        for agent_branch in branch.children:
                            if agent_name in str(agent_branch.label):
                                agent_branch.label = f"✅ {agent_name}: {summary}"

            elif content.get("type") == "research_complete":
                # Update final nodes
                for branch in tree.children:
                    if "Synthesis" in str(branch.label):
                        branch.children[0].label = "✅ Completed"
                    elif "Quality Evaluation" in str(branch.label):
                        branch.children[0].label = "✅ Completed"

    async def _display_final_result(self, result: ResearchResult, query: str):
        """Display final research result.

        Args:
            result: Research result
            query: Original query
        """
        self.console.print(f"\n[bold green]✅ Research Complete![/bold green]\n")

        # Create result panel
        result_panel = Panel(
            result.synthesis,
            title=f"Research Results: {query}",
            border_style="green"
        )
        self.console.print(result_panel)

        # Display paper count if available
        if result.papers:
            papers_table = Table(title="📚 Research Papers Found")
            papers_table.add_column("Title", style="cyan")
            papers_table.add_column("Relevance", style="magenta")
            papers_table.add_column("Summary", style="white")

            for paper in result.papers[:5]:  # Show top 5 papers
                papers_table.add_row(
                    paper.get("title", "Unknown"),
                    f"{paper.get('relevance', 0):.2f}",
                    paper.get("summary", "No summary available")[:100] + "..."
                )

            self.console.print(papers_table)

        # Display trace information
        if result.trace_id:
            self.console.print(f"\n[dim]Trace ID: {result.trace_id}[/dim]")

        # Display session statistics
        if self.trace_collector:
            session_stats = self.trace_collector.get_session_summary()
            self.console.print(f"\n[dim]Session Stats: {session_stats['trace_count']} traces, "
                             f"{session_stats['episode_count']} episodes[/dim]")

    def display_training_stats(self):
        """Display GRPO training statistics."""
        if not self.trace_collector:
            return

        stats = self.trace_collector.get_session_summary()

        stats_panel = Panel(
            f"""Traces: {stats['trace_count']}
Episodes: {stats['episode_count']}
Average Score: {stats['average_score']:.2f}
Session Duration: {stats['duration']:.1f}s""",
            title="📊 Session Statistics",
            border_style="blue"
        )

        self.console.print(stats_panel)

    def display_orchestration_insights(self, insights: Dict[str, Any]):
        """Display orchestration insights from evaluation.

        Args:
            insights: Orchestration insights
        """
        insights_panel = Panel(
            f"""Best Approach: {insights.get('best_approach', {}).get('name', 'Unknown')}
Score: {insights.get('best_approach', {}).get('score', 0):.1f}/100

Key Learnings:
{chr(10).join(f'• {learning}' for learning in insights.get('key_learnings', []))}

Recommendations:
{chr(10).join(f'• {rec}' for rec in insights.get('recommendations', {}).values())}""",
            title="🧠 Orchestration Insights",
            border_style="yellow"
        )

        self.console.print(insights_panel)

    async def interactive_mode(self):
        """Run interactive CLI mode."""
        self.console.print("[bold blue]🔬 Interactive Research Mode[/bold blue]")
        self.console.print("Type 'quit' or 'exit' to end session\n")

        while True:
            try:
                # Get user input
                query = await self._get_user_input()

                if query.lower() in ['quit', 'exit', 'q']:
                    break

                if not query.strip():
                    continue

                # Execute research
                await self.research(query, streaming=True)

                self.console.print("\n" + "="*80 + "\n")

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                self.logger.error(f"Interactive mode error: {e}")

        self.console.print("[green]Goodbye! 👋[/green]")

    async def _get_user_input(self) -> str:
        """Get user input with prompt.

        Returns:
            User input string
        """
        from rich.prompt import Prompt

        return Prompt.ask(
            "\n[bold cyan]Research Query[/bold cyan]",
            default=""
        )

    def display_help(self):
        """Display CLI help information."""
        help_text = """
[bold blue]🔬 Research CLI Commands[/bold blue]

[bold]Basic Usage:[/bold]
  research <query>          Execute research on a query
  research --stream <query> Show streaming output
  interactive               Start interactive mode

[bold]Training:[/bold]
  train <episodes>          Collect training episodes for GRPO
  stats                     Show session statistics
  insights                  Display orchestration insights

[bold]Configuration:[/bold]
  config                    Show current configuration
  agents                    List available agents

[bold]Examples:[/bold]
  research "quantum computing applications in healthcare"
  research --stream "AI ethics and governance frameworks"
  train 100                 Collect 100 training episodes

[bold]Options:[/bold]
  --help, -h               Show this help message
  --version, -v            Show version information
  --config <file>          Load configuration from file

Type 'interactive' for guided research mode!
        """

        self.console.print(Panel(help_text, title="Help", border_style="blue"))

    def display_config(self):
        """Display current configuration."""
        config_info = f"""
[bold]🔧 Configuration[/bold]

Orchestrator: {'Initialized' if self.orchestrator else 'Not initialized'}
Trace Collector: {'Enabled' if self.trace_collector else 'Disabled'}
Session ID: {self.trace_collector.session_id if self.trace_collector else 'N/A'}

Available Agents: {len(self.orchestrator.agent_registry) if self.orchestrator else 0}
Context Tree: {'Active' if hasattr(self.orchestrator, 'context_tree') else 'Not available'}

Streaming: Enabled
Real-time Updates: Enabled
        """

        self.console.print(Panel(config_info, title="Configuration", border_style="cyan"))

    def display_agents(self):
        """Display available agents."""
        if not self.orchestrator:
            self.console.print("[red]Orchestrator not initialized[/red]")
            return

        agents = self.orchestrator.get_registered_agents()

        if not agents:
            self.console.print("[yellow]No agents registered[/yellow]")
            return

        agents_table = Table(title="🤖 Available Agents")
        agents_table.add_column("Agent Name", style="cyan")
        agents_table.add_column("Status", style="green")
        agents_table.add_column("Description", style="white")

        for agent in agents:
            agents_table.add_row(
                agent,
                "✅ Active",
                f"Research agent: {agent}"
            )

        self.console.print(agents_table)


# CLI command handlers
class CLICommandHandler:
    """Handle CLI commands and routing."""

    def __init__(self, cli: ResearchCLI):
        """Initialize command handler.

        Args:
            cli: ResearchCLI instance
        """
        self.cli = cli
        self.commands = {
            'research': self._handle_research,
            'train': self._handle_train,
            'stats': self._handle_stats,
            'insights': self._handle_insights,
            'interactive': self._handle_interactive,
            'config': self._handle_config,
            'agents': self._handle_agents,
            'help': self._handle_help,
            'quit': self._handle_quit,
            'exit': self._handle_quit
        }

    async def handle_command(self, command: str, args: List[str]) -> bool:
        """Handle CLI command.

        Args:
            command: Command name
            args: Command arguments

        Returns:
            True if should continue, False if should exit
        """
        if command in self.commands:
            should_continue = await self.commands[command](args)
            return should_continue
        else:
            self.cli.console.print(f"[red]Unknown command: {command}[/red]")
            self.cli.display_help()
            return True

    async def _handle_research(self, args: List[str]) -> bool:
        """Handle research command."""
        if not args:
            self.cli.console.print("[red]Research query required[/red]")
            return True

        query = " ".join(args)
        streaming = "--stream" in args

        await self.cli.research(query, streaming=streaming)
        return True

    async def _handle_train(self, args: List[str]) -> bool:
        """Handle training command."""
        if not self.trace_collector:
            self.cli.console.print("[red]Trace collector not available[/red]")
            return True

        episodes = 100
        if args:
            try:
                episodes = int(args[0])
            except ValueError:
                self.cli.console.print("[red]Invalid episode count[/red]")
                return True

        # This would trigger GRPO training
        self.cli.console.print(f"[yellow]Training with {episodes} episodes (not yet implemented)[/yellow]")
        return True

    async def _handle_stats(self, args: List[str]) -> bool:
        """Handle stats command."""
        self.cli.display_training_stats()
        return True

    async def _handle_insights(self, args: List[str]) -> bool:
        """Handle insights command."""
        self.cli.console.print("[yellow]Orchestration insights (not yet implemented)[/yellow]")
        return True

    async def _handle_interactive(self, args: List[str]) -> bool:
        """Handle interactive command."""
        await self.cli.interactive_mode()
        return False  # Exit after interactive mode

    async def _handle_config(self, args: List[str]) -> bool:
        """Handle config command."""
        self.cli.display_config()
        return True

    async def _handle_agents(self, args: List[str]) -> bool:
        """Handle agents command."""
        self.cli.display_agents()
        return True

    async def _handle_help(self, args: List[str]) -> bool:
        """Handle help command."""
        self.cli.display_help()
        return True

    async def _handle_quit(self, args: List[str]) -> bool:
        """Handle quit command."""
        return False  # Exit


# Main CLI application
async def main():
    """Main CLI application entry point."""
    cli = ResearchCLI()
    await cli.initialize()

    try:
        # Parse command line arguments
        import sys

        if len(sys.argv) < 2:
            await cli.interactive_mode()
        else:
            command_handler = CLICommandHandler(cli)
            should_continue = await command_handler.handle_command(sys.argv[1], sys.argv[2:])
            if should_continue:
                await main()

    finally:
        await cli.close()


if __name__ == "__main__":
    asyncio.run(main())
