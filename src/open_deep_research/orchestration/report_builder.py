"""Running report builder for incremental research report generation.

This module builds reports incrementally as agents complete their tasks,
maintaining context and insights throughout the sequential workflow.
"""

import logging
from datetime import datetime
from typing import Dict, List

from open_deep_research.state import AgentExecutionReport, RunningReport

logger = logging.getLogger(__name__)


class RunningReportBuilder:
    """Build reports incrementally as agents complete their tasks."""
    
    @staticmethod
    def initialize_report(
        research_topic: str,
        sequence_name: str,
        planned_agents: List[str]
    ) -> RunningReport:
        """Initialize a new running report for a research sequence.
        
        Args:
            research_topic: The main research topic
            sequence_name: Name of the agent sequence being used
            planned_agents: List of agents planned for execution
            
        Returns:
            Initialized RunningReport instance
        """
        logger.info(f"Initializing running report for '{research_topic}' with sequence '{sequence_name}'")
        
        return RunningReport(
            research_topic=research_topic,
            sequence_name=sequence_name,
            start_time=datetime.utcnow(),
            agent_reports=[],
            all_insights=[],
            insight_connections=[],
            executive_summary="",
            detailed_findings=[],
            recommendations=[],
            total_agents_executed=0,
            total_execution_time=0.0,
            completion_status="running"
        )
    
    @staticmethod
    def add_agent_execution(
        running_report: RunningReport,
        agent_report: AgentExecutionReport
    ) -> RunningReport:
        """Add an agent execution report to the running report.
        
        Args:
            running_report: Current running report
            agent_report: Report from completed agent execution
            
        Returns:
            Updated running report with new agent results
        """
        logger.info(f"Adding execution report for agent '{agent_report.agent_name}' to running report")
        
        # Update agent reports
        updated_reports = running_report.agent_reports + [agent_report]
        
        # Update cumulative insights
        updated_insights = running_report.all_insights + agent_report.insights
        
        # Track insight connections between agents
        new_connections = RunningReportBuilder._analyze_insight_connections(
            agent_report, updated_reports[:-1]  # Previous agent reports
        )
        updated_connections = running_report.insight_connections + new_connections
        
        # Update detailed findings
        agent_findings = [
            f"[{agent_report.agent_type.upper()}] {insight}" 
            for insight in agent_report.insights
        ]
        updated_detailed_findings = running_report.detailed_findings + agent_findings
        
        # Update metadata
        total_execution_time = running_report.total_execution_time + agent_report.execution_duration
        total_agents_executed = running_report.total_agents_executed + 1
        
        # Create updated running report
        updated_report = RunningReport(
            research_topic=running_report.research_topic,
            sequence_name=running_report.sequence_name,
            start_time=running_report.start_time,
            agent_reports=updated_reports,
            all_insights=updated_insights,
            insight_connections=updated_connections,
            executive_summary=running_report.executive_summary,  # Updated separately
            detailed_findings=updated_detailed_findings,
            recommendations=running_report.recommendations,  # Updated separately
            total_agents_executed=total_agents_executed,
            total_execution_time=total_execution_time,
            completion_status="running"
        )
        
        logger.debug(f"Running report updated: {total_agents_executed} agents, "
                    f"{len(updated_insights)} total insights, "
                    f"{total_execution_time:.1f}s total time")
        
        return updated_report
    
    @staticmethod
    def _analyze_insight_connections(
        current_agent_report: AgentExecutionReport,
        previous_reports: List[AgentExecutionReport]
    ) -> List[Dict[str, str]]:
        """Analyze connections between current agent insights and previous insights.
        
        Args:
            current_agent_report: Report from current agent
            previous_reports: Reports from all previous agents
            
        Returns:
            List of insight connection dictionaries
        """
        connections = []
        
        # Simple keyword-based connection detection
        current_insights_text = " ".join(current_agent_report.insights).lower()
        
        for prev_report in previous_reports:
            for prev_insight in prev_report.insights:
                # Look for shared keywords/concepts
                prev_words = set(prev_insight.lower().split())
                current_words = set(current_insights_text.split())
                
                # Find overlapping significant words (length > 3)
                significant_overlap = {
                    word for word in prev_words & current_words 
                    if len(word) > 3 and word not in {'research', 'analysis', 'study', 'finding'}
                }
                
                if len(significant_overlap) >= 2:  # Threshold for connection
                    connections.append({
                        "from_agent": prev_report.agent_name,
                        "to_agent": current_agent_report.agent_name,
                        "connection_type": "concept_overlap",
                        "shared_concepts": list(significant_overlap),
                        "source_insight": prev_insight[:100] + "..." if len(prev_insight) > 100 else prev_insight,
                        "target_insights": current_agent_report.insights[:2]  # First 2 insights
                    })
        
        return connections
    
    @staticmethod
    def update_executive_summary(
        running_report: RunningReport,
        force_regenerate: bool = False
    ) -> RunningReport:
        """Update the executive summary based on current insights.
        
        Args:
            running_report: Current running report
            force_regenerate: Whether to regenerate even if summary exists
            
        Returns:
            Running report with updated executive summary
        """
        # Only update if we have insights and no summary exists (or forced)
        if not running_report.all_insights:
            return running_report
        
        if running_report.executive_summary and not force_regenerate:
            return running_report
        
        logger.info(f"Updating executive summary with {len(running_report.all_insights)} insights")
        
        # Generate summary from insights
        top_insights = running_report.all_insights[:5]  # Top 5 insights
        agent_types = list(set(report.agent_type for report in running_report.agent_reports))
        
        summary_parts = [
            f"Research on '{running_report.research_topic}' has been conducted using the "
            f"'{running_report.sequence_name}' sequence with {running_report.total_agents_executed} "
            f"specialized agents ({', '.join(agent_types)}).",
            "",
            "Key findings include:"
        ]
        
        for i, insight in enumerate(top_insights, 1):
            summary_parts.append(f"{i}. {insight}")
        
        if len(running_report.all_insights) > 5:
            summary_parts.append(f"...and {len(running_report.all_insights) - 5} additional insights.")
        
        summary_parts.extend([
            "",
            f"Total research time: {running_report.total_execution_time:.1f} seconds. "
            f"Insight connections identified: {len(running_report.insight_connections)}."
        ])
        
        updated_summary = "\n".join(summary_parts)
        
        # Create updated running report
        updated_report = RunningReport(
            research_topic=running_report.research_topic,
            sequence_name=running_report.sequence_name,
            start_time=running_report.start_time,
            agent_reports=running_report.agent_reports,
            all_insights=running_report.all_insights,
            insight_connections=running_report.insight_connections,
            executive_summary=updated_summary,
            detailed_findings=running_report.detailed_findings,
            recommendations=running_report.recommendations,
            total_agents_executed=running_report.total_agents_executed,
            total_execution_time=running_report.total_execution_time,
            completion_status=running_report.completion_status
        )
        
        return updated_report
    
    @staticmethod
    def generate_recommendations(
        running_report: RunningReport
    ) -> RunningReport:
        """Generate actionable recommendations from agent insights.
        
        Args:
            running_report: Current running report
            
        Returns:
            Running report with generated recommendations
        """
        if not running_report.all_insights:
            return running_report
        
        logger.info(f"Generating recommendations from {len(running_report.all_insights)} insights")
        
        # Extract actionable insights
        actionable_keywords = [
            'recommend', 'suggest', 'should', 'could', 'implement',
            'apply', 'strategy', 'approach', 'method', 'framework',
            'next steps', 'action', 'solution', 'opportunity'
        ]
        
        recommendations = []
        
        for insight in running_report.all_insights:
            insight_lower = insight.lower()
            if any(keyword in insight_lower for keyword in actionable_keywords):
                # Extract recommendation from insight
                if len(insight) > 200:
                    recommendation = insight[:200] + "..."
                else:
                    recommendation = insight
                recommendations.append(recommendation)
        
        # Add general recommendations if none found
        if not recommendations:
            recommendations = [
                f"Continue researching {running_report.research_topic} with focus on identified gaps",
                "Validate findings through additional sources and expert consultation",
                "Consider implementing insights in a pilot or test environment"
            ]
        
        # Limit to top 10 recommendations
        final_recommendations = recommendations[:10]
        
        # Create updated running report
        updated_report = RunningReport(
            research_topic=running_report.research_topic,
            sequence_name=running_report.sequence_name,
            start_time=running_report.start_time,
            agent_reports=running_report.agent_reports,
            all_insights=running_report.all_insights,
            insight_connections=running_report.insight_connections,
            executive_summary=running_report.executive_summary,
            detailed_findings=running_report.detailed_findings,
            recommendations=final_recommendations,
            total_agents_executed=running_report.total_agents_executed,
            total_execution_time=running_report.total_execution_time,
            completion_status=running_report.completion_status
        )
        
        return updated_report
    
    @staticmethod
    def finalize_report(
        running_report: RunningReport
    ) -> RunningReport:
        """Finalize the running report when all agents have completed.
        
        Args:
            running_report: Current running report
            
        Returns:
            Finalized running report
        """
        logger.info(f"Finalizing running report for '{running_report.research_topic}'")
        
        # Update executive summary and recommendations
        updated_report = RunningReportBuilder.update_executive_summary(
            running_report, force_regenerate=True
        )
        updated_report = RunningReportBuilder.generate_recommendations(updated_report)
        
        # Mark as completed
        final_report = RunningReport(
            research_topic=updated_report.research_topic,
            sequence_name=updated_report.sequence_name,
            start_time=updated_report.start_time,
            agent_reports=updated_report.agent_reports,
            all_insights=updated_report.all_insights,
            insight_connections=updated_report.insight_connections,
            executive_summary=updated_report.executive_summary,
            detailed_findings=updated_report.detailed_findings,
            recommendations=updated_report.recommendations,
            total_agents_executed=updated_report.total_agents_executed,
            total_execution_time=updated_report.total_execution_time,
            completion_status="completed"
        )
        
        logger.info(f"Report finalized: {final_report.total_agents_executed} agents, "
                   f"{len(final_report.all_insights)} insights, "
                   f"{final_report.total_execution_time:.1f}s total time")
        
        return final_report
    
    @staticmethod
    def format_report_as_markdown(running_report: RunningReport) -> str:
        """Format the running report as a comprehensive Markdown document.
        
        Args:
            running_report: The running report to format
            
        Returns:
            Formatted Markdown report
        """
        report_lines = [
            f"# Research Report: {running_report.research_topic}",
            "",
            f"**Sequence:** {running_report.sequence_name}  ",
            f"**Generated:** {running_report.start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC  ",
            f"**Status:** {running_report.completion_status.title()}  ",
            f"**Total Execution Time:** {running_report.total_execution_time:.1f} seconds  ",
            f"**Agents Executed:** {running_report.total_agents_executed}  ",
            "",
            "## Executive Summary",
            "",
            running_report.executive_summary,
            "",
            "## Detailed Findings by Agent",
            ""
        ]
        
        # Add agent-specific sections
        for i, agent_report in enumerate(running_report.agent_reports, 1):
            agent_name = agent_report.agent_name.replace('-', ' ').replace('_', ' ').title()
            report_lines.extend([
                f"### {i}. {agent_name} ({agent_report.agent_type})",
                "",
                f"**Execution Time:** {agent_report.execution_duration:.1f} seconds  ",
                f"**Completion Confidence:** {agent_report.completion_confidence:.2f}  ",
                f"**Research Depth:** {agent_report.research_depth_score:.2f}  ",
                f"**Questions Addressed:** {len(agent_report.questions_addressed)}  ",
                "",
                "#### Key Insights:",
                ""
            ])
            
            for insight in agent_report.insights:
                report_lines.append(f"- {insight}")
            
            if agent_report.research_content:
                report_lines.extend([
                    "",
                    "#### Research Summary:",
                    "",
                    agent_report.research_content[:500] + ("..." if len(agent_report.research_content) > 500 else ""),
                    ""
                ])
        
        # Add insight connections
        if running_report.insight_connections:
            report_lines.extend([
                "## Insight Connections",
                "",
                "The following connections were identified between agent insights:",
                ""
            ])
            
            for connection in running_report.insight_connections:
                shared_concepts = ", ".join(connection.get("shared_concepts", []))
                report_lines.extend([
                    f"**{connection['from_agent']} → {connection['to_agent']}**  ",
                    f"Connection: {connection.get('connection_type', 'unknown')}  ",
                    f"Shared concepts: {shared_concepts}  ",
                    ""
                ])
        
        # Add recommendations
        if running_report.recommendations:
            report_lines.extend([
                "## Recommendations",
                "",
                "Based on the research findings, the following recommendations are provided:",
                ""
            ])
            
            for i, recommendation in enumerate(running_report.recommendations, 1):
                report_lines.append(f"{i}. {recommendation}")
            
            report_lines.append("")
        
        # Add summary statistics
        unique_insights = len(set(running_report.all_insights))
        redundancy = ((len(running_report.all_insights) - unique_insights) / 
                     max(len(running_report.all_insights), 1) * 100)
        
        report_lines.extend([
            "## Research Statistics",
            "",
            f"- **Total Insights Generated:** {len(running_report.all_insights)}",
            f"- **Unique Insights:** {unique_insights}",
            f"- **Insight Redundancy:** {redundancy:.1f}%",
            f"- **Average Execution Time per Agent:** {running_report.total_execution_time / max(running_report.total_agents_executed, 1):.1f}s",
            f"- **Insight Connections Discovered:** {len(running_report.insight_connections)}",
            "",
            "---",
            "",
            "*Report generated by Sequential Multi-Agent Supervisor*"
        ])
        
        return "\n".join(report_lines)