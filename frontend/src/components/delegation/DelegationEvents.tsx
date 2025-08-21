import { ProcessedEvent } from '@/components/ActivityTimeline';

/**
 * Delegation-specific event types extending the base ProcessedEvent interface
 */

export enum DelegationEventType {
  SEQUENCE_START = 'sequence_start',
  AGENT_TRANSITION = 'agent_transition',
  INSIGHT_GENERATION = 'insight_generation',
  TOOL_PRODUCTIVITY_UPDATE = 'tool_productivity_update',
  SEQUENCE_COMPARISON = 'sequence_comparison',
  FINAL_SYNTHESIS = 'final_synthesis',
}

export enum AgentType {
  ACADEMIC = 'academic',
  INDUSTRY = 'industry',
  TECHNICAL_TRENDS = 'technical_trends',
}

export enum SequenceStrategy {
  THEORY_FIRST = 'theory_first',
  MARKET_FIRST = 'market_first',
  FUTURE_BACK = 'future_back',
}

export interface DelegationEventData {
  sequenceId: string;
  strategy: SequenceStrategy;
  currentAgent?: AgentType;
  previousAgent?: AgentType;
  nextAgent?: AgentType;
  executionOrder?: number;
  insights?: string[];
  toolProductivity?: number;
  agentEfficiency?: number;
  timeToValue?: number;
  timestamp: number;
}

export interface DelegationProcessedEvent extends ProcessedEvent {
  eventType: DelegationEventType;
  delegationData: DelegationEventData;
  sequenceId: string;
}

/**
 * Process delegation-specific events from the backend into DelegationProcessedEvent format
 */
export function processDelegationEvent(
  event: Record<string, unknown>,
  _selectedAgentId?: string
): DelegationProcessedEvent | null {
  // Check for sequence start events
  if ('sequence_start' in event && event.sequence_start) {
    const sequenceData = event.sequence_start as {
      sequence_id: string;
      strategy: string;
      agent_order: string[];
    };
    
    return {
      title: `Starting ${sequenceData.strategy.replace('_', ' ').toUpperCase()} Sequence`,
      data: `Agent order: ${sequenceData.agent_order.join(' → ')}`,
      eventType: DelegationEventType.SEQUENCE_START,
      delegationData: {
        sequenceId: sequenceData.sequence_id,
        strategy: sequenceData.strategy as SequenceStrategy,
        timestamp: Date.now(),
      },
      sequenceId: sequenceData.sequence_id,
    };
  }

  // Check for agent transition events
  if ('agent_transition' in event && event.agent_transition) {
    const transitionData = event.agent_transition as {
      sequence_id: string;
      from_agent: string;
      to_agent: string;
      execution_order: number;
      insights_transferred: string[];
    };

    return {
      title: `Agent Transition: ${transitionData.from_agent} → ${transitionData.to_agent}`,
      data: `Transferring ${transitionData.insights_transferred.length} insights`,
      eventType: DelegationEventType.AGENT_TRANSITION,
      delegationData: {
        sequenceId: transitionData.sequence_id,
        strategy: SequenceStrategy.THEORY_FIRST, // This would be determined from context
        previousAgent: transitionData.from_agent as AgentType,
        currentAgent: transitionData.to_agent as AgentType,
        executionOrder: transitionData.execution_order,
        insights: transitionData.insights_transferred,
        timestamp: Date.now(),
      },
      sequenceId: transitionData.sequence_id,
    };
  }

  // Check for insight generation events
  if ('insight_generation' in event && event.insight_generation) {
    const insightData = event.insight_generation as {
      sequence_id: string;
      agent_type: string;
      insights_generated: string[];
      quality_score: number;
    };

    return {
      title: `${insightData.agent_type.toUpperCase()} Agent Generated Insights`,
      data: `${insightData.insights_generated.length} insights (Quality: ${(insightData.quality_score * 100).toFixed(1)}%)`,
      eventType: DelegationEventType.INSIGHT_GENERATION,
      delegationData: {
        sequenceId: insightData.sequence_id,
        strategy: SequenceStrategy.THEORY_FIRST, // This would be determined from context
        currentAgent: insightData.agent_type as AgentType,
        insights: insightData.insights_generated,
        timestamp: Date.now(),
      },
      sequenceId: insightData.sequence_id,
    };
  }

  // Check for tool productivity updates
  if ('tool_productivity_update' in event && event.tool_productivity_update) {
    const productivityData = event.tool_productivity_update as {
      sequence_id: string;
      tool_productivity: number;
      agent_efficiency: number;
      time_to_value: number;
    };

    return {
      title: 'Tool Productivity Update',
      data: `Productivity: ${productivityData.tool_productivity.toFixed(2)}, Efficiency: ${(productivityData.agent_efficiency * 100).toFixed(1)}%`,
      eventType: DelegationEventType.TOOL_PRODUCTIVITY_UPDATE,
      delegationData: {
        sequenceId: productivityData.sequence_id,
        strategy: SequenceStrategy.THEORY_FIRST, // This would be determined from context
        toolProductivity: productivityData.tool_productivity,
        agentEfficiency: productivityData.agent_efficiency,
        timeToValue: productivityData.time_to_value,
        timestamp: Date.now(),
      },
      sequenceId: productivityData.sequence_id,
    };
  }

  // Check for sequence comparison events
  if ('sequence_comparison' in event && event.sequence_comparison) {
    const comparisonData = event.sequence_comparison as {
      comparison_id: string;
      highest_productivity_sequence: string;
      productivity_variance: number;
    };

    return {
      title: 'Sequence Comparison Update',
      data: `Best: ${comparisonData.highest_productivity_sequence.replace('_', ' ').toUpperCase()}, Variance: ${(comparisonData.productivity_variance * 100).toFixed(1)}%`,
      eventType: DelegationEventType.SEQUENCE_COMPARISON,
      delegationData: {
        sequenceId: comparisonData.comparison_id,
        strategy: comparisonData.highest_productivity_sequence as SequenceStrategy,
        timestamp: Date.now(),
      },
      sequenceId: comparisonData.comparison_id,
    };
  }

  // Check for final synthesis events
  if ('final_synthesis' in event && event.final_synthesis) {
    const synthesisData = event.final_synthesis as {
      sequence_id: string;
      strategy: string;
      final_quality_score: number;
      unique_insights_count: number;
    };

    return {
      title: 'Final Research Synthesis Complete',
      data: `Quality: ${(synthesisData.final_quality_score * 100).toFixed(1)}%, Unique insights: ${synthesisData.unique_insights_count}`,
      eventType: DelegationEventType.FINAL_SYNTHESIS,
      delegationData: {
        sequenceId: synthesisData.sequence_id,
        strategy: synthesisData.strategy as SequenceStrategy,
        timestamp: Date.now(),
      },
      sequenceId: synthesisData.sequence_id,
    };
  }

  return null;
}

/**
 * Group delegation events by sequence ID for dashboard display
 */
export function groupEventsBySequence(
  events: DelegationProcessedEvent[]
): Record<string, DelegationProcessedEvent[]> {
  return events.reduce((groups, event) => {
    const sequenceId = event.sequenceId;
    if (!groups[sequenceId]) {
      groups[sequenceId] = [];
    }
    groups[sequenceId].push(event);
    return groups;
  }, {} as Record<string, DelegationProcessedEvent[]>);
}

/**
 * Extract latest metrics from delegation events for a specific sequence
 */
export function extractLatestMetrics(
  events: DelegationProcessedEvent[],
  sequenceId: string
): {
  toolProductivity: number | null;
  agentEfficiency: number | null;
  timeToValue: number | null;
  insightCount: number;
} {
  const sequenceEvents = events.filter(e => e.sequenceId === sequenceId);
  
  let latestProductivity: number | null = null;
  let latestEfficiency: number | null = null;
  let latestTimeToValue: number | null = null;
  let totalInsights = 0;

  for (const event of sequenceEvents) {
    if (event.eventType === DelegationEventType.TOOL_PRODUCTIVITY_UPDATE) {
      latestProductivity = event.delegationData.toolProductivity || null;
      latestEfficiency = event.delegationData.agentEfficiency || null;
      latestTimeToValue = event.delegationData.timeToValue || null;
    }
    if (event.eventType === DelegationEventType.INSIGHT_GENERATION) {
      totalInsights += event.delegationData.insights?.length || 0;
    }
  }

  return {
    toolProductivity: latestProductivity,
    agentEfficiency: latestEfficiency,
    timeToValue: latestTimeToValue,
    insightCount: totalInsights,
  };
}

/**
 * Get current agent and execution order for a sequence
 */
export function getCurrentSequenceState(
  events: DelegationProcessedEvent[],
  sequenceId: string
): {
  currentAgent: AgentType | null;
  executionOrder: number;
  strategy: SequenceStrategy | null;
} {
  const sequenceEvents = events
    .filter(e => e.sequenceId === sequenceId)
    .sort((a, b) => a.delegationData.timestamp - b.delegationData.timestamp);

  let currentAgent: AgentType | null = null;
  let executionOrder = 0;
  let strategy: SequenceStrategy | null = null;

  for (const event of sequenceEvents) {
    if (event.eventType === DelegationEventType.SEQUENCE_START) {
      strategy = event.delegationData.strategy;
    }
    if (event.eventType === DelegationEventType.AGENT_TRANSITION) {
      currentAgent = event.delegationData.currentAgent || null;
      executionOrder = event.delegationData.executionOrder || 0;
    }
  }

  return { currentAgent, executionOrder, strategy };
}