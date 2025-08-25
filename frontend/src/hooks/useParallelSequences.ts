/**
 * Simplified React hook for managing in-place parallel sequences.
 * 
 * This streamlined version focuses solely on in-place tab functionality,
 * removing dual client management and complex standalone interface support.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  SequenceState,
  LLMGeneratedSequence,
  ParallelSequencesState,
  RealTimeMetrics,
  UseParallelSequencesReturn,
  ConnectionState,
  RoutedMessage,
  SequenceMetrics,
  ErrorMessage,
} from '@/types/parallel';

interface UseParallelSequencesConfig {
  enableMetrics?: boolean;
}

const DEFAULT_CONFIG: Required<UseParallelSequencesConfig> = {
  enableMetrics: true,
};

/**
 * Create initial sequence state for LLM-generated sequences
 */
const createInitialSequenceState = (sequence: LLMGeneratedSequence): SequenceState => ({
  sequence_id: sequence.sequence_id,
  sequence: sequence,
  progress: {
    sequence_id: sequence.sequence_id,
    sequence_name: sequence.sequence_name,
    current_agent: null,
    current_agent_name: undefined,
    agents_completed: 0,
    total_agents: sequence.agent_names.length,
    completion_percentage: 0,
    estimated_time_remaining: undefined,
    last_activity: Date.now(),
    status: 'initializing',
  },
  messages: [],
  current_agent: null,
  current_agent_name: undefined,
  agent_transitions: [],
  errors: [],
  start_time: Date.now(),
  metrics: {
    sequence_id: sequence.sequence_id,
    message_count: 0,
    research_duration: 0,
    tokens_used: 0,
    average_response_time: 0,
  },
  status: 'initializing',
  last_activity: new Date().toISOString(),
});

/**
 * Simplified hook for parallel sequences management - in-place tabs only
 */
export function useParallelSequences(
  config: UseParallelSequencesConfig = {}
): UseParallelSequencesReturn {
  const finalConfig = useMemo(() => ({ ...DEFAULT_CONFIG, ...config }), [config]);

  // Core state
  const [sequences, setSequences] = useState<SequenceState[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.CONNECTED);
  
  // Simple metrics
  const [metrics] = useState<RealTimeMetrics>({
    messages_per_second: 0,
    average_latency: 0,
    connection_health: 1,
    buffer_utilization: 0,
    error_rate: 0,
    throughput_efficiency: 1,
    last_updated: Date.now(),
  });

  // Current state refs
  const activeSequenceId = useRef<string>('');
  const researchQuery = useRef<string>('');

  /**
   * Calculated progress state
   */
  const progress: ParallelSequencesState = useMemo(() => {
    const activeSequences = sequences.filter(s => s.status === 'running').length;
    const completedSequences = sequences.filter(s => s.status === 'completed').length;
    const totalMessages = sequences.reduce((sum, s) => sum + s.messages.length, 0);
    const overallProgress = sequences.length > 0 
      ? sequences.reduce((sum, s) => sum + s.progress.completion_percentage, 0) / sequences.length 
      : 0;

    const earliestStartTime = sequences.length > 0 
      ? Math.min(...sequences.map(s => s.start_time))
      : Date.now();

    return {
      sequences,
      overall_progress: overallProgress,
      active_sequences: activeSequences,
      completed_sequences: completedSequences,
      total_messages: totalMessages,
      start_time: earliestStartTime,
      research_query: researchQuery.current,
      activeSequenceId: activeSequenceId.current || (sequences.length > 0 ? sequences[0].sequence_id : ''),
      executionStartTime: sequences.length > 0 ? new Date(earliestStartTime).toISOString() : undefined,
      connectionState,
      metrics,
      status: isLoading 
        ? 'running' 
        : completedSequences === sequences.length && sequences.length > 0
          ? 'completed' 
          : activeSequences > 0 
            ? 'running' 
            : 'initializing',
    };
  }, [sequences, isLoading, connectionState, metrics]);

  /**
   * Handle incoming message routing for in-place tabs
   */
  const handleIncomingMessage = useCallback((message: RoutedMessage) => {
    setSequences(prev => prev.map(seq => {
      if (seq.sequence_id === message.sequence_id) {
        const updatedMessages = [...seq.messages, message];
        const updatedMetrics: SequenceMetrics = {
          ...seq.metrics,
          message_count: updatedMessages.length,
          research_duration: Date.now() - seq.start_time,
        };

        // Update progress based on message type
        let updatedProgress = { ...seq.progress };
        
        switch (message.message_type) {
          case 'progress':
            updatedProgress = {
              ...updatedProgress,
              messages_received: (updatedProgress.messages_received || 0) + 1,
              last_activity: Date.now(),
            };
            break;
          
          case 'agent_transition':
            if (message.current_agent || message.agent_type) {
              const agentName = message.current_agent || message.agent_type;
              const totalAgents = seq.sequence?.agent_names.length || 3;
              const agentsCompleted = Math.min(updatedProgress.agents_completed + 1, totalAgents);
              
              updatedProgress = {
                ...updatedProgress,
                current_agent: message.agent_type || null,
                current_agent_name: agentName,
                agents_completed: agentsCompleted,
                completion_percentage: Math.min((agentsCompleted / totalAgents) * 100, 100),
              };
            }
            break;
          
          case 'completion':
            updatedProgress = {
              ...updatedProgress,
              status: 'completed',
              completion_percentage: 100,
            };
            break;
          
          case 'error':
            updatedProgress = {
              ...updatedProgress,
              status: 'failed',
            };
            break;
        }

        return {
          ...seq,
          messages: updatedMessages,
          progress: updatedProgress,
          metrics: updatedMetrics,
          current_agent: updatedProgress.current_agent,
          current_agent_name: updatedProgress.current_agent_name,
          last_activity: new Date().toISOString(),
        };
      }
      return seq;
    }));
  }, []);

  /**
   * Start parallel research sequences with LLM-generated sequences
   * Simplified for in-place tabs only - uses main stream routing
   */
  const start = useCallback(async (query: string, llmSequences?: LLMGeneratedSequence[]) => {
    if (!query.trim()) {
      throw new Error('Research query is required');
    }

    if (!llmSequences || llmSequences.length === 0) {
      throw new Error('LLM sequences are required for in-place tabs');
    }

    try {
      setIsLoading(true);
      setError(null);
      setSequences([]);
      researchQuery.current = query;
      
      // Initialize sequences from LLM data
      const initialSequences = llmSequences.map(seq => createInitialSequenceState(seq));
      setSequences(initialSequences);
      
      // Set first sequence as active
      if (initialSequences.length > 0) {
        activeSequenceId.current = initialSequences[0].sequence_id;
      }

      if (import.meta.env.DEV) {
        console.log(`Initialized ${initialSequences.length} sequences for in-place tabs`);
      }

    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to start parallel sequences');
      if (import.meta.env.DEV) {
        console.error('Failed to start parallel sequences:', error);
      }
      setError(error);
      setIsLoading(false);
      throw error;
    }
  }, []);

  /**
   * Stop all sequences
   */
  const stop = useCallback(() => {
    setIsLoading(false);
    setConnectionState(ConnectionState.DISCONNECTED);
  }, []);

  /**
   * Restart sequences with same query
   */
  const restart = useCallback(() => {
    const lastQuery = progress.research_query;
    stop();
    if (lastQuery) {
      setTimeout(() => start(lastQuery), 1000);
    }
  }, [progress.research_query, stop, start]);

  /**
   * Change active sequence being viewed
   */
  const changeActiveSequence = useCallback((sequenceId: string) => {
    activeSequenceId.current = sequenceId;
  }, []);

  /**
   * Get messages for specific sequence
   */
  const getSequenceMessages = useCallback((sequenceId: string): RoutedMessage[] => {
    const sequence = sequences.find(s => s.sequence_id === sequenceId);
    return sequence?.messages || [];
  }, [sequences]);

  /**
   * Handle message routing from main stream (used by App.tsx)
   */
  const routeMessage = useCallback((message: RoutedMessage) => {
    handleIncomingMessage(message);
  }, [handleIncomingMessage]);

  /**
   * Mark loading as false when all sequences complete
   */
  useEffect(() => {
    const hasSequences = sequences.length > 0;
    const allSequencesCompleted = hasSequences && 
      sequences.every(seq => seq.progress.status === 'completed' || seq.progress.status === 'failed');
    
    if (allSequencesCompleted && isLoading) {
      setIsLoading(false);
    }
  }, [sequences, isLoading]);

  return {
    // State
    sequences,
    isLoading,
    error,
    progress,
    metrics,
    
    // Control functions
    start,
    stop,
    restart,
    
    // Sequence-specific controls (simplified)
    pauseSequence: () => {}, // No-op for simplified version
    resumeSequence: () => {}, // No-op for simplified version
    changeActiveSequence,
    
    // Message access
    getSequenceMessages,
    getSequenceProgress: () => null, // No-op for simplified version
    routeMessage,
    
    // Connection status
    connectionState,
  };
}