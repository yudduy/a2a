/**
 * Production-ready React hook for managing 3 concurrent research sequences.
 * 
 * This hook provides real-time state management using LangGraph SDK streaming,
 * message routing, and performance optimization for parallel sequence execution.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Client } from '@langchain/langgraph-sdk';
import {
  SequenceState,
  SequenceStrategy,
  LLMGeneratedSequence,
  ParallelSequencesState,
  RealTimeMetrics,
  UseParallelSequencesReturn,
  ConnectionState,
  RoutedMessage,
  SequenceProgress,
  SequenceMetrics,
  ErrorMessage,
} from '@/types/parallel';

/**
 * Configuration for the parallel sequences hook
 */
interface UseParallelSequencesConfig {
  apiUrl?: string;
  assistantId?: string;
  enableMetrics?: boolean;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Required<UseParallelSequencesConfig> = {
  apiUrl: import.meta.env.DEV ? 'http://localhost:2024' : 'http://localhost:8123',
  assistantId: 'Deep Researcher', // Use the exact registered graph name
  enableMetrics: true,
};

/**
 * Create initial sequence state for LLM-generated sequences
 */
const createInitialSequenceState = (sequence: LLMGeneratedSequence): SequenceState => ({
  sequence_id: sequence.sequence_id,
  sequence: sequence,  // Store full LLM sequence instead of strategy enum
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
 * Initialize sequences from LLM-generated sequences
 */
export const initializeSequences = (llmSequences: LLMGeneratedSequence[]): SequenceState[] => {
  return llmSequences.map(seq => createInitialSequenceState(seq));
};

/**
 * Production hook for parallel sequences management
 */
export function useParallelSequences(
  config: UseParallelSequencesConfig = {}
): UseParallelSequencesReturn {
  // Configuration
  const finalConfig = useMemo(() => ({ ...DEFAULT_CONFIG, ...config }), [config]);

  // Core state
  const [sequences, setSequences] = useState<SequenceState[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [subscriptionStatus, setSubscriptionStatus] = useState<Record<string, 'active' | 'pending' | 'failed'>>({});

  // Metrics and progress
  const [metrics] = useState<RealTimeMetrics>({
    messages_per_second: 0,
    average_latency: 0,
    connection_health: 0,
    buffer_utilization: 0,
    error_rate: 0,
    throughput_efficiency: 0,
    last_updated: Date.now(),
  });

  // Refs for stable references
  const clientRef = useRef<Client | null>(null);
  const activeStreamsRef = useRef<Map<string, AbortController>>(new Map());
  const sequenceThreadsRef = useRef<Map<string, string>>(new Map());

  // Additional state for new interface
  const [activeSequenceId, setActiveSequenceId] = useState<string>('');
  const [researchQuery, setResearchQuery] = useState<string>('');

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
      research_query: researchQuery,
      activeSequenceId: activeSequenceId || (sequences.length > 0 ? sequences[0].sequence_id : ''),
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
  }, [sequences, isLoading, activeSequenceId, researchQuery, connectionState, metrics]);

  /**
   * Initialize LangGraph client
   */
  const initializeClient = useCallback(() => {
    if (!clientRef.current) {
      clientRef.current = new Client({
        apiUrl: finalConfig.apiUrl,
      });
      setConnectionState(ConnectionState.CONNECTED);
    }
    return clientRef.current;
  }, [finalConfig.apiUrl]);

  /**
   * Start a sequence stream for LLM-generated sequence
   */
  const startSequenceStream = useCallback(async (
    query: string, 
    sequence: LLMGeneratedSequence, 
    index: number
  ): Promise<string> => {
    const client = initializeClient();
    
    // Create thread for this sequence
    const thread = await client.threads.create();
    const sequenceId = sequence.sequence_id;
    
    sequenceThreadsRef.current.set(sequenceId, thread.thread_id);
    
    // Create abort controller for this stream
    const abortController = new AbortController();
    activeStreamsRef.current.set(sequenceId, abortController);
    
    // Initialize sequence state
    const initialSequence = createInitialSequenceState(sequence);
    
    setSequences(prev => [...prev, initialSequence]);
    
    // Set first sequence as active if none selected
    if (!activeSequenceId && index === 0) {
      setActiveSequenceId(sequenceId);
    }
    
    // Start streaming directly to the graph
    try {
      if (import.meta.env.DEV) {
        console.log(`Starting stream for sequence ${sequenceId} (${sequence.sequence_name})`);
      }
      
      const stream = client.runs.stream(
        thread.thread_id,
        finalConfig.assistantId, // This should be the graph_id 'deep_researcher'
        {
          input: { 
            messages: [{ role: "human", content: query }],
            // Add sequence-specific metadata
            config: {
              sequence_id: sequenceId,
              sequence_name: sequence.sequence_name,
              agent_names: sequence.agent_names,
              research_focus: sequence.research_focus
            }
          },
          signal: abortController.signal,
        }
      );

      // Process stream in background
      processSequenceStream(stream, sequenceId, sequence, index);
      
    } catch (error) {
      if (import.meta.env.DEV) {
        console.error(`Failed to start stream for sequence ${sequenceId}:`, error);
      }
      handleError({
        error_id: `stream_start_${sequenceId}`,
        error_type: 'connection',
        message: `Failed to start sequence stream: ${error instanceof Error ? error.message : 'Unknown error'}`,
        sequence_id: sequenceId,
        recoverable: true,
        timestamp: Date.now(),
      });
    }
    
    return sequenceId;
  }, [finalConfig.assistantId, initializeClient, activeSequenceId]);

  /**
   * Handle incoming WebSocket message
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
   * Handle WebSocket errors
   */
  const handleError = useCallback((errorMessage: ErrorMessage) => {
    console.error('Streaming error:', errorMessage);
    
    // Add error to relevant sequence
    if (errorMessage.sequence_id) {
      setSequences(prev => prev.map(seq => {
        if (seq.sequence_id === errorMessage.sequence_id) {
          return {
            ...seq,
            errors: [...seq.errors, errorMessage],
            progress: {
              ...seq.progress,
              status: errorMessage.recoverable ? seq.progress.status : 'failed',
            },
          };
        }
        return seq;
      }));
    } else {
      // Global error
      setError(new Error(errorMessage.message));
    }
  }, []);

  /**
   * Process stream events for a sequence
   */
  const processSequenceStream = useCallback(async (
    stream: any,
    sequenceId: string,
    sequence: LLMGeneratedSequence,
    index: number
  ) => {
    try {
      // Update sequence status to running
      setSequences(prev => prev.map(seq => 
        seq.sequence_id === sequenceId 
          ? { ...seq, status: 'running' }
          : seq
      ));

      for await (const chunk of stream) {
        // Convert LangGraph chunk to our message format
        const routedMessage: RoutedMessage = {
          message_id: `msg_${sequenceId}_${Date.now()}`,
          sequence_id: sequenceId,
          sequence_name: sequence.sequence_name,
          message_type: 'progress',
          timestamp: Date.now(),
          content: chunk.data || chunk,
          sequence_index: index,
          routing_timestamp: Date.now(),
        };

        // Add message type detection based on chunk content
        if (chunk.event === 'messages') {
          routedMessage.message_type = 'result';
        } else if (chunk.event === 'error') {
          routedMessage.message_type = 'error';
        } else if (chunk.event === 'end') {
          routedMessage.message_type = 'completion';
        }

        handleIncomingMessage(routedMessage);
      }

      // Mark sequence as completed
      setSequences(prev => prev.map(seq => 
        seq.sequence_id === sequenceId 
          ? { ...seq, status: 'completed', end_time: Date.now() }
          : seq
      ));

    } catch (error) {
      if ((error as any)?.name !== 'AbortError') {
        console.error(`Stream processing error for sequence ${sequenceId}:`, error);
        
        // Mark sequence as failed
        setSequences(prev => prev.map(seq => 
          seq.sequence_id === sequenceId 
            ? { ...seq, status: 'failed' }
            : seq
        ));

        handleError({
          error_id: `stream_error_${sequenceId}`,
          error_type: 'processing',
          message: `Stream processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          sequence_id: sequenceId,
          recoverable: false,
          timestamp: Date.now(),
        });
      }
    }
  }, [handleIncomingMessage, handleError]);

  /**
   * Start parallel research sequences with LLM-generated sequences
   * Enhanced for in-place tabs support
   */
  const start = useCallback(async (query: string, llmSequences?: LLMGeneratedSequence[]) => {
    if (!query.trim()) {
      throw new Error('Research query is required');
    }

    try {
      setIsLoading(true);
      setError(null);
      setSequences([]);
      setResearchQuery(query);
      
      // Clear any existing streams
      activeStreamsRef.current.forEach(controller => controller.abort());
      activeStreamsRef.current.clear();
      sequenceThreadsRef.current.clear();

      // Use provided LLM sequences (prioritized for in-place tabs) or fallback to legacy strategies
      let sequencesToUse: LLMGeneratedSequence[];
      
      if (llmSequences && llmSequences.length > 0) {
        sequencesToUse = llmSequences;
        if (import.meta.env.DEV) {
          console.log('Using provided LLM sequences for in-place tabs:', sequencesToUse);
        }
      } else {
        // Fallback to legacy strategies (for backward compatibility)
        const legacyStrategies = [
          SequenceStrategy.THEORY_FIRST,
          SequenceStrategy.MARKET_FIRST,
          SequenceStrategy.FUTURE_BACK,
        ];
        
        sequencesToUse = legacyStrategies.map((strategy, index) => ({
          sequence_id: `seq_${strategy}_${Date.now()}_${index}`,
          sequence_name: strategy.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
          agent_names: ['research_agent', 'analysis_agent', 'synthesis_agent'],
          rationale: `Legacy ${strategy} strategy`,
          research_focus: `${strategy} approach`,
          confidence_score: 0.7,
          approach_description: `Traditional ${strategy} research approach`,
          expected_outcomes: ['Research insights', 'Analysis results'],
          created_at: new Date().toISOString(),
        }));
        if (import.meta.env.DEV) {
          console.log('Using legacy sequences as fallback:', sequencesToUse);
        }
      }

      const sequencePromises = sequencesToUse.map((sequence, index) => 
        startSequenceStream(query, sequence, index)
      );

      const sequenceIds = await Promise.all(sequencePromises);

      // Update subscription status
      const initialSubscriptionStatus: Record<string, 'active' | 'pending' | 'failed'> = {};
      sequenceIds.forEach(id => {
        initialSubscriptionStatus[id] = 'active';
      });
      setSubscriptionStatus(initialSubscriptionStatus);

      if (import.meta.env.DEV) {
        console.log(`Started ${sequenceIds.length} parallel sequences for in-place tabs`);
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
  }, [startSequenceStream]);

  /**
   * Stop all sequences
   */
  const stop = useCallback(() => {
    setIsLoading(false);
    
    // Abort all active streams
    activeStreamsRef.current.forEach(controller => controller.abort());
    activeStreamsRef.current.clear();
    sequenceThreadsRef.current.clear();
    
    setConnectionState(ConnectionState.DISCONNECTED);
    setSubscriptionStatus({});
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
   * Change active sequence being viewed (enhanced for in-place tabs)
   */
  const changeActiveSequence = useCallback((sequenceId: string) => {
    if (import.meta.env.DEV) {
      console.log(`Changing active sequence to: ${sequenceId}`);
    }
    setActiveSequenceId(sequenceId);
  }, []);

  /**
   * Pause specific sequence
   */
  const pauseSequence = useCallback((sequenceId: string) => {
    setSequences(prev => prev.map(seq => 
      seq.sequence_id === sequenceId 
        ? { ...seq, progress: { ...seq.progress, status: 'pending' } }
        : seq
    ));
  }, []);

  /**
   * Resume specific sequence
   */
  const resumeSequence = useCallback((sequenceId: string) => {
    setSequences(prev => prev.map(seq => 
      seq.sequence_id === sequenceId 
        ? { ...seq, progress: { ...seq.progress, status: 'active' } }
        : seq
    ));
  }, []);

  /**
   * Get messages for specific sequence
   */
  const getSequenceMessages = useCallback((sequenceId: string): RoutedMessage[] => {
    const sequence = sequences.find(s => s.sequence_id === sequenceId);
    return sequence?.messages || [];
  }, [sequences]);

  /**
   * Get progress for specific sequence
   */
  const getSequenceProgress = useCallback((sequenceId: string): SequenceProgress | null => {
    const sequence = sequences.find(s => s.sequence_id === sequenceId);
    return sequence?.progress || null;
  }, [sequences]);


  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  /**
   * Mark loading as false when all sequences complete or fail
   * Enhanced for variable sequence counts (not just 3)
   */
  useEffect(() => {
    const hasSequences = sequences.length > 0;
    const allSequencesCompleted = hasSequences && 
      sequences.every(seq => seq.progress.status === 'completed' || seq.progress.status === 'failed');
    
    if (allSequencesCompleted && isLoading) {
      if (import.meta.env.DEV) {
        console.log(`All ${sequences.length} sequences completed, stopping loading`);
      }
      setIsLoading(false);
    }
  }, [sequences, isLoading]);

  /**
   * Log sequence state changes for debugging in-place tabs
   */
  useEffect(() => {
    if (sequences.length > 0) {
      const activeCount = sequences.filter(s => s.status === 'running').length;
      const completedCount = sequences.filter(s => s.status === 'completed').length;
      const failedCount = sequences.filter(s => s.status === 'failed').length;
      
      if (import.meta.env.DEV) {
        console.log(`Sequence status update: ${activeCount} active, ${completedCount} completed, ${failedCount} failed`);
      }
    }
  }, [sequences]);

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
    
    // Sequence-specific controls
    pauseSequence,
    resumeSequence,
    changeActiveSequence,
    
    // Message access
    getSequenceMessages,
    getSequenceProgress,
    
    // Connection status
    connectionState,
    subscriptionStatus,
  };
}

export default useParallelSequences;