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
 * Create initial sequence state
 */
const createInitialSequenceState = (strategy: SequenceStrategy): SequenceState => ({
  sequence_id: `seq_${strategy}_${Date.now()}`,
  strategy,
  progress: {
    sequence_id: `seq_${strategy}_${Date.now()}`,
    strategy,
    current_agent: null,
    agent_index: 0,
    completion_percentage: 0,
    messages_received: 0,
    last_activity: Date.now(),
    status: 'pending',
  },
  messages: [],
  current_agent: null,
  agent_transitions: [],
  errors: [],
  start_time: Date.now(),
  metrics: {
    total_messages: 0,
    processing_time: 0,
    agent_calls: 0,
    insights_generated: 0,
    last_updated: Date.now(),
  },
});

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

  /**
   * Calculated progress state
   */
  const progress: ParallelSequencesState = useMemo(() => {
    const activeSequences = sequences.filter(s => s.progress.status === 'active').length;
    const completedSequences = sequences.filter(s => s.progress.status === 'completed').length;
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
      research_query: '', // This would be set when starting research
      status: isLoading 
        ? 'running' 
        : completedSequences === 3 
          ? 'completed' 
          : activeSequences > 0 
            ? 'running' 
            : 'initializing',
    };
  }, [sequences, isLoading]);

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
   * Start a sequence stream for a specific strategy
   */
  const startSequenceStream = useCallback(async (
    query: string, 
    strategy: SequenceStrategy, 
    index: number
  ): Promise<string> => {
    const client = initializeClient();
    
    // Create thread for this sequence
    const thread = await client.threads.create();
    const sequenceId = `seq_${strategy}_${Date.now()}_${index}`;
    
    sequenceThreadsRef.current.set(sequenceId, thread.thread_id);
    
    // Create abort controller for this stream
    const abortController = new AbortController();
    activeStreamsRef.current.set(sequenceId, abortController);
    
    // Initialize sequence state
    const initialSequence = createInitialSequenceState(strategy);
    initialSequence.sequence_id = sequenceId;
    
    setSequences(prev => [...prev, initialSequence]);
    
    // Start streaming directly to the graph
    try {
      console.log(`Starting stream for sequence ${sequenceId} with strategy ${strategy}`);
      
      const stream = client.runs.stream(
        thread.thread_id,
        finalConfig.assistantId, // This should be the graph_id 'deep_researcher'
        {
          input: { 
            messages: [{ role: "human", content: query }],
            // Add sequence-specific metadata if needed
            config: {
              sequence_strategy: strategy,
              sequence_id: sequenceId
            }
          },
          signal: abortController.signal,
        }
      );

      // Process stream in background
      processSequenceStream(stream, sequenceId, strategy, index);
      
    } catch (error) {
      console.error(`Failed to start stream for sequence ${sequenceId}:`, error);
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
  }, [finalConfig.assistantId, initializeClient]);

  /**
   * Handle incoming WebSocket message
   */
  const handleIncomingMessage = useCallback((message: RoutedMessage) => {
    setSequences(prev => prev.map(seq => {
      if (seq.sequence_id === message.sequence_id) {
        const updatedMessages = [...seq.messages, message];
        const updatedMetrics: SequenceMetrics = {
          ...seq.metrics,
          total_messages: updatedMessages.length,
          last_updated: Date.now(),
        };

        // Update progress based on message type
        let updatedProgress = { ...seq.progress };
        
        switch (message.message_type) {
          case 'progress':
            updatedProgress = {
              ...updatedProgress,
              messages_received: updatedProgress.messages_received + 1,
              last_activity: Date.now(),
            };
            break;
          
          case 'agent_transition':
            if (message.agent_type) {
              updatedProgress = {
                ...updatedProgress,
                current_agent: message.agent_type,
                agent_index: Math.min(updatedProgress.agent_index + 1, 2),
                completion_percentage: Math.min((updatedProgress.agent_index + 1) * 33.33, 100),
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
    strategy: SequenceStrategy,
    index: number
  ) => {
    try {
      for await (const chunk of stream) {
        // Convert LangGraph chunk to our message format
        const routedMessage: RoutedMessage = {
          message_id: `msg_${sequenceId}_${Date.now()}`,
          sequence_id: sequenceId,
          sequence_strategy: strategy,
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
    } catch (error) {
      if ((error as any)?.name !== 'AbortError') {
        console.error(`Stream processing error for sequence ${sequenceId}:`, error);
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
   * Start parallel research sequences
   */
  const start = useCallback(async (query: string) => {
    if (!query.trim()) {
      throw new Error('Research query is required');
    }

    try {
      setIsLoading(true);
      setError(null);
      setSequences([]);
      
      // Clear any existing streams
      activeStreamsRef.current.forEach(controller => controller.abort());
      activeStreamsRef.current.clear();
      sequenceThreadsRef.current.clear();

      // Start all three sequences in parallel
      const strategies = [
        SequenceStrategy.THEORY_FIRST,
        SequenceStrategy.MARKET_FIRST,
        SequenceStrategy.FUTURE_BACK,
      ];

      const sequencePromises = strategies.map((strategy, index) => 
        startSequenceStream(query, strategy, index)
      );

      const sequenceIds = await Promise.all(sequencePromises);

      // Update subscription status
      const initialSubscriptionStatus: Record<string, 'active' | 'pending' | 'failed'> = {};
      sequenceIds.forEach(id => {
        initialSubscriptionStatus[id] = 'active';
      });
      setSubscriptionStatus(initialSubscriptionStatus);

    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to start parallel sequences');
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
   */
  useEffect(() => {
    const allSequencesCompleted = sequences.length === 3 && 
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
    
    // Sequence-specific controls
    pauseSequence,
    resumeSequence,
    
    // Message access
    getSequenceMessages,
    getSequenceProgress,
    
    // Connection status
    connectionState,
    subscriptionStatus,
  };
}

export default useParallelSequences;