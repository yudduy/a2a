/**
 * Production-ready React hook for managing 3 concurrent research sequences.
 * 
 * This hook provides real-time state management, WebSocket connection handling,
 * message routing, and performance optimization for parallel sequence execution.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  SequenceState,
  SequenceStrategy,
  ParallelSequencesState,
  RealTimeMetrics,
  UseParallelSequencesReturn,
  ConnectionState,
  RoutedMessage,
  SequenceProgress,
  AgentTransition,
  SequenceMetrics,
  ErrorMessage,
  WebSocketEventHandlers,
  WebSocketClientConfig,
} from '@/types/parallel';
import { OptimizedParallelWebSocketClient } from '@/utils/optimizedWebSocketClient';
import { ParallelWebSocketClient } from '@/utils/websocketClient';

/**
 * Configuration for the parallel sequences hook
 */
interface UseParallelSequencesConfig {
  apiUrl?: string;
  assistantId?: string;
  enableAutoReconnect?: boolean;
  enableMetrics?: boolean;
  bufferSize?: number;
  maxReconnectAttempts?: number;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Required<UseParallelSequencesConfig> = {
  apiUrl: import.meta.env.DEV ? 'http://localhost:2024' : 'http://localhost:8123',
  assistantId: 'parallel_deep_researcher',
  enableAutoReconnect: true,
  enableMetrics: true,
  bufferSize: 1000,
  maxReconnectAttempts: 5,
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
  const [metrics, setMetrics] = useState<RealTimeMetrics>({
    messages_per_second: 0,
    average_latency: 0,
    connection_health: 0,
    buffer_utilization: 0,
    error_rate: 0,
    throughput_efficiency: 0,
    last_updated: Date.now(),
  });

  // Refs for stable references
  const clientRef = useRef<OptimizedParallelWebSocketClient | null>(null);
  const fallbackClientRef = useRef<ParallelWebSocketClient | null>(null);
  const sequenceIdsRef = useRef<string[]>([]);
  const isInitializedRef = useRef(false);

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
   * WebSocket event handlers
   */
  const eventHandlers: WebSocketEventHandlers = useMemo(() => ({
    onMessage: (message: RoutedMessage) => {
      handleIncomingMessage(message);
    },
    onConnectionStateChange: (state: ConnectionState) => {
      setConnectionState(state);
    },
    onError: (errorMessage: ErrorMessage) => {
      handleError(errorMessage);
    },
    onMetricsUpdate: (newMetrics: RealTimeMetrics) => {
      setMetrics(newMetrics);
    },
    onAgentTransition: (transition: AgentTransition) => {
      handleAgentTransition(transition);
    },
  }), []);

  /**
   * Initialize WebSocket client
   */
  const initializeClient = useCallback(async () => {
    if (clientRef.current || isInitializedRef.current) {
      return;
    }

    try {
      const clientConfig: WebSocketClientConfig = {
        apiUrl: finalConfig.apiUrl,
        assistantId: finalConfig.assistantId,
        maxConnections: 3,
        reconnectAttempts: finalConfig.maxReconnectAttempts,
        reconnectDelay: 1000,
        heartbeatInterval: 30000,
        messageTimeout: 10000,
        bufferSize: finalConfig.bufferSize,
        compressionEnabled: true,
        enableMetrics: finalConfig.enableMetrics,
      };

      // Try optimized client first, fallback to regular client
      let client: OptimizedParallelWebSocketClient | ParallelWebSocketClient;
      try {
        client = new OptimizedParallelWebSocketClient(clientConfig, eventHandlers);
        await client.start();
        clientRef.current = client;
      } catch (optimizedError) {
        console.warn('Optimized WebSocket client failed, falling back to regular client:', optimizedError);
        client = new ParallelWebSocketClient(clientConfig, eventHandlers);
        await client.start();
        fallbackClientRef.current = client;
      }
      
      isInitializedRef.current = true;
      setConnectionState(ConnectionState.CONNECTED);
      
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to initialize WebSocket client');
      setError(error);
      setConnectionState(ConnectionState.FAILED);
      throw error;
    }
  }, [finalConfig, eventHandlers]);

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

      // Initialize client if needed
      await initializeClient();

      const activeClient = clientRef.current || fallbackClientRef.current;
      if (!activeClient) {
        throw new Error('WebSocket client not initialized');
      }

      // Initialize sequence states
      const initialSequences = [
        createInitialSequenceState(SequenceStrategy.THEORY_FIRST),
        createInitialSequenceState(SequenceStrategy.MARKET_FIRST),
        createInitialSequenceState(SequenceStrategy.FUTURE_BACK),
      ];

      setSequences(initialSequences);

      // Start parallel sequences
      const sequenceIds = await activeClient.startParallelSequences(query);
      sequenceIdsRef.current = sequenceIds;

      // Update sequence states with actual IDs
      setSequences(prev => prev.map((seq, index) => ({
        ...seq,
        sequence_id: sequenceIds[index] || seq.sequence_id,
        progress: {
          ...seq.progress,
          sequence_id: sequenceIds[index] || seq.sequence_id,
          status: 'active',
        },
      })));

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
  }, [initializeClient]);

  /**
   * Stop all sequences
   */
  const stop = useCallback(() => {
    setIsLoading(false);
    
    if (clientRef.current) {
      clientRef.current.stop().catch(console.error);
      clientRef.current = null;
    }
    if (fallbackClientRef.current) {
      fallbackClientRef.current.stop().catch(console.error);
      fallbackClientRef.current = null;
    }

    isInitializedRef.current = false;
    sequenceIdsRef.current = [];
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
    const activeClient = clientRef.current || fallbackClientRef.current;
    if (!activeClient) {
      return [];
    }
    return activeClient.getSequenceMessages(sequenceId);
  }, []);

  /**
   * Get progress for specific sequence
   */
  const getSequenceProgress = useCallback((sequenceId: string): SequenceProgress | null => {
    const sequence = sequences.find(s => s.sequence_id === sequenceId);
    return sequence?.progress || null;
  }, [sequences]);

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
    console.error('WebSocket error:', errorMessage);
    
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
   * Handle agent transitions
   */
  const handleAgentTransition = useCallback((transition: AgentTransition) => {
    // Find the sequence this transition belongs to
    setSequences(prev => prev.map(seq => {
      // Simple matching - in production, you'd have more sophisticated routing
      const shouldUpdate = seq.current_agent === transition.from_agent || 
                          seq.progress.status === 'active';
      
      if (shouldUpdate) {
        return {
          ...seq,
          agent_transitions: [...seq.agent_transitions, transition],
          current_agent: transition.to_agent,
        };
      }
      return seq;
    }));
  }, []);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      if (clientRef.current) {
        clientRef.current.stop().catch(console.error);
      }
      if (fallbackClientRef.current) {
        fallbackClientRef.current.stop().catch(console.error);
      }
    };
  }, []);

  /**
   * Auto-reconnection logic
   */
  useEffect(() => {
    if (finalConfig.enableAutoReconnect && 
        connectionState === ConnectionState.FAILED && 
        isLoading) {
      const reconnectTimer = setTimeout(() => {
        initializeClient().catch(console.error);
      }, 5000);

      return () => clearTimeout(reconnectTimer);
    }
  }, [connectionState, isLoading, finalConfig.enableAutoReconnect, initializeClient]);

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