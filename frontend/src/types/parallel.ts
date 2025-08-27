/**
 * Production-ready TypeScript interfaces for parallel streaming support.
 * 
 * This module defines comprehensive type safety for real-time parallel sequence execution,
 * WebSocket message routing, and state management across 3 concurrent research streams.
 */


// ========================================
// Core Parallel Streaming Types
// ========================================

/**
 * LLM-generated sequence interface (replaces hardcoded strategies)
 */
export interface LLMGeneratedSequence {
  sequence_id: string;
  sequence_name: string;        // LLM-generated name like "Deep Technical Analysis"
  agent_names: string[];        // e.g., ["research_agent", "technical_agent", "analysis_agent"]
  rationale: string;           // LLM reasoning for this sequence
  research_focus: string;      // Focus area description
  confidence_score: number;    // 0.0-1.0
  approach_description: string; // High-level description of the research approach
  expected_outcomes: string[]; // Expected outcomes from this sequence
  created_at: string;         // Timestamp
}

/**
 * Legacy sequence strategies (kept for backward compatibility)
 * @deprecated Use LLMGeneratedSequence instead
 */
export enum SequenceStrategy {
  THEORY_FIRST = "theory_first",
  MARKET_FIRST = "market_first", 
  FUTURE_BACK = "future_back"
}

/**
 * Agent types in the research process
 */
export enum AgentType {
  ACADEMIC = "academic",
  INDUSTRY = "industry", 
  TECHNICAL_TRENDS = "technical_trends"
}

/**
 * WebSocket connection states
 */
export enum ConnectionState {
  CONNECTING = "connecting",
  CONNECTED = "connected",
  DISCONNECTED = "disconnected", 
  RECONNECTING = "reconnecting",
  FAILED = "failed"
}

/**
 * Message delivery guarantee levels
 */
export enum DeliveryGuarantee {
  AT_MOST_ONCE = "at_most_once",
  AT_LEAST_ONCE = "at_least_once",
  EXACTLY_ONCE = "exactly_once"
}

// ========================================
// Message Structure Types
// ========================================

/**
 * Core stream message structure from backend
 */
export interface StreamMessage {
  message_id: string;
  sequence_id: string;
  sequence_name?: string;        // LLM-generated sequence name
  sequence_strategy?: SequenceStrategy; // Legacy field for backward compatibility
  agent_type?: AgentType;
  current_agent?: string;        // Current agent name
  message_type: 'progress' | 'result' | 'error' | 'completion' | 'agent_transition';
  timestamp: number;
  content: any;
  metadata?: Record<string, any>;
  requires_ack?: boolean;
}

/**
 * Routed message with sequence information
 */
export interface RoutedMessage extends StreamMessage {
  sequence_index: number; // 0, 1, or 2 for the three parallel sequences
  routing_timestamp: number;
  tool_calls?: any[];     // Tool calls associated with this message
}

/**
 * WebSocket frame message wrapper
 */
export interface WebSocketFrame {
  type: 'message' | 'ack' | 'ping' | 'pong' | 'error' | 'subscription';
  payload?: StreamMessage | SubscriptionMessage | AckMessage | ErrorMessage;
  timestamp: number;
  frame_id: string;
}

/**
 * Subscription configuration message
 */
export interface SubscriptionMessage {
  subscription_id: string;
  sequence_strategies: SequenceStrategy[];
  message_types: string[];
  delivery_guarantee: DeliveryGuarantee;
  client_id: string;
}

/**
 * Acknowledgment message
 */
export interface AckMessage {
  message_id: string;
  sequence_id: string;
  status: 'received' | 'processed' | 'failed';
  timestamp: number;
}

/**
 * Error message structure
 */
export interface ErrorMessage {
  error_id: string;
  error_type: 'connection' | 'routing' | 'processing' | 'timeout';
  message: string;
  sequence_id?: string;
  recoverable: boolean;
  timestamp: number;
}

// ========================================
// Sequence State Management
// ========================================

/**
 * Progress tracking for individual sequence
 */
export interface SequenceProgress {
  sequence_id: string;
  sequence_name?: string;        // LLM-generated sequence name
  strategy?: SequenceStrategy;   // Legacy field for backward compatibility
  current_agent: AgentType | null;
  current_agent_name?: string;   // Current agent name in LLM sequence
  agents_completed: number;      // Number of agents completed
  total_agents: number;          // Total agents in sequence
  completion_percentage: number;
  estimated_time_remaining?: number; // seconds
  messages_received?: number;    // Made optional for backward compatibility
  last_activity: number;
  status: 'initializing' | 'pending' | 'active' | 'completed' | 'failed' | 'cancelled';
}

/**
 * Complete state for a single research sequence
 */
export interface SequenceState {
  sequence_id: string;
  sequence?: LLMGeneratedSequence;  // LLM-generated sequence details
  strategy?: SequenceStrategy;      // Legacy field for backward compatibility
  progress: SequenceProgress;
  messages: RoutedMessage[];
  current_agent: AgentType | null;
  current_agent_name?: string;      // Current agent name in LLM sequence
  agent_transitions: AgentTransition[];
  errors: ErrorMessage[];
  start_time: number;
  end_time?: number;
  final_result?: string;
  metrics: SequenceMetrics;
  status: 'initializing' | 'running' | 'completed' | 'failed';
  last_activity: string;
}

/**
 * Agent transition tracking
 */
export interface AgentTransition {
  from_agent: AgentType | null;
  to_agent: AgentType;
  transition_time: number;
  insights_transferred: string[];
  transition_context: Record<string, any>;
}

/**
 * Performance metrics for sequence execution
 */
export interface SequenceMetrics {
  sequence_id: string;
  message_count: number;           // Renamed from total_messages for consistency
  research_duration: number;       // milliseconds
  tokens_used: number;            // Token usage tracking
  average_response_time: number;  // Average response time
  agent_calls?: number;           // Made optional
  insights_generated?: number;    // Made optional
  quality_score?: number;
  efficiency_score?: number;
  last_updated?: number;          // Made optional
}

// ========================================
// Real-time State Management
// ========================================

/**
 * Aggregated state for all parallel sequences
 */
export interface ParallelSequencesState {
  sequences: SequenceState[];     // Now contains 3 LLM-generated sequences
  overall_progress: number;       // 0-100
  active_sequences: number;
  completed_sequences: number;
  total_messages: number;
  start_time: number;
  research_query: string;
  activeSequenceId: string;       // Currently viewed sequence
  executionStartTime?: string;
  connectionState: ConnectionState;
  metrics: RealTimeMetrics;
  status: 'initializing' | 'running' | 'completing' | 'completed' | 'failed';
}

/**
 * Real-time metrics across all sequences
 */
export interface RealTimeMetrics {
  messages_per_second: number;
  average_latency: number; // milliseconds
  connection_health: number; // 0-100
  buffer_utilization: number; // 0-100
  error_rate: number; // percentage
  throughput_efficiency: number; // percentage
  last_updated: number;
}

// ========================================
// WebSocket Client Configuration
// ========================================

/**
 * Configuration for WebSocket client manager
 */
export interface WebSocketClientConfig {
  apiUrl: string;
  assistantId: string;
  maxConnections: number;
  reconnectAttempts: number;
  reconnectDelay: number; // milliseconds
  heartbeatInterval: number; // milliseconds
  messageTimeout: number; // milliseconds
  bufferSize: number;
  compressionEnabled: boolean;
  enableMetrics: boolean;
}

/**
 * Connection metrics tracking
 */
export interface ConnectionMetrics {
  messages_sent: number;
  messages_received: number;
  messages_failed: number;
  reconnection_count: number;
  total_connection_time: number; // milliseconds
  last_activity: number;
  latency_samples: number[];
  average_latency: number;
  success_rate: number; // percentage
}

/**
 * Subscription configuration for stream filtering
 */
export interface StreamSubscription {
  subscription_id: string;
  client_id: string;
  sequence_strategies: Set<SequenceStrategy>;
  message_types: Set<string>;
  delivery_guarantee: DeliveryGuarantee;
  include_progress: boolean;
  include_errors: boolean;
  include_results: boolean;
  max_message_rate?: number; // messages per second
  buffer_size: number;
  created_at: number;
  last_activity: number;
}

// ========================================
// Hook Return Types
// ========================================

/**
 * Return type for useParallelSequences hook
 */
export interface UseParallelSequencesReturn {
  // State
  sequences: SequenceState[];
  isLoading: boolean;
  error: Error | null;
  progress: ParallelSequencesState;
  metrics: RealTimeMetrics;
  
  // Control functions
  start: (query: string, llmSequences?: LLMGeneratedSequence[]) => Promise<void>;
  stop: () => void;
  restart: () => void;
  
  // Sequence-specific controls
  pauseSequence: (sequenceId: string) => void;
  resumeSequence: (sequenceId: string) => void;
  changeActiveSequence: (sequenceId: string) => void;
  
  // Message access
  getSequenceMessages: (sequenceId: string) => RoutedMessage[];
  getSequenceProgress: (sequenceId: string) => SequenceProgress | null;
  routeMessage: (message: RoutedMessage) => void;
  
  // Connection status
  connectionState: ConnectionState;
  subscriptionStatus?: Record<string, 'active' | 'pending' | 'failed'>;
}

// ========================================
// Event Handlers
// ========================================

/**
 * Event handler types for WebSocket events
 */
export interface WebSocketEventHandlers {
  onMessage?: (message: RoutedMessage) => void;
  onSequenceProgress?: (progress: SequenceProgress) => void;
  onSequenceComplete?: (sequenceId: string, result: string) => void;
  onError?: (error: ErrorMessage) => void;
  onConnectionStateChange?: (state: ConnectionState) => void;
  onMetricsUpdate?: (metrics: RealTimeMetrics) => void;
  onAgentTransition?: (transition: AgentTransition) => void;
}

// ========================================
// Utility Types
// ========================================

/**
 * Message buffer configuration
 */
export interface MessageBufferConfig {
  max_size: number;
  overflow_strategy: 'drop_oldest' | 'drop_newest' | 'block';
  enable_compression: boolean;
  enable_deduplication: boolean;
}

/**
 * Health check result
 */
export interface HealthCheckResult {
  connection_healthy: boolean;
  latency: number; // milliseconds
  last_message_time: number;
  buffer_utilization: number; // percentage
  error_count: number;
  timestamp: number;
}

/**
 * Performance optimization hints
 */
export interface PerformanceHints {
  enable_message_batching: boolean;
  batch_size: number;
  compression_threshold: number; // bytes
  enable_lazy_rendering: boolean;
  message_retention_limit: number;
}

// ========================================
// Integration with Existing Types
// ========================================

/**
 * Extended message type compatible with existing LangGraph messages
 */
export interface ExtendedParallelMessage {
  id?: string;
  type: string;
  content: unknown;
  sequence_id?: string;
  sequence_strategy?: SequenceStrategy;
  agent_type?: AgentType;
  routing_metadata?: Record<string, any>;
}

/**
 * Processed event for activity timeline with sequence context
 */
export interface ProcessedSequenceEvent {
  sequence_id: string;
  sequence_name?: string;         // LLM-generated sequence name
  strategy?: SequenceStrategy;    // Legacy field for backward compatibility
  agent_type?: AgentType;
  agent_name?: string;           // Current agent name
  title: string;
  data: string;
  timestamp: number;
  event_type: 'agent_start' | 'agent_progress' | 'agent_complete' | 'transition' | 'error';
}

