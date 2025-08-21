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
 * Sequence strategies for specialized agent ordering
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
  sequence_strategy: SequenceStrategy;
  agent_type?: AgentType;
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
  strategy: SequenceStrategy;
  current_agent: AgentType | null;
  agent_index: number; // 0, 1, 2
  completion_percentage: number;
  estimated_remaining_time?: number; // seconds
  messages_received: number;
  last_activity: number;
  status: 'pending' | 'active' | 'completed' | 'failed' | 'cancelled';
}

/**
 * Complete state for a single research sequence
 */
export interface SequenceState {
  sequence_id: string;
  strategy: SequenceStrategy;
  progress: SequenceProgress;
  messages: RoutedMessage[];
  current_agent: AgentType | null;
  agent_transitions: AgentTransition[];
  errors: ErrorMessage[];
  start_time: number;
  end_time?: number;
  final_result?: string;
  metrics: SequenceMetrics;
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
  total_messages: number;
  processing_time: number; // milliseconds
  agent_calls: number;
  insights_generated: number;
  quality_score?: number;
  efficiency_score?: number;
  last_updated: number;
}

// ========================================
// Real-time State Management
// ========================================

/**
 * Aggregated state for all parallel sequences
 */
export interface ParallelSequencesState {
  sequences: SequenceState[];
  overall_progress: number; // 0-100
  active_sequences: number;
  completed_sequences: number;
  total_messages: number;
  start_time: number;
  research_query: string;
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
  start: (query: string) => Promise<void>;
  stop: () => void;
  restart: () => void;
  
  // Sequence-specific controls
  pauseSequence: (sequenceId: string) => void;
  resumeSequence: (sequenceId: string) => void;
  
  // Message access
  getSequenceMessages: (sequenceId: string) => RoutedMessage[];
  getSequenceProgress: (sequenceId: string) => SequenceProgress | null;
  
  // Connection status
  connectionState: ConnectionState;
  subscriptionStatus: Record<string, 'active' | 'pending' | 'failed'>;
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
  strategy: SequenceStrategy;
  agent_type?: AgentType;
  title: string;
  data: string;
  timestamp: number;
  event_type: 'agent_start' | 'agent_progress' | 'agent_complete' | 'transition' | 'error';
}

