/**
 * Production-ready WebSocket client manager for parallel sequence streaming.
 * 
 * This module provides robust WebSocket connection management, message routing,
 * reconnection logic, and performance optimization for 3 concurrent research streams.
 */

import {
  ConnectionState,
  DeliveryGuarantee,
  StreamMessage,
  RoutedMessage,
  WebSocketFrame,
  SubscriptionMessage,
  AckMessage,
  ErrorMessage,
  ConnectionMetrics,
  StreamSubscription,
  WebSocketClientConfig,
  WebSocketEventHandlers,
  HealthCheckResult,
  MessageBufferConfig,
  SequenceStrategy,
  AgentType,
} from '@/types/parallel';

/**
 * Message buffer with overflow handling and performance optimization
 */
class MessageBuffer {
  private buffer: RoutedMessage[] = [];
  private readonly maxSize: number;
  private readonly overflowStrategy: string;
  private droppedCount = 0;
  private readonly enableDeduplication: boolean;
  private readonly messageIds = new Set<string>();

  constructor(config: MessageBufferConfig) {
    this.maxSize = config.max_size;
    this.overflowStrategy = config.overflow_strategy;
    this.enableDeduplication = config.enable_deduplication;
  }

  /**
   * Add message to buffer with overflow handling
   */
  addMessage(message: RoutedMessage): boolean {
    // Deduplication check
    if (this.enableDeduplication && this.messageIds.has(message.message_id)) {
      return false;
    }

    // Handle buffer overflow
    if (this.buffer.length >= this.maxSize) {
      switch (this.overflowStrategy) {
        case 'drop_oldest':
          const oldest = this.buffer.shift();
          if (oldest) {
            this.messageIds.delete(oldest.message_id);
          }
          break;
        case 'drop_newest':
          this.droppedCount++;
          return false;
        case 'block':
          // In production, this would implement backpressure
          return false;
        default:
          this.buffer.shift();
      }
    }

    this.buffer.push(message);
    this.messageIds.add(message.message_id);
    return true;
  }

  /**
   * Get messages by sequence ID
   */
  getSequenceMessages(sequenceId: string): RoutedMessage[] {
    return this.buffer.filter(msg => msg.sequence_id === sequenceId);
  }

  /**
   * Get all messages and clear buffer
   */
  getAllMessages(): RoutedMessage[] {
    const messages = [...this.buffer];
    this.clear();
    return messages;
  }

  /**
   * Get recent messages without clearing
   */
  getRecentMessages(count: number): RoutedMessage[] {
    return this.buffer.slice(-count);
  }

  /**
   * Clear buffer
   */
  clear(): void {
    this.buffer = [];
    this.messageIds.clear();
  }

  /**
   * Get buffer statistics
   */
  getStats() {
    return {
      size: this.buffer.length,
      maxSize: this.maxSize,
      droppedCount: this.droppedCount,
      utilization: (this.buffer.length / this.maxSize) * 100,
    };
  }
}

/**
 * WebSocket connection wrapper with state management
 */
class WebSocketConnection {
  private websocket: WebSocket | null = null;
  private state: ConnectionState = ConnectionState.DISCONNECTED;
  private reconnectAttempts = 0;
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private metrics: ConnectionMetrics;
  private pendingAcks = new Map<string, number>();
  private messageQueue: WebSocketFrame[] = [];

  constructor(
    private url: string,
    private config: WebSocketClientConfig,
    private eventHandlers: WebSocketEventHandlers
  ) {
    this.metrics = {
      messages_sent: 0,
      messages_received: 0,
      messages_failed: 0,
      reconnection_count: 0,
      total_connection_time: 0,
      last_activity: Date.now(),
      latency_samples: [],
      average_latency: 0,
      success_rate: 100,
    };
  }

  /**
   * Establish WebSocket connection
   */
  async connect(): Promise<void> {
    if (this.state === ConnectionState.CONNECTING || this.state === ConnectionState.CONNECTED) {
      return;
    }

    this.setState(ConnectionState.CONNECTING);

    try {
      this.websocket = new WebSocket(this.url);
      this.setupEventListeners();
      
      // Wait for connection to open
      await new Promise<void>((resolve, reject) => {
        if (!this.websocket) {
          reject(new Error('WebSocket not initialized'));
          return;
        }

        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
        }, this.config.messageTimeout);

        this.websocket.onopen = () => {
          clearTimeout(timeout);
          resolve();
        };

        this.websocket.onerror = (event) => {
          clearTimeout(timeout);
          reject(new Error(`WebSocket connection failed: ${event}`));
        };
      });

      this.setState(ConnectionState.CONNECTED);
      this.resetReconnectAttempts();
      this.startHeartbeat();
      this.flushMessageQueue();

    } catch (error) {
      this.setState(ConnectionState.FAILED);
      await this.scheduleReconnect();
      throw error;
    }
  }

  /**
   * Close WebSocket connection
   */
  async disconnect(): Promise<void> {
    this.setState(ConnectionState.DISCONNECTED);
    this.stopHeartbeat();
    this.clearReconnectTimer();

    if (this.websocket) {
      this.websocket.close(1000, 'Client disconnect');
      this.websocket = null;
    }
  }

  /**
   * Send message through WebSocket
   */
  async sendMessage(message: StreamMessage): Promise<boolean> {
    const frame: WebSocketFrame = {
      type: 'message',
      payload: message,
      timestamp: Date.now(),
      frame_id: `frame_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };

    return this.sendFrame(frame);
  }

  /**
   * Send subscription message
   */
  async subscribe(subscription: SubscriptionMessage): Promise<boolean> {
    const frame: WebSocketFrame = {
      type: 'subscription',
      payload: subscription,
      timestamp: Date.now(),
      frame_id: `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };

    return this.sendFrame(frame);
  }

  /**
   * Send acknowledgment
   */
  async sendAck(ack: AckMessage): Promise<boolean> {
    const frame: WebSocketFrame = {
      type: 'ack',
      payload: ack,
      timestamp: Date.now(),
      frame_id: `ack_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };

    return this.sendFrame(frame);
  }

  /**
   * Get connection state
   */
  getState(): ConnectionState {
    return this.state;
  }

  /**
   * Get connection metrics
   */
  getMetrics(): ConnectionMetrics {
    return { ...this.metrics };
  }

  /**
   * Perform health check
   */
  async healthCheck(): Promise<HealthCheckResult> {
    const pingStart = Date.now();
    
    try {
      await this.sendPing();
      const latency = Date.now() - pingStart;
      
      return {
        connection_healthy: this.state === ConnectionState.CONNECTED,
        latency,
        last_message_time: this.metrics.last_activity,
        buffer_utilization: (this.messageQueue.length / 100) * 100,
        error_count: this.metrics.messages_failed,
        timestamp: Date.now(),
      };
    } catch (error) {
      return {
        connection_healthy: false,
        latency: -1,
        last_message_time: this.metrics.last_activity,
        buffer_utilization: (this.messageQueue.length / 100) * 100,
        error_count: this.metrics.messages_failed + 1,
        timestamp: Date.now(),
      };
    }
  }

  /**
   * Send WebSocket frame
   */
  private async sendFrame(frame: WebSocketFrame): Promise<boolean> {
    if (this.state !== ConnectionState.CONNECTED || !this.websocket) {
      // Queue message for later delivery
      this.messageQueue.push(frame);
      return false;
    }

    try {
      const serialized = JSON.stringify(frame);
      this.websocket.send(serialized);
      
      this.metrics.messages_sent++;
      this.metrics.last_activity = Date.now();
      
      // Track acknowledgment if required
      if (frame.payload && 'requires_ack' in frame.payload && frame.payload.requires_ack) {
        this.pendingAcks.set(frame.frame_id, Date.now());
      }
      
      return true;
    } catch (error) {
      this.metrics.messages_failed++;
      console.error('Failed to send WebSocket frame:', error);
      return false;
    }
  }

  /**
   * Setup WebSocket event listeners
   */
  private setupEventListeners(): void {
    if (!this.websocket) return;

    this.websocket.onmessage = (event) => {
      this.handleIncomingMessage(event.data);
    };

    this.websocket.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      this.setState(ConnectionState.DISCONNECTED);
      this.scheduleReconnect();
    };

    this.websocket.onerror = (event) => {
      console.error('WebSocket error:', event);
      this.setState(ConnectionState.FAILED);
      this.eventHandlers.onError?.({
        error_id: `ws_error_${Date.now()}`,
        error_type: 'connection',
        message: 'WebSocket connection error',
        recoverable: true,
        timestamp: Date.now(),
      });
    };
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleIncomingMessage(data: string): void {
    try {
      const frame: WebSocketFrame = JSON.parse(data);
      this.metrics.messages_received++;
      this.metrics.last_activity = Date.now();

      switch (frame.type) {
        case 'message':
          if (frame.payload) {
            this.eventHandlers.onMessage?.(frame.payload as RoutedMessage);
          }
          break;

        case 'ack':
          this.handleAcknowledgment(frame.payload as AckMessage);
          break;

        case 'pong':
          // Heartbeat response - update latency
          const latency = Date.now() - frame.timestamp;
          this.updateLatencyMetrics(latency);
          break;

        case 'error':
          this.eventHandlers.onError?.(frame.payload as ErrorMessage);
          break;

        default:
          console.warn('Unknown frame type:', frame.type);
      }
    } catch (error) {
      console.error('Failed to parse incoming message:', error);
      this.metrics.messages_failed++;
    }
  }

  /**
   * Handle message acknowledgment
   */
  private handleAcknowledgment(ack: AckMessage): void {
    const sendTime = this.pendingAcks.get(ack.message_id);
    if (sendTime) {
      const latency = Date.now() - sendTime;
      this.updateLatencyMetrics(latency);
      this.pendingAcks.delete(ack.message_id);
    }
  }

  /**
   * Update latency metrics
   */
  private updateLatencyMetrics(latency: number): void {
    this.metrics.latency_samples.push(latency);
    
    // Keep only recent samples
    if (this.metrics.latency_samples.length > 100) {
      this.metrics.latency_samples = this.metrics.latency_samples.slice(-50);
    }
    
    this.metrics.average_latency = 
      this.metrics.latency_samples.reduce((a, b) => a + b, 0) / 
      this.metrics.latency_samples.length;
  }

  /**
   * Send ping message
   */
  private async sendPing(): Promise<void> {
    const frame: WebSocketFrame = {
      type: 'ping',
      timestamp: Date.now(),
      frame_id: `ping_${Date.now()}`,
    };
    
    await this.sendFrame(frame);
  }

  /**
   * Start heartbeat monitoring
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = window.setInterval(() => {
      this.sendPing().catch(console.error);
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat monitoring
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Set connection state and notify handlers
   */
  private setState(newState: ConnectionState): void {
    if (this.state !== newState) {
      this.state = newState;
      this.eventHandlers.onConnectionStateChange?.(newState);
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private async scheduleReconnect(): Promise<void> {
    if (this.reconnectAttempts >= this.config.reconnectAttempts) {
      this.setState(ConnectionState.FAILED);
      return;
    }

    this.setState(ConnectionState.RECONNECTING);
    this.reconnectAttempts++;
    this.metrics.reconnection_count++;

    const delay = Math.min(
      this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      30000 // Max 30 seconds
    );

    this.reconnectTimer = window.setTimeout(() => {
      this.connect().catch(console.error);
    }, delay);
  }

  /**
   * Clear reconnect timer
   */
  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /**
   * Reset reconnect attempts counter
   */
  private resetReconnectAttempts(): void {
    this.reconnectAttempts = 0;
  }

  /**
   * Flush queued messages
   */
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.state === ConnectionState.CONNECTED) {
      const frame = this.messageQueue.shift();
      if (frame) {
        this.sendFrame(frame).catch(console.error);
      }
    }
  }
}

/**
 * Production WebSocket client manager for parallel sequence streaming
 */
export class ParallelWebSocketClient {
  private connections = new Map<string, WebSocketConnection>();
  private messageBuffer: MessageBuffer;
  private subscriptions = new Map<string, StreamSubscription>();
  private sequenceRouting = new Map<string, number>(); // sequence_id -> connection_index
  private isStarted = false;
  private metricsTimer: number | null = null;

  constructor(
    private config: WebSocketClientConfig,
    private eventHandlers: WebSocketEventHandlers
  ) {
    this.messageBuffer = new MessageBuffer({
      max_size: config.bufferSize,
      overflow_strategy: 'drop_oldest',
      enable_compression: true,
      enable_deduplication: true,
    });
  }

  /**
   * Initialize client and establish connections
   */
  async start(): Promise<void> {
    if (this.isStarted) {
      return;
    }

    try {
      // Create 3 connections for parallel sequences
      for (let i = 0; i < 3; i++) {
        const connectionId = `connection_${i}`;
        const connection = new WebSocketConnection(
          this.config.apiUrl,
          this.config,
          {
            ...this.eventHandlers,
            onMessage: (message) => this.handleRoutedMessage(message, i),
          }
        );

        await connection.connect();
        this.connections.set(connectionId, connection);
      }

      this.isStarted = true;
      
      if (this.config.enableMetrics) {
        this.startMetricsCollection();
      }

      console.log('ParallelWebSocketClient started with 3 connections');
    } catch (error) {
      console.error('Failed to start ParallelWebSocketClient:', error);
      await this.stop();
      throw error;
    }
  }

  /**
   * Stop client and close all connections
   */
  async stop(): Promise<void> {
    this.isStarted = false;
    this.stopMetricsCollection();

    // Close all connections
    const closePromises = Array.from(this.connections.values()).map(
      connection => connection.disconnect()
    );

    await Promise.all(closePromises);
    this.connections.clear();
    this.subscriptions.clear();
    this.sequenceRouting.clear();
    this.messageBuffer.clear();

    console.log('ParallelWebSocketClient stopped');
  }

  /**
   * Create subscription for sequence strategy
   */
  async createSubscription(
    sequenceStrategy: SequenceStrategy,
    connectionIndex: number
  ): Promise<string> {
    const subscription: StreamSubscription = {
      subscription_id: `sub_${sequenceStrategy}_${Date.now()}`,
      client_id: `client_${Date.now()}`,
      sequence_strategies: new Set([sequenceStrategy]),
      message_types: new Set(['progress', 'result', 'error', 'completion', 'agent_transition']),
      delivery_guarantee: DeliveryGuarantee.AT_LEAST_ONCE,
      include_progress: true,
      include_errors: true,
      include_results: true,
      buffer_size: this.config.bufferSize,
      created_at: Date.now(),
      last_activity: Date.now(),
    };

    this.subscriptions.set(subscription.subscription_id, subscription);

    // Send subscription to specific connection
    const connection = this.getConnection(connectionIndex);
    if (connection) {
      const subscriptionMessage: SubscriptionMessage = {
        subscription_id: subscription.subscription_id,
        sequence_strategies: Array.from(subscription.sequence_strategies),
        message_types: Array.from(subscription.message_types),
        delivery_guarantee: subscription.delivery_guarantee,
        client_id: subscription.client_id,
      };

      await connection.subscribe(subscriptionMessage);
    }

    return subscription.subscription_id;
  }

  /**
   * Start parallel sequence execution
   */
  async startParallelSequences(query: string): Promise<string[]> {
    if (!this.isStarted) {
      throw new Error('Client not started');
    }

    const sequenceIds: string[] = [];
    const strategies = [
      SequenceStrategy.THEORY_FIRST,
      SequenceStrategy.MARKET_FIRST,
      SequenceStrategy.FUTURE_BACK,
    ];

    // Create subscriptions and start sequences
    for (let i = 0; i < 3; i++) {
      const strategy = strategies[i];
      await this.createSubscription(strategy, i);
      
      // Send research query to initiate sequence
      const startMessage: StreamMessage = {
        message_id: `start_${Date.now()}_${i}`,
        sequence_id: `seq_${strategy}_${Date.now()}`,
        sequence_strategy: strategy,
        message_type: 'progress',
        timestamp: Date.now(),
        content: {
          type: 'start_sequence',
          query,
          strategy,
          agent_order: this.getAgentOrder(strategy),
        },
        requires_ack: true,
      };

      const connection = this.getConnection(i);
      if (connection) {
        await connection.sendMessage(startMessage);
        sequenceIds.push(startMessage.sequence_id);
        this.sequenceRouting.set(startMessage.sequence_id, i);
      }
    }

    return sequenceIds;
  }

  /**
   * Get messages for specific sequence
   */
  getSequenceMessages(sequenceId: string): RoutedMessage[] {
    return this.messageBuffer.getSequenceMessages(sequenceId);
  }

  /**
   * Get all messages from buffer
   */
  getAllMessages(): RoutedMessage[] {
    return this.messageBuffer.getAllMessages();
  }

  /**
   * Get client health status
   */
  async getHealthStatus(): Promise<HealthCheckResult[]> {
    const healthChecks = Array.from(this.connections.values()).map(
      connection => connection.healthCheck()
    );

    return Promise.all(healthChecks);
  }

  /**
   * Get aggregated metrics
   */
  getAggregatedMetrics(): ConnectionMetrics {
    const allMetrics = Array.from(this.connections.values()).map(
      connection => connection.getMetrics()
    );

    return {
      messages_sent: allMetrics.reduce((sum, m) => sum + m.messages_sent, 0),
      messages_received: allMetrics.reduce((sum, m) => sum + m.messages_received, 0),
      messages_failed: allMetrics.reduce((sum, m) => sum + m.messages_failed, 0),
      reconnection_count: allMetrics.reduce((sum, m) => sum + m.reconnection_count, 0),
      total_connection_time: allMetrics.reduce((sum, m) => sum + m.total_connection_time, 0),
      last_activity: Math.max(...allMetrics.map(m => m.last_activity)),
      latency_samples: allMetrics.flatMap(m => m.latency_samples),
      average_latency: allMetrics.reduce((sum, m) => sum + m.average_latency, 0) / allMetrics.length,
      success_rate: allMetrics.reduce((sum, m) => sum + m.success_rate, 0) / allMetrics.length,
    };
  }

  /**
   * Handle routed message from connection
   */
  private handleRoutedMessage(message: StreamMessage, connectionIndex: number): void {
    const routedMessage: RoutedMessage = {
      ...message,
      sequence_index: connectionIndex,
      routing_timestamp: Date.now(),
    };

    this.messageBuffer.addMessage(routedMessage);
    this.eventHandlers.onMessage?.(routedMessage);
  }

  /**
   * Get connection by index
   */
  private getConnection(index: number): WebSocketConnection | undefined {
    return this.connections.get(`connection_${index}`);
  }

  /**
   * Get agent order for strategy
   */
  private getAgentOrder(strategy: SequenceStrategy): AgentType[] {
    switch (strategy) {
      case SequenceStrategy.THEORY_FIRST:
        return [AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS];
      case SequenceStrategy.MARKET_FIRST:
        return [AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS];
      case SequenceStrategy.FUTURE_BACK:
        return [AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY];
      default:
        return [AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS];
    }
  }

  /**
   * Start metrics collection
   */
  private startMetricsCollection(): void {
    this.metricsTimer = window.setInterval(() => {
      const metrics = this.getAggregatedMetrics();
      const bufferStats = this.messageBuffer.getStats();
      
      this.eventHandlers.onMetricsUpdate?.({
        messages_per_second: this.calculateMessageRate(metrics),
        average_latency: metrics.average_latency,
        connection_health: this.calculateConnectionHealth(),
        buffer_utilization: bufferStats.utilization,
        error_rate: this.calculateErrorRate(metrics),
        throughput_efficiency: this.calculateThroughputEfficiency(metrics),
        last_updated: Date.now(),
      });
    }, 1000);
  }

  /**
   * Stop metrics collection
   */
  private stopMetricsCollection(): void {
    if (this.metricsTimer) {
      clearInterval(this.metricsTimer);
      this.metricsTimer = null;
    }
  }

  /**
   * Calculate message rate
   */
  private calculateMessageRate(metrics: ConnectionMetrics): number {
    const timeWindow = 60; // 60 seconds
    const recentSamples = metrics.latency_samples.slice(-timeWindow);
    return recentSamples.length / timeWindow;
  }

  /**
   * Calculate connection health percentage
   */
  private calculateConnectionHealth(): number {
    const totalConnections = this.connections.size;
    const healthyConnections = Array.from(this.connections.values()).filter(
      connection => connection.getState() === ConnectionState.CONNECTED
    ).length;

    return totalConnections > 0 ? (healthyConnections / totalConnections) * 100 : 0;
  }

  /**
   * Calculate error rate percentage
   */
  private calculateErrorRate(metrics: ConnectionMetrics): number {
    const total = metrics.messages_sent + metrics.messages_failed;
    return total > 0 ? (metrics.messages_failed / total) * 100 : 0;
  }

  /**
   * Calculate throughput efficiency
   */
  private calculateThroughputEfficiency(metrics: ConnectionMetrics): number {
    return 100 - this.calculateErrorRate(metrics);
  }
}

export default ParallelWebSocketClient;