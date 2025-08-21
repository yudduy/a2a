/**
 * Production-Optimized WebSocket Client for Parallel Research Sequences
 * 
 * Features:
 * - Connection pooling with load balancing
 * - Memory-efficient message buffering with circular buffers
 * - Advanced reconnection strategies with jitter
 * - Message compression and batching
 * - Circuit breaker pattern for error handling
 * - Performance monitoring and metrics collection
 * - Resource cleanup and leak prevention
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
  RealTimeMetrics,
} from '@/types/parallel';

/**
 * Circular buffer for memory-efficient message storage
 */
class CircularMessageBuffer {
  private buffer: RoutedMessage[];
  private head = 0;
  private tail = 0;
  private size = 0;
  private readonly capacity: number;
  private readonly messageIndex = new Map<string, number>();
  private droppedMessages = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }

  add(message: RoutedMessage): boolean {
    // Check for duplicates
    if (this.messageIndex.has(message.message_id)) {
      return false;
    }

    // Remove old message if buffer is full
    if (this.size === this.capacity) {
      const oldMessage = this.buffer[this.tail];
      if (oldMessage) {
        this.messageIndex.delete(oldMessage.message_id);
      }
      this.tail = (this.tail + 1) % this.capacity;
      this.droppedMessages++;
    } else {
      this.size++;
    }

    // Add new message
    this.buffer[this.head] = message;
    this.messageIndex.set(message.message_id, this.head);
    this.head = (this.head + 1) % this.capacity;

    return true;
  }

  getBySequence(sequenceId: string): RoutedMessage[] {
    const result: RoutedMessage[] = [];
    for (let i = 0; i < this.size; i++) {
      const index = (this.tail + i) % this.capacity;
      const message = this.buffer[index];
      if (message?.sequence_id === sequenceId) {
        result.push(message);
      }
    }
    return result;
  }

  getRecent(count: number): RoutedMessage[] {
    const result: RoutedMessage[] = [];
    const actualCount = Math.min(count, this.size);
    
    for (let i = 0; i < actualCount; i++) {
      const index = (this.head - 1 - i + this.capacity) % this.capacity;
      const message = this.buffer[index];
      if (message) {
        result.unshift(message);
      }
    }
    
    return result;
  }

  clear(): void {
    this.head = 0;
    this.tail = 0;
    this.size = 0;
    this.messageIndex.clear();
    this.droppedMessages = 0;
  }

  getStats() {
    return {
      size: this.size,
      capacity: this.capacity,
      utilization: (this.size / this.capacity) * 100,
      droppedMessages: this.droppedMessages,
    };
  }
}

/**
 * Circuit breaker for connection reliability
 */
class CircuitBreaker {
  private failures = 0;
  private lastFailure = 0;
  private state: 'closed' | 'open' | 'half-open' = 'closed';
  
  constructor(
    private failureThreshold = 5,
    private resetTimeout = 30000
  ) {}

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() - this.lastFailure > this.resetTimeout) {
        this.state = 'half-open';
      } else {
        throw new Error('Circuit breaker is open');
      }
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failures = 0;
    this.state = 'closed';
  }

  private onFailure(): void {
    this.failures++;
    this.lastFailure = Date.now();
    
    if (this.failures >= this.failureThreshold) {
      this.state = 'open';
    }
  }

  getState() {
    return this.state;
  }
}

/**
 * Message compression utility
 */
class MessageCompressor {
  private static async compress(data: string): Promise<ArrayBuffer> {
    if ('CompressionStream' in window) {
      const stream = new CompressionStream('gzip');
      const writer = stream.writable.getWriter();
      const reader = stream.readable.getReader();
      
      writer.write(new TextEncoder().encode(data));
      writer.close();
      
      const chunks: Uint8Array[] = [];
      let result = await reader.read();
      
      while (!result.done) {
        chunks.push(result.value);
        result = await reader.read();
      }
      
      const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
      const compressed = new Uint8Array(totalLength);
      let offset = 0;
      
      for (const chunk of chunks) {
        compressed.set(chunk, offset);
        offset += chunk.length;
      }
      
      return compressed.buffer;
    }
    
    // Fallback to no compression
    return new TextEncoder().encode(data).buffer;
  }

  private static async decompress(data: ArrayBuffer): Promise<string> {
    if ('DecompressionStream' in window) {
      const stream = new DecompressionStream('gzip');
      const writer = stream.writable.getWriter();
      const reader = stream.readable.getReader();
      
      writer.write(data);
      writer.close();
      
      const chunks: Uint8Array[] = [];
      let result = await reader.read();
      
      while (!result.done) {
        chunks.push(result.value);
        result = await reader.read();
      }
      
      const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
      const decompressed = new Uint8Array(totalLength);
      let offset = 0;
      
      for (const chunk of chunks) {
        decompressed.set(chunk, offset);
        offset += chunk.length;
      }
      
      return new TextDecoder().decode(decompressed);
    }
    
    // Fallback to no decompression
    return new TextDecoder().decode(data);
  }

  static async compressMessage(message: WebSocketFrame): Promise<ArrayBuffer | string> {
    const serialized = JSON.stringify(message);
    
    // Only compress if message is large enough
    if (serialized.length > 1024) {
      return this.compress(serialized);
    }
    
    return serialized;
  }

  static async decompressMessage(data: ArrayBuffer | string): Promise<WebSocketFrame> {
    let messageStr: string;
    
    if (data instanceof ArrayBuffer) {
      messageStr = await this.decompress(data);
    } else {
      messageStr = data;
    }
    
    return JSON.parse(messageStr);
  }
}

/**
 * Optimized WebSocket connection with advanced features
 */
class OptimizedWebSocketConnection {
  private websocket: WebSocket | null = null;
  private state: ConnectionState = ConnectionState.DISCONNECTED;
  private reconnectAttempts = 0;
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private metrics: ConnectionMetrics;
  private circuitBreaker: CircuitBreaker;
  private messageQueue: WebSocketFrame[] = [];
  private pendingAcks = new Map<string, { timestamp: number; resolve: Function; reject: Function }>();
  private performanceObserver: PerformanceObserver | null = null;
  
  // Connection pooling
  private connectionId: string;
  private isActive = true;

  constructor(
    private url: string,
    private config: WebSocketClientConfig,
    private eventHandlers: WebSocketEventHandlers,
    connectionId: string
  ) {
    this.connectionId = connectionId;
    this.circuitBreaker = new CircuitBreaker(5, 30000);
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

    this.setupPerformanceObserver();
  }

  private setupPerformanceObserver(): void {
    if ('PerformanceObserver' in window) {
      this.performanceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.name.includes('websocket')) {
            console.log(`WebSocket performance: ${entry.name} took ${entry.duration}ms`);
          }
        });
      });
      
      try {
        this.performanceObserver.observe({ entryTypes: ['measure'] });
      } catch (error) {
        console.warn('Performance observer not supported:', error);
      }
    }
  }

  async connect(): Promise<void> {
    if (!this.isActive) {
      throw new Error('Connection is deactivated');
    }

    return this.circuitBreaker.execute(async () => {
      if (this.state === ConnectionState.CONNECTING || this.state === ConnectionState.CONNECTED) {
        return;
      }

      this.setState(ConnectionState.CONNECTING);
      performance.mark(`websocket-connect-start-${this.connectionId}`);

      try {
        // Add jitter to prevent thundering herd
        const jitter = Math.random() * 1000;
        await new Promise(resolve => setTimeout(resolve, jitter));

        this.websocket = new WebSocket(this.url);
        this.setupEventListeners();
        
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
            performance.mark(`websocket-connect-end-${this.connectionId}`);
            performance.measure(
              `websocket-connect-${this.connectionId}`,
              `websocket-connect-start-${this.connectionId}`,
              `websocket-connect-end-${this.connectionId}`
            );
            resolve();
          };

          this.websocket.onerror = (event) => {
            clearTimeout(timeout);
            reject(new Error(`WebSocket connection failed: ${event.type}`));
          };
        });

        this.setState(ConnectionState.CONNECTED);
        this.resetReconnectAttempts();
        this.startHeartbeat();
        await this.flushMessageQueue();

      } catch (error) {
        this.setState(ConnectionState.FAILED);
        this.scheduleReconnect();
        throw error;
      }
    });
  }

  async disconnect(): Promise<void> {
    this.isActive = false;
    this.setState(ConnectionState.DISCONNECTED);
    this.stopHeartbeat();
    this.clearReconnectTimer();
    this.clearPendingAcks();

    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }

    if (this.websocket) {
      this.websocket.close(1000, 'Client disconnect');
      this.websocket = null;
    }
  }

  async sendMessage(message: StreamMessage): Promise<boolean> {
    const frame: WebSocketFrame = {
      type: 'message',
      payload: message,
      timestamp: Date.now(),
      frame_id: `frame_${this.connectionId}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };

    return this.sendFrame(frame);
  }

  async sendMessageWithAck(message: StreamMessage): Promise<void> {
    const frame: WebSocketFrame = {
      type: 'message',
      payload: { ...message, requires_ack: true },
      timestamp: Date.now(),
      frame_id: `frame_${this.connectionId}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };

    return new Promise<void>((resolve, reject) => {
      this.pendingAcks.set(frame.frame_id, {
        timestamp: Date.now(),
        resolve,
        reject,
      });

      // Set timeout for acknowledgment
      setTimeout(() => {
        const pending = this.pendingAcks.get(frame.frame_id);
        if (pending) {
          this.pendingAcks.delete(frame.frame_id);
          pending.reject(new Error('Acknowledgment timeout'));
        }
      }, this.config.messageTimeout);

      this.sendFrame(frame).then((sent) => {
        if (!sent) {
          this.pendingAcks.delete(frame.frame_id);
          reject(new Error('Failed to send message'));
        }
      });
    });
  }

  private async sendFrame(frame: WebSocketFrame): Promise<boolean> {
    if (this.state !== ConnectionState.CONNECTED || !this.websocket) {
      this.messageQueue.push(frame);
      return false;
    }

    try {
      const compressed = await MessageCompressor.compressMessage(frame);
      
      if (compressed instanceof ArrayBuffer) {
        this.websocket.send(compressed);
      } else {
        this.websocket.send(compressed);
      }
      
      this.metrics.messages_sent++;
      this.metrics.last_activity = Date.now();
      
      return true;
    } catch (error) {
      this.metrics.messages_failed++;
      console.error('Failed to send WebSocket frame:', error);
      return false;
    }
  }

  private setupEventListeners(): void {
    if (!this.websocket) return;

    this.websocket.onmessage = async (event) => {
      try {
        const frame = await MessageCompressor.decompressMessage(
          event.data instanceof ArrayBuffer ? event.data : event.data
        );
        
        this.handleIncomingFrame(frame);
      } catch (error) {
        console.error('Failed to process WebSocket message:', error);
        this.metrics.messages_failed++;
      }
    };

    this.websocket.onclose = (event) => {
      console.log(`WebSocket closed: ${event.code} - ${event.reason}`);
      this.setState(ConnectionState.DISCONNECTED);
      
      if (this.isActive && event.code !== 1000) {
        this.scheduleReconnect();
      }
    };

    this.websocket.onerror = (event) => {
      console.error('WebSocket error:', event);
      this.setState(ConnectionState.FAILED);
      this.eventHandlers.onError?.({
        error_id: `ws_error_${this.connectionId}_${Date.now()}`,
        error_type: 'connection',
        message: 'WebSocket connection error',
        recoverable: true,
        timestamp: Date.now(),
      });
    };
  }

  private handleIncomingFrame(frame: WebSocketFrame): void {
    this.metrics.messages_received++;
    this.metrics.last_activity = Date.now();

    switch (frame.type) {
      case 'message':
        if (frame.payload) {
          this.eventHandlers.onMessage?.(frame.payload as RoutedMessage);
        }
        break;

      case 'ack':
        this.handleAcknowledgment(frame);
        break;

      case 'pong':
        this.updateLatencyMetrics(Date.now() - frame.timestamp);
        break;

      case 'error':
        this.eventHandlers.onError?.(frame.payload as ErrorMessage);
        break;

      default:
        console.warn('Unknown frame type:', frame.type);
    }
  }

  private handleAcknowledgment(frame: WebSocketFrame): void {
    const pending = this.pendingAcks.get(frame.frame_id);
    if (pending) {
      const latency = Date.now() - pending.timestamp;
      this.updateLatencyMetrics(latency);
      this.pendingAcks.delete(frame.frame_id);
      pending.resolve();
    }
  }

  private updateLatencyMetrics(latency: number): void {
    this.metrics.latency_samples.push(latency);
    
    // Keep only recent samples (sliding window)
    if (this.metrics.latency_samples.length > 100) {
      this.metrics.latency_samples = this.metrics.latency_samples.slice(-50);
    }
    
    this.metrics.average_latency = 
      this.metrics.latency_samples.reduce((a, b) => a + b, 0) / 
      this.metrics.latency_samples.length;
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = window.setInterval(() => {
      if (this.state === ConnectionState.CONNECTED) {
        this.sendPing();
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private async sendPing(): Promise<void> {
    const frame: WebSocketFrame = {
      type: 'ping',
      timestamp: Date.now(),
      frame_id: `ping_${this.connectionId}_${Date.now()}`,
    };
    
    await this.sendFrame(frame);
  }

  private setState(newState: ConnectionState): void {
    if (this.state !== newState) {
      this.state = newState;
      this.eventHandlers.onConnectionStateChange?.(newState);
    }
  }

  private scheduleReconnect(): void {
    if (!this.isActive || this.reconnectAttempts >= this.config.reconnectAttempts) {
      this.setState(ConnectionState.FAILED);
      return;
    }

    this.setState(ConnectionState.RECONNECTING);
    this.reconnectAttempts++;
    this.metrics.reconnection_count++;

    // Exponential backoff with jitter
    const baseDelay = this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    const jitter = Math.random() * 1000;
    const delay = Math.min(baseDelay + jitter, 30000);

    this.reconnectTimer = window.setTimeout(() => {
      this.connect().catch(console.error);
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private resetReconnectAttempts(): void {
    this.reconnectAttempts = 0;
  }

  private clearPendingAcks(): void {
    this.pendingAcks.forEach((pending) => {
      pending.reject(new Error('Connection closed'));
    });
    this.pendingAcks.clear();
  }

  private async flushMessageQueue(): Promise<void> {
    const batch = this.messageQueue.splice(0, 10); // Process in batches
    
    for (const frame of batch) {
      await this.sendFrame(frame);
    }
    
    if (this.messageQueue.length > 0) {
      // Schedule next batch
      setTimeout(() => this.flushMessageQueue(), 100);
    }
  }

  getState(): ConnectionState {
    return this.state;
  }

  getMetrics(): ConnectionMetrics {
    return { ...this.metrics };
  }

  getConnectionId(): string {
    return this.connectionId;
  }

  isHealthy(): boolean {
    return this.state === ConnectionState.CONNECTED && 
           this.circuitBreaker.getState() === 'closed' &&
           this.metrics.average_latency < 1000;
  }
}

/**
 * Production-optimized WebSocket client manager
 */
export class OptimizedParallelWebSocketClient {
  private connectionPool: Map<string, OptimizedWebSocketConnection> = new Map();
  private messageBuffer: CircularMessageBuffer;
  private subscriptions = new Map<string, StreamSubscription>();
  private sequenceRouting = new Map<string, string>(); // sequence_id -> connection_id
  private isStarted = false;
  private metricsTimer: number | null = null;
  private loadBalancerIndex = 0;
  private readonly maxConnectionsPerPool = 3;

  constructor(
    private config: WebSocketClientConfig,
    private eventHandlers: WebSocketEventHandlers
  ) {
    this.messageBuffer = new CircularMessageBuffer(config.bufferSize);
  }

  async start(): Promise<void> {
    if (this.isStarted) {
      return;
    }

    try {
      // Create connection pool
      for (let i = 0; i < this.maxConnectionsPerPool; i++) {
        const connectionId = `connection_${i}`;
        const connection = new OptimizedWebSocketConnection(
          this.config.apiUrl,
          this.config,
          {
            ...this.eventHandlers,
            onMessage: (message) => this.handleRoutedMessage(message, connectionId),
          },
          connectionId
        );

        await connection.connect();
        this.connectionPool.set(connectionId, connection);
      }

      this.isStarted = true;
      
      if (this.config.enableMetrics) {
        this.startMetricsCollection();
      }

      console.log(`OptimizedParallelWebSocketClient started with ${this.maxConnectionsPerPool} connections`);
    } catch (error) {
      console.error('Failed to start OptimizedParallelWebSocketClient:', error);
      await this.stop();
      throw error;
    }
  }

  async stop(): Promise<void> {
    this.isStarted = false;
    this.stopMetricsCollection();

    // Close all connections
    const closePromises = Array.from(this.connectionPool.values()).map(
      connection => connection.disconnect()
    );

    await Promise.all(closePromises);
    this.connectionPool.clear();
    this.subscriptions.clear();
    this.sequenceRouting.clear();
    this.messageBuffer.clear();

    console.log('OptimizedParallelWebSocketClient stopped');
  }

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

    // Use load balancing to distribute sequences across connections
    for (let i = 0; i < strategies.length; i++) {
      const strategy = strategies[i];
      const connectionId = this.getNextConnection();
      const connection = this.connectionPool.get(connectionId);
      
      if (!connection) {
        throw new Error(`Connection ${connectionId} not found`);
      }

      const sequenceId = `seq_${strategy}_${Date.now()}_${i}`;
      this.sequenceRouting.set(sequenceId, connectionId);

      const startMessage: StreamMessage = {
        message_id: `start_${Date.now()}_${i}`,
        sequence_id: sequenceId,
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

      try {
        await connection.sendMessageWithAck(startMessage);
        sequenceIds.push(sequenceId);
      } catch (error) {
        console.error(`Failed to start sequence ${strategy}:`, error);
        // Try with a different connection
        const fallbackConnectionId = this.getHealthyConnection();
        if (fallbackConnectionId && fallbackConnectionId !== connectionId) {
          const fallbackConnection = this.connectionPool.get(fallbackConnectionId);
          if (fallbackConnection) {
            this.sequenceRouting.set(sequenceId, fallbackConnectionId);
            await fallbackConnection.sendMessageWithAck(startMessage);
            sequenceIds.push(sequenceId);
          }
        }
      }
    }

    return sequenceIds;
  }

  getSequenceMessages(sequenceId: string): RoutedMessage[] {
    return this.messageBuffer.getBySequence(sequenceId);
  }

  getRecentMessages(count: number = 50): RoutedMessage[] {
    return this.messageBuffer.getRecent(count);
  }

  async getHealthStatus(): Promise<HealthCheckResult[]> {
    const healthChecks = Array.from(this.connectionPool.values()).map(
      async (connection) => {
        const isHealthy = connection.isHealthy();
        const metrics = connection.getMetrics();
        
        return {
          connection_healthy: isHealthy,
          latency: metrics.average_latency,
          last_message_time: metrics.last_activity,
          buffer_utilization: this.messageBuffer.getStats().utilization,
          error_count: metrics.messages_failed,
          timestamp: Date.now(),
        };
      }
    );

    return Promise.all(healthChecks);
  }

  getAggregatedMetrics(): ConnectionMetrics & { bufferStats: any } {
    const allMetrics = Array.from(this.connectionPool.values()).map(
      connection => connection.getMetrics()
    );

    const aggregated = {
      messages_sent: allMetrics.reduce((sum, m) => sum + m.messages_sent, 0),
      messages_received: allMetrics.reduce((sum, m) => sum + m.messages_received, 0),
      messages_failed: allMetrics.reduce((sum, m) => sum + m.messages_failed, 0),
      reconnection_count: allMetrics.reduce((sum, m) => sum + m.reconnection_count, 0),
      total_connection_time: allMetrics.reduce((sum, m) => sum + m.total_connection_time, 0),
      last_activity: Math.max(...allMetrics.map(m => m.last_activity)),
      latency_samples: allMetrics.flatMap(m => m.latency_samples),
      average_latency: allMetrics.reduce((sum, m) => sum + m.average_latency, 0) / allMetrics.length,
      success_rate: allMetrics.reduce((sum, m) => sum + m.success_rate, 0) / allMetrics.length,
      bufferStats: this.messageBuffer.getStats(),
    };

    return aggregated;
  }

  private handleRoutedMessage(message: StreamMessage, connectionId: string): void {
    const routedMessage: RoutedMessage = {
      ...message,
      sequence_index: this.getConnectionIndex(connectionId),
      routing_timestamp: Date.now(),
    };

    this.messageBuffer.add(routedMessage);
    this.eventHandlers.onMessage?.(routedMessage);
  }

  private getNextConnection(): string {
    const connectionIds = Array.from(this.connectionPool.keys());
    const connectionId = connectionIds[this.loadBalancerIndex % connectionIds.length];
    this.loadBalancerIndex++;
    return connectionId;
  }

  private getHealthyConnection(): string | null {
    for (const [connectionId, connection] of this.connectionPool) {
      if (connection.isHealthy()) {
        return connectionId;
      }
    }
    return null;
  }

  private getConnectionIndex(connectionId: string): number {
    return parseInt(connectionId.split('_')[1]) || 0;
  }

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

  private startMetricsCollection(): void {
    this.metricsTimer = window.setInterval(() => {
      const metrics = this.getAggregatedMetrics();
      
      const realTimeMetrics: RealTimeMetrics = {
        messages_per_second: this.calculateMessageRate(metrics),
        average_latency: metrics.average_latency,
        connection_health: this.calculateConnectionHealth(),
        buffer_utilization: metrics.bufferStats.utilization,
        error_rate: this.calculateErrorRate(metrics),
        throughput_efficiency: this.calculateThroughputEfficiency(metrics),
        last_updated: Date.now(),
      };

      this.eventHandlers.onMetricsUpdate?.(realTimeMetrics);
    }, 1000);
  }

  private stopMetricsCollection(): void {
    if (this.metricsTimer) {
      clearInterval(this.metricsTimer);
      this.metricsTimer = null;
    }
  }

  private calculateMessageRate(metrics: ConnectionMetrics): number {
    const timeWindow = 60;
    const recentSamples = metrics.latency_samples.slice(-timeWindow);
    return recentSamples.length / timeWindow;
  }

  private calculateConnectionHealth(): number {
    const totalConnections = this.connectionPool.size;
    const healthyConnections = Array.from(this.connectionPool.values()).filter(
      connection => connection.isHealthy()
    ).length;

    return totalConnections > 0 ? (healthyConnections / totalConnections) * 100 : 0;
  }

  private calculateErrorRate(metrics: ConnectionMetrics): number {
    const total = metrics.messages_sent + metrics.messages_failed;
    return total > 0 ? (metrics.messages_failed / total) * 100 : 0;
  }

  private calculateThroughputEfficiency(metrics: ConnectionMetrics): number {
    return Math.max(0, 100 - this.calculateErrorRate(metrics));
  }
}

export default OptimizedParallelWebSocketClient;