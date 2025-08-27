import React from 'react';

/**
 * Production Memory Management Utilities
 * 
 * Features:
 * - Memory leak detection and prevention
 * - Component cleanup utilities
 * - Resource pooling and recycling
 * - Memory usage monitoring and optimization
 * - Garbage collection hints and optimization
 */

interface MemoryConfig {
  maxMessageBuffer: number;
  maxErrorHistory: number;
  gcHintInterval: number;
  memoryThreshold: number;
  enableMemoryProfiling: boolean;
  autoCleanupInterval: number;
}

interface MemoryMetrics {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
  utilizationPercentage: number;
  timestamp: number;
}

interface ResourceHandle {
  id: string;
  type: 'websocket' | 'interval' | 'timeout' | 'observer' | 'listener';
  resource: any;
  createdAt: number;
  lastUsed: number;
}

class MemoryManager {
  private static instance: MemoryManager;
  private config: MemoryConfig;
  private resourceRegistry = new Map<string, ResourceHandle>();
  private memoryHistory: MemoryMetrics[] = [];
  private gcTimer: number | null = null;
  private cleanupTimer: number | null = null;
  private memoryObserver: PerformanceObserver | null = null;
  private weakRefs = new Set<any>(); // WeakRef not available in all environments

  private constructor(config: Partial<MemoryConfig> = {}) {
    this.config = {
      maxMessageBuffer: 1000,
      maxErrorHistory: 50,
      gcHintInterval: 30000, // 30 seconds
      memoryThreshold: 0.8, // 80% of heap limit
      enableMemoryProfiling: process.env.NODE_ENV === 'development',
      autoCleanupInterval: 60000, // 1 minute
      ...config,
    };

    this.initialize();
  }

  static getInstance(config?: Partial<MemoryConfig>): MemoryManager {
    if (!MemoryManager.instance) {
      MemoryManager.instance = new MemoryManager(config);
    }
    return MemoryManager.instance;
  }

  private initialize(): void {
    this.startMemoryMonitoring();
    this.startPeriodicCleanup();
    this.setupMemoryObserver();
    
    // Listen for page visibility changes to trigger cleanup
    document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));
    
    // Listen for beforeunload to perform final cleanup
    window.addEventListener('beforeunload', this.performFinalCleanup.bind(this));
  }

  private setupMemoryObserver(): void {
    if ('PerformanceObserver' in window && this.config.enableMemoryProfiling) {
      try {
        this.memoryObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          entries.forEach((entry) => {
            if (entry.entryType === 'measure' && entry.name.includes('memory')) {
              console.log(`Memory operation: ${entry.name} took ${entry.duration}ms`);
            }
          });
        });
        
        this.memoryObserver.observe({ entryTypes: ['measure'] });
      } catch (error) {
        console.warn('Memory observer setup failed:', error);
      }
    }
  }

  /**
   * Register a resource for automatic cleanup
   */
  registerResource(
    id: string,
    type: ResourceHandle['type'],
    resource: any,
    onCleanup?: () => void
  ): string {
    const handle: ResourceHandle = {
      id,
      type,
      resource,
      createdAt: Date.now(),
      lastUsed: Date.now(),
    };

    // Add cleanup function if provided
    if (onCleanup) {
      (handle as any).cleanup = onCleanup;
    }

    this.resourceRegistry.set(id, handle);
    
    if (this.config.enableMemoryProfiling) {
      console.log(`Registered ${type} resource: ${id}`);
    }

    return id;
  }

  /**
   * Unregister and cleanup a resource
   */
  unregisterResource(id: string): boolean {
    const handle = this.resourceRegistry.get(id);
    if (!handle) {
      return false;
    }

    this.cleanupResource(handle);
    this.resourceRegistry.delete(id);
    
    if (this.config.enableMemoryProfiling) {
      console.log(`Unregistered ${handle.type} resource: ${id}`);
    }

    return true;
  }

  /**
   * Update last used timestamp for a resource
   */
  touchResource(id: string): void {
    const handle = this.resourceRegistry.get(id);
    if (handle) {
      handle.lastUsed = Date.now();
    }
  }

  /**
   * Create a circular buffer with automatic memory management
   */
  createCircularBuffer<T>(capacity: number): CircularBuffer<T> {
    return new CircularBuffer<T>(capacity, this);
  }

  /**
   * Create a weak reference that's automatically tracked
   */
  createWeakRef<T extends object>(target: T): any {
    // WeakRef not available in all environments
    if (typeof window !== 'undefined' && 'WeakRef' in window) {
      const weakRef = new (window as any).WeakRef(target);
      this.weakRefs.add(weakRef);
      return weakRef;
    }
    return null;
  }

  /**
   * Optimize component for memory efficiency
   */
  optimizeComponent<T extends Record<string, any>>(component: T): T {
    // Add memory-aware cleanup methods
    if (!(component as any)._memoryOptimized) {
      const originalComponentWillUnmount = (component as any).componentWillUnmount;
      const originalCleanup = (component as any).cleanup;

      (component as any).componentWillUnmount = function() {
        // Clean up tracked resources
        if ((this as any)._resourceIds) {
          (this as any)._resourceIds.forEach((id: string) => {
            MemoryManager.getInstance().unregisterResource(id);
          });
        }

        // Call original cleanup methods
        if (originalCleanup) {
          originalCleanup.call(this);
        }
        if (originalComponentWillUnmount) {
          originalComponentWillUnmount.call(this);
        }
      };

      (component as any)._memoryOptimized = true;
      (component as any)._resourceIds = new Set<string>();
    }

    return component;
  }

  /**
   * Get current memory metrics
   */
  getMemoryMetrics(): MemoryMetrics | null {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        usedJSHeapSize: memory.usedJSHeapSize,
        totalJSHeapSize: memory.totalJSHeapSize,
        jsHeapSizeLimit: memory.jsHeapSizeLimit,
        utilizationPercentage: (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100,
        timestamp: Date.now(),
      };
    }
    return null;
  }

  /**
   * Check if memory usage is above threshold
   */
  isMemoryPressure(): boolean {
    const metrics = this.getMemoryMetrics();
    if (!metrics) return false;
    
    return metrics.utilizationPercentage > (this.config.memoryThreshold * 100);
  }

  /**
   * Force garbage collection hint
   */
  suggestGarbageCollection(): void {
    if (this.config.enableMemoryProfiling) {
      performance.mark('memory-gc-hint-start');
    }

    // Create and immediately discard objects to trigger GC
    try {
      const temp = new Array(1000).fill(null).map(() => ({ data: new Array(100) }));
      temp.length = 0;
    } catch (error) {
      console.warn('GC hint failed:', error);
    }

    if (this.config.enableMemoryProfiling) {
      performance.mark('memory-gc-hint-end');
      performance.measure('memory-gc-hint', 'memory-gc-hint-start', 'memory-gc-hint-end');
    }
  }

  /**
   * Cleanup stale resources
   */
  cleanupStaleResources(maxAge: number = 300000): number { // 5 minutes default
    const now = Date.now();
    let cleanedCount = 0;

    for (const [id, handle] of this.resourceRegistry) {
      if (now - handle.lastUsed > maxAge) {
        this.cleanupResource(handle);
        this.resourceRegistry.delete(id);
        cleanedCount++;
      }
    }

    if (this.config.enableMemoryProfiling && cleanedCount > 0) {
      console.log(`Cleaned up ${cleanedCount} stale resources`);
    }

    return cleanedCount;
  }

  /**
   * Clean up weak references
   */
  cleanupWeakRefs(): number {
    let cleanedCount = 0;
    
    for (const weakRef of this.weakRefs) {
      if (weakRef.deref() === undefined) {
        this.weakRefs.delete(weakRef);
        cleanedCount++;
      }
    }

    return cleanedCount;
  }

  /**
   * Get memory usage report
   */
  getMemoryReport(): {
    metrics: MemoryMetrics | null;
    resourceCount: number;
    weakRefCount: number;
    memoryPressure: boolean;
    recommendations: string[];
  } {
    const metrics = this.getMemoryMetrics();
    const memoryPressure = this.isMemoryPressure();
    const resourceCount = this.resourceRegistry.size;
    const weakRefCount = this.weakRefs.size;

    const recommendations: string[] = [];
    
    if (memoryPressure) {
      recommendations.push('High memory usage detected - consider reducing message buffer sizes');
    }
    
    if (resourceCount > 100) {
      recommendations.push('Large number of registered resources - check for resource leaks');
    }
    
    if (weakRefCount > 500) {
      recommendations.push('Many weak references - consider cleanup');
    }

    return {
      metrics,
      resourceCount,
      weakRefCount,
      memoryPressure,
      recommendations,
    };
  }

  private cleanupResource(handle: ResourceHandle): void {
    try {
      switch (handle.type) {
        case 'websocket':
          if (handle.resource && typeof handle.resource.close === 'function') {
            handle.resource.close();
          }
          break;
        
        case 'interval':
          if (typeof handle.resource === 'number') {
            clearInterval(handle.resource);
          }
          break;
        
        case 'timeout':
          if (typeof handle.resource === 'number') {
            clearTimeout(handle.resource);
          }
          break;
        
        case 'observer':
          if (handle.resource && typeof handle.resource.disconnect === 'function') {
            handle.resource.disconnect();
          }
          break;
        
        case 'listener':
          // Custom cleanup function should be provided
          break;
      }

      // Call custom cleanup function if provided
      if ((handle as any).cleanup) {
        (handle as any).cleanup();
      }
    } catch (error) {
      console.error(`Failed to cleanup resource ${handle.id}:`, error);
    }
  }

  private startMemoryMonitoring(): void {
    if (this.config.gcHintInterval > 0) {
      this.gcTimer = window.setInterval(() => {
        const metrics = this.getMemoryMetrics();
        if (metrics) {
          this.memoryHistory.push(metrics);
          
          // Keep only recent history
          if (this.memoryHistory.length > 100) {
            this.memoryHistory = this.memoryHistory.slice(-50);
          }

          // Suggest GC if memory pressure is high
          if (this.isMemoryPressure()) {
            this.suggestGarbageCollection();
          }
        }
      }, this.config.gcHintInterval);

      this.registerResource('memory-monitor', 'interval', this.gcTimer);
    }
  }

  private startPeriodicCleanup(): void {
    if (this.config.autoCleanupInterval > 0) {
      this.cleanupTimer = window.setInterval(() => {
        this.cleanupStaleResources();
        this.cleanupWeakRefs();
      }, this.config.autoCleanupInterval);

      this.registerResource('auto-cleanup', 'interval', this.cleanupTimer);
    }
  }

  private handleVisibilityChange(): void {
    if (document.hidden) {
      // Page is hidden, perform cleanup
      this.cleanupStaleResources(60000); // More aggressive cleanup when hidden
      this.suggestGarbageCollection();
    }
  }

  private performFinalCleanup(): void {
    // Cleanup all registered resources
    for (const [id, handle] of this.resourceRegistry) {
      this.cleanupResource(handle);
    }
    
    this.resourceRegistry.clear();
    this.weakRefs.clear();
    
    if (this.memoryObserver) {
      this.memoryObserver.disconnect();
    }
  }

  /**
   * Destroy the memory manager instance
   */
  destroy(): void {
    this.performFinalCleanup();
    
    if (this.gcTimer) {
      clearInterval(this.gcTimer);
    }
    
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
    }

    document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    window.removeEventListener('beforeunload', this.performFinalCleanup);
  }
}

/**
 * Memory-efficient circular buffer
 */
export class CircularBuffer<T> {
  private buffer: (T | undefined)[];
  private head = 0;
  private tail = 0;
  private size = 0;
  private readonly capacity: number;
  private readonly memoryManager: MemoryManager;

  constructor(capacity: number, memoryManager: MemoryManager) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
    this.memoryManager = memoryManager;
  }

  add(item: T): boolean {
    // Remove old item if buffer is full
    if (this.size === this.capacity) {
      this.buffer[this.tail] = undefined; // Help GC
      this.tail = (this.tail + 1) % this.capacity;
    } else {
      this.size++;
    }

    this.buffer[this.head] = item;
    this.head = (this.head + 1) % this.capacity;

    return true;
  }

  getAll(): T[] {
    const result: T[] = [];
    for (let i = 0; i < this.size; i++) {
      const index = (this.tail + i) % this.capacity;
      const item = this.buffer[index];
      if (item !== undefined) {
        result.push(item);
      }
    }
    return result;
  }

  getRecent(count: number): T[] {
    const result: T[] = [];
    const actualCount = Math.min(count, this.size);
    
    for (let i = 0; i < actualCount; i++) {
      const index = (this.head - 1 - i + this.capacity) % this.capacity;
      const item = this.buffer[index];
      if (item !== undefined) {
        result.unshift(item);
      }
    }
    
    return result;
  }

  clear(): void {
    // Help GC by clearing references
    for (let i = 0; i < this.capacity; i++) {
      this.buffer[i] = undefined;
    }
    
    this.head = 0;
    this.tail = 0;
    this.size = 0;
  }

  getStats() {
    return {
      size: this.size,
      capacity: this.capacity,
      utilization: (this.size / this.capacity) * 100,
    };
  }
}

/**
 * React hook for memory management
 */
export function useMemoryManager(config?: Partial<MemoryConfig>) {
  const memoryManager = MemoryManager.getInstance(config);
  
  React.useEffect(() => {
    return () => {
      // Component cleanup
      memoryManager.cleanupStaleResources();
    };
  }, [memoryManager]);

  return {
    memoryManager,
    registerResource: memoryManager.registerResource.bind(memoryManager),
    unregisterResource: memoryManager.unregisterResource.bind(memoryManager),
    createCircularBuffer: memoryManager.createCircularBuffer.bind(memoryManager),
    createWeakRef: memoryManager.createWeakRef.bind(memoryManager),
    getMemoryReport: memoryManager.getMemoryReport.bind(memoryManager),
    isMemoryPressure: memoryManager.isMemoryPressure.bind(memoryManager),
  };
}

// Export the main instance
export const memoryManager = MemoryManager.getInstance();
export default MemoryManager;