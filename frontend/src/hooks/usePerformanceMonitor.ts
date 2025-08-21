/**
 * Performance Monitoring Hook
 * 
 * Provides comprehensive performance monitoring for React components including:
 * - Render performance tracking
 * - Memory usage monitoring
 * - Network performance metrics
 * - User interaction timing
 * - Core Web Vitals measurement
 * - Custom performance marks
 */

import { useEffect, useRef, useState, useCallback } from 'react';

interface PerformanceMetrics {
  // Render performance
  renderCount: number;
  averageRenderTime: number;
  slowRenders: number;
  lastRenderTime: number;
  
  // Memory metrics
  memoryUsage: {
    used: number;
    total: number;
    percentage: number;
  };
  
  // Network metrics
  networkMetrics: {
    rtt: number;
    downlink: number;
    effectiveType: string;
  };
  
  // Core Web Vitals
  webVitals: {
    fcp: number | null; // First Contentful Paint
    lcp: number | null; // Largest Contentful Paint
    fid: number | null; // First Input Delay
    cls: number | null; // Cumulative Layout Shift
    ttfb: number | null; // Time to First Byte
  };
  
  // Custom timings
  customTimings: Record<string, number>;
  
  // Interaction metrics
  interactions: {
    totalInteractions: number;
    averageInteractionTime: number;
    slowInteractions: number;
  };
}

interface PerformanceConfig {
  enableMemoryMonitoring?: boolean;
  enableNetworkMonitoring?: boolean;
  enableWebVitals?: boolean;
  enableCustomTimings?: boolean;
  sampleRate?: number; // 0-1, for production use
  slowRenderThreshold?: number; // ms
  slowInteractionThreshold?: number; // ms
  reportingInterval?: number; // ms
}

const DEFAULT_CONFIG: Required<PerformanceConfig> = {
  enableMemoryMonitoring: true,
  enableNetworkMonitoring: true,
  enableWebVitals: true,
  enableCustomTimings: true,
  sampleRate: 1.0,
  slowRenderThreshold: 16, // 60fps = 16.67ms per frame
  slowInteractionThreshold: 100,
  reportingInterval: 5000,
};

export function usePerformanceMonitor(
  componentName: string,
  config: PerformanceConfig = {}
) {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  
  // Skip monitoring based on sample rate
  const shouldMonitor = Math.random() < finalConfig.sampleRate;
  
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    renderCount: 0,
    averageRenderTime: 0,
    slowRenders: 0,
    lastRenderTime: 0,
    memoryUsage: { used: 0, total: 0, percentage: 0 },
    networkMetrics: { rtt: 0, downlink: 0, effectiveType: 'unknown' },
    webVitals: { fcp: null, lcp: null, fid: null, cls: null, ttfb: null },
    customTimings: {},
    interactions: { totalInteractions: 0, averageInteractionTime: 0, slowInteractions: 0 },
  });

  const renderStartTime = useRef<number>(0);
  const renderTimes = useRef<number[]>([]);
  const interactionTimes = useRef<number[]>([]);
  const customMarks = useRef<Map<string, number>>(new Map());
  const reportingTimer = useRef<NodeJS.Timeout>();

  // Performance observer for Web Vitals
  const webVitalsObserver = useRef<PerformanceObserver>();

  // Track render start
  const markRenderStart = useCallback(() => {
    if (!shouldMonitor) return;
    renderStartTime.current = performance.now();
  }, [shouldMonitor]);

  // Track render end
  const markRenderEnd = useCallback(() => {
    if (!shouldMonitor || renderStartTime.current === 0) return;
    
    const renderTime = performance.now() - renderStartTime.current;
    renderTimes.current.push(renderTime);
    
    // Keep only recent render times (last 100)
    if (renderTimes.current.length > 100) {
      renderTimes.current = renderTimes.current.slice(-50);
    }
    
    const slowRenders = renderTimes.current.filter(time => time > finalConfig.slowRenderThreshold).length;
    const averageRenderTime = renderTimes.current.reduce((a, b) => a + b, 0) / renderTimes.current.length;
    
    setMetrics(prev => ({
      ...prev,
      renderCount: prev.renderCount + 1,
      averageRenderTime,
      slowRenders,
      lastRenderTime: renderTime,
    }));
    
    renderStartTime.current = 0;
  }, [shouldMonitor, finalConfig.slowRenderThreshold]);

  // Custom timing utilities
  const startTiming = useCallback((name: string) => {
    if (!shouldMonitor || !finalConfig.enableCustomTimings) return;
    customMarks.current.set(name, performance.now());
  }, [shouldMonitor, finalConfig.enableCustomTimings]);

  const endTiming = useCallback((name: string) => {
    if (!shouldMonitor || !finalConfig.enableCustomTimings) return;
    
    const startTime = customMarks.current.get(name);
    if (startTime) {
      const duration = performance.now() - startTime;
      setMetrics(prev => ({
        ...prev,
        customTimings: {
          ...prev.customTimings,
          [name]: duration,
        },
      }));
      customMarks.current.delete(name);
    }
  }, [shouldMonitor, finalConfig.enableCustomTimings]);

  // Track user interactions
  const trackInteraction = useCallback((interactionName: string, duration: number) => {
    if (!shouldMonitor) return;
    
    interactionTimes.current.push(duration);
    
    // Keep only recent interactions
    if (interactionTimes.current.length > 100) {
      interactionTimes.current = interactionTimes.current.slice(-50);
    }
    
    const slowInteractions = interactionTimes.current.filter(
      time => time > finalConfig.slowInteractionThreshold
    ).length;
    
    const averageInteractionTime = interactionTimes.current.reduce((a, b) => a + b, 0) / 
      interactionTimes.current.length;
    
    setMetrics(prev => ({
      ...prev,
      interactions: {
        totalInteractions: prev.interactions.totalInteractions + 1,
        averageInteractionTime,
        slowInteractions,
      },
    }));
  }, [shouldMonitor, finalConfig.slowInteractionThreshold]);

  // Memory monitoring
  const updateMemoryMetrics = useCallback(() => {
    if (!shouldMonitor || !finalConfig.enableMemoryMonitoring) return;
    
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      const used = memory.usedJSHeapSize / 1024 / 1024; // MB
      const total = memory.totalJSHeapSize / 1024 / 1024; // MB
      const percentage = (used / total) * 100;
      
      setMetrics(prev => ({
        ...prev,
        memoryUsage: { used, total, percentage },
      }));
    }
  }, [shouldMonitor, finalConfig.enableMemoryMonitoring]);

  // Network monitoring
  const updateNetworkMetrics = useCallback(() => {
    if (!shouldMonitor || !finalConfig.enableNetworkMonitoring) return;
    
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      setMetrics(prev => ({
        ...prev,
        networkMetrics: {
          rtt: connection.rtt || 0,
          downlink: connection.downlink || 0,
          effectiveType: connection.effectiveType || 'unknown',
        },
      }));
    }
  }, [shouldMonitor, finalConfig.enableNetworkMonitoring]);

  // Web Vitals monitoring
  const setupWebVitalsMonitoring = useCallback(() => {
    if (!shouldMonitor || !finalConfig.enableWebVitals || !window.PerformanceObserver) return;

    try {
      // LCP (Largest Contentful Paint)
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1] as any;
        setMetrics(prev => ({
          ...prev,
          webVitals: { ...prev.webVitals, lcp: lastEntry.startTime },
        }));
      });
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

      // FCP (First Contentful Paint)
      const fcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          if (entry.name === 'first-contentful-paint') {
            setMetrics(prev => ({
              ...prev,
              webVitals: { ...prev.webVitals, fcp: entry.startTime },
            }));
          }
        });
      });
      fcpObserver.observe({ entryTypes: ['paint'] });

      // FID (First Input Delay)
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          setMetrics(prev => ({
            ...prev,
            webVitals: { ...prev.webVitals, fid: entry.processingStart - entry.startTime },
          }));
        });
      });
      fidObserver.observe({ entryTypes: ['first-input'] });

      // CLS (Cumulative Layout Shift)
      let clsValue = 0;
      const clsObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          if (!entry.hadRecentInput) {
            clsValue += entry.value;
            setMetrics(prev => ({
              ...prev,
              webVitals: { ...prev.webVitals, cls: clsValue },
            }));
          }
        });
      });
      clsObserver.observe({ entryTypes: ['layout-shift'] });

      webVitalsObserver.current = lcpObserver; // Store reference for cleanup
    } catch (error) {
      console.warn('Failed to setup Web Vitals monitoring:', error);
    }
  }, [shouldMonitor, finalConfig.enableWebVitals]);

  // Periodic reporting
  const startPeriodicReporting = useCallback(() => {
    if (!shouldMonitor) return;
    
    reportingTimer.current = setInterval(() => {
      updateMemoryMetrics();
      updateNetworkMetrics();
      
      // Custom reporting logic here
      if (process.env.NODE_ENV === 'development') {
        console.log(`[${componentName}] Performance Metrics:`, metrics);
      }
    }, finalConfig.reportingInterval);
  }, [shouldMonitor, finalConfig.reportingInterval, componentName, metrics, updateMemoryMetrics, updateNetworkMetrics]);

  // Setup effects
  useEffect(() => {
    if (!shouldMonitor) return;
    
    setupWebVitalsMonitoring();
    startPeriodicReporting();
    
    return () => {
      if (webVitalsObserver.current) {
        webVitalsObserver.current.disconnect();
      }
      if (reportingTimer.current) {
        clearInterval(reportingTimer.current);
      }
    };
  }, [shouldMonitor, setupWebVitalsMonitoring, startPeriodicReporting]);

  // Auto-track renders
  useEffect(() => {
    markRenderStart();
    return markRenderEnd;
  });

  // Expose utility functions
  const createInteractionTracker = useCallback((interactionName: string) => {
    const startTime = performance.now();
    return () => {
      const duration = performance.now() - startTime;
      trackInteraction(interactionName, duration);
    };
  }, [trackInteraction]);

  const measureAsync = useCallback(async <T>(
    name: string,
    fn: () => Promise<T>
  ): Promise<T> => {
    startTiming(name);
    try {
      const result = await fn();
      endTiming(name);
      return result;
    } catch (error) {
      endTiming(name);
      throw error;
    }
  }, [startTiming, endTiming]);

  const measureSync = useCallback(<T>(
    name: string,
    fn: () => T
  ): T => {
    startTiming(name);
    try {
      const result = fn();
      endTiming(name);
      return result;
    } catch (error) {
      endTiming(name);
      throw error;
    }
  }, [startTiming, endTiming]);

  return {
    metrics,
    
    // Timing utilities
    startTiming,
    endTiming,
    measureAsync,
    measureSync,
    
    // Interaction tracking
    trackInteraction,
    createInteractionTracker,
    
    // Manual updates
    updateMemoryMetrics,
    updateNetworkMetrics,
    
    // Performance score (0-100)
    performanceScore: Math.max(0, Math.min(100, 
      100 - (metrics.averageRenderTime * 2) - (metrics.memoryUsage.percentage * 0.5)
    )),
    
    // Health indicators
    isHealthy: metrics.averageRenderTime < finalConfig.slowRenderThreshold &&
              metrics.memoryUsage.percentage < 80 &&
              metrics.interactions.averageInteractionTime < finalConfig.slowInteractionThreshold,
  };
}

export default usePerformanceMonitor;