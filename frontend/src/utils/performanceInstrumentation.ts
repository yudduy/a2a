/**
 * Production Performance Instrumentation System
 * 
 * Features:
 * - Real-time performance monitoring
 * - Error tracking and analytics
 * - User interaction metrics
 * - Network performance monitoring
 * - Custom performance marks and measures
 * - Automated performance budgets
 * - Health check endpoints
 */

interface PerformanceConfig {
  apiEndpoint?: string;
  sampleRate: number;
  enableWebVitals: boolean;
  enableUserTiming: boolean;
  enableResourceTiming: boolean;
  enableNavigationTiming: boolean;
  enableErrorTracking: boolean;
  enableUserInteractionTracking: boolean;
  performanceBudgets: PerformanceBudgets;
  reportingInterval: number;
  maxRetries: number;
}

interface PerformanceBudgets {
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  firstInputDelay: number;
  cumulativeLayoutShift: number;
  timeToInteractive: number;
  totalBlockingTime: number;
  memoryUsage: number;
  bundleSize: number;
}

interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: number;
  category: 'webvitals' | 'custom' | 'network' | 'memory' | 'interaction';
  metadata?: Record<string, any>;
}

interface ErrorMetric {
  message: string;
  stack?: string;
  source: string;
  line?: number;
  column?: number;
  timestamp: number;
  userAgent: string;
  url: string;
  userId?: string;
  sessionId: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  tags?: string[];
}

interface UserInteractionMetric {
  type: 'click' | 'scroll' | 'input' | 'navigation' | 'custom';
  element?: string;
  timestamp: number;
  duration?: number;
  metadata?: Record<string, any>;
}

interface NetworkMetric {
  url: string;
  method: string;
  status: number;
  duration: number;
  size: number;
  timestamp: number;
  cached: boolean;
  metadata?: Record<string, any>;
}

class PerformanceInstrumentation {
  private static instance: PerformanceInstrumentation;
  private config: PerformanceConfig;
  private metrics: PerformanceMetric[] = [];
  private errors: ErrorMetric[] = [];
  private interactions: UserInteractionMetric[] = [];
  private networkMetrics: NetworkMetric[] = [];
  private sessionId: string;
  private isEnabled = true;
  private reportingTimer: number | null = null;
  private observers: PerformanceObserver[] = [];
  private retryCount = 0;

  private constructor(config: Partial<PerformanceConfig>) {
    this.config = {
      sampleRate: 0.1, // 10% sampling in production
      enableWebVitals: true,
      enableUserTiming: true,
      enableResourceTiming: true,
      enableNavigationTiming: true,
      enableErrorTracking: true,
      enableUserInteractionTracking: true,
      performanceBudgets: {
        firstContentfulPaint: 1800,
        largestContentfulPaint: 2500,
        firstInputDelay: 100,
        cumulativeLayoutShift: 0.1,
        timeToInteractive: 3800,
        totalBlockingTime: 300,
        memoryUsage: 50, // MB
        bundleSize: 1000, // KB
      },
      reportingInterval: 30000, // 30 seconds
      maxRetries: 3,
      ...config,
    };

    this.sessionId = this.generateSessionId();
    this.initialize();
  }

  static getInstance(config?: Partial<PerformanceConfig>): PerformanceInstrumentation {
    if (!PerformanceInstrumentation.instance) {
      PerformanceInstrumentation.instance = new PerformanceInstrumentation(config || {});
    }
    return PerformanceInstrumentation.instance;
  }

  private initialize(): void {
    // Check if we should monitor based on sample rate
    if (Math.random() > this.config.sampleRate && process.env.NODE_ENV === 'production') {
      this.isEnabled = false;
      return;
    }

    this.setupPerformanceObservers();
    this.setupErrorTracking();
    this.setupUserInteractionTracking();
    this.setupNetworkMonitoring();
    this.startPeriodicReporting();

    console.log('Performance instrumentation initialized');
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private setupPerformanceObservers(): void {
    if (!this.isEnabled || !('PerformanceObserver' in window)) return;

    // Web Vitals Observer
    if (this.config.enableWebVitals) {
      this.setupWebVitalsObserver();
    }

    // User Timing Observer
    if (this.config.enableUserTiming) {
      this.setupUserTimingObserver();
    }

    // Resource Timing Observer
    if (this.config.enableResourceTiming) {
      this.setupResourceTimingObserver();
    }

    // Navigation Timing Observer
    if (this.config.enableNavigationTiming) {
      this.setupNavigationTimingObserver();
    }
  }

  private setupWebVitalsObserver(): void {
    try {
      // Largest Contentful Paint
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries() as any[];
        const lastEntry = entries[entries.length - 1];
        
        this.recordMetric({
          name: 'largest-contentful-paint',
          value: lastEntry.startTime,
          unit: 'ms',
          timestamp: Date.now(),
          category: 'webvitals',
          metadata: {
            element: lastEntry.element?.tagName,
            url: lastEntry.url,
          },
        });

        this.checkPerformanceBudget('largestContentfulPaint', lastEntry.startTime);
      });
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
      this.observers.push(lcpObserver);

      // First Contentful Paint
      const fcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          if (entry.name === 'first-contentful-paint') {
            this.recordMetric({
              name: 'first-contentful-paint',
              value: entry.startTime,
              unit: 'ms',
              timestamp: Date.now(),
              category: 'webvitals',
            });

            this.checkPerformanceBudget('firstContentfulPaint', entry.startTime);
          }
        });
      });
      fcpObserver.observe({ entryTypes: ['paint'] });
      this.observers.push(fcpObserver);

      // First Input Delay
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries() as any[];
        entries.forEach((entry) => {
          const fid = entry.processingStart - entry.startTime;
          
          this.recordMetric({
            name: 'first-input-delay',
            value: fid,
            unit: 'ms',
            timestamp: Date.now(),
            category: 'webvitals',
            metadata: {
              eventType: entry.name,
            },
          });

          this.checkPerformanceBudget('firstInputDelay', fid);
        });
      });
      fidObserver.observe({ entryTypes: ['first-input'] });
      this.observers.push(fidObserver);

      // Cumulative Layout Shift
      let clsValue = 0;
      const clsObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries() as any[];
        entries.forEach((entry) => {
          if (!entry.hadRecentInput) {
            clsValue += entry.value;
            
            this.recordMetric({
              name: 'cumulative-layout-shift',
              value: clsValue,
              unit: 'score',
              timestamp: Date.now(),
              category: 'webvitals',
              metadata: {
                sources: entry.sources?.map((s: any) => s.node?.tagName),
              },
            });

            this.checkPerformanceBudget('cumulativeLayoutShift', clsValue);
          }
        });
      });
      clsObserver.observe({ entryTypes: ['layout-shift'] });
      this.observers.push(clsObserver);

    } catch (error) {
      console.warn('Failed to setup Web Vitals observer:', error);
    }
  }

  private setupUserTimingObserver(): void {
    try {
      const userTimingObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          this.recordMetric({
            name: entry.name,
            value: entry.duration || entry.startTime,
            unit: 'ms',
            timestamp: Date.now(),
            category: 'custom',
            metadata: {
              entryType: entry.entryType,
              detail: (entry as any).detail,
            },
          });
        });
      });
      userTimingObserver.observe({ entryTypes: ['mark', 'measure'] });
      this.observers.push(userTimingObserver);
    } catch (error) {
      console.warn('Failed to setup User Timing observer:', error);
    }
  }

  private setupResourceTimingObserver(): void {
    try {
      const resourceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries() as PerformanceResourceTiming[];
        entries.forEach((entry) => {
          const networkMetric: NetworkMetric = {
            url: entry.name,
            method: 'GET', // PerformanceResourceTiming doesn't include method
            status: 200, // Assume success if loaded
            duration: entry.duration,
            size: entry.transferSize || entry.encodedBodySize || 0,
            timestamp: Date.now(),
            cached: entry.transferSize === 0 && entry.encodedBodySize > 0,
            metadata: {
              initiatorType: entry.initiatorType,
              nextHopProtocol: entry.nextHopProtocol,
              renderBlockingStatus: (entry as any).renderBlockingStatus,
            },
          };

          this.recordNetworkMetric(networkMetric);
        });
      });
      resourceObserver.observe({ entryTypes: ['resource'] });
      this.observers.push(resourceObserver);
    } catch (error) {
      console.warn('Failed to setup Resource Timing observer:', error);
    }
  }

  private setupNavigationTimingObserver(): void {
    try {
      const navigationObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries() as PerformanceNavigationTiming[];
        entries.forEach((entry) => {
          // Record various navigation metrics
          const metrics = [
            { name: 'dns-lookup', value: entry.domainLookupEnd - entry.domainLookupStart },
            { name: 'tcp-connection', value: entry.connectEnd - entry.connectStart },
            { name: 'request-response', value: entry.responseEnd - entry.requestStart },
            { name: 'dom-processing', value: entry.domComplete - entry.domLoading },
            { name: 'load-complete', value: entry.loadEventEnd - entry.loadEventStart },
          ];

          metrics.forEach((metric) => {
            if (metric.value > 0) {
              this.recordMetric({
                name: metric.name,
                value: metric.value,
                unit: 'ms',
                timestamp: Date.now(),
                category: 'network',
                metadata: {
                  navigationType: entry.type,
                  redirectCount: entry.redirectCount,
                },
              });
            }
          });
        });
      });
      navigationObserver.observe({ entryTypes: ['navigation'] });
      this.observers.push(navigationObserver);
    } catch (error) {
      console.warn('Failed to setup Navigation Timing observer:', error);
    }
  }

  private setupErrorTracking(): void {
    if (!this.config.enableErrorTracking || !this.isEnabled) return;

    // Global error handler
    window.addEventListener('error', (event) => {
      this.recordError({
        message: event.message,
        stack: event.error?.stack,
        source: event.filename,
        line: event.lineno,
        column: event.colno,
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        url: window.location.href,
        sessionId: this.sessionId,
        severity: 'high',
        tags: ['javascript', 'runtime'],
      });
    });

    // Promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
      this.recordError({
        message: event.reason?.message || 'Unhandled Promise Rejection',
        stack: event.reason?.stack,
        source: 'promise',
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        url: window.location.href,
        sessionId: this.sessionId,
        severity: 'medium',
        tags: ['promise', 'rejection'],
      });
    });

    // Console error wrapper
    const originalConsoleError = console.error;
    console.error = (...args) => {
      this.recordError({
        message: args.join(' '),
        source: 'console',
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        url: window.location.href,
        sessionId: this.sessionId,
        severity: 'low',
        tags: ['console'],
      });
      originalConsoleError.apply(console, args);
    };
  }

  private setupUserInteractionTracking(): void {
    if (!this.config.enableUserInteractionTracking || !this.isEnabled) return;

    // Click tracking
    document.addEventListener('click', (event) => {
      const target = event.target as HTMLElement;
      this.recordInteraction({
        type: 'click',
        element: this.getElementSelector(target),
        timestamp: Date.now(),
        metadata: {
          x: event.clientX,
          y: event.clientY,
          button: event.button,
        },
      });
    });

    // Input tracking
    document.addEventListener('input', (event) => {
      const target = event.target as HTMLElement;
      this.recordInteraction({
        type: 'input',
        element: this.getElementSelector(target),
        timestamp: Date.now(),
        metadata: {
          inputType: (event as any).inputType,
          isComposing: (event as any).isComposing,
        },
      });
    });

    // Scroll tracking with throttling
    let scrollTimeout: number;
    document.addEventListener('scroll', () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = window.setTimeout(() => {
        this.recordInteraction({
          type: 'scroll',
          timestamp: Date.now(),
          metadata: {
            scrollY: window.scrollY,
            scrollX: window.scrollX,
          },
        });
      }, 100);
    });
  }

  private setupNetworkMonitoring(): void {
    if (!this.isEnabled) return;

    // Monkey patch fetch
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const startTime = performance.now();
      const url = typeof args[0] === 'string' ? args[0] : args[0].url;
      const method = args[1]?.method || 'GET';

      try {
        const response = await originalFetch(...args);
        const duration = performance.now() - startTime;

        this.recordNetworkMetric({
          url,
          method,
          status: response.status,
          duration,
          size: parseInt(response.headers.get('content-length') || '0'),
          timestamp: Date.now(),
          cached: response.headers.get('cache-control')?.includes('max-age') || false,
          metadata: {
            contentType: response.headers.get('content-type'),
            fromCache: response.headers.has('cf-cache-status'),
          },
        });

        return response;
      } catch (error) {
        const duration = performance.now() - startTime;
        
        this.recordNetworkMetric({
          url,
          method,
          status: 0,
          duration,
          size: 0,
          timestamp: Date.now(),
          cached: false,
          metadata: {
            error: error instanceof Error ? error.message : 'Unknown error',
          },
        });

        throw error;
      }
    };
  }

  private recordMetric(metric: PerformanceMetric): void {
    if (!this.isEnabled) return;

    this.metrics.push(metric);
    
    // Keep only recent metrics
    if (this.metrics.length > 1000) {
      this.metrics = this.metrics.slice(-500);
    }

    // Log in development
    if (process.env.NODE_ENV === 'development') {
      console.log('Performance metric:', metric);
    }
  }

  private recordError(error: ErrorMetric): void {
    if (!this.isEnabled) return;

    this.errors.push(error);
    
    // Keep only recent errors
    if (this.errors.length > 100) {
      this.errors = this.errors.slice(-50);
    }

    console.error('Error tracked:', error);
  }

  private recordInteraction(interaction: UserInteractionMetric): void {
    if (!this.isEnabled) return;

    this.interactions.push(interaction);
    
    // Keep only recent interactions
    if (this.interactions.length > 500) {
      this.interactions = this.interactions.slice(-250);
    }
  }

  private recordNetworkMetric(metric: NetworkMetric): void {
    if (!this.isEnabled) return;

    this.networkMetrics.push(metric);
    
    // Keep only recent network metrics
    if (this.networkMetrics.length > 200) {
      this.networkMetrics = this.networkMetrics.slice(-100);
    }
  }

  private checkPerformanceBudget(budgetKey: keyof PerformanceBudgets, value: number): void {
    const budget = this.config.performanceBudgets[budgetKey];
    if (value > budget) {
      console.warn(`Performance budget exceeded for ${budgetKey}: ${value} > ${budget}`);
      
      this.recordError({
        message: `Performance budget exceeded: ${budgetKey}`,
        source: 'performance-budget',
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        url: window.location.href,
        sessionId: this.sessionId,
        severity: 'medium',
        tags: ['performance', 'budget'],
      });
    }
  }

  private getElementSelector(element: HTMLElement): string {
    if (element.id) return `#${element.id}`;
    if (element.className) return `.${element.className.split(' ')[0]}`;
    return element.tagName.toLowerCase();
  }

  private startPeriodicReporting(): void {
    if (!this.isEnabled || !this.config.apiEndpoint) return;

    this.reportingTimer = window.setInterval(() => {
      this.sendMetrics();
    }, this.config.reportingInterval);
  }

  private async sendMetrics(): Promise<void> {
    if (!this.config.apiEndpoint || this.retryCount >= this.config.maxRetries) return;

    const payload = {
      sessionId: this.sessionId,
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      metrics: this.metrics.splice(0), // Clear after getting
      errors: this.errors.splice(0),
      interactions: this.interactions.splice(0),
      networkMetrics: this.networkMetrics.splice(0),
      memoryUsage: this.getMemoryUsage(),
    };

    try {
      const response = await fetch(this.config.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      this.retryCount = 0; // Reset on success
    } catch (error) {
      this.retryCount++;
      console.error('Failed to send performance metrics:', error);
      
      // Put metrics back if failed
      this.metrics.unshift(...payload.metrics);
      this.errors.unshift(...payload.errors);
      this.interactions.unshift(...payload.interactions);
      this.networkMetrics.unshift(...payload.networkMetrics);
    }
  }

  private getMemoryUsage(): number {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return memory.usedJSHeapSize / 1024 / 1024; // MB
    }
    return 0;
  }

  // Public API
  public mark(name: string, detail?: any): void {
    if (!this.isEnabled) return;
    
    performance.mark(name, { detail });
  }

  public measure(name: string, startMark?: string, endMark?: string): void {
    if (!this.isEnabled) return;
    
    try {
      if (startMark && endMark) {
        performance.measure(name, startMark, endMark);
      } else if (startMark) {
        performance.measure(name, startMark);
      } else {
        performance.measure(name);
      }
    } catch (error) {
      console.warn('Failed to create performance measure:', error);
    }
  }

  public recordCustomMetric(name: string, value: number, unit = 'ms', metadata?: Record<string, any>): void {
    this.recordMetric({
      name,
      value,
      unit,
      timestamp: Date.now(),
      category: 'custom',
      metadata,
    });
  }

  public getReport(): {
    metrics: PerformanceMetric[];
    errors: ErrorMetric[];
    interactions: UserInteractionMetric[];
    networkMetrics: NetworkMetric[];
    summary: Record<string, any>;
  } {
    const summary = {
      totalMetrics: this.metrics.length,
      totalErrors: this.errors.length,
      totalInteractions: this.interactions.length,
      totalNetworkRequests: this.networkMetrics.length,
      sessionDuration: Date.now() - parseInt(this.sessionId.split('_')[1]),
      averagePageLoadTime: this.networkMetrics
        .filter(m => m.url === window.location.href)
        .reduce((sum, m) => sum + m.duration, 0) / Math.max(1, this.networkMetrics.length),
      errorRate: (this.errors.length / (this.metrics.length + this.errors.length)) * 100,
      memoryUsage: this.getMemoryUsage(),
    };

    return {
      metrics: [...this.metrics],
      errors: [...this.errors],
      interactions: [...this.interactions],
      networkMetrics: [...this.networkMetrics],
      summary,
    };
  }

  public destroy(): void {
    this.isEnabled = false;
    
    if (this.reportingTimer) {
      clearInterval(this.reportingTimer);
    }

    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];

    // Send final metrics
    if (this.config.apiEndpoint) {
      this.sendMetrics();
    }
  }
}

// React hook for performance instrumentation
export function usePerformanceInstrumentation(config?: Partial<PerformanceConfig>) {
  const instrumentation = React.useMemo(
    () => PerformanceInstrumentation.getInstance(config),
    [config]
  );

  React.useEffect(() => {
    return () => {
      // Don't destroy singleton on unmount
    };
  }, []);

  return {
    mark: instrumentation.mark.bind(instrumentation),
    measure: instrumentation.measure.bind(instrumentation),
    recordCustomMetric: instrumentation.recordCustomMetric.bind(instrumentation),
    getReport: instrumentation.getReport.bind(instrumentation),
  };
}

// Export singleton instance
export const performanceInstrumentation = PerformanceInstrumentation.getInstance({
  apiEndpoint: process.env.NODE_ENV === 'production' ? '/api/performance' : undefined,
  sampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
});

export default PerformanceInstrumentation;