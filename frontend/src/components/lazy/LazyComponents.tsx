/**
 * Lazy-loaded components for bundle optimization
 * 
 * Features:
 * - Code splitting for major components
 * - Loading states and error boundaries
 * - Preloading strategies
 * - Performance monitoring
 * - Resource hints for optimal loading
 */

import React, { Suspense, lazy, ComponentType } from 'react';
import { Loader2, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import ResearchErrorBoundary from '@/components/ui/research-error-boundary';

// Loading component with performance monitoring
const LoadingFallback: React.FC<{ componentName?: string; isPreloading?: boolean }> = ({ 
  componentName = 'Component',
  isPreloading = false 
}) => {
  React.useEffect(() => {
    if (!isPreloading) {
      performance.mark(`${componentName}-load-start`);
      
      return () => {
        performance.mark(`${componentName}-load-end`);
        performance.measure(
          `${componentName}-load-time`,
          `${componentName}-load-start`,
          `${componentName}-load-end`
        );
      };
    }
  }, [componentName, isPreloading]);

  return (
    <Card className="border-neutral-700 bg-neutral-800/80 animate-pulse">
      <CardContent className="flex items-center justify-center h-64">
        <div className="text-center text-neutral-400">
          <Loader2 className="h-8 w-8 mx-auto mb-3 animate-spin" />
          <p className="text-sm">Loading {componentName}...</p>
          {isPreloading && (
            <p className="text-xs text-neutral-500 mt-1">Preloading in background</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Error fallback component
const ErrorFallback: React.FC<{ 
  error?: Error; 
  retry?: () => void; 
  componentName?: string; 
}> = ({ error, retry, componentName = 'Component' }) => (
  <Card className="border-red-500/30 bg-red-500/5">
    <CardContent className="flex items-center justify-center h-64">
      <div className="text-center text-red-300">
        <AlertCircle className="h-8 w-8 mx-auto mb-3" />
        <p className="text-sm mb-3">Failed to load {componentName}</p>
        {error && (
          <p className="text-xs text-red-400 mb-3 font-mono">{error.message}</p>
        )}
        {retry && (
          <Button 
            onClick={retry} 
            size="sm" 
            variant="outline"
            className="border-red-500/30 text-red-300 hover:bg-red-500/10"
          >
            Retry
          </Button>
        )}
      </div>
    </CardContent>
  </Card>
);

// Higher-order component for lazy loading with enhanced features
function createLazyComponent<T extends ComponentType<any>>(
  importFn: () => Promise<{ default: T }>,
  componentName: string,
  preloadStrategy?: 'hover' | 'visible' | 'idle' | 'immediate'
) {
  const LazyComponent = lazy(importFn);
  
  const WrappedComponent = React.forwardRef<any, React.ComponentProps<T>>((props, ref) => {
    const [isPreloading, setIsPreloading] = React.useState(false);
    const [hasError, setHasError] = React.useState(false);
    const [retryCount, setRetryCount] = React.useState(0);
    const componentRef = React.useRef<HTMLDivElement>(null);

    // Preload component based on strategy
    const preloadComponent = React.useCallback(() => {
      if (!isPreloading) {
        setIsPreloading(true);
        importFn().catch((error) => {
          console.error(`Failed to preload ${componentName}:`, error);
        });
      }
    }, [isPreloading]);

    // Setup preloading strategies
    React.useEffect(() => {
      if (preloadStrategy === 'immediate') {
        preloadComponent();
      } else if (preloadStrategy === 'idle') {
        if ('requestIdleCallback' in window) {
          requestIdleCallback(() => preloadComponent());
        } else {
          setTimeout(preloadComponent, 100);
        }
      } else if (preloadStrategy === 'visible') {
        const observer = new IntersectionObserver(
          (entries) => {
            if (entries[0]?.isIntersecting) {
              preloadComponent();
              observer.disconnect();
            }
          },
          { threshold: 0.1 }
        );
        
        if (componentRef.current) {
          observer.observe(componentRef.current);
        }
        
        return () => observer.disconnect();
      }
    }, [preloadComponent, preloadStrategy]);

    // Retry mechanism
    const handleRetry = React.useCallback(() => {
      setHasError(false);
      setRetryCount(prev => prev + 1);
    }, []);

    // Error boundary integration
    const handleError = React.useCallback((error: Error) => {
      setHasError(true);
      console.error(`Error loading ${componentName}:`, error);
    }, []);

    const hoverHandlers = preloadStrategy === 'hover' ? {
      onMouseEnter: preloadComponent,
      onFocus: preloadComponent,
    } : {};

    return (
      <div ref={componentRef} {...hoverHandlers}>
        <ResearchErrorBoundary 
          onError={handleError}
          enableGracefulDegradation={true}
          fallbackComponent={
            <ErrorFallback 
              error={hasError ? new Error('Component failed to load') : undefined}
              retry={handleRetry}
              componentName={componentName}
            />
          }
        >
          <Suspense 
            fallback={<LoadingFallback componentName={componentName} isPreloading={isPreloading} />}
          >
            <LazyComponent {...props} ref={ref} key={retryCount} />
          </Suspense>
        </ResearchErrorBoundary>
      </div>
    );
  });

  WrappedComponent.displayName = `LazyLoaded${componentName}`;
  
  // Add preload method for manual preloading
  (WrappedComponent as any).preload = importFn;
  
  return WrappedComponent;
}

// Lazy-loaded research components
export const LazyParallelChatGrid = createLazyComponent(
  () => import('@/components/research/ParallelChatGrid'),
  'ParallelChatGrid',
  'visible'
);

export const LazySequenceChat = createLazyComponent(
  () => import('@/components/research/SequenceChat'),
  'SequenceChat',
  'immediate'
);

export const LazyQueryAnalyzer = createLazyComponent(
  () => import('@/components/research/QueryAnalyzer'),
  'QueryAnalyzer',
  'hover'
);

export const LazyComparisonSummary = createLazyComponent(
  () => import('@/components/research/ComparisonSummary'),
  'ComparisonSummary',
  'idle'
);

export const LazyLiveMetricsBar = createLazyComponent(
  () => import('@/components/research/LiveMetricsBar'),
  'LiveMetricsBar',
  'immediate'
);


// Delegation components
export const LazyDelegationDashboard = createLazyComponent(
  () => import('@/components/delegation/DelegationDashboard'),
  'DelegationDashboard',
  'hover'
);

export const LazyMetricsPanel = createLazyComponent(
  () => import('@/components/delegation/MetricsPanel'),
  'MetricsPanel',
  'visible'
);

// Utility components
export const LazyWelcomeScreen = createLazyComponent(
  () => import('@/components/WelcomeScreen'),
  'WelcomeScreen',
  'immediate'
);

export const LazyChatInterface = createLazyComponent(
  () => import('@/components/ChatInterface'),
  'ChatInterface',
  'hover'
);

export const LazyActivityTimeline = createLazyComponent(
  () => import('@/components/ActivityTimeline'),
  'ActivityTimeline',
  'idle'
);

// Hook for preloading components
export function useComponentPreloader() {
  const [preloadedComponents, setPreloadedComponents] = React.useState<Set<string>>(new Set());

  const preloadComponent = React.useCallback(async (componentName: string, importFn: () => Promise<any>) => {
    if (preloadedComponents.has(componentName)) {
      return;
    }

    try {
      performance.mark(`preload-${componentName}-start`);
      await importFn();
      performance.mark(`preload-${componentName}-end`);
      performance.measure(
        `preload-${componentName}`,
        `preload-${componentName}-start`,
        `preload-${componentName}-end`
      );
      
      setPreloadedComponents(prev => new Set(prev).add(componentName));
      console.log(`Preloaded ${componentName}`);
    } catch (error) {
      console.error(`Failed to preload ${componentName}:`, error);
    }
  }, [preloadedComponents]);

  const preloadResearchComponents = React.useCallback(() => {
    const componentsToPreload = [
      { name: 'ParallelChatGrid', fn: LazyParallelChatGrid.preload },
      { name: 'SequenceChat', fn: LazySequenceChat.preload },
      { name: 'LiveMetricsBar', fn: LazyLiveMetricsBar.preload },
    ];

    componentsToPreload.forEach(({ name, fn }) => {
      preloadComponent(name, fn);
    });
  }, [preloadComponent]);

  const preloadDelegationComponents = React.useCallback(() => {
    const componentsToPreload = [
      { name: 'DelegationDashboard', fn: LazyDelegationDashboard.preload },
      { name: 'MetricsPanel', fn: LazyMetricsPanel.preload },
      { name: 'ChatInterface', fn: LazyChatInterface.preload },
    ];

    componentsToPreload.forEach(({ name, fn }) => {
      preloadComponent(name, fn);
    });
  }, [preloadComponent]);

  return {
    preloadComponent,
    preloadResearchComponents,
    preloadDelegationComponents,
    preloadedComponents: Array.from(preloadedComponents),
  };
}

// Resource hints hook for optimal loading
export function useResourceHints() {
  React.useEffect(() => {
    // Add DNS prefetch for external resources
    const createResourceHint = (rel: string, href: string, as?: string) => {
      const link = document.createElement('link');
      link.rel = rel;
      link.href = href;
      if (as) link.as = as;
      document.head.appendChild(link);
      return link;
    };

    const hints: HTMLLinkElement[] = [];

    // Prefetch critical resources
    hints.push(createResourceHint('prefetch', '/api/research-state', 'fetch'));
    hints.push(createResourceHint('prefetch', '/api/metrics', 'fetch'));
    
    // Preload critical stylesheets
    hints.push(createResourceHint('preload', '/assets/research-grid.css', 'style'));
    
    // DNS prefetch for external APIs
    if (process.env.NODE_ENV === 'production') {
      hints.push(createResourceHint('dns-prefetch', 'https://api.research.ai'));
      hints.push(createResourceHint('dns-prefetch', 'https://metrics.research.ai'));
    }

    return () => {
      // Cleanup resource hints
      hints.forEach(link => {
        if (link.parentNode) {
          link.parentNode.removeChild(link);
        }
      });
    };
  }, []);
}

// Performance monitoring for lazy components
export function useLazyComponentMetrics() {
  const [metrics, setMetrics] = React.useState<{
    loadTimes: Record<string, number>;
    preloadSuccess: Record<string, boolean>;
    errorCounts: Record<string, number>;
  }>({
    loadTimes: {},
    preloadSuccess: {},
    errorCounts: {},
  });

  React.useEffect(() => {
    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      
      entries.forEach((entry) => {
        if (entry.name.includes('-load-time')) {
          const componentName = entry.name.replace('-load-time', '');
          setMetrics(prev => ({
            ...prev,
            loadTimes: {
              ...prev.loadTimes,
              [componentName]: entry.duration,
            },
          }));
        }
        
        if (entry.name.includes('preload-')) {
          const componentName = entry.name.replace('preload-', '');
          setMetrics(prev => ({
            ...prev,
            preloadSuccess: {
              ...prev.preloadSuccess,
              [componentName]: true,
            },
          }));
        }
      });
    });

    observer.observe({ entryTypes: ['measure'] });

    return () => observer.disconnect();
  }, []);

  return metrics;
}

export default {
  LazyParallelChatGrid,
  LazySequenceChat,
  LazyQueryAnalyzer,
  LazyComparisonSummary,
  LazyLiveMetricsBar,
  LazyDelegationDashboard,
  LazyMetricsPanel,
  LazyWelcomeScreen,
  LazyChatInterface,
  LazyActivityTimeline,
  useComponentPreloader,
  useResourceHints,
  useLazyComponentMetrics,
};