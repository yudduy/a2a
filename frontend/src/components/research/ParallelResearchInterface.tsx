/**
 * ParallelResearchInterface - Complete production-ready parallel research interface
 * 
 * This component serves as the main container for the parallel research experience,
 * integrating all components with real-time WebSocket data and providing a cohesive
 * user experience for running multiple research sequences simultaneously.
 */

import React, { useState, useCallback, useEffect, useRef, useMemo, memo } from 'react';
import { EnhancedErrorBoundary } from '@/components/ui/enhanced-error-boundary';
import { usePerformanceMonitor } from '@/hooks/usePerformanceMonitor';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import {
  GitCompare,
  Home,
  Pause,
  Play,
  RotateCcw,
  BarChart3,
  MessageSquare,
  TrendingUp,
  Activity,
  Zap,
  CheckCircle,
  AlertTriangle,
  Loader2,
  Monitor,
  Smartphone,
  Eye,
  EyeOff,
  RefreshCw,
} from 'lucide-react';

import { QueryAnalyzer } from './QueryAnalyzer';
import { ParallelChatGrid } from './ParallelChatGrid';
import { LiveMetricsBar } from './LiveMetricsBar';
import { ComparisonSummary } from './ComparisonSummary';
import { SequenceChat } from './SequenceChat';

import { useParallelSequences } from '@/hooks/useParallelSequences';
import { 
  SequenceStrategy,
  LLMGeneratedSequence,
  ConnectionState,
  SequenceState,
  UseParallelSequencesReturn
} from '@/types/parallel';
import { cn } from '@/lib/utils';

interface ParallelResearchInterfaceProps {
  query: string;
  onReset: () => void;
  className?: string;
}

export function ParallelResearchInterface({ 
  query, 
  onReset, 
  className 
}: ParallelResearchInterfaceProps) {
  // Performance monitoring
  const performanceMonitor = usePerformanceMonitor('ParallelResearchInterface', {
    enableMemoryMonitoring: true,
    enableNetworkMonitoring: true,
    enableWebVitals: true,
    sampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
  });

  // Initialize parallel sequences hook
  const parallelSequences = useParallelSequences({
    enableAutoReconnect: true,
    enableMetrics: true,
    bufferSize: 1000,
  });

  // Local state for UI controls
  const [selectedView, setSelectedView] = useState<'grid' | 'overview' | 'metrics'>('grid');
  const [isMobileView, setIsMobileView] = useState(false);
  const [showQueryAnalysis, setShowQueryAnalysis] = useState(true);
  const [isInitialized, setIsInitialized] = useState(false);
  const [hasStartedOnce, setHasStartedOnce] = useState(false);
  
  // Refs for stable behavior
  const initializeTimeoutRef = useRef<NodeJS.Timeout>();

  // Destructure parallel sequences return values
  const {
    sequences,
    isLoading,
    error,
    progress,
    metrics,
    start,
    stop,
    restart,
    connectionState,
  } = parallelSequences;

  // Initialize parallel sequences when component mounts
  useEffect(() => {
    if (query && !isInitialized && !hasStartedOnce) {
      // Add a slight delay to ensure all components are ready
      initializeTimeoutRef.current = setTimeout(async () => {
        const finishTracking = performanceMonitor.createInteractionTracker('parallel-sequences-init');
        try {
          await start(query);
          setIsInitialized(true);
          setHasStartedOnce(true);
          finishTracking();
        } catch (err) {
          console.error('Failed to initialize parallel sequences:', err);
          finishTracking();
        }
      }, 100);
    }

    return () => {
      if (initializeTimeoutRef.current) {
        clearTimeout(initializeTimeoutRef.current);
      }
    };
  }, [query, start, isInitialized, hasStartedOnce, performanceMonitor]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  // Responsive behavior
  useEffect(() => {
    const checkIsMobile = () => {
      setIsMobileView(window.innerWidth < 1024);
    };

    checkIsMobile();
    window.addEventListener('resize', checkIsMobile);
    return () => window.removeEventListener('resize', checkIsMobile);
  }, []);

  // Memoized computed values
  const computedState = useMemo(() => {
    const hasActiveSequences = sequences.some(seq => seq.progress.status === 'active');
    const allCompleted = sequences.length === 3 && 
      sequences.every(seq => seq.progress.status === 'completed');
    const hasErrors = sequences.some(seq => seq.errors.length > 0) || error !== null;
    const overallProgress = sequences.length > 0 
      ? Math.round(sequences.reduce((sum, s) => sum + s.progress.completion_percentage, 0) / sequences.length)
      : 0;

    return {
      hasActiveSequences,
      allCompleted,
      hasErrors,
      overallProgress,
      canRestart: !isLoading && sequences.length > 0,
      canPause: hasActiveSequences && isLoading,
      canResume: !hasActiveSequences && sequences.length > 0 && !allCompleted,
    };
  }, [sequences, error, isLoading]);

  // Control handlers
  const handlePause = useCallback(() => {
    stop();
  }, [stop]);

  const handleResume = useCallback(() => {
    if (query) {
      start(query);
    }
  }, [start, query]);

  const handleRestart = useCallback(() => {
    restart();
  }, [restart]);

  const handleReset = useCallback(() => {
    stop();
    onReset();
  }, [stop, onReset]);

  const handleToggleQueryAnalysis = useCallback(() => {
    setShowQueryAnalysis(prev => !prev);
  }, []);

  // Strategy configurations for display
  const strategyConfig = {
    [SequenceStrategy.THEORY_FIRST]: {
      title: 'Theory First',
      description: 'Academic → Industry → Technical',
      icon: <Activity className="h-4 w-4" />,
      color: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    },
    [SequenceStrategy.MARKET_FIRST]: {
      title: 'Market First', 
      description: 'Industry → Academic → Technical',
      icon: <TrendingUp className="h-4 w-4" />,
      color: 'bg-green-500/20 text-green-300 border-green-500/30',
    },
    [SequenceStrategy.FUTURE_BACK]: {
      title: 'Future Back',
      description: 'Technical → Academic → Industry',
      icon: <Zap className="h-4 w-4" />,
      color: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    },
  };

  const getConnectionStatusInfo = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return { color: 'text-green-400', text: 'Connected', icon: '●' };
      case ConnectionState.CONNECTING:
      case ConnectionState.RECONNECTING:
        return { color: 'text-yellow-400', text: 'Connecting...', icon: '◐' };
      case ConnectionState.FAILED:
        return { color: 'text-red-400', text: 'Connection Failed', icon: '●' };
      default:
        return { color: 'text-neutral-400', text: 'Disconnected', icon: '○' };
    }
  };

  const connectionInfo = getConnectionStatusInfo();

  // Loading state while initializing
  if (!isInitialized && !hasStartedOnce) {
    return (
      <div className={cn("h-full flex flex-col bg-neutral-900 text-neutral-100", className)}>
        <Card className="h-full border-neutral-700 bg-neutral-800 flex items-center justify-center">
          <div className="text-center max-w-md mx-auto p-8">
            <Loader2 className="h-16 w-16 mx-auto mb-6 animate-spin text-blue-400" />
            <h2 className="text-2xl font-bold text-neutral-100 mb-3">
              Starting Parallel Research
            </h2>
            <p className="text-neutral-400 mb-4 leading-relaxed">
              Initializing 3 research sequences to analyze your query from different perspectives...
            </p>
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm text-neutral-300">
                <Activity className="h-4 w-4 text-blue-400" />
                <span>Theory First: Academic → Industry → Technical</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-neutral-300">
                <TrendingUp className="h-4 w-4 text-green-400" />
                <span>Market First: Industry → Academic → Technical</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-neutral-300">
                <Zap className="h-4 w-4 text-purple-400" />
                <span>Future Back: Technical → Academic → Industry</span>
              </div>
            </div>
            <Button 
              onClick={handleReset} 
              variant="ghost" 
              className="mt-6 text-neutral-400 hover:text-neutral-200"
            >
              Cancel
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <EnhancedErrorBoundary
      level="feature"
      enableRetry={true}
      maxRetries={3}
      resetKeys={[query]}
      onError={(error, errorInfo) => {
        console.error('ParallelResearchInterface Error:', error, errorInfo);
        // Report to monitoring service in production
        if (process.env.NODE_ENV === 'production') {
          // Send to error reporting service
        }
      }}
    >
      <div className={cn("h-full flex flex-col bg-neutral-900 text-neutral-100", className)}>
        {/* Header */}
        <div className="border-b border-neutral-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <GitCompare className="h-6 w-6 text-blue-400" />
              <div>
                <h1 className="text-xl font-bold text-neutral-100">
                  Parallel Research Analysis
                </h1>
                <p className="text-sm text-neutral-400">
                  Real-time comparison of 3 research strategies
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1 text-sm">
                <span className={connectionInfo.color}>{connectionInfo.icon}</span>
                <span className="text-neutral-400">{connectionInfo.text}</span>
              </div>
              
              {computedState.overallProgress > 0 && (
                <div className="flex items-center gap-2">
                  <Progress value={computedState.overallProgress} className="w-20 h-2" />
                  <span className="text-sm text-neutral-300 min-w-[3rem]">
                    {computedState.overallProgress}%
                  </span>
                </div>
              )}

              <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30">
                {sequences.filter(s => s.progress.status === 'active').length} active
              </Badge>
            </div>
          </div>

          {/* Control buttons */}
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsMobileView(!isMobileView)}
              className="text-neutral-400 hover:text-neutral-200"
            >
              {isMobileView ? <Monitor className="h-4 w-4" /> : <Smartphone className="h-4 w-4" />}
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={handleToggleQueryAnalysis}
              className="text-neutral-400 hover:text-neutral-200"
            >
              {showQueryAnalysis ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>

            {computedState.canPause && (
              <Button
                onClick={handlePause}
                variant="outline"
                size="sm"
                className="border-amber-600 text-amber-400 hover:bg-amber-600/10"
              >
                <Pause className="h-4 w-4 mr-1" />
                Pause
              </Button>
            )}

            {computedState.canResume && (
              <Button
                onClick={handleResume}
                variant="outline"
                size="sm"
                className="border-green-600 text-green-400 hover:bg-green-600/10"
              >
                <Play className="h-4 w-4 mr-1" />
                Resume
              </Button>
            )}

            {computedState.canRestart && (
              <Button
                onClick={handleRestart}
                variant="outline"
                size="sm"
                className="border-blue-600 text-blue-400 hover:bg-blue-600/10"
              >
                <RefreshCw className="h-4 w-4 mr-1" />
                Restart
              </Button>
            )}

            <Button
              onClick={handleReset}
              variant="outline"
              size="sm"
              className="border-neutral-600 text-neutral-300 hover:bg-neutral-600/10"
            >
              <Home className="h-4 w-4 mr-1" />
              New Query
            </Button>
          </div>
        </div>

        {/* Live metrics bar */}
        <div className="mt-4">
          <LiveMetricsBar
            progress={progress}
            metrics={metrics}
            connectionState={connectionState}
            isLoading={isLoading}
            error={error}
          />
        </div>
      </div>

      {/* Query Analysis (collapsible) */}
      {showQueryAnalysis && (
        <div className="border-b border-neutral-700">
          <QueryAnalyzer
            query={query}
            sequences={sequences}
            isLoading={isLoading}
            defaultExpanded={true}
          />
        </div>
      )}

      {/* Main content area */}
      <div className="flex-1 overflow-hidden">
        <Tabs value={selectedView} onValueChange={(value) => setSelectedView(value as any)} className="h-full flex flex-col">
          <TabsList className="bg-neutral-700 border-b border-neutral-600 rounded-none justify-start p-1 mx-6 mt-4 mb-0">
            <TabsTrigger value="grid" className="data-[state=active]:bg-neutral-600">
              <MessageSquare className="h-4 w-4 mr-1" />
              Chat Grid
            </TabsTrigger>
            <TabsTrigger value="overview" className="data-[state=active]:bg-neutral-600">
              <BarChart3 className="h-4 w-4 mr-1" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="metrics" className="data-[state=active]:bg-neutral-600">
              <Activity className="h-4 w-4 mr-1" />
              Metrics
            </TabsTrigger>
          </TabsList>

          {/* Chat Grid View */}
          <TabsContent value="grid" className="flex-1 p-6 mt-0 overflow-hidden">
            {sequences.length === 0 && !isLoading ? (
              <Card className="h-full border-neutral-700 bg-neutral-800 flex items-center justify-center">
                <div className="text-center text-neutral-500">
                  <MessageSquare className="h-16 w-16 mx-auto mb-4 opacity-50" />
                  <h3 className="text-lg font-medium mb-2">No Sequences Available</h3>
                  <p className="text-sm text-neutral-600 mb-4">
                    Something went wrong while starting the parallel sequences.
                  </p>
                  <Button onClick={handleRestart} className="bg-blue-600 hover:bg-blue-700 text-white">
                    Retry
                  </Button>
                </div>
              </Card>
            ) : (
              <div className="h-full">
                {isMobileView ? (
                  <MobileChatView sequences={sequences} isLoading={isLoading} />
                ) : (
                  <DesktopChatGrid sequences={sequences} isLoading={isLoading} />
                )}
              </div>
            )}
          </TabsContent>

          {/* Overview View */}
          <TabsContent value="overview" className="flex-1 p-6 mt-0 overflow-hidden">
            <div className="h-full grid grid-cols-1 lg:grid-cols-3 gap-6">
              {Object.values(SequenceStrategy).map((strategy) => {
                const sequence = sequences.find(s => s.strategy === strategy);
                const config = strategyConfig[strategy];
                const isActive = sequence?.progress.status === 'active';
                
                return (
                  <Card key={strategy} className="border-neutral-700 bg-neutral-800/80 backdrop-blur-sm">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <Badge className={config.color}>
                          {config.icon}
                          <span className="ml-1">{config.title}</span>
                        </Badge>
                        {isActive && (
                          <div className="flex items-center gap-1">
                            <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse" />
                            <span className="text-xs text-green-400">Live</span>
                          </div>
                        )}
                      </div>
                      
                      <p className="text-sm text-neutral-400 mb-4">
                        {config.description}
                      </p>
                      
                      {sequence ? (
                        <div className="space-y-3">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-neutral-400">Progress:</span>
                            <span className="text-neutral-200">
                              {Math.round(sequence.progress.completion_percentage)}%
                            </span>
                          </div>
                          <Progress value={sequence.progress.completion_percentage} className="h-2" />
                          
                          <div className="grid grid-cols-2 gap-4 text-xs">
                            <div>
                              <span className="text-neutral-400">Messages:</span>
                              <span className="text-neutral-200 ml-1">
                                {sequence.progress.messages_received}
                              </span>
                            </div>
                            <div>
                              <span className="text-neutral-400">Status:</span>
                              <span className="text-neutral-200 ml-1 capitalize">
                                {sequence.progress.status}
                              </span>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center text-neutral-500 py-4">
                          <Loader2 className="h-6 w-6 mx-auto mb-2 animate-spin" />
                          <p className="text-xs">Initializing...</p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>

          {/* Metrics View */}
          <TabsContent value="metrics" className="flex-1 p-6 mt-0 overflow-hidden">
            <Card className="h-full border-neutral-700 bg-neutral-800">
              <CardContent className="p-6 h-full">
                {/* This could be expanded with detailed metrics */}
                <div className="text-center text-neutral-500 py-12">
                  <BarChart3 className="h-16 w-16 mx-auto mb-4 opacity-50" />
                  <h3 className="text-lg font-medium mb-2">Detailed Metrics</h3>
                  <p className="text-sm text-neutral-600">
                    Advanced analytics and performance metrics coming soon.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Comparison Summary (when completed) */}
      {computedState.allCompleted && (
        <div className="border-t border-neutral-700">
          <ComparisonSummary
            sequences={sequences}
            progress={progress}
            metrics={metrics}
          />
        </div>
      )}

      {/* Error Display */}
      {computedState.hasErrors && (
        <div className="border-t border-red-500/30 bg-red-500/5 px-6 py-3">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-red-400" />
            <span className="text-sm text-red-200">
              {error?.message || 'Some sequences encountered errors. Check individual sequence details.'}
            </span>
            {error && (
              <Button
                onClick={handleRestart}
                variant="ghost"
                size="sm"
                className="text-red-300 hover:text-red-200 ml-auto"
              >
                Retry
              </Button>
            )}
          </div>
        </div>
      )}
      </div>
    </EnhancedErrorBoundary>
  );
}

// Memory cleanup hook
function useMemoryCleanup() {
  const cleanupRefs = useRef<Array<() => void>>([]);

  const addCleanup = useCallback((cleanupFn: () => void) => {
    cleanupRefs.current.push(cleanupFn);
  }, []);

  useEffect(() => {
    return () => {
      // Execute all cleanup functions
      cleanupRefs.current.forEach(cleanup => {
        try {
          cleanup();
        } catch (error) {
          console.error('Cleanup error:', error);
        }
      });
      cleanupRefs.current = [];
    };
  }, []);

  return { addCleanup };
}

// Mobile chat view component (memoized for performance)
const MobileChatView = memo(({ sequences, isLoading }: { 
  sequences: SequenceState[], 
  isLoading: boolean 
}) => {
  const [activeTab, setActiveTab] = useState<SequenceStrategy>(SequenceStrategy.THEORY_FIRST);

  const strategyConfig = {
    [SequenceStrategy.THEORY_FIRST]: { title: 'Theory', icon: <Activity className="h-4 w-4" /> },
    [SequenceStrategy.MARKET_FIRST]: { title: 'Market', icon: <TrendingUp className="h-4 w-4" /> },
    [SequenceStrategy.FUTURE_BACK]: { title: 'Future', icon: <Zap className="h-4 w-4" /> },
  };

  return (
    <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as SequenceStrategy)} className="h-full flex flex-col">
      <TabsList className="grid w-full grid-cols-3 bg-neutral-700">
        {Object.entries(strategyConfig).map(([strategy, config]) => {
          const sequence = sequences.find(s => s.strategy === strategy as SequenceStrategy);
          const isActive = sequence?.progress.status === 'active';
          
          return (
            <TabsTrigger 
              key={strategy}
              value={strategy} 
              className="data-[state=active]:bg-neutral-600 flex items-center gap-2"
            >
              {config.icon}
              {config.title}
              {isActive && <div className="h-1.5 w-1.5 bg-green-400 rounded-full animate-pulse" />}
            </TabsTrigger>
          );
        })}
      </TabsList>

      {Object.values(SequenceStrategy).map((strategy) => {
        const sequence = sequences.find(s => s.strategy === strategy);
        
        return (
          <TabsContent key={strategy} value={strategy} className="flex-1 mt-0 overflow-hidden">
            <SequenceChat
              sequence={sequence}
              strategy={strategy}
              isActive={sequence?.progress.status === 'active'}
              isLoading={isLoading}
              showHeader={true}
              enableExpansion={true}
            />
          </TabsContent>
        );
      })}
    </Tabs>
  );
});

// Desktop chat grid component (memoized for performance)
const DesktopChatGrid = memo(({ sequences, isLoading }: { 
  sequences: SequenceState[], 
  isLoading: boolean 
}) => {
  const strategyConfig = {
    [SequenceStrategy.THEORY_FIRST]: {
      title: 'Theory First',
      description: 'Academic → Industry → Technical',
      icon: <Activity className="h-4 w-4" />,
      color: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    },
    [SequenceStrategy.MARKET_FIRST]: {
      title: 'Market First', 
      description: 'Industry → Academic → Technical',
      icon: <TrendingUp className="h-4 w-4" />,
      color: 'bg-green-500/20 text-green-300 border-green-500/30',
    },
    [SequenceStrategy.FUTURE_BACK]: {
      title: 'Future Back',
      description: 'Technical → Academic → Industry',
      icon: <Zap className="h-4 w-4" />,
      color: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    },
  };

  return (
    <div className="h-full grid grid-cols-1 lg:grid-cols-3 gap-6">
      {Object.values(SequenceStrategy).map((strategy) => {
        const sequence = sequences.find(s => s.strategy === strategy);
        const config = strategyConfig[strategy];
        
        return (
          <div key={strategy} className="flex flex-col">
            <div className="mb-4 p-4 bg-neutral-800/50 rounded-lg border border-neutral-700">
              <div className="flex items-center justify-between mb-2">
                <Badge className={config.color}>
                  {config.icon}
                  <span className="ml-1">{config.title}</span>
                </Badge>
                {sequence?.progress.status === 'active' && (
                  <div className="flex items-center gap-1">
                    <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse" />
                    <span className="text-xs text-green-400">Live</span>
                  </div>
                )}
                {sequence?.progress.status === 'completed' && (
                  <CheckCircle className="h-4 w-4 text-green-400" />
                )}
              </div>
              <p className="text-xs text-neutral-400">{config.description}</p>
            </div>
            
            <div className="flex-1 min-h-0">
              <SequenceChat
                sequence={sequence}
                strategy={strategy}
                isActive={sequence?.progress.status === 'active'}
                isLoading={isLoading}
                showHeader={false}
                enableExpansion={true}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
});

// Memoize the main component for performance
export default memo(ParallelResearchInterface);