/**
 * ParallelChatGrid - Production-ready 3-column parallel research interface
 * 
 * Features:
 * - Responsive 3-column layout for Theory First | Market First | Future Back
 * - Real-time message streaming in each column
 * - Live metrics and performance tracking
 * - Mobile-friendly tabbed interface
 * - Professional ShadCN UI components
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  GitCompare, 
  Monitor, 
  Smartphone, 
  BarChart3,
  Activity,
  TrendingUp,
  Zap,
  Loader2,
  AlertCircle,
  CheckCircle,
  Pause,
  Play,
  RotateCcw
} from 'lucide-react';

import { SequenceChat } from './SequenceChat';
import { QueryAnalyzer } from './QueryAnalyzer';
import { LiveMetricsBar } from './LiveMetricsBar';
import { ComparisonSummary } from './ComparisonSummary';

import { 
  SequenceState, 
  SequenceStrategy, 
  ParallelSequencesState, 
  RealTimeMetrics,
  UseParallelSequencesReturn 
} from '@/types/parallel';
import { cn } from '@/lib/utils';

interface ParallelChatGridProps {
  parallelSequences: UseParallelSequencesReturn;
  className?: string;
  onQuerySubmit?: (query: string) => void;
  enableMobileOptimization?: boolean;
  showQueryAnalyzer?: boolean;
  showComparisonSummary?: boolean;
}

export function ParallelChatGrid({
  parallelSequences,
  className,
  onQuerySubmit,
  enableMobileOptimization = true,
  showQueryAnalyzer = true,
  showComparisonSummary = true,
}: ParallelChatGridProps) {
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

  // State for responsive behavior
  const [isMobile, setIsMobile] = useState(false);
  const [activeSequenceTab, setActiveSequenceTab] = useState<SequenceStrategy>(SequenceStrategy.THEORY_FIRST);
  const [showControls, setShowControls] = useState(true);

  // Check if we're on mobile
  useEffect(() => {
    if (!enableMobileOptimization) return;

    const checkIsMobile = () => {
      setIsMobile(window.innerWidth < 1024); // lg breakpoint
    };

    checkIsMobile();
    window.addEventListener('resize', checkIsMobile);
    return () => window.removeEventListener('resize', checkIsMobile);
  }, [enableMobileOptimization]);

  // Get sequence by strategy
  const getSequenceByStrategy = useCallback((strategy: SequenceStrategy): SequenceState | undefined => {
    return sequences.find(seq => seq.strategy === strategy);
  }, [sequences]);

  // Strategy configurations
  const strategyConfig = {
    [SequenceStrategy.THEORY_FIRST]: {
      title: 'Theory First',
      description: 'Academic → Industry → Technical',
      icon: <Activity className="h-4 w-4" />,
      color: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
      bgColor: 'bg-blue-500/5',
    },
    [SequenceStrategy.MARKET_FIRST]: {
      title: 'Market First', 
      description: 'Industry → Academic → Technical',
      icon: <TrendingUp className="h-4 w-4" />,
      color: 'bg-green-500/20 text-green-300 border-green-500/30',
      bgColor: 'bg-green-500/5',
    },
    [SequenceStrategy.FUTURE_BACK]: {
      title: 'Future Back',
      description: 'Technical → Academic → Industry',
      icon: <Zap className="h-4 w-4" />,
      color: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
      bgColor: 'bg-purple-500/5',
    },
  };

  // Control handlers
  const handlePause = useCallback(() => {
    stop();
  }, [stop]);

  const handleResume = useCallback(() => {
    if (progress.research_query) {
      start(progress.research_query);
    }
  }, [start, progress.research_query]);

  const handleRestart = useCallback(() => {
    restart();
  }, [restart]);

  // Check if all sequences are completed
  const allCompleted = sequences.length === 3 && 
    sequences.every(seq => seq.progress.status === 'completed');

  const hasActiveSequences = sequences.some(seq => seq.progress.status === 'active');

  // Desktop 3-column layout
  const DesktopLayout = () => (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
      {Object.values(SequenceStrategy).map((strategy) => {
        const sequence = getSequenceByStrategy(strategy);
        const config = strategyConfig[strategy];
        
        return (
          <div key={strategy} className={cn("flex flex-col", config.bgColor, "rounded-lg")}>
            <div className="p-4 border-b border-neutral-700">
              <div className="flex items-center justify-between mb-2">
                <Badge className={cn("text-xs", config.color)}>
                  {config.icon}
                  <span className="ml-1">{config.title}</span>
                </Badge>
                <div className="flex items-center gap-2">
                  {sequence?.progress.status === 'active' && (
                    <div className="flex items-center gap-1">
                      <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse" />
                      <span className="text-xs text-green-400">Live</span>
                    </div>
                  )}
                  {sequence?.progress.status === 'completed' && (
                    <CheckCircle className="h-4 w-4 text-green-400" />
                  )}
                  {sequence?.progress.status === 'failed' && (
                    <AlertCircle className="h-4 w-4 text-red-400" />
                  )}
                </div>
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
              />
            </div>
          </div>
        );
      })}
    </div>
  );

  // Mobile tabbed layout
  const MobileLayout = () => (
    <Tabs value={activeSequenceTab} onValueChange={(value) => setActiveSequenceTab(value as SequenceStrategy)} className="h-full flex flex-col">
      <TabsList className="grid w-full grid-cols-3 bg-neutral-700">
        {Object.values(SequenceStrategy).map((strategy) => {
          const config = strategyConfig[strategy];
          const sequence = getSequenceByStrategy(strategy);
          
          return (
            <TabsTrigger 
              key={strategy}
              value={strategy} 
              className="data-[state=active]:bg-neutral-600 flex items-center gap-1 text-xs"
            >
              {config.icon}
              <span className="hidden sm:inline">{config.title}</span>
              {sequence?.progress.status === 'active' && (
                <div className="h-1.5 w-1.5 bg-green-400 rounded-full animate-pulse" />
              )}
            </TabsTrigger>
          );
        })}
      </TabsList>

      {Object.values(SequenceStrategy).map((strategy) => {
        const sequence = getSequenceByStrategy(strategy);
        
        return (
          <TabsContent key={strategy} value={strategy} className="flex-1 mt-0">
            <SequenceChat
              sequence={sequence}
              strategy={strategy}
              isActive={sequence?.progress.status === 'active'}
              isLoading={isLoading}
              showHeader={true}
            />
          </TabsContent>
        );
      })}
    </Tabs>
  );

  return (
    <div className={cn("h-full flex flex-col bg-neutral-900 text-neutral-100", className)}>
      {/* Header with controls */}
      <div className="border-b border-neutral-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <GitCompare className="h-6 w-6 text-blue-400" />
            <div>
              <h1 className="text-xl font-bold text-neutral-100">
                Parallel Research Chat
              </h1>
              <p className="text-sm text-neutral-400">
                Real-time 3-sequence analysis comparison
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {enableMobileOptimization && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsMobile(!isMobile)}
                className="text-neutral-400 hover:text-neutral-200"
              >
                {isMobile ? <Monitor className="h-4 w-4" /> : <Smartphone className="h-4 w-4" />}
              </Button>
            )}
            
            {showControls && hasActiveSequences && (
              <Button
                variant="outline"
                size="sm"
                onClick={handlePause}
                className="border-amber-600 text-amber-400 hover:bg-amber-600/10"
              >
                <Pause className="h-4 w-4 mr-1" />
                Pause
              </Button>
            )}
            
            {showControls && !hasActiveSequences && sequences.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleResume}
                className="border-green-600 text-green-400 hover:bg-green-600/10"
              >
                <Play className="h-4 w-4 mr-1" />
                Resume
              </Button>
            )}
            
            {showControls && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleRestart}
                className="border-blue-600 text-blue-400 hover:bg-blue-600/10"
              >
                <RotateCcw className="h-4 w-4 mr-1" />
                Restart
              </Button>
            )}
          </div>
        </div>

        {/* Live metrics bar */}
        <div className="mt-3">
          <LiveMetricsBar
            progress={progress}
            metrics={metrics}
            connectionState={connectionState}
            isLoading={isLoading}
            error={error}
          />
        </div>
      </div>

      {/* Query analyzer */}
      {showQueryAnalyzer && progress.research_query && (
        <div className="border-b border-neutral-700">
          <QueryAnalyzer
            query={progress.research_query}
            sequences={sequences}
            isLoading={isLoading}
          />
        </div>
      )}

      {/* Main chat grid */}
      <div className="flex-1 p-4 min-h-0">
        {sequences.length === 0 ? (
          <Card className="h-full flex items-center justify-center border-neutral-700 bg-neutral-800">
            <div className="text-center text-neutral-500">
              <BarChart3 className="h-16 w-16 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-medium mb-2">Ready for Parallel Research</h3>
              <p className="text-sm text-neutral-600 mb-4 max-w-md">
                Start a research query to see 3 parallel sequences analyzing your topic 
                with different approaches in real-time.
              </p>
              {onQuerySubmit && (
                <Button 
                  onClick={() => onQuerySubmit('')}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  Start Research
                </Button>
              )}
            </div>
          </Card>
        ) : isLoading && sequences.every(seq => seq.messages.length === 0) ? (
          <Card className="h-full flex items-center justify-center border-neutral-700 bg-neutral-800">
            <div className="text-center text-neutral-300">
              <Loader2 className="h-8 w-8 mx-auto mb-4 animate-spin" />
              <h3 className="text-lg font-medium mb-2">Initializing Sequences</h3>
              <p className="text-sm text-neutral-500">
                Starting 3 parallel research sequences...
              </p>
            </div>
          </Card>
        ) : (
          <>
            {isMobile ? <MobileLayout /> : <DesktopLayout />}
          </>
        )}
      </div>

      {/* Comparison summary */}
      {showComparisonSummary && allCompleted && (
        <div className="border-t border-neutral-700">
          <ComparisonSummary
            sequences={sequences}
            progress={progress}
            metrics={metrics}
          />
        </div>
      )}
    </div>
  );
}

export default ParallelChatGrid;