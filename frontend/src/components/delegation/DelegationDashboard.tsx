import React, { useState, useCallback } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  GitCompare,
  Pause,
  RotateCcw,
  TrendingUp,
  BarChart3,
  Activity,
  Info,
  Home,
  Wifi,
  WifiOff,
  AlertCircle,
} from 'lucide-react';

import { MetricsPanel } from './MetricsPanel';
import { SequenceColumn } from './SequenceColumn';
import {
  SequenceState,
  ParallelSequencesState,
  RealTimeMetrics,
  ConnectionState,
  SequenceStrategy,
} from '@/types/parallel';

interface DelegationDashboardProps {
  sequences: SequenceState[];
  progress: ParallelSequencesState;
  metrics: RealTimeMetrics;
  isLoading: boolean;
  error: Error | null;
  connectionState: ConnectionState;
  onReset: () => void;
  onStop: () => void;
  onRestart: () => void;
}

export function DelegationDashboard({
  sequences,
  progress,
  metrics,
  isLoading,
  error,
  connectionState,
  onReset,
  onStop,
  onRestart,
}: DelegationDashboardProps) {
  const [selectedTab, setSelectedTab] = useState<'overview' | 'sequences' | 'metrics'>('overview');

  // Calculate derived state
  const isComparisonActive = progress.status === 'running' || isLoading;

  const handlePauseComparison = useCallback(() => {
    onStop();
  }, [onStop]);

  const handleResetComparison = useCallback(() => {
    onReset();
  }, [onReset]);

  const handleRestartComparison = useCallback(() => {
    onRestart();
  }, [onRestart]);

  const formatStrategy = (strategy?: SequenceStrategy): string => {
    return strategy ? strategy.replace('_', ' ').toUpperCase() : 'UNKNOWN';
  };

  const getStrategyDescription = (strategy?: SequenceStrategy): string => {
    if (!strategy) return 'Unknown analysis pattern';
    switch (strategy) {
      case SequenceStrategy.THEORY_FIRST:
        return 'Academic → Industry → Technical analysis pattern';
      case SequenceStrategy.MARKET_FIRST:
        return 'Industry → Academic → Technical analysis pattern';
      case SequenceStrategy.FUTURE_BACK:
        return 'Technical → Academic → Industry analysis pattern';
      default:
        return 'Unknown analysis pattern';
    }
  };

  const getStrategyIcon = (strategy?: SequenceStrategy): React.JSX.Element => {
    if (!strategy) return <></>;  
    switch (strategy) {
      case SequenceStrategy.THEORY_FIRST:
        return <Activity className="h-4 w-4" />;
      case SequenceStrategy.MARKET_FIRST:
        return <TrendingUp className="h-4 w-4" />;
      case SequenceStrategy.FUTURE_BACK:
        return <BarChart3 className="h-4 w-4" />;
      default:
        return <GitCompare className="h-4 w-4" />;
    }
  };

  const getConnectionStatusIcon = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return <Wifi className="h-4 w-4 text-green-400" />;
      case ConnectionState.CONNECTING:
      case ConnectionState.RECONNECTING:
        return <Wifi className="h-4 w-4 text-yellow-400 animate-pulse" />;
      case ConnectionState.DISCONNECTED:
      case ConnectionState.FAILED:
        return <WifiOff className="h-4 w-4 text-red-400" />;
      default:
        return <WifiOff className="h-4 w-4 text-neutral-400" />;
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return 'Connected';
      case ConnectionState.CONNECTING:
        return 'Connecting';
      case ConnectionState.RECONNECTING:
        return 'Reconnecting';
      case ConnectionState.DISCONNECTED:
        return 'Disconnected';
      case ConnectionState.FAILED:
        return 'Failed';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className="h-full flex flex-col bg-neutral-800 text-neutral-100">
      {/* Header */}
      <div className="border-b border-neutral-700 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-neutral-100 flex items-center gap-2">
              <GitCompare className="h-6 w-6" />
              Parallel Research Dashboard
            </h1>
            <p className="text-neutral-300 mt-1">
              Real-time tracking of 3 concurrent research sequences with live metrics
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <Button
              onClick={handleResetComparison}
              variant="outline"
              className="border-neutral-600 text-neutral-300 hover:bg-neutral-600/10"
            >
              <Home className="h-4 w-4 mr-2" />
              New Query
            </Button>
            
            {isComparisonActive ? (
              <>
                <Button
                  onClick={handlePauseComparison}
                  variant="outline"
                  className="border-amber-600 text-amber-400 hover:bg-amber-600/10"
                >
                  <Pause className="h-4 w-4 mr-2" />
                  Stop
                </Button>
                <Button
                  onClick={handleRestartComparison}
                  variant="outline"
                  className="border-blue-600 text-blue-400 hover:bg-blue-600/10"
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Restart
                </Button>
              </>
            ) : null}
          </div>
        </div>
        
        {/* Status Bar */}
        <div className="flex items-center gap-4 mt-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-neutral-400">Connection:</span>
            {getConnectionStatusIcon()}
            <Badge 
              className={connectionState === ConnectionState.CONNECTED
                ? 'bg-green-500/20 text-green-300 border-green-500/30'
                : connectionState === ConnectionState.FAILED
                  ? 'bg-red-500/20 text-red-300 border-red-500/30'
                  : 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30'
              }
            >
              {getConnectionStatusText()}
            </Badge>
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-neutral-400">Status:</span>
            <Badge 
              className={progress.status === 'running'
                ? 'bg-green-500/20 text-green-300 border-green-500/30'
                : progress.status === 'completed'
                  ? 'bg-blue-500/20 text-blue-300 border-blue-500/30'
                  : progress.status === 'failed'
                    ? 'bg-red-500/20 text-red-300 border-red-500/30'
                    : 'bg-neutral-500/20 text-neutral-300 border-neutral-500/30'
              }
            >
              {progress.status.charAt(0).toUpperCase() + progress.status.slice(1)}
            </Badge>
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-neutral-400">Progress:</span>
            <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30">
              {Math.round(progress.overall_progress)}%
            </Badge>
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-neutral-400">Messages:</span>
            <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30">
              {progress.total_messages}
            </Badge>
          </div>
          
          {error && (
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-red-400" />
              <span className="text-sm text-red-400">Error: {error.message}</span>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <Tabs value={selectedTab} onValueChange={(value) => setSelectedTab(value as any)} className="h-full flex flex-col">
          <TabsList className="bg-neutral-700 border-b border-neutral-600 rounded-none justify-start p-1 m-6 mb-0">
            <TabsTrigger value="overview" className="data-[state=active]:bg-neutral-600">
              Overview
            </TabsTrigger>
            <TabsTrigger value="sequences" className="data-[state=active]:bg-neutral-600">
              Sequences
            </TabsTrigger>
            <TabsTrigger value="metrics" className="data-[state=active]:bg-neutral-600">
              Metrics
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="flex-1 p-6 overflow-hidden">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
              {/* Strategy Overview */}
              <div className="lg:col-span-2">
                <Card className="border-none rounded-lg bg-neutral-700 h-full flex flex-col">
                  <CardHeader>
                    <CardTitle className="text-neutral-100">Sequence Overview</CardTitle>
                    <CardDescription className="text-neutral-300">
                      Real-time status of parallel research sequences
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="flex-1 overflow-y-auto space-y-4">
                    {sequences.map((sequence) => {
                      const isActive = sequence.progress.status === 'active';
                      const isCompleted = sequence.progress.status === 'completed';
                      const isFailed = sequence.progress.status === 'failed';
                      
                      return (
                        <div
                          key={sequence.sequence_id}
                          className={`border rounded-lg p-4 transition-all ${
                            isCompleted
                              ? 'border-green-500/30 bg-green-500/5'
                              : isActive
                                ? 'border-blue-500/30 bg-blue-500/5'
                                : isFailed
                                  ? 'border-red-500/30 bg-red-500/5'
                                  : 'border-neutral-600 bg-neutral-800'
                          }`}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-2">
                                {getStrategyIcon(sequence.strategy)}
                                <h3 className="text-lg font-semibold text-neutral-100">
                                  {formatStrategy(sequence.strategy)}
                                </h3>
                                <Badge className={`text-xs ${
                                  isCompleted
                                    ? 'bg-green-500/20 text-green-300 border-green-500/30'
                                    : isActive
                                      ? 'bg-blue-500/20 text-blue-300 border-blue-500/30'
                                      : isFailed
                                        ? 'bg-red-500/20 text-red-300 border-red-500/30'
                                        : 'bg-neutral-500/20 text-neutral-300 border-neutral-500/30'
                                }`}>
                                  {sequence.progress.status.charAt(0).toUpperCase() + sequence.progress.status.slice(1)}
                                </Badge>
                              </div>
                              <p className="text-neutral-300 text-sm mb-3">
                                {getStrategyDescription(sequence.strategy)}
                              </p>
                              
                              {/* Progress and metrics */}
                              <div className="space-y-2">
                                <div className="flex items-center gap-4">
                                  <span className="text-xs font-medium text-neutral-400">Progress:</span>
                                  <div className="flex-1 bg-neutral-800 rounded-full h-2">
                                    <div
                                      className={`h-2 rounded-full transition-all ${
                                        isCompleted ? 'bg-green-500' : isActive ? 'bg-blue-500' : 'bg-neutral-600'
                                      }`}
                                      style={{ width: `${sequence.progress.completion_percentage}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-xs text-neutral-300">
                                    {Math.round(sequence.progress.completion_percentage)}%
                                  </span>
                                </div>
                                
                                <div className="grid grid-cols-3 gap-4 text-xs">
                                  <div>
                                    <span className="text-neutral-400">Agent:</span>
                                    <span className="text-neutral-300 ml-1">
                                      {sequence.current_agent || 'None'}
                                    </span>
                                  </div>
                                  <div>
                                    <span className="text-neutral-400">Messages:</span>
                                    <span className="text-neutral-300 ml-1">
                                      {sequence.progress.messages_received}
                                    </span>
                                  </div>
                                  <div>
                                    <span className="text-neutral-400">Errors:</span>
                                    <span className="text-neutral-300 ml-1">
                                      {sequence.errors.length}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </CardContent>
                </Card>
              </div>

              {/* Quick Metrics */}
              <div>
                <MetricsPanel
                  sequences={sequences}
                  metrics={metrics}
                  isLoading={isLoading}
                />
              </div>
            </div>
          </TabsContent>

          {/* Sequences Tab */}
          <TabsContent value="sequences" className="flex-1 p-6 overflow-hidden">
            {sequences.length === 0 ? (
              <Card className="border-none rounded-lg bg-neutral-700 h-full flex items-center justify-center">
                <div className="text-center text-neutral-500">
                  <GitCompare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <h3 className="text-lg font-medium mb-2">No Active Sequences</h3>
                  <p className="text-sm text-neutral-600 mb-4">
                    Start a new research query to see parallel sequence progression
                  </p>
                  <Button
                    onClick={handleResetComparison}
                    className="bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    <Home className="h-4 w-4 mr-2" />
                    New Query
                  </Button>
                </div>
              </Card>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
                {sequences.map((sequence) => (
                  <SequenceColumn
                    key={sequence.sequence_id}
                    sequenceId={sequence.sequence_id}
                    sequence={sequence}
                    isActive={isComparisonActive}
                    isLoading={isLoading}
                  />
                ))}
              </div>
            )}
          </TabsContent>

          {/* Metrics Tab */}
          <TabsContent value="metrics" className="flex-1 p-6 overflow-hidden">
            <MetricsPanel
              sequences={sequences}
              metrics={metrics}
              isLoading={isLoading}
            />
          </TabsContent>
        </Tabs>
      </div>

      {/* Info Banner for No Data */}
      {sequences.length === 0 && !isLoading && (
        <div className="border-t border-neutral-700 p-4">
          <div className="flex items-center gap-2 text-neutral-400">
            <Info className="h-4 w-4" />
            <span className="text-sm">
              Start a new research query to see real-time parallel sequence analysis
            </span>
          </div>
        </div>
      )}
    </div>
  );
}