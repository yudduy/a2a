/**
 * LiveMetricsBar - Real-time metrics display for parallel sequences
 * 
 * Features:
 * - Live connection health and status
 * - Real-time performance metrics
 * - Sequence progress indicators
 * - Error rate and latency tracking
 * - Responsive design with collapsible details
 */

import React, { useState, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import {
  Wifi,
  WifiOff,
  Activity,
  Clock,
  AlertTriangle,
  TrendingUp,
  Gauge,
  MessageSquare,
  Zap,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  Loader2,
  BarChart3,
  Signal,
} from 'lucide-react';

import { 
  ParallelSequencesState, 
  RealTimeMetrics, 
  ConnectionState 
} from '@/types/parallel';
import { cn } from '@/lib/utils';

interface LiveMetricsBarProps {
  progress: ParallelSequencesState;
  metrics: RealTimeMetrics;
  connectionState: ConnectionState;
  isLoading: boolean;
  error?: Error | null;
  className?: string;
  showDetails?: boolean;
}

export function LiveMetricsBar({
  progress,
  metrics,
  connectionState,
  isLoading,
  error,
  className,
  showDetails = false,
}: LiveMetricsBarProps) {
  const [showDetailedMetrics, setShowDetailedMetrics] = useState(showDetails);
  const [animatedProgress, setAnimatedProgress] = useState(0);

  // Animate progress changes
  useEffect(() => {
    const targetProgress = progress.overall_progress;
    const current = animatedProgress;
    const diff = targetProgress - current;
    
    if (Math.abs(diff) > 0.1) {
      const increment = diff > 0 ? Math.min(diff * 0.1, 2) : Math.max(diff * 0.1, -2);
      const timer = setTimeout(() => {
        setAnimatedProgress(prev => prev + increment);
      }, 16); // ~60fps
      
      return () => clearTimeout(timer);
    }
  }, [progress.overall_progress, animatedProgress]);

  // Connection status helpers
  const getConnectionIcon = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return <Wifi className="h-4 w-4 text-green-400" />;
      case ConnectionState.CONNECTING:
      case ConnectionState.RECONNECTING:
        return <Wifi className="h-4 w-4 text-yellow-400 animate-pulse" />;
      case ConnectionState.FAILED:
      case ConnectionState.DISCONNECTED:
        return <WifiOff className="h-4 w-4 text-red-400" />;
      default:
        return <WifiOff className="h-4 w-4 text-neutral-400" />;
    }
  };

  const getConnectionStatus = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return 'Connected';
      case ConnectionState.CONNECTING:
        return 'Connecting';
      case ConnectionState.RECONNECTING:
        return 'Reconnecting';
      case ConnectionState.FAILED:
        return 'Failed';
      case ConnectionState.DISCONNECTED:
        return 'Disconnected';
      default:
        return 'Unknown';
    }
  };

  const getConnectionColor = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return 'bg-green-500/20 text-green-300 border-green-500/30';
      case ConnectionState.CONNECTING:
      case ConnectionState.RECONNECTING:
        return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
      case ConnectionState.FAILED:
      case ConnectionState.DISCONNECTED:
        return 'bg-red-500/20 text-red-300 border-red-500/30';
      default:
        return 'bg-neutral-500/20 text-neutral-300 border-neutral-500/30';
    }
  };

  // Status helpers
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500/20 text-green-300 border-green-500/30';
      case 'running':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      case 'failed':
        return 'bg-red-500/20 text-red-300 border-red-500/30';
      default:
        return 'bg-neutral-500/20 text-neutral-300 border-neutral-500/30';
    }
  };

  const formatLatency = (latency: number) => {
    if (latency < 1000) return `${Math.round(latency)}ms`;
    return `${(latency / 1000).toFixed(1)}s`;
  };

  const formatDuration = (startTime: number) => {
    const duration = (Date.now() - startTime) / 1000;
    if (duration < 60) return `${Math.floor(duration)}s`;
    if (duration < 3600) return `${Math.floor(duration / 60)}m ${Math.floor(duration % 60)}s`;
    return `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`;
  };

  const getHealthColor = (health: number) => {
    if (health >= 80) return 'text-green-400';
    if (health >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className={cn("bg-neutral-800/50 backdrop-blur-sm border-t border-neutral-700", className)}>
      {/* Main metrics bar */}
      <div className="flex items-center gap-6 p-3">
        {/* Connection status */}
        <div className="flex items-center gap-2">
          {getConnectionIcon()}
          <Badge className={getConnectionColor()}>
            {getConnectionStatus()}
          </Badge>
        </div>

        {/* Overall status */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            {progress.status === 'running' && (
              <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />
            )}
            {progress.status === 'completed' && (
              <CheckCircle className="h-4 w-4 text-green-400" />
            )}
            {progress.status === 'failed' && (
              <AlertTriangle className="h-4 w-4 text-red-400" />
            )}
            {progress.status === 'initializing' && (
              <Activity className="h-4 w-4 text-neutral-400" />
            )}
          </div>
          <Badge className={getStatusColor(progress.status)}>
            {progress.status.charAt(0).toUpperCase() + progress.status.slice(1)}
          </Badge>
        </div>

        {/* Progress */}
        <div className="flex items-center gap-2 min-w-[120px]">
          <div className="flex-1">
            <Progress 
              value={animatedProgress} 
              className="h-2 bg-neutral-700"
            />
          </div>
          <span className="text-sm font-medium text-neutral-300 min-w-[40px]">
            {Math.round(animatedProgress)}%
          </span>
        </div>

        {/* Active sequences */}
        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-blue-400" />
          <span className="text-sm text-neutral-300">
            {progress.active_sequences}/{progress.sequences.length}
          </span>
        </div>

        {/* Total messages */}
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4 text-purple-400" />
          <span className="text-sm text-neutral-300">
            {progress.total_messages}
          </span>
        </div>

        {/* Duration */}
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-neutral-400" />
          <span className="text-sm text-neutral-300">
            {formatDuration(progress.start_time)}
          </span>
        </div>

        {/* Error indicator */}
        {error && (
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-red-400" />
            <Badge className="bg-red-500/20 text-red-300 border-red-500/30">
              Error
            </Badge>
          </div>
        )}

        {/* Expand/collapse button */}
        <div className="ml-auto">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowDetailedMetrics(!showDetailedMetrics)}
            className="text-neutral-400 hover:text-neutral-200"
          >
            {showDetailedMetrics ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Detailed metrics */}
      {showDetailedMetrics && (
        <>
          <Separator className="bg-neutral-700" />
          <div className="p-4 space-y-4">
            {/* Performance metrics */}
            <div>
              <h4 className="text-sm font-medium text-neutral-200 mb-3 flex items-center gap-2">
                <Gauge className="h-4 w-4" />
                Performance Metrics
              </h4>
              
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {/* Messages per second */}
                <div className="bg-neutral-700/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Activity className="h-3 w-3 text-blue-400" />
                    <span className="text-xs text-neutral-400">Msg/sec</span>
                  </div>
                  <span className="text-lg font-medium text-neutral-200">
                    {metrics.messages_per_second.toFixed(1)}
                  </span>
                </div>

                {/* Average latency */}
                <div className="bg-neutral-700/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Clock className="h-3 w-3 text-yellow-400" />
                    <span className="text-xs text-neutral-400">Latency</span>
                  </div>
                  <span className="text-lg font-medium text-neutral-200">
                    {formatLatency(metrics.average_latency)}
                  </span>
                </div>

                {/* Connection health */}
                <div className="bg-neutral-700/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Signal className="h-3 w-3 text-green-400" />
                    <span className="text-xs text-neutral-400">Health</span>
                  </div>
                  <span className={cn("text-lg font-medium", getHealthColor(metrics.connection_health))}>
                    {Math.round(metrics.connection_health)}%
                  </span>
                </div>

                {/* Buffer utilization */}
                <div className="bg-neutral-700/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Zap className="h-3 w-3 text-purple-400" />
                    <span className="text-xs text-neutral-400">Buffer</span>
                  </div>
                  <span className="text-lg font-medium text-neutral-200">
                    {Math.round(metrics.buffer_utilization)}%
                  </span>
                </div>

                {/* Error rate */}
                <div className="bg-neutral-700/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <AlertTriangle className="h-3 w-3 text-red-400" />
                    <span className="text-xs text-neutral-400">Errors</span>
                  </div>
                  <span className="text-lg font-medium text-neutral-200">
                    {metrics.error_rate.toFixed(1)}%
                  </span>
                </div>

                {/* Throughput efficiency */}
                <div className="bg-neutral-700/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <TrendingUp className="h-3 w-3 text-blue-400" />
                    <span className="text-xs text-neutral-400">Efficiency</span>
                  </div>
                  <span className="text-lg font-medium text-neutral-200">
                    {Math.round(metrics.throughput_efficiency)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Sequence status */}
            <div>
              <h4 className="text-sm font-medium text-neutral-200 mb-3 flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Sequence Status
              </h4>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {progress.sequences.map((sequence) => {
                  const strategyName = sequence.strategy.replace('_', ' ').split(' ')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
                  
                  return (
                    <div key={sequence.sequence_id} className="bg-neutral-700/50 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-neutral-200">
                          {strategyName}
                        </span>
                        <Badge className={getStatusColor(sequence.progress.status)} variant="outline">
                          {sequence.progress.status}
                        </Badge>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-neutral-400">Progress</span>
                          <span className="text-neutral-300">{Math.round(sequence.progress.completion_percentage)}%</span>
                        </div>
                        <Progress 
                          value={sequence.progress.completion_percentage} 
                          className="h-1.5 bg-neutral-600"
                        />
                        
                        <div className="flex justify-between text-xs text-neutral-400">
                          <span>{sequence.messages.length} msgs</span>
                          <span>{sequence.current_agent || 'Idle'}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Error details */}
            {error && (
              <div>
                <h4 className="text-sm font-medium text-red-300 mb-2 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Error Details
                </h4>
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                  <p className="text-sm text-red-200">{error.message}</p>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default LiveMetricsBar;