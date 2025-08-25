import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Activity,
  TrendingUp,
  Clock,
  Target,
  BarChart3,
  Wifi,
  MessageSquare,
} from 'lucide-react';
import { 
  SequenceState,
  RealTimeMetrics,
  SequenceStrategy,
} from '@/types/parallel';

interface MetricsPanelProps {
  sequences: SequenceState[];
  metrics: RealTimeMetrics;
  isLoading: boolean;
}

export function MetricsPanel({
  sequences,
  metrics,
  isLoading,
}: MetricsPanelProps) {
  // Calculate derived metrics
  const totalMessages = sequences.reduce((sum, seq) => sum + seq.messages.length, 0);
  const avgCompletion = sequences.length > 0 
    ? sequences.reduce((sum, seq) => sum + seq.progress.completion_percentage, 0) / sequences.length 
    : 0;
  
  const bestPerformingSequence = sequences
    .filter(seq => seq.metrics.efficiency_score !== undefined)
    .sort((a, b) => (b.metrics.efficiency_score || 0) - (a.metrics.efficiency_score || 0))[0];
    
  const activeSequences = sequences.filter(seq => seq.progress.status === 'active').length;

  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    const seconds = ms / 1000;
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    return `${(seconds / 60).toFixed(1)}m`;
  };

  const formatStrategy = (strategy?: SequenceStrategy): string => {
    return strategy ? strategy.replace('_', ' ').toUpperCase() : 'UNKNOWN';
  };

  const getStrategyColor = (strategy?: SequenceStrategy): string => {
    if (!strategy) return 'bg-neutral-500/20 text-neutral-300 border-neutral-500/30';
    switch (strategy) {
      case SequenceStrategy.THEORY_FIRST:
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      case SequenceStrategy.MARKET_FIRST:
        return 'bg-green-500/20 text-green-300 border-green-500/30';
      case SequenceStrategy.FUTURE_BACK:
        return 'bg-purple-500/20 text-purple-300 border-purple-500/30';
      default:
        return 'bg-neutral-500/20 text-neutral-300 border-neutral-500/30';
    }
  };

  const getHealthColor = (health: number): string => {
    if (health >= 80) return 'text-green-400';
    if (health >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <Card className="border-none rounded-lg bg-neutral-700 h-full flex flex-col">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-neutral-100">
          <BarChart3 className="h-5 w-5" />
          Real-time Metrics
        </CardTitle>
        <CardDescription className="text-neutral-300">
          Real-time metrics and parallel sequence performance
        </CardDescription>
      </CardHeader>

      <ScrollArea className="flex-1">
        <CardContent className="space-y-6">
          {/* Real-time Connection Metrics */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-neutral-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Wifi className={`h-4 w-4 ${getHealthColor(metrics.connection_health)}`} />
                <span className="text-xs text-neutral-400">Connection Health</span>
              </div>
              <div className="text-lg font-semibold text-neutral-100">
                {metrics.connection_health.toFixed(0)}%
              </div>
            </div>

            <div className="bg-neutral-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <MessageSquare className="h-4 w-4 text-blue-400" />
                <span className="text-xs text-neutral-400">Messages/sec</span>
              </div>
              <div className="text-lg font-semibold text-neutral-100">
                {metrics.messages_per_second.toFixed(1)}
              </div>
            </div>

            <div className="bg-neutral-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="h-4 w-4 text-amber-400" />
                <span className="text-xs text-neutral-400">Avg Latency</span>
              </div>
              <div className="text-lg font-semibold text-neutral-100">
                {formatDuration(metrics.average_latency)}
              </div>
            </div>

            <div className="bg-neutral-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="h-4 w-4 text-green-400" />
                <span className="text-xs text-neutral-400">Throughput</span>
              </div>
              <div className="text-lg font-semibold text-neutral-100">
                {metrics.throughput_efficiency.toFixed(1)}%
              </div>
            </div>
          </div>

          <Separator className="bg-neutral-600" />

          {/* Overall Performance Summary */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-neutral-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="h-4 w-4 text-blue-400" />
                <span className="text-xs text-neutral-400">Active Sequences</span>
              </div>
              <div className="text-lg font-semibold text-neutral-100">
                {activeSequences}/{sequences.length}
              </div>
            </div>

            <div className="bg-neutral-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="h-4 w-4 text-green-400" />
                <span className="text-xs text-neutral-400">Avg Progress</span>
              </div>
              <div className="text-lg font-semibold text-neutral-100">
                {avgCompletion.toFixed(0)}%
              </div>
            </div>

            <div className="bg-neutral-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <MessageSquare className="h-4 w-4 text-purple-400" />
                <span className="text-xs text-neutral-400">Total Messages</span>
              </div>
              <div className="text-lg font-semibold text-neutral-100">
                {totalMessages}
              </div>
            </div>
          </div>

          <Separator className="bg-neutral-600" />

          {/* Individual Sequence Metrics */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-neutral-200 flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Sequence Performance
            </h4>

            {sequences.length === 0 && !isLoading && (
              <div className="text-center py-8 text-neutral-500">
                <BarChart3 className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No active sequences</p>
                <p className="text-xs text-neutral-600 mt-1">
                  Start a new research query to see metrics
                </p>
              </div>
            )}

            {isLoading && sequences.length === 0 && (
              <div className="text-center py-8 text-neutral-500">
                <Activity className="h-8 w-8 mx-auto mb-2 animate-spin opacity-50" />
                <p className="text-sm">Initializing sequences...</p>
              </div>
            )}

            {sequences.map((sequence) => (
              <div
                key={sequence.sequence_id}
                className="bg-neutral-800 rounded-lg p-4 space-y-3"
              >
                <div className="flex items-center justify-between">
                  <Badge className={`text-xs ${getStrategyColor(sequence.strategy)}`}>
                    {formatStrategy(sequence.strategy)}
                  </Badge>
                  {sequence.progress.status === 'active' && (
                    <div className="flex items-center gap-1">
                      <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse" />
                      <span className="text-xs text-green-400">Active</span>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="space-y-1">
                    <span className="text-neutral-400">Progress</span>
                    <div className="font-medium text-neutral-200">
                      {sequence.progress.completion_percentage.toFixed(0)}%
                    </div>
                  </div>

                  <div className="space-y-1">
                    <span className="text-neutral-400">Current Agent</span>
                    <div className="font-medium text-neutral-200">
                      {sequence.current_agent || 'None'}
                    </div>
                  </div>

                  <div className="space-y-1">
                    <span className="text-neutral-400">Messages</span>
                    <div className="font-medium text-neutral-200">
                      {sequence.messages.length}
                    </div>
                  </div>

                  <div className="space-y-1">
                    <span className="text-neutral-400">Errors</span>
                    <div className={`font-medium ${
                      sequence.errors.length > 0 ? 'text-red-300' : 'text-neutral-200'
                    }`}>
                      {sequence.errors.length}
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3 text-neutral-400" />
                      <span className="text-neutral-400">Processing Time</span>
                    </div>
                    <div className="font-medium text-neutral-200">
                      {formatDuration(sequence.metrics.research_duration)}
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center gap-1">
                      <TrendingUp className="h-3 w-3 text-neutral-400" />
                      <span className="text-neutral-400">Quality Score</span>
                    </div>
                    <div className="font-medium text-neutral-200">
                      {sequence.metrics.quality_score 
                        ? `${(sequence.metrics.quality_score * 100).toFixed(0)}%`
                        : 'N/A'
                      }
                    </div>
                  </div>
                </div>

                {/* Progress Indicator */}
                <div className="mt-3">
                  <div className="flex justify-between text-xs text-neutral-400 mb-1">
                    <span>Completion</span>
                    <span>{sequence.progress.completion_percentage.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-neutral-700 rounded-full h-1.5">
                    <div
                      className={`h-1.5 rounded-full transition-all duration-300 ${
                        sequence.progress.status === 'completed' ? 'bg-green-400' :
                        sequence.progress.status === 'active' ? 'bg-blue-400' :
                        sequence.progress.status === 'failed' ? 'bg-red-400' : 'bg-neutral-600'
                      }`}
                      style={{
                        width: `${sequence.progress.completion_percentage}%`
                      }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Performance Insights */}
          {sequences.length > 1 && (
            <>
              <Separator className="bg-neutral-600" />
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-neutral-200">
                  Performance Insights
                </h4>
                
                {bestPerformingSequence && (
                  <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <TrendingUp className="h-4 w-4 text-green-400" />
                      <span className="text-sm font-medium text-green-300">
                        Top Performer
                      </span>
                    </div>
                    <p className="text-xs text-green-200">
                      {formatStrategy(bestPerformingSequence.strategy)} sequence is leading with
                      {bestPerformingSequence.progress.completion_percentage.toFixed(0)}% completion
                    </p>
                  </div>
                )}

                {/* System Health */}
                <div className={`border rounded-lg p-3 ${
                  metrics.connection_health >= 80
                    ? 'bg-green-500/10 border-green-500/20'
                    : metrics.connection_health >= 60
                      ? 'bg-yellow-500/10 border-yellow-500/20'
                      : 'bg-red-500/10 border-red-500/20'
                }`}>
                  <div className="flex items-center gap-2 mb-1">
                    <Wifi className={`h-4 w-4 ${getHealthColor(metrics.connection_health)}`} />
                    <span className="text-sm font-medium">
                      System Health
                    </span>
                  </div>
                  <p className="text-xs">
                    Connection health: {metrics.connection_health.toFixed(0)}% |
                    Error rate: {metrics.error_rate.toFixed(1)}% |
                    Buffer utilization: {metrics.buffer_utilization.toFixed(0)}%
                  </p>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </ScrollArea>
    </Card>
  );
}