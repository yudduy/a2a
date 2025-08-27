import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Trophy, Activity, TrendingUp, Zap, CheckCircle, Clock, MessageSquare, Wrench } from 'lucide-react';
import { SequenceState, SequenceStrategy } from '@/types/parallel';
import { cn } from '@/lib/utils';

interface SimpleMetricsComparisonProps {
  sequences: SequenceState[];
  onNewQuery: () => void;
  className?: string;
}

interface SequenceScore {
  sequence: SequenceState;
  score: number;
  productivity: number;
  efficiency: number;
  reliability: number;
}

const SimpleMetricsComparison: React.FC<SimpleMetricsComparisonProps> = ({
  sequences,
  onNewQuery,
  className
}) => {
  // Strategy display configuration
  const getStrategyConfig = (strategy: SequenceStrategy) => {
    switch (strategy) {
      case SequenceStrategy.THEORY_FIRST:
        return { 
          name: 'Theory First', 
          color: 'bg-blue-100 text-blue-800 border-blue-200',
          bgColor: 'bg-blue-50 border-blue-200'
        };
      case SequenceStrategy.MARKET_FIRST:
        return { 
          name: 'Market First', 
          color: 'bg-green-100 text-green-800 border-green-200',
          bgColor: 'bg-green-50 border-green-200'
        };
      case SequenceStrategy.FUTURE_BACK:
        return { 
          name: 'Future Back', 
          color: 'bg-purple-100 text-purple-800 border-purple-200',
          bgColor: 'bg-purple-50 border-purple-200'
        };
      default:
        return { 
          name: 'Unknown', 
          color: 'bg-gray-100 text-gray-800 border-gray-200',
          bgColor: 'bg-gray-50 border-gray-200'
        };
    }
  };

  // Calculate comprehensive scores for each sequence
  const calculateSequenceScores = (): SequenceScore[] => {
    return sequences.map(sequence => {
      const { metrics, start_time, end_time, errors } = sequence;
      
      // Productivity: Messages and insights generated
      const productivity = Math.min(100, (metrics.message_count * 2) + ((metrics.insights_generated || 0) * 5));
      
      // Efficiency: Speed and processing time (lower is better, so invert)
      const processingTimeSeconds = metrics.research_duration / 1000;
      const efficiency = Math.max(0, 100 - Math.min(100, processingTimeSeconds / 60 * 10)); // Penalty for >6 minutes
      
      // Reliability: Error rate and completion status
      const errorCount = errors.length;
      const isCompleted = sequence.progress.status === 'completed';
      const reliability = isCompleted ? Math.max(0, 100 - (errorCount * 20)) : 50;
      
      // Overall weighted score
      const score = (productivity * 0.4) + (efficiency * 0.35) + (reliability * 0.25);
      
      return {
        sequence,
        score: Math.round(score),
        productivity: Math.round(productivity),
        efficiency: Math.round(efficiency),
        reliability: Math.round(reliability)
      };
    });
  };

  // Get winner (highest score)
  const sequenceScores = calculateSequenceScores();
  const winner = sequenceScores.reduce((prev, current) => 
    prev.score > current.score ? prev : current
  );

  // Format completion time
  const formatDuration = (startTime: number, endTime?: number): string => {
    if (!endTime) return 'In progress...';
    const durationMs = endTime - startTime;
    const minutes = Math.floor(durationMs / 60000);
    const seconds = Math.floor((durationMs % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  // Get completed sequences for comparison
  const completedSequences = sequenceScores.filter(s => 
    s.sequence.progress.status === 'completed'
  );

  const otherSequences = sequenceScores.filter(s => s.sequence.sequence_id !== winner.sequence.sequence_id);

  return (
    <div className={cn("w-full max-w-4xl mx-auto space-y-6", className)}>
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-2 text-2xl font-bold text-foreground">
          <Trophy className="w-7 h-7 text-yellow-500" />
          Research Complete
        </div>
        <p className="text-muted-foreground">
          All sequences have finished processing. Here's how they performed:
        </p>
      </div>

      {/* Winner Section */}
      <Card className={cn("relative overflow-hidden", getStrategyConfig(winner.sequence.strategy || SequenceStrategy.THEORY_FIRST).bgColor)}>
        <CardHeader className="relative">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Trophy className="w-5 h-5 text-yellow-500" />
                <CardTitle className="text-lg">Winner</CardTitle>
                <Badge className={getStrategyConfig(winner.sequence.strategy || SequenceStrategy.THEORY_FIRST).color}>
                  {getStrategyConfig(winner.sequence.strategy || SequenceStrategy.THEORY_FIRST).name}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                Best overall performance with {winner.score}% efficiency score
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-blue-600" />
              <div className="text-sm">
                <div className="font-medium">{winner.sequence.metrics.message_count}</div>
                <div className="text-muted-foreground">Messages</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-green-600" />
              <div className="text-sm">
                <div className="font-medium">
                  {formatDuration(winner.sequence.start_time, winner.sequence.end_time)}
                </div>
                <div className="text-muted-foreground">Duration</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Wrench className="w-4 h-4 text-purple-600" />
              <div className="text-sm">
                <div className="font-medium">{winner.sequence.metrics.agent_calls}</div>
                <div className="text-muted-foreground">Tool calls</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-emerald-600" />
              <div className="text-sm">
                <div className="font-medium">{winner.sequence.errors.length}</div>
                <div className="text-muted-foreground">Errors</div>
              </div>
            </div>
          </div>
          
          {/* Performance bars */}
          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>Productivity</span>
              <span className="font-medium">{winner.productivity}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${winner.productivity}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-sm">
              <span>Efficiency</span>
              <span className="font-medium">{winner.efficiency}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${winner.efficiency}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-sm">
              <span>Reliability</span>
              <span className="font-medium">{winner.reliability}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-purple-500 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${winner.reliability}%` }}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Other Results */}
      {otherSequences.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-foreground">Other Results</h3>
          <div className="grid gap-3 md:grid-cols-2">
            {otherSequences.map(({ sequence, score, productivity, efficiency, reliability }) => (
              <Card key={sequence.sequence_id} className="relative">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge 
                        variant="outline" 
                        className={getStrategyConfig(sequence.strategy || SequenceStrategy.THEORY_FIRST).color}
                      >
                        {getStrategyConfig(sequence.strategy || SequenceStrategy.THEORY_FIRST).name}
                      </Badge>
                      <span className="text-sm text-muted-foreground">
                        {score}% score
                      </span>
                    </div>
                    {sequence.progress.status === 'completed' ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <Activity className="w-4 h-4 text-yellow-500" />
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-3 text-sm">
                    <div className="text-center">
                      <div className="font-medium">{sequence.metrics.message_count}</div>
                      <div className="text-muted-foreground">Messages</div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium">
                        {formatDuration(sequence.start_time, sequence.end_time)}
                      </div>
                      <div className="text-muted-foreground">Duration</div>
                    </div>
                    <div className="text-center">
                      <div className="font-medium">{sequence.errors.length}</div>
                      <div className="text-muted-foreground">Errors</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Summary Stats */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Research Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="text-center">
              <div className="font-semibold text-lg">
                {sequenceScores.reduce((sum, s) => sum + s.sequence.metrics.message_count, 0)}
              </div>
              <div className="text-muted-foreground">Total Messages</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-lg">
                {completedSequences.length}/{sequences.length}
              </div>
              <div className="text-muted-foreground">Completed</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-lg">
                {sequenceScores.reduce((sum, s) => sum + (s.sequence.metrics.agent_calls || 0), 0)}
              </div>
              <div className="text-muted-foreground">Tool Calls</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-lg">
                {Math.round(sequenceScores.reduce((sum, s) => sum + s.score, 0) / sequenceScores.length)}%
              </div>
              <div className="text-muted-foreground">Avg Score</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Action Button */}
      <div className="flex justify-center pt-4">
        <Button onClick={onNewQuery} size="lg" className="px-8">
          <Zap className="w-4 h-4 mr-2" />
          New Query
        </Button>
      </div>
    </div>
  );
};

export default SimpleMetricsComparison;