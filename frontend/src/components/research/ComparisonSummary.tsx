/**
 * ComparisonSummary - Final winner analysis and sequence comparison
 * 
 * Features:
 * - Comparative analysis of all 3 sequences
 * - Winner declaration and reasoning
 * - Performance metrics comparison
 * - Key insights extraction
 * - Downloadable summary report
 * - Visual comparison charts
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import {
  Trophy,
  Activity,
  TrendingUp,
  Zap,
  BarChart3,
  Download,
  Star,
  Clock,
  MessageSquare,
  Target,
  Lightbulb,
  Award,
  CheckCircle,
  AlertCircle,
  Copy,
  CopyCheck,
  Share2,
  FileText,
  TrendingDown,
  Users,
  Brain,
  Gauge,
} from 'lucide-react';

import { 
  SequenceState, 
  SequenceStrategy, 
  ParallelSequencesState, 
  RealTimeMetrics 
} from '@/types/parallel';
import { cn } from '@/lib/utils';

interface ComparisonSummaryProps {
  sequences: SequenceState[];
  progress: ParallelSequencesState;
  metrics: RealTimeMetrics;
  className?: string;
}

interface SequenceAnalysis {
  sequence: SequenceState;
  score: number;
  strengths: string[];
  weaknesses: string[];
  keyInsights: string[];
  efficiency: number;
  completeness: number;
  quality: number;
  rank: number;
}

export function ComparisonSummary({
  sequences,
  progress,
  metrics,
  className,
}: ComparisonSummaryProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'details' | 'export'>('overview');
  const [copiedContent, setCopiedContent] = useState<string | null>(null);

  // Strategy configurations
  const strategyConfig = {
    [SequenceStrategy.THEORY_FIRST]: {
      title: 'Theory First',
      icon: <Activity className="h-4 w-4" />,
      color: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
      description: 'Academic → Industry → Technical'
    },
    [SequenceStrategy.MARKET_FIRST]: {
      title: 'Market First',
      icon: <TrendingUp className="h-4 w-4" />,
      color: 'bg-green-500/20 text-green-300 border-green-500/30',
      description: 'Industry → Academic → Technical'
    },
    [SequenceStrategy.FUTURE_BACK]: {
      title: 'Future Back',
      icon: <Zap className="h-4 w-4" />,
      color: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
      description: 'Technical → Academic → Industry'
    },
  };

  // Analyze sequences and calculate scores
  const analysis = useMemo((): SequenceAnalysis[] => {
    const analyses: SequenceAnalysis[] = sequences.map(sequence => {
      // Calculate metrics
      const completeness = sequence.progress.completion_percentage;
      const messageCount = sequence.messages.length;
      const errorCount = sequence.errors.length;
      const processingTime = sequence.end_time ? 
        (sequence.end_time - sequence.start_time) / 1000 : 
        (Date.now() - sequence.start_time) / 1000;

      // Calculate efficiency (messages per minute with error penalty)
      const efficiency = Math.max(0, (messageCount / (processingTime / 60)) - (errorCount * 0.5));
      
      // Calculate quality score (completeness with error penalty)
      const quality = Math.max(0, completeness - (errorCount * 5));
      
      // Overall score (weighted average)
      const score = (completeness * 0.4) + (efficiency * 0.3) + (quality * 0.3);

      // Generate insights based on strategy and performance
      const keyInsights = generateKeyInsights(sequence);
      const strengths = generateStrengths(sequence);
      const weaknesses = generateWeaknesses(sequence);

      return {
        sequence,
        score,
        efficiency,
        completeness,
        quality,
        keyInsights,
        strengths,
        weaknesses,
        rank: 0, // Will be set after sorting
      };
    });

    // Sort by score and assign ranks
    analyses.sort((a, b) => b.score - a.score);
    analyses.forEach((analysis, index) => {
      analysis.rank = index + 1;
    });

    return analyses;
  }, [sequences]);

  // Generate insights based on sequence performance
  function generateKeyInsights(sequence: SequenceState): string[] {
    const insights: string[] = [];
    const messageCount = sequence.messages.length;
    const errorCount = sequence.errors.length;
    const completion = sequence.progress.completion_percentage;

    if (completion === 100) {
      insights.push('Successfully completed full research sequence');
    }
    if (messageCount > 20) {
      insights.push('Generated comprehensive analysis with rich detail');
    }
    if (errorCount === 0) {
      insights.push('Executed without errors - high reliability');
    }
    if (sequence.agent_transitions.length >= 2) {
      insights.push('Successful multi-agent collaboration and knowledge transfer');
    }

    // Strategy-specific insights
    switch (sequence.strategy) {
      case SequenceStrategy.THEORY_FIRST:
        insights.push('Strong academic foundation with theoretical depth');
        break;
      case SequenceStrategy.MARKET_FIRST:
        insights.push('Practical market-driven approach with actionable insights');
        break;
      case SequenceStrategy.FUTURE_BACK:
        insights.push('Forward-thinking analysis with innovation focus');
        break;
    }

    return insights.slice(0, 4);
  }

  function generateStrengths(sequence: SequenceState): string[] {
    const strengths: string[] = [];
    const messageCount = sequence.messages.length;
    const errorCount = sequence.errors.length;

    if (sequence.progress.completion_percentage > 90) {
      strengths.push('High completion rate');
    }
    if (messageCount > 15) {
      strengths.push('Rich content generation');
    }
    if (errorCount <= 1) {
      strengths.push('Reliable execution');
    }
    if (sequence.agent_transitions.length >= 2) {
      strengths.push('Effective agent coordination');
    }

    return strengths.slice(0, 3);
  }

  function generateWeaknesses(sequence: SequenceState): string[] {
    const weaknesses: string[] = [];
    const messageCount = sequence.messages.length;
    const errorCount = sequence.errors.length;

    if (sequence.progress.completion_percentage < 80) {
      weaknesses.push('Lower completion rate');
    }
    if (messageCount < 10) {
      weaknesses.push('Limited content generation');
    }
    if (errorCount > 2) {
      weaknesses.push('Multiple execution errors');
    }
    if (sequence.progress.status === 'failed') {
      weaknesses.push('Failed to complete sequence');
    }

    return weaknesses.slice(0, 2);
  }

  const winner = analysis[0];
  const totalDuration = (Date.now() - progress.start_time) / 1000;

  // Export functions
  const handleCopy = async (content: string, type: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedContent(type);
      setTimeout(() => setCopiedContent(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const generateSummaryText = () => {
    const winnerConfig = strategyConfig[winner.sequence.strategy];
    return `# Parallel Research Analysis Summary

## Query: ${progress.research_query}

## Winner: ${winnerConfig.title} Strategy
**Score:** ${winner.score.toFixed(1)}/100
**Completion:** ${winner.completeness.toFixed(1)}%
**Efficiency:** ${winner.efficiency.toFixed(1)} msg/min

### Key Insights:
${winner.keyInsights.map(insight => `- ${insight}`).join('\n')}

### Performance Comparison:
${analysis.map(a => {
  const config = strategyConfig[a.sequence.strategy];
  return `**${config.title}:** ${a.score.toFixed(1)}/100 (${a.completeness.toFixed(1)}% complete)`;
}).join('\n')}

### Analysis Details:
- Total Duration: ${Math.floor(totalDuration / 60)}m ${Math.floor(totalDuration % 60)}s
- Total Messages: ${progress.total_messages}
- Sequences Completed: ${progress.completed_sequences}/3
- Average Latency: ${metrics.average_latency.toFixed(0)}ms

Generated at: ${new Date().toISOString()}`;
  };

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1:
        return <Trophy className="h-5 w-5 text-yellow-400" />;
      case 2:
        return <Award className="h-5 w-5 text-neutral-300" />;
      case 3:
        return <Target className="h-5 w-5 text-amber-600" />;
      default:
        return <BarChart3 className="h-5 w-5 text-neutral-400" />;
    }
  };

  const getRankColor = (rank: number) => {
    switch (rank) {
      case 1:
        return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
      case 2:
        return 'bg-neutral-500/20 text-neutral-300 border-neutral-500/30';
      case 3:
        return 'bg-amber-600/20 text-amber-300 border-amber-600/30';
      default:
        return 'bg-neutral-600/20 text-neutral-400 border-neutral-600/30';
    }
  };

  return (
    <Card className={cn("border-neutral-700 bg-neutral-800/80 backdrop-blur-sm", className)}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-yellow-500/20 flex items-center justify-center">
              <Trophy className="h-5 w-5 text-yellow-400" />
            </div>
            <div>
              <CardTitle className="text-xl text-neutral-100">
                Analysis Complete
              </CardTitle>
              <CardDescription className="text-neutral-400">
                Comparative analysis of 3 parallel research sequences
              </CardDescription>
            </div>
          </div>
          
          <Badge className="bg-green-500/20 text-green-300 border-green-500/30">
            <CheckCircle className="h-3 w-3 mr-1" />
            Completed
          </Badge>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as any)}>
          <TabsList className="grid w-full grid-cols-3 bg-neutral-700">
            <TabsTrigger value="overview" className="data-[state=active]:bg-neutral-600">
              Overview
            </TabsTrigger>
            <TabsTrigger value="details" className="data-[state=active]:bg-neutral-600">
              Details
            </TabsTrigger>
            <TabsTrigger value="export" className="data-[state=active]:bg-neutral-600">
              Export
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Winner Declaration */}
            <div className="bg-gradient-to-r from-yellow-500/10 to-yellow-600/5 border border-yellow-500/20 rounded-lg p-6">
              <div className="flex items-center gap-4 mb-4">
                <Trophy className="h-8 w-8 text-yellow-400" />
                <div>
                  <h3 className="text-xl font-bold text-yellow-100">
                    {strategyConfig[winner.sequence.strategy].title} Strategy Wins!
                  </h3>
                  <p className="text-yellow-200/80">
                    {strategyConfig[winner.sequence.strategy].description}
                  </p>
                </div>
                <div className="ml-auto text-right">
                  <div className="text-2xl font-bold text-yellow-300">
                    {winner.score.toFixed(1)}
                  </div>
                  <div className="text-sm text-yellow-400">Overall Score</div>
                </div>
              </div>
              
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-lg font-semibold text-neutral-200">
                    {winner.completeness.toFixed(1)}%
                  </div>
                  <div className="text-sm text-neutral-400">Completion</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-semibold text-neutral-200">
                    {winner.efficiency.toFixed(1)}
                  </div>
                  <div className="text-sm text-neutral-400">Efficiency</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-semibold text-neutral-200">
                    {winner.sequence.messages.length}
                  </div>
                  <div className="text-sm text-neutral-400">Messages</div>
                </div>
              </div>
            </div>

            {/* Key Insights */}
            <div>
              <h4 className="text-lg font-semibold text-neutral-200 mb-3 flex items-center gap-2">
                <Lightbulb className="h-5 w-5 text-yellow-400" />
                Key Insights
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {winner.keyInsights.map((insight, index) => (
                  <div key={index} className="bg-neutral-700/50 rounded-lg p-3">
                    <div className="flex items-start gap-2">
                      <Star className="h-4 w-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                      <p className="text-sm text-neutral-300">{insight}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Rankings */}
            <div>
              <h4 className="text-lg font-semibold text-neutral-200 mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Final Rankings
              </h4>
              <div className="space-y-3">
                {analysis.map((a, index) => {
                  const config = strategyConfig[a.sequence.strategy];
                  
                  return (
                    <div key={a.sequence.sequence_id} className="bg-neutral-700/50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <Badge className={getRankColor(a.rank)}>
                            {getRankIcon(a.rank)}
                            <span className="ml-1">#{a.rank}</span>
                          </Badge>
                          <div>
                            <h5 className="font-semibold text-neutral-200">{config.title}</h5>
                            <p className="text-sm text-neutral-400">{config.description}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-bold text-neutral-200">
                            {a.score.toFixed(1)}
                          </div>
                          <div className="text-xs text-neutral-400">Score</div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="text-neutral-400">Completion</span>
                            <span className="text-neutral-300">{a.completeness.toFixed(1)}%</span>
                          </div>
                          <Progress value={a.completeness} className="h-1.5 bg-neutral-600" />
                        </div>
                        <div>
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="text-neutral-400">Quality</span>
                            <span className="text-neutral-300">{a.quality.toFixed(1)}%</span>
                          </div>
                          <Progress value={a.quality} className="h-1.5 bg-neutral-600" />
                        </div>
                        <div>
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="text-neutral-400">Messages</span>
                            <span className="text-neutral-300">{a.sequence.messages.length}</span>
                          </div>
                          <Progress 
                            value={(a.sequence.messages.length / Math.max(...sequences.map(s => s.messages.length))) * 100} 
                            className="h-1.5 bg-neutral-600" 
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </TabsContent>

          {/* Details Tab */}
          <TabsContent value="details" className="space-y-6">
            {analysis.map((a) => {
              const config = strategyConfig[a.sequence.strategy];
              
              return (
                <Card key={a.sequence.sequence_id} className="border-neutral-600 bg-neutral-700/50">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Badge className={config.color}>
                          {config.icon}
                          <span className="ml-1">{config.title}</span>
                        </Badge>
                        <Badge className={getRankColor(a.rank)}>
                          Rank #{a.rank}
                        </Badge>
                      </div>
                      <div className="text-right">
                        <div className="text-xl font-bold text-neutral-200">
                          {a.score.toFixed(1)}
                        </div>
                        <div className="text-xs text-neutral-400">Overall Score</div>
                      </div>
                    </div>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    {/* Metrics */}
                    <div className="grid grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-lg font-semibold text-neutral-200">
                          {a.completeness.toFixed(1)}%
                        </div>
                        <div className="text-xs text-neutral-400">Completion</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold text-neutral-200">
                          {a.efficiency.toFixed(1)}
                        </div>
                        <div className="text-xs text-neutral-400">Efficiency</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold text-neutral-200">
                          {a.sequence.messages.length}
                        </div>
                        <div className="text-xs text-neutral-400">Messages</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-semibold text-neutral-200">
                          {a.sequence.errors.length}
                        </div>
                        <div className="text-xs text-neutral-400">Errors</div>
                      </div>
                    </div>

                    <Separator className="bg-neutral-600" />

                    {/* Strengths and Weaknesses */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h5 className="font-medium text-green-300 mb-2 flex items-center gap-2">
                          <CheckCircle className="h-4 w-4" />
                          Strengths
                        </h5>
                        <ul className="space-y-1">
                          {a.strengths.map((strength, index) => (
                            <li key={index} className="text-sm text-neutral-300 flex items-center gap-2">
                              <div className="w-1 h-1 bg-green-400 rounded-full" />
                              {strength}
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      {a.weaknesses.length > 0 && (
                        <div>
                          <h5 className="font-medium text-amber-300 mb-2 flex items-center gap-2">
                            <AlertCircle className="h-4 w-4" />
                            Areas for Improvement
                          </h5>
                          <ul className="space-y-1">
                            {a.weaknesses.map((weakness, index) => (
                              <li key={index} className="text-sm text-neutral-300 flex items-center gap-2">
                                <div className="w-1 h-1 bg-amber-400 rounded-full" />
                                {weakness}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </TabsContent>

          {/* Export Tab */}
          <TabsContent value="export" className="space-y-6">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-neutral-200 mb-2">
                Export Analysis Results
              </h3>
              <p className="text-neutral-400 mb-6">
                Download or copy the complete analysis summary
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Button 
                onClick={() => handleCopy(generateSummaryText(), 'summary')}
                className="bg-blue-600 hover:bg-blue-700 text-white h-auto p-4"
              >
                <div className="flex items-center gap-3">
                  {copiedContent === 'summary' ? (
                    <CopyCheck className="h-5 w-5" />
                  ) : (
                    <Copy className="h-5 w-5" />
                  )}
                  <div className="text-left">
                    <div className="font-medium">Copy Summary</div>
                    <div className="text-sm opacity-80">
                      {copiedContent === 'summary' ? 'Copied!' : 'Copy to clipboard'}
                    </div>
                  </div>
                </div>
              </Button>

              <Button 
                onClick={() => {
                  const blob = new Blob([generateSummaryText()], { type: 'text/markdown' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `research-analysis-${Date.now()}.md`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
                variant="outline"
                className="border-neutral-600 h-auto p-4"
              >
                <div className="flex items-center gap-3">
                  <Download className="h-5 w-5" />
                  <div className="text-left">
                    <div className="font-medium">Download Report</div>
                    <div className="text-sm opacity-80">Markdown format</div>
                  </div>
                </div>
              </Button>
            </div>

            <div className="bg-neutral-700/50 rounded-lg p-4">
              <h4 className="font-medium text-neutral-200 mb-2">Analysis Summary</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-neutral-400">Duration:</span>
                  <div className="font-medium text-neutral-200">
                    {Math.floor(totalDuration / 60)}m {Math.floor(totalDuration % 60)}s
                  </div>
                </div>
                <div>
                  <span className="text-neutral-400">Messages:</span>
                  <div className="font-medium text-neutral-200">{progress.total_messages}</div>
                </div>
                <div>
                  <span className="text-neutral-400">Sequences:</span>
                  <div className="font-medium text-neutral-200">{progress.completed_sequences}/3</div>
                </div>
                <div>
                  <span className="text-neutral-400">Winner:</span>
                  <div className="font-medium text-yellow-300">
                    {strategyConfig[winner.sequence.strategy].title}
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

export default ComparisonSummary;