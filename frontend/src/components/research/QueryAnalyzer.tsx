/**
 * QueryAnalyzer - Shows sequence selection reasoning and query analysis
 * 
 * Features:
 * - Query breakdown and analysis
 * - Sequence strategy explanation
 * - Expected outcomes per sequence
 * - Real-time sequence status
 * - Collapsible interface
 */

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Separator } from '@/components/ui/separator';
import {
  Search,
  Brain,
  ChevronDown,
  ChevronUp,
  Activity,
  TrendingUp,
  Zap,
  Target,
  Lightbulb,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader2,
  ArrowRight,
  Info,
} from 'lucide-react';

import { 
  SequenceState, 
  SequenceStrategy,
} from '@/types/parallel';
import { cn } from '@/lib/utils';

interface QueryAnalyzerProps {
  query: string;
  sequences: SequenceState[];
  isLoading: boolean;
  className?: string;
  defaultExpanded?: boolean;
}

interface SequenceAnalysis {
  strategy: SequenceStrategy;
  title: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  expectedOutcomes: string[];
  strengths: string[];
  approach: string;
  timelinePhases: string[];
}

export function QueryAnalyzer({
  query,
  sequences,
  isLoading,
  className,
  defaultExpanded = false,
}: QueryAnalyzerProps) {
  const [isOpen, setIsOpen] = useState(defaultExpanded);

  // Analyze query for keywords and topics
  const analyzeQuery = (query: string) => {
    const keywords = query.toLowerCase().split(' ').filter(word => word.length > 3);
    const hasAcademicTerms = keywords.some(word => 
      ['research', 'study', 'analysis', 'academic', 'theory', 'framework', 'model'].includes(word)
    );
    const hasMarketTerms = keywords.some(word => 
      ['market', 'business', 'industry', 'commercial', 'economics', 'financial'].includes(word)
    );
    const hasTechTerms = keywords.some(word => 
      ['technology', 'tech', 'innovation', 'future', 'trends', 'emerging', 'ai', 'digital'].includes(word)
    );

    return {
      keywords: keywords.slice(0, 5),
      hasAcademicTerms,
      hasMarketTerms,
      hasTechTerms,
      complexity: keywords.length > 10 ? 'high' : keywords.length > 5 ? 'medium' : 'low',
      estimatedDuration: keywords.length > 10 ? '8-12 min' : keywords.length > 5 ? '5-8 min' : '3-5 min',
    };
  };

  const queryAnalysis = analyzeQuery(query);

  // Get sequence analysis configurations
  const sequenceAnalyses: SequenceAnalysis[] = [
    {
      strategy: SequenceStrategy.THEORY_FIRST,
      title: 'Theory First',
      description: 'Academic-driven approach starting with theoretical foundations',
      icon: <Activity className="h-4 w-4" />,
      color: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
      approach: 'Academic → Industry → Technical',
      expectedOutcomes: [
        'Comprehensive theoretical framework',
        'Peer-reviewed research citations',
        'Foundational principles and models',
        'Academic perspectives and debates'
      ],
      strengths: [
        'Deep theoretical understanding',
        'Research-backed insights',
        'Long-term perspective',
        'Rigorous methodology'
      ],
      timelinePhases: [
        'Literature Review & Theory',
        'Industry Applications',
        'Future Technical Trends'
      ]
    },
    {
      strategy: SequenceStrategy.MARKET_FIRST,
      title: 'Market First',
      description: 'Industry-focused approach emphasizing practical applications',
      icon: <TrendingUp className="h-4 w-4" />,
      color: 'bg-green-500/20 text-green-300 border-green-500/30',
      approach: 'Industry → Academic → Technical',
      expectedOutcomes: [
        'Current market dynamics',
        'Industry case studies',
        'Practical implementation strategies',
        'Economic impact analysis'
      ],
      strengths: [
        'Real-world applicability',
        'Market-driven insights',
        'Immediate actionability',
        'Industry best practices'
      ],
      timelinePhases: [
        'Market Analysis & Trends',
        'Academic Validation',
        'Technical Implementation'
      ]
    },
    {
      strategy: SequenceStrategy.FUTURE_BACK,
      title: 'Future Back',
      description: 'Forward-looking approach working backwards from future possibilities',
      icon: <Zap className="h-4 w-4" />,
      color: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
      approach: 'Technical → Academic → Industry',
      expectedOutcomes: [
        'Emerging technology trends',
        'Future scenario analysis',
        'Innovation opportunities',
        'Disruptive potential assessment'
      ],
      strengths: [
        'Forward-thinking insights',
        'Innovation focus',
        'Disruptive opportunities',
        'Technology roadmaps'
      ],
      timelinePhases: [
        'Future Tech Trends',
        'Academic Research',
        'Industry Implications'
      ]
    }
  ];

  const getSequenceStatus = (strategy: SequenceStrategy) => {
    const sequence = sequences.find(seq => seq.strategy === strategy);
    if (!sequence) return { status: 'pending', progress: 0, messages: 0 };
    
    return {
      status: sequence.progress.status,
      progress: sequence.progress.completion_percentage,
      messages: sequence.progress.messages_received,
      currentAgent: sequence.current_agent,
    };
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'active':
        return <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-400" />;
      default:
        return <Clock className="h-4 w-4 text-neutral-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500/20 text-green-300 border-green-500/30';
      case 'active':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      case 'failed':
        return 'bg-red-500/20 text-red-300 border-red-500/30';
      default:
        return 'bg-neutral-500/20 text-neutral-300 border-neutral-500/30';
    }
  };

  return (
    <Card className={cn("border-neutral-700 bg-neutral-800/80 backdrop-blur-sm", className)}>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-neutral-700/50 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="h-8 w-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <Brain className="h-4 w-4 text-blue-400" />
                </div>
                <div>
                  <CardTitle className="text-lg text-neutral-100 flex items-center gap-2">
                    Query Analysis
                    <Badge className="bg-blue-500/20 text-blue-300 text-xs">
                      {sequences.length}/3 sequences
                    </Badge>
                  </CardTitle>
                  <p className="text-sm text-neutral-400 mt-1">
                    Understanding your research query and sequence strategies
                  </p>
                </div>
              </div>
              
              <Button variant="ghost" size="sm" className="text-neutral-400">
                {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </Button>
            </div>
          </CardHeader>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="pt-0">
            {/* Query Analysis */}
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <Search className="h-4 w-4 text-neutral-400" />
                <h3 className="font-medium text-neutral-200">Research Query</h3>
              </div>
              
              <div className="bg-neutral-700/50 rounded-lg p-4 mb-4">
                <p className="text-neutral-100 font-medium mb-3 leading-relaxed">
                  "{query}"
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div>
                    <span className="text-xs text-neutral-400 block mb-1">Keywords</span>
                    <div className="flex flex-wrap gap-1">
                      {queryAnalysis.keywords.map((keyword, index) => (
                        <Badge key={index} variant="outline" className="text-xs bg-neutral-600/50 border-neutral-500">
                          {keyword}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <span className="text-xs text-neutral-400 block mb-1">Complexity</span>
                    <Badge className={cn(
                      "text-xs capitalize",
                      queryAnalysis.complexity === 'high' && "bg-red-500/20 text-red-300 border-red-500/30",
                      queryAnalysis.complexity === 'medium' && "bg-yellow-500/20 text-yellow-300 border-yellow-500/30",
                      queryAnalysis.complexity === 'low' && "bg-green-500/20 text-green-300 border-green-500/30"
                    )}>
                      {queryAnalysis.complexity}
                    </Badge>
                  </div>
                  
                  <div>
                    <span className="text-xs text-neutral-400 block mb-1">Est. Duration</span>
                    <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30 text-xs">
                      <Clock className="h-3 w-3 mr-1" />
                      {queryAnalysis.estimatedDuration}
                    </Badge>
                  </div>
                  
                  <div>
                    <span className="text-xs text-neutral-400 block mb-1">Focus Areas</span>
                    <div className="flex gap-1">
                      {queryAnalysis.hasAcademicTerms && (
                        <Badge className="bg-blue-500/20 text-blue-300 text-xs">Academic</Badge>
                      )}
                      {queryAnalysis.hasMarketTerms && (
                        <Badge className="bg-green-500/20 text-green-300 text-xs">Market</Badge>
                      )}
                      {queryAnalysis.hasTechTerms && (
                        <Badge className="bg-purple-500/20 text-purple-300 text-xs">Tech</Badge>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <Separator className="mb-6 bg-neutral-700" />

            {/* Sequence Strategies */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Target className="h-4 w-4 text-neutral-400" />
                <h3 className="font-medium text-neutral-200">Sequence Strategies</h3>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {sequenceAnalyses.map((analysis) => {
                  const status = getSequenceStatus(analysis.strategy);
                  
                  return (
                    <Card key={analysis.strategy} className="border-neutral-600 bg-neutral-700/50">
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Badge className={analysis.color}>
                              {analysis.icon}
                              <span className="ml-1">{analysis.title}</span>
                            </Badge>
                          </div>
                          {getStatusIcon(status.status)}
                        </div>
                        
                        <div>
                          <p className="text-sm text-neutral-300 mb-2">{analysis.description}</p>
                          <Badge className={getStatusColor(status.status)} variant="outline">
                            {status.status} • {Math.round(status.progress)}% • {status.messages} msgs
                          </Badge>
                        </div>
                      </CardHeader>
                      
                      <CardContent className="pt-0 space-y-4">
                        {/* Approach Timeline */}
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <ArrowRight className="h-3 w-3 text-neutral-400" />
                            <span className="text-xs font-medium text-neutral-300">Approach</span>
                          </div>
                          <p className="text-xs text-neutral-400 font-mono bg-neutral-800/50 px-2 py-1 rounded">
                            {analysis.approach}
                          </p>
                        </div>
                        
                        {/* Expected Outcomes */}
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <Lightbulb className="h-3 w-3 text-neutral-400" />
                            <span className="text-xs font-medium text-neutral-300">Expected Outcomes</span>
                          </div>
                          <ul className="text-xs text-neutral-400 space-y-1">
                            {analysis.expectedOutcomes.slice(0, 2).map((outcome, index) => (
                              <li key={index} className="flex items-start gap-1">
                                <span className="text-neutral-600 mt-0.5">•</span>
                                <span>{outcome}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        {/* Current Phase */}
                        {status.status === 'active' && status.currentAgent && (
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <Activity className="h-3 w-3 text-blue-400" />
                              <span className="text-xs font-medium text-blue-300">Current Phase</span>
                            </div>
                            <p className="text-xs text-blue-200 bg-blue-500/10 px-2 py-1 rounded border border-blue-500/30">
                              {status.currentAgent?.replace('_', ' ').split(' ').map(word => 
                                word.charAt(0).toUpperCase() + word.slice(1)
                              ).join(' ')} Agent
                            </p>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </div>

            {/* Info footer */}
            <div className="mt-6 p-3 bg-blue-500/5 border border-blue-500/20 rounded-lg">
              <div className="flex items-start gap-2">
                <Info className="h-4 w-4 text-blue-400 mt-0.5 flex-shrink-0" />
                <div className="text-xs text-blue-200">
                  <p className="font-medium mb-1">How it works</p>
                  <p className="text-blue-200/80 leading-relaxed">
                    Each sequence runs independently with specialized agents in different orders. 
                    This provides diverse perspectives and comprehensive coverage of your research topic.
                    Results will be compared at the end to identify the most effective approach.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

export default QueryAnalyzer;