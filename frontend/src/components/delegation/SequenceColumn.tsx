import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Loader2,
  GraduationCap,
  TrendingUp,
  Zap,
  ArrowRight,
  CheckCircle,
  Clock,
  Activity,
  AlertCircle,
  MessageSquare,
} from 'lucide-react';
import { 
  SequenceState,
  SequenceStrategy,
  AgentType,
} from '@/types/parallel';

interface SequenceColumnProps {
  sequenceId: string;
  sequence: SequenceState;
  isActive: boolean;
  isLoading: boolean;
}

export function SequenceColumn({
  sequenceId: _sequenceId,
  sequence,
  isActive,
  isLoading,
}: SequenceColumnProps) {
  const { strategy, progress, messages, current_agent, errors } = sequence;
  
  // Get agent progression order
  const agentProgression = getAgentProgression(strategy);
  const currentAgentIndex = current_agent ? agentProgression.indexOf(current_agent) : -1;

// Define the agent progression based on strategy
function getAgentProgression(strategy?: SequenceStrategy): AgentType[] {
  if (!strategy) return [];
  switch (strategy) {
    case SequenceStrategy.THEORY_FIRST:
      return [AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS];
    case SequenceStrategy.MARKET_FIRST:
      return [AgentType.INDUSTRY, AgentType.ACADEMIC, AgentType.TECHNICAL_TRENDS];
    case SequenceStrategy.FUTURE_BACK:
      return [AgentType.TECHNICAL_TRENDS, AgentType.ACADEMIC, AgentType.INDUSTRY];
    default:
      return [AgentType.ACADEMIC, AgentType.INDUSTRY, AgentType.TECHNICAL_TRENDS];
  }
}

  const formatStrategy = (strategy?: SequenceStrategy): string => {
    if (!strategy) return 'UNKNOWN';
    return strategy.replace('_', ' ').toUpperCase();
  };

  const formatAgentName = (agent: AgentType): string => {
    switch (agent) {
      case AgentType.ACADEMIC:
        return 'Academic';
      case AgentType.INDUSTRY:
        return 'Industry';
      case AgentType.TECHNICAL_TRENDS:
        return 'Technical';
      default:
        // TypeScript exhaustiveness check - this should never happen
        const _exhaustiveCheck: never = agent;
        return _exhaustiveCheck;
    }
  };

  const getAgentIcon = (agent: AgentType): React.JSX.Element => {
    switch (agent) {
      case AgentType.ACADEMIC:
        return <GraduationCap className="h-4 w-4" />;
      case AgentType.INDUSTRY:
        return <TrendingUp className="h-4 w-4" />;
      case AgentType.TECHNICAL_TRENDS:
        return <Zap className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
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

  const getAgentStatus = (agent: AgentType, index: number): 'completed' | 'active' | 'pending' => {
    if (index < currentAgentIndex) return 'completed';
    if (index === currentAgentIndex && agent === current_agent) return 'active';
    return 'pending';
  };

  const getMessageIcon = (messageType: string): React.JSX.Element => {
    switch (messageType) {
      case 'progress':
        return <Activity className="h-4 w-4 text-blue-400" />;
      case 'agent_transition':
        return <ArrowRight className="h-4 w-4 text-amber-400" />;
      case 'result':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-400" />;
      case 'completion':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      default:
        return <MessageSquare className="h-4 w-4 text-neutral-400" />;
    }
  };

  const formatTimeAgo = (timestamp: number): string => {
    const now = Date.now();
    const diff = (now - timestamp) / 1000; // seconds
    
    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  };

  return (
    <Card className="border-none rounded-lg bg-neutral-700 h-full flex flex-col">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <Badge className={`text-xs ${getStrategyColor(strategy)}`}>
            {formatStrategy(strategy)}
          </Badge>
          {isActive && (
            <div className="flex items-center gap-1">
              <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-xs text-green-400">Active</span>
            </div>
          )}
        </div>
        <CardTitle className="text-neutral-100 text-lg">
          Sequence Progress
        </CardTitle>
        <CardDescription className="text-neutral-300 text-sm">
          Agent progression and timeline
        </CardDescription>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col space-y-6">
        {/* Progress Summary */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-neutral-200">Overall Progress</span>
            <span className="text-sm text-neutral-300">{Math.round(progress.completion_percentage)}%</span>
          </div>
          <div className="w-full bg-neutral-800 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all ${
                progress.status === 'completed' ? 'bg-green-500' : 
                progress.status === 'active' ? 'bg-blue-500' : 
                progress.status === 'failed' ? 'bg-red-500' : 'bg-neutral-600'
              }`}
              style={{ width: `${progress.completion_percentage}%` }}
            ></div>
          </div>
        </div>

        {/* Agent Progression Timeline */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-neutral-200 flex items-center gap-2">
            <ArrowRight className="h-4 w-4" />
            Agent Progression
          </h4>
          
          <div className="space-y-3">
            {agentProgression.map((agent, index) => {
              const status = getAgentStatus(agent, index);
              const isCurrentAgent = agent === current_agent && status === 'active';
              
              return (
                <div key={agent} className="relative">
                  {index < agentProgression.length - 1 && (
                    <div className={`absolute left-6 top-8 h-6 w-0.5 ${
                      status === 'completed' 
                        ? 'bg-green-500' 
                        : status === 'active' 
                          ? 'bg-blue-500' 
                          : 'bg-neutral-600'
                    }`} />
                  )}
                  
                  <div className="flex items-center gap-3">
                    <div className={`h-6 w-6 rounded-full flex items-center justify-center ring-4 ${
                      status === 'completed'
                        ? 'bg-green-500 ring-green-500/20'
                        : status === 'active'
                          ? 'bg-blue-500 ring-blue-500/20'
                          : 'bg-neutral-600 ring-neutral-600/20'
                    }`}>
                      {status === 'completed' ? (
                        <CheckCircle className="h-3 w-3 text-white" />
                      ) : isCurrentAgent && isLoading ? (
                        <Loader2 className="h-3 w-3 text-white animate-spin" />
                      ) : (
                        getAgentIcon(agent)
                      )}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <span className={`text-sm font-medium ${
                          status === 'completed' 
                            ? 'text-green-300'
                            : status === 'active'
                              ? 'text-blue-300'
                              : 'text-neutral-400'
                        }`}>
                          {formatAgentName(agent)} Agent
                        </span>
                        
                        <span className="text-xs text-neutral-500">
                          Step {index + 1}
                        </span>
                      </div>
                      
                      {isCurrentAgent && (
                        <p className="text-xs text-blue-200 mt-1">
                          Currently executing...
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Activity Timeline */}
        <div className="flex-1 flex flex-col min-h-0">
          <h4 className="text-sm font-medium text-neutral-200 flex items-center gap-2 mb-4">
            <Clock className="h-4 w-4" />
            Activity Timeline
          </h4>
          
          <ScrollArea className="flex-1">
            {messages.length === 0 && !isLoading ? (
              <div className="text-center py-8 text-neutral-500">
                <Activity className="h-6 w-6 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No activity yet</p>
                <p className="text-xs text-neutral-600 mt-1">
                  Timeline will update during execution
                </p>
              </div>
            ) : isLoading && messages.length === 0 ? (
              <div className="text-center py-8 text-neutral-500">
                <Loader2 className="h-6 w-6 mx-auto mb-2 animate-spin opacity-50" />
                <p className="text-sm">Initializing sequence...</p>
              </div>
            ) : (
              <div className="space-y-3">
                {messages.map((message, index) => (
                  <div key={`${message.message_id}-${index}`} className="relative">
                    {index < messages.length - 1 && (
                      <div className="absolute left-3 top-6 h-full w-0.5 bg-neutral-600" />
                    )}
                    
                    <div className="flex gap-3">
                      <div className="h-6 w-6 rounded-full bg-neutral-600 flex items-center justify-center ring-4 ring-neutral-700 flex-shrink-0">
                        {getMessageIcon(message.message_type)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <h5 className="text-sm font-medium text-neutral-200 truncate">
                            {message.message_type.replace('_', ' ').toUpperCase()}
                          </h5>
                          <span className="text-xs text-neutral-500 flex-shrink-0">
                            {formatTimeAgo(message.timestamp)}
                          </span>
                        </div>
                        
                        <p className="text-xs text-neutral-300 leading-relaxed break-words">
                          {typeof message.content === 'string' 
                            ? message.content 
                            : typeof message.content === 'object'
                              ? JSON.stringify(message.content)
                              : String(message.content)
                          }
                        </p>
                        
                        {/* Show agent type if available */}
                        {message.agent_type && (
                          <div className="mt-2">
                            <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-300 border-blue-500/30">
                              {formatAgentName(message.agent_type)}
                            </Badge>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                
                {/* Show errors if any */}
                {errors.map((error, index) => (
                  <div key={`error-${index}`} className="relative">
                    <div className="flex gap-3">
                      <div className="h-6 w-6 rounded-full bg-red-600 flex items-center justify-center ring-4 ring-red-600/20 flex-shrink-0">
                        <AlertCircle className="h-3 w-3 text-white" />
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <h5 className="text-sm font-medium text-red-300">
                            Error
                          </h5>
                          <span className="text-xs text-neutral-500">
                            {formatTimeAgo(error.timestamp)}
                          </span>
                        </div>
                        
                        <p className="text-xs text-red-200 leading-relaxed break-words">
                          {error.message}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
                
                {/* Loading indicator at the end if sequence is active */}
                {isActive && isLoading && (
                  <div className="relative">
                    <div className="flex gap-3">
                      <div className="h-6 w-6 rounded-full bg-neutral-600 flex items-center justify-center ring-4 ring-neutral-700">
                        <Loader2 className="h-3 w-3 text-neutral-400 animate-spin" />
                      </div>
                      
                      <div className="flex-1">
                        <p className="text-sm text-neutral-300 font-medium">
                          Processing...
                        </p>
                        <p className="text-xs text-neutral-500">
                          Sequence execution in progress
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </ScrollArea>
        </div>
      </CardContent>
    </Card>
  );
}