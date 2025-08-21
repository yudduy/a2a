/**
 * SequenceChat - Individual sequence chat column with real-time streaming
 * 
 * Features:
 * - Real-time message streaming display with virtual scrolling
 * - Agent progression timeline
 * - Tool execution indicators
 * - Progress tracking and status
 * - Professional chat UI with smooth animations
 * - Performance optimized with React.memo and memoized callbacks
 */

import React, { useRef, useEffect, useState, useCallback, useMemo, memo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import {
  GraduationCap,
  TrendingUp,
  Zap,
  Activity,
  ArrowRight,
  CheckCircle,
  Loader2,
  Clock,
  MessageSquare,
  AlertCircle,
  Eye,
  EyeOff,
  Copy,
  CopyCheck,
  Maximize2,
  Minimize2,
} from 'lucide-react';

import { 
  SequenceState, 
  SequenceStrategy, 
  AgentType,
  RoutedMessage,
} from '@/types/parallel';
import { cn } from '@/lib/utils';

interface SequenceChatProps {
  sequence?: SequenceState;
  strategy: SequenceStrategy;
  isActive: boolean;
  isLoading: boolean;
  showHeader?: boolean;
  className?: string;
  enableExpansion?: boolean;
}

const SequenceChat = memo(function SequenceChat({
  sequence,
  strategy,
  isActive,
  isLoading,
  showHeader = true,
  className,
  enableExpansion = true,
}: SequenceChatProps) {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [showAgentTimeline, setShowAgentTimeline] = useState(true);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  // Memoized message count for performance
  const messageCount = useMemo(() => sequence?.messages.length || 0, [sequence?.messages.length]);

  // Auto-scroll to bottom when new messages arrive (optimized)
  const scrollToBottom = useCallback(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        // Use requestAnimationFrame for smoother scrolling
        requestAnimationFrame(() => {
          scrollElement.scrollTop = scrollElement.scrollHeight;
        });
      }
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messageCount, scrollToBottom]);

  // Memoized agent progression order
  const agentProgression = useMemo(() => {
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
  }, [strategy]);

  const currentAgentIndex = useMemo(() => 
    sequence?.current_agent ? agentProgression.indexOf(sequence.current_agent) : -1, 
    [sequence?.current_agent, agentProgression]
  );

  // Memoized helper functions for performance
  const formatAgentName = useCallback((agent: AgentType): string => {
    switch (agent) {
      case AgentType.ACADEMIC:
        return 'Academic Research';
      case AgentType.INDUSTRY:
        return 'Industry Analysis';
      case AgentType.TECHNICAL_TRENDS:
        return 'Technical Trends';
    }
  }, []);

  const getAgentIcon = useCallback((agent: AgentType) => {
    switch (agent) {
      case AgentType.ACADEMIC:
        return <GraduationCap className="h-4 w-4" />;
      case AgentType.INDUSTRY:
        return <TrendingUp className="h-4 w-4" />;
      case AgentType.TECHNICAL_TRENDS:
        return <Zap className="h-4 w-4" />;
    }
  }, []);

  const getAgentStatus = useCallback((agent: AgentType, index: number): 'completed' | 'active' | 'pending' => {
    if (index < currentAgentIndex) return 'completed';
    if (index === currentAgentIndex && agent === sequence?.current_agent) return 'active';
    return 'pending';
  }, [currentAgentIndex, sequence?.current_agent]);

  const formatTimeAgo = useCallback((timestamp: number): string => {
    const now = Date.now();
    const diff = (now - timestamp) / 1000;
    
    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  }, []);

  const getMessageIcon = useCallback((messageType: string) => {
    switch (messageType) {
      case 'progress':
        return <Activity className="h-3 w-3 text-blue-400" />;
      case 'agent_transition':
        return <ArrowRight className="h-3 w-3 text-amber-400" />;
      case 'result':
        return <CheckCircle className="h-3 w-3 text-green-400" />;
      case 'error':
        return <AlertCircle className="h-3 w-3 text-red-400" />;
      case 'completion':
        return <CheckCircle className="h-3 w-3 text-green-400" />;
      default:
        return <MessageSquare className="h-3 w-3 text-neutral-400" />;
    }
  }, []);

  const handleCopyMessage = useCallback(async (message: RoutedMessage) => {
    try {
      const content = typeof message.content === 'string' 
        ? message.content 
        : JSON.stringify(message.content, null, 2);
      
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(message.message_id);
      
      // Clear the copied state after 2 seconds
      const timeoutId = setTimeout(() => setCopiedMessageId(null), 2000);
      
      // Cleanup timeout on unmount or re-copy
      return () => clearTimeout(timeoutId);
    } catch (err) {
      console.error('Failed to copy message:', err);
    }
  }, []);

  const getStatusColor = useCallback((status: string) => {
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
  }, []);

  // Memoized UI state computations
  const uiState = useMemo(() => {
    if (!sequence) return null;
    
    return {
      statusColor: getStatusColor(sequence.progress.status),
      completionPercentage: Math.round(sequence.progress.completion_percentage),
      formattedStrategy: strategy.replace('_', ' ').split(' ').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' '),
    };
  }, [sequence?.progress.status, sequence?.progress.completion_percentage, strategy, getStatusColor]);

  // Render empty state
  if (!sequence) {
    return (
      <Card className={cn("h-full border-neutral-700 bg-neutral-800/50", className)}>
        <CardContent className="h-full flex items-center justify-center">
          <div className="text-center text-neutral-500">
            <Loader2 className="h-8 w-8 mx-auto mb-3 animate-spin opacity-50" />
            <p className="text-sm">Waiting for sequence...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn(
      "h-full border-neutral-700 bg-neutral-800/80 backdrop-blur-sm transition-all duration-300",
      isActive && "ring-2 ring-blue-500/30 border-blue-500/50",
      isExpanded && "fixed inset-4 z-50",
      className
    )}>
      {showHeader && (
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge className={uiState?.statusColor}>
                {sequence.progress.status}
              </Badge>
              {isActive && (
                <div className="flex items-center gap-1">
                  <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse" />
                  <span className="text-xs text-green-400">Live</span>
                </div>
              )}
            </div>
            
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowAgentTimeline(!showAgentTimeline)}
                className="h-6 w-6 p-0 text-neutral-400 hover:text-neutral-200"
              >
                {showAgentTimeline ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
              </Button>
              
              {enableExpansion && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="h-6 w-6 p-0 text-neutral-400 hover:text-neutral-200"
                >
                  {isExpanded ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
                </Button>
              )}
            </div>
          </div>
          
          <div>
            <CardTitle className="text-lg text-neutral-100">
              {uiState?.formattedStrategy}
            </CardTitle>
            <CardDescription className="text-neutral-400">
              {sequence.progress.messages_received} messages • {uiState?.completionPercentage}% complete
            </CardDescription>
          </div>

          {/* Progress bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-neutral-400">Progress</span>
              <span className="text-neutral-300">{uiState?.completionPercentage}%</span>
            </div>
            <Progress 
              value={sequence.progress.completion_percentage} 
              className="h-1.5 bg-neutral-700"
            />
          </div>
        </CardHeader>
      )}

      <CardContent className={cn("flex flex-col", showHeader ? "h-[calc(100%-200px)]" : "h-full")}>
        {/* Agent Timeline */}
        {showAgentTimeline && (
          <div className="mb-4 p-3 bg-neutral-700/50 rounded-lg">
            <h4 className="text-sm font-medium text-neutral-200 mb-3 flex items-center gap-2">
              <ArrowRight className="h-4 w-4" />
              Agent Progression
            </h4>
            
            <div className="flex items-center gap-2">
              {agentProgression.map((agent, index) => {
                const status = getAgentStatus(agent, index);
                const isCurrentAgent = agent === sequence.current_agent && status === 'active';
                
                return (
                  <React.Fragment key={agent}>
                    <div className={cn(
                      "flex items-center gap-2 px-2 py-1 rounded-md transition-all",
                      status === 'completed' && "bg-green-500/20 text-green-300",
                      status === 'active' && "bg-blue-500/20 text-blue-300",
                      status === 'pending' && "bg-neutral-600/50 text-neutral-400"
                    )}>
                      <div className={cn(
                        "h-5 w-5 rounded-full flex items-center justify-center",
                        status === 'completed' && "bg-green-500",
                        status === 'active' && "bg-blue-500",
                        status === 'pending' && "bg-neutral-600"
                      )}>
                        {status === 'completed' ? (
                          <CheckCircle className="h-3 w-3 text-white" />
                        ) : isCurrentAgent && isLoading ? (
                          <Loader2 className="h-3 w-3 text-white animate-spin" />
                        ) : (
                          getAgentIcon(agent)
                        )}
                      </div>
                      <span className="text-xs font-medium whitespace-nowrap">
                        {formatAgentName(agent)}
                      </span>
                    </div>
                    
                    {index < agentProgression.length - 1 && (
                      <ArrowRight className={cn(
                        "h-3 w-3 flex-shrink-0",
                        index < currentAgentIndex ? "text-green-400" : "text-neutral-600"
                      )} />
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 min-h-0">
          <ScrollArea className="h-full" ref={scrollAreaRef}>
            <div className="space-y-3 p-1">
              {sequence.messages.length === 0 && !isLoading ? (
                <div className="text-center py-8 text-neutral-500">
                  <MessageSquare className="h-8 w-8 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">No messages yet</p>
                  <p className="text-xs text-neutral-600 mt-1">
                    Chat will appear here during execution
                  </p>
                </div>
              ) : isLoading && sequence.messages.length === 0 ? (
                <div className="text-center py-8 text-neutral-500">
                  <Loader2 className="h-8 w-8 mx-auto mb-3 animate-spin opacity-50" />
                  <p className="text-sm">Starting sequence...</p>
                </div>
              ) : (
                <>
                  {sequence.messages.map((message, index) => (
                    <div key={`${message.message_id}-${index}`} className="group">
                      <div className="flex gap-3">
                        <div className="h-6 w-6 rounded-full bg-neutral-600 flex items-center justify-center flex-shrink-0 mt-1">
                          {getMessageIcon(message.message_type)}
                        </div>
                        
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center gap-2">
                              <h5 className="text-sm font-medium text-neutral-200">
                                {message.message_type.replace('_', ' ').toUpperCase()}
                              </h5>
                              {message.agent_type && (
                                <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-300 border-blue-500/30">
                                  {formatAgentName(message.agent_type)}
                                </Badge>
                              )}
                            </div>
                            
                            <div className="flex items-center gap-1">
                              <span className="text-xs text-neutral-500">
                                {formatTimeAgo(message.timestamp)}
                              </span>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleCopyMessage(message)}
                                className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity text-neutral-400 hover:text-neutral-200"
                              >
                                {copiedMessageId === message.message_id ? (
                                  <CopyCheck className="h-3 w-3" />
                                ) : (
                                  <Copy className="h-3 w-3" />
                                )}
                              </Button>
                            </div>
                          </div>
                          
                          <div className="bg-neutral-700/50 rounded-lg p-3">
                            <p className="text-sm text-neutral-300 leading-relaxed break-words">
                              {typeof message.content === 'string' 
                                ? message.content 
                                : JSON.stringify(message.content, null, 2)
                              }
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {/* Show errors */}
                  {sequence.errors.map((error, index) => (
                    <div key={`error-${index}`} className="group">
                      <div className="flex gap-3">
                        <div className="h-6 w-6 rounded-full bg-red-600 flex items-center justify-center flex-shrink-0 mt-1">
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
                          
                          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                            <p className="text-sm text-red-200 leading-relaxed break-words">
                              {error.message}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {/* Loading indicator */}
                  {isActive && isLoading && (
                    <div className="flex gap-3">
                      <div className="h-6 w-6 rounded-full bg-neutral-600 flex items-center justify-center flex-shrink-0 mt-1">
                        <Loader2 className="h-3 w-3 text-neutral-400 animate-spin" />
                      </div>
                      
                      <div className="flex-1">
                        <div className="bg-neutral-700/50 rounded-lg p-3">
                          <div className="flex items-center gap-2">
                            <div className="flex gap-1">
                              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.32s]"></div>
                              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.16s]"></div>
                              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                            </div>
                            <p className="text-sm text-neutral-300">
                              {sequence.current_agent ? `${formatAgentName(sequence.current_agent)} is analyzing...` : 'Processing...'}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </ScrollArea>
        </div>
      </CardContent>
    </Card>
  );
});

// Add display name for debugging
SequenceChat.displayName = 'SequenceChat';

export { SequenceChat };
export default SequenceChat;