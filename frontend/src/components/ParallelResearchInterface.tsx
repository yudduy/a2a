/**
 * ParallelResearchInterface - Sophisticated 3-column side-by-side parallel research interface
 * 
 * Combines:
 * - ChatGPT's clean, minimal design with subtle message bubbles
 * - Claude's progressive disclosure with collapsible reasoning sections
 * - Independent scrolling for each sequence column
 * - Real-time comparison of research approaches
 * 
 * Features:
 * - 3 equal-width columns (33% each) for side-by-side comparison
 * - Clean message bubbles with minimal borders (ChatGPT style)
 * - Progressive disclosure with collapsible sections (Claude style)
 * - Tool usage blocks resembling Claude's code execution blocks
 * - Independent scroll containers for each sequence
 * - Responsive design that stacks on mobile
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
// import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
// import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { 
  ChevronRight, 
  Brain, 
  Code, 
  Activity, 
  CheckCircle, 
  Loader2,
  Copy,
  CopyCheck,
  Eye,
  EyeOff,
  MessageSquare,
  Zap
} from 'lucide-react';

import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';
import { MessageContentParser, ThinkingSection } from '@/types/messages';

// ============================================================================
// INTERFACES AND TYPES
// ============================================================================

interface ParallelResearchInterfaceProps {
  sequences: LLMGeneratedSequence[];
  parallelMessages: Record<string, RoutedMessage[]>;
  activeTabId?: string;
  onTabChange?: (tabId: string) => void;
  isLoading?: boolean;
  className?: string;
}

interface MessageBubbleProps {
  message: RoutedMessage;
  isUser?: boolean;
  onCopy?: (content: string) => void;
  copiedMessageId?: string | null;
}

interface ToolUsageBlockProps {
  toolCall: {
    name: string;
    input: any;
    output?: any;
    status: 'running' | 'completed' | 'error';
  };
  isExpanded: boolean;
  onToggle: () => void;
}

interface ReasoningSectionProps {
  section: ThinkingSection;
  isExpanded: boolean;
  onToggle: () => void;
}

interface SequenceColumnProps {
  sequence: LLMGeneratedSequence;
  messages: RoutedMessage[];
  isActive?: boolean;
  isLoading?: boolean;
  onCopy?: (content: string) => void;
  copiedMessageId?: string | null;
  className?: string;
}

// ============================================================================
// CLAUDE-STYLE REASONING SECTION COMPONENT
// ============================================================================

const ReasoningSection: React.FC<ReasoningSectionProps> = ({
  section,
  isExpanded,
  onToggle,
}) => {
  return (
    <div className="border border-blue-500/20 bg-blue-50/5 rounded-lg overflow-hidden my-2">
      <button
        onClick={onToggle}
        className="w-full px-3 py-2 hover:bg-blue-50/10 transition-colors text-left focus:outline-none focus:ring-1 focus:ring-blue-500/50"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-1 rounded bg-blue-500/20">
              <Brain className="h-3 w-3 text-blue-400" />
            </div>
            <span className="text-sm font-medium text-blue-300">
              thinking...
            </span>
            <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/30">
              {section.content.length} chars
            </Badge>
          </div>
          <ChevronRight
            className={cn(
              'h-3 w-3 text-blue-400 transition-transform duration-200',
              isExpanded && 'rotate-90'
            )}
          />
        </div>
      </button>
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-blue-500/20 bg-blue-950/10">
          <div className="pt-2">
            <div className="font-mono text-xs text-blue-100 leading-relaxed whitespace-pre-wrap">
              {section.content}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// CLAUDE-STYLE TOOL USAGE BLOCK COMPONENT  
// ============================================================================

const ToolUsageBlock: React.FC<ToolUsageBlockProps> = ({
  toolCall,
  isExpanded,
  onToggle,
}) => {
  const getStatusIcon = () => {
    switch (toolCall.status) {
      case 'running':
        return <Loader2 className="h-3 w-3 text-amber-400 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-3 w-3 text-green-400" />;
      case 'error':
        return <div className="h-3 w-3 rounded-full bg-red-400" />;
      default:
        return <Code className="h-3 w-3 text-neutral-400" />;
    }
  };

  const getStatusColor = () => {
    switch (toolCall.status) {
      case 'running':
        return 'border-amber-500/30 bg-amber-50/5';
      case 'completed':
        return 'border-green-500/30 bg-green-50/5';
      case 'error':
        return 'border-red-500/30 bg-red-50/5';
      default:
        return 'border-neutral-500/30 bg-neutral-50/5';
    }
  };

  return (
    <div className={cn('border rounded-lg overflow-hidden my-2', getStatusColor())}>
      <button
        onClick={onToggle}
        className="w-full px-3 py-2 hover:bg-white/5 transition-colors text-left focus:outline-none focus:ring-1 focus:ring-blue-500/50"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            <span className="text-sm font-medium text-neutral-200">
              {toolCall.name}
            </span>
            <Badge variant="outline" className="text-xs bg-neutral-500/10 text-neutral-400">
              Tool
            </Badge>
          </div>
          <ChevronRight
            className={cn(
              'h-3 w-3 text-neutral-400 transition-transform duration-200',
              isExpanded && 'rotate-90'
            )}
          />
        </div>
      </button>
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-neutral-500/20 bg-neutral-950/20">
          <div className="pt-2 space-y-2">
            <div>
              <div className="text-xs font-medium text-neutral-400 mb-1">Input:</div>
              <div className="bg-neutral-900 rounded p-2 font-mono text-xs text-neutral-200">
                {JSON.stringify(toolCall.input, null, 2)}
              </div>
            </div>
            {toolCall.output && (
              <div>
                <div className="text-xs font-medium text-neutral-400 mb-1">Output:</div>
                <div className="bg-neutral-900 rounded p-2 font-mono text-xs text-neutral-200">
                  {typeof toolCall.output === 'string' ? toolCall.output : JSON.stringify(toolCall.output, null, 2)}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// CHATGPT-STYLE MESSAGE BUBBLE COMPONENT
// ============================================================================

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isUser = false,
  onCopy,
  copiedMessageId,
}) => {
  // const [showDetails, setShowDetails] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());

  const handleCopy = useCallback(async () => {
    if (!onCopy) return;
    const content = typeof message.content === 'string' 
      ? message.content 
      : JSON.stringify(message.content, null, 2);
    onCopy(content);
  }, [message.content, onCopy]);

  const toggleSection = useCallback((sectionId: string) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId);
      } else {
        newSet.add(sectionId);
      }
      return newSet;
    });
  }, []);

  const toggleTool = useCallback((toolId: string) => {
    setExpandedTools(prev => {
      const newSet = new Set(prev);
      if (newSet.has(toolId)) {
        newSet.delete(toolId);
      } else {
        newSet.add(toolId);
      }
      return newSet;
    });
  }, []);

  // Parse actual thinking sections from message content
  const messageContent = typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2);
  const parsedContent = MessageContentParser.parse({
    id: message.message_id,
    type: 'ai',
    content: messageContent
  });
  
  // Get actual tool calls if available (implement based on your data structure)
  const actualToolCalls: any[] = message.tool_calls || [];

  const content = typeof message.content === 'string' 
    ? message.content 
    : JSON.stringify(message.content, null, 2);

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="bg-blue-600 text-white rounded-2xl px-4 py-3 max-w-[80%] shadow-sm">
          <div className="text-sm leading-relaxed">
            {content}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mb-6">
      <div className="flex items-start gap-3">
        <div className="w-7 h-7 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0 mt-1">
          <Activity className="w-4 h-4 text-green-400" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="bg-neutral-50/5 rounded-2xl p-4 shadow-sm border border-neutral-700/50">
            {/* Agent badge */}
            {message.current_agent && (
              <div className="mb-3">
                <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/30">
                  {message.current_agent}
                </Badge>
              </div>
            )}

            {/* Thinking sections (Claude-style progressive disclosure) */}
            {parsedContent.thinkingSections.length > 0 && (
              <div className="mb-3">
                {parsedContent.thinkingSections.map(section => (
                  <ReasoningSection
                    key={section.id}
                    section={section}
                    isExpanded={expandedSections.has(section.id)}
                    onToggle={() => toggleSection(section.id)}
                  />
                ))}
              </div>
            )}

            {/* Tool usage blocks */}
            {actualToolCalls.length > 0 && (
              <div className="mb-3">
                {actualToolCalls.map((tool: any, index: number) => (
                  <ToolUsageBlock
                    key={`tool-${tool.id || index}`}
                    toolCall={{
                      name: tool.name || 'unknown_tool',
                      input: tool.input || {},
                      output: tool.output,
                      status: tool.status || 'completed'
                    }}
                    isExpanded={expandedTools.has(`tool-${tool.id || index}`)}
                    onToggle={() => toggleTool(`tool-${tool.id || index}`)}
                  />
                ))}
              </div>
            )}

            {/* Main content with proper thinking section parsing */}
            <div className="prose prose-invert max-w-none text-sm text-neutral-200 leading-relaxed">
              {parsedContent.preThinking && (
                <div className="mb-3">{parsedContent.preThinking}</div>
              )}
              
              {parsedContent.postThinking && (
                <div className="mt-3">{parsedContent.postThinking}</div>
              )}
              
              {/* Fallback to show full content if no thinking sections */}
              {parsedContent.thinkingSections.length === 0 && !parsedContent.preThinking && !parsedContent.postThinking && (
                <div>{content}</div>
              )}
            </div>

            {/* Metadata and copy button */}
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-neutral-700/50">
              <div className="flex items-center gap-2 text-xs text-neutral-500">
                <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
                {message.message_type && (
                  <>
                    <span>•</span>
                    <span>{message.message_type.replace('_', ' ')}</span>
                  </>
                )}
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                className="text-neutral-400 hover:text-neutral-200 h-7 px-2"
              >
                {copiedMessageId === message.message_id ? (
                  <CopyCheck className="w-3 h-3" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// SEQUENCE COLUMN COMPONENT
// ============================================================================

const SequenceColumn: React.FC<SequenceColumnProps> = ({
  sequence,
  messages,
  isActive = false,
  isLoading = false,
  onCopy,
  copiedMessageId,
  className,
}) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [showProgressDetails, setShowProgressDetails] = useState(false);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const viewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (viewport) {
        viewport.scrollTo({
          top: viewport.scrollHeight,
          behavior: 'smooth',
        });
      }
    }
  }, [messages.length]);

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Column header */}
      <div className="p-4 border-b border-neutral-700/50 bg-neutral-900/50">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Badge className={cn(
              'text-xs font-medium',
              isActive ? 'bg-blue-500/20 text-blue-300 border-blue-500/50' : 'bg-neutral-500/20 text-neutral-300'
            )}>
              {sequence.sequence_name}
            </Badge>
            {isActive && (
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-xs text-green-400">Live</span>
              </div>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowProgressDetails(!showProgressDetails)}
            className="text-neutral-400 hover:text-neutral-200 h-6 w-6 p-0"
          >
            {showProgressDetails ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
          </Button>
        </div>

        <div className="space-y-2">
          <h3 className="text-sm font-medium text-neutral-200">
            {sequence.approach_description}
          </h3>
          <p className="text-xs text-neutral-400 leading-relaxed">
            {sequence.rationale}
          </p>
        </div>

        {/* Progress details */}
        {showProgressDetails && (
          <div className="mt-3 pt-3 border-t border-neutral-700/50">
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <span className="text-neutral-500">Confidence:</span>
                <span className="ml-2 text-neutral-300 font-medium">
                  {Math.round(sequence.confidence_score * 100)}%
                </span>
              </div>
              <div>
                <span className="text-neutral-500">Messages:</span>
                <span className="ml-2 text-neutral-300 font-medium">
                  {messages.length}
                </span>
              </div>
            </div>
            {sequence.agent_names.length > 0 && (
              <div className="mt-2">
                <span className="text-xs text-neutral-500">Agents:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {sequence.agent_names.map(agent => (
                    <Badge key={agent} variant="outline" className="text-xs bg-neutral-500/10">
                      {agent.replace('_', ' ')}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Messages area with independent scrolling */}
      <div className="flex-1 min-h-0">
        <ScrollArea className="h-full" ref={scrollAreaRef}>
          <div className="p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-32 text-neutral-500">
                <div className="text-center">
                  <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Waiting for messages...</p>
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <MessageBubble
                  key={`${message.message_id}-${index}`}
                  message={message}
                  onCopy={onCopy}
                  copiedMessageId={copiedMessageId}
                />
              ))
            )}

            {/* Loading indicator */}
            {isActive && isLoading && (
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0 mt-1">
                  <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                </div>
                <div className="flex-1">
                  <div className="bg-neutral-50/5 rounded-2xl p-4 shadow-sm border border-neutral-700/50">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.32s]"></div>
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.16s]"></div>
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                      </div>
                      <p className="text-sm text-neutral-400">Generating response...</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
};

// ============================================================================
// MAIN PARALLEL RESEARCH INTERFACE COMPONENT
// ============================================================================

const ParallelResearchInterface: React.FC<ParallelResearchInterfaceProps> = ({
  sequences,
  parallelMessages,
  activeTabId,
  onTabChange,
  isLoading = false,
  className,
}) => {
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [showMobileView, setShowMobileView] = useState(false);

  // Handle copy functionality
  const handleCopy = useCallback(async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(Date.now().toString());
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  }, []);

  // Check if we're on mobile
  useEffect(() => {
    const checkIsMobile = () => {
      setShowMobileView(window.innerWidth < 1024);
    };

    checkIsMobile();
    window.addEventListener('resize', checkIsMobile);
    return () => window.removeEventListener('resize', checkIsMobile);
  }, []);

  if (sequences.length === 0) {
    return (
      <div className={cn('h-full flex items-center justify-center', className)}>
        <div className="text-center text-neutral-500">
          <Zap className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No Sequences Available</h3>
          <p className="text-sm text-neutral-600">
            Sequences will appear here when parallel research begins
          </p>
        </div>
      </div>
    );
  }

  // Mobile stacked view
  if (showMobileView) {
    return (
      <div className={cn('h-full', className)}>
        {/* Mobile tab selector */}
        <div className="border-b border-neutral-700 bg-neutral-900">
          <div className="flex overflow-x-auto">
            {sequences.map((sequence) => (
              <button
                key={sequence.sequence_id}
                onClick={() => onTabChange?.(sequence.sequence_id)}
                className={cn(
                  'flex-shrink-0 px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                  activeTabId === sequence.sequence_id
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-neutral-400 hover:text-neutral-200'
                )}
              >
                {sequence.sequence_name}
              </button>
            ))}
          </div>
        </div>

        {/* Active sequence content */}
        <div className="h-[calc(100%-57px)]">
          {sequences.map(sequence => (
            <div
              key={sequence.sequence_id}
              className={cn(
                'h-full',
                activeTabId === sequence.sequence_id ? 'block' : 'hidden'
              )}
            >
              <SequenceColumn
                sequence={sequence}
                messages={parallelMessages[sequence.sequence_id] || []}
                isActive={activeTabId === sequence.sequence_id}
                isLoading={isLoading}
                onCopy={handleCopy}
                copiedMessageId={copiedMessageId}
              />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Desktop 3-column side-by-side view (truly horizontal)
  return (
    <div className={cn('h-full w-full', className)}>
      <div className="flex h-full divide-x divide-neutral-700/50">
        {sequences.slice(0, 3).map((sequence) => (
          <div 
            key={sequence.sequence_id} 
            className="flex-1 min-w-0 h-full overflow-hidden"
            style={{ width: `${100 / Math.min(sequences.length, 3)}%` }}
          >
            <SequenceColumn
              sequence={sequence}
              messages={parallelMessages[sequence.sequence_id] || []}
              isActive={activeTabId === sequence.sequence_id}
              isLoading={isLoading}
              onCopy={handleCopy}
              copiedMessageId={copiedMessageId}
              className="h-full"
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export { ParallelResearchInterface };
export default ParallelResearchInterface;