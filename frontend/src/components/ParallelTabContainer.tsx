import React, { useState, useMemo, useCallback } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { CheckCircle, Loader2, Pause, Play, RotateCcw, GitBranch } from 'lucide-react';
import { cn } from '@/lib/utils';
import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';
import { TypedMarkdown } from '@/components/ui/typed-markdown';
import { ActivityTimeline, ProcessedEvent } from '@/components/ActivityTimeline';
import ReactMarkdown from 'react-markdown';

// ========================================
// Component Interfaces
// ========================================

interface TabState {
  id: string;
  name: string;
  status: 'initializing' | 'typing' | 'paused' | 'completed' | 'error';
  messages: RoutedMessage[];
  progress: number;
  isActive: boolean;
}

interface ParallelTabContainerProps {
  sequences: LLMGeneratedSequence[];
  parallelMessages: Record<string, RoutedMessage[]>;
  activeTabId: string;
  onTabChange: (tabId: string) => void;
  isLoading: boolean;
  className?: string;
}

interface TabButtonProps {
  sequence: LLMGeneratedSequence;
  index: number;
  isActive: boolean;
  status: 'initializing' | 'typing' | 'paused' | 'completed' | 'error';
  messageCount: number;
  onClick: () => void;
}

interface SequenceTabContentProps {
  sequence: LLMGeneratedSequence;
  messages: RoutedMessage[];
  isActive: boolean;
  isLoading: boolean;
  index: number;
}

// ========================================
// Utility Functions
// ========================================

const getTabStatus = (
  sequenceId: string, 
  parallelMessages: Record<string, RoutedMessage[]>
): 'initializing' | 'typing' | 'paused' | 'completed' | 'error' => {
  const messages = parallelMessages[sequenceId] || [];
  
  if (messages.length === 0) {
    return 'initializing';
  }
  
  const lastMessage = messages[messages.length - 1];
  
  if (lastMessage.message_type === 'completion') {
    return 'completed';
  }
  
  if (lastMessage.message_type === 'error') {
    return 'error';
  }
  
  return 'typing';
};

// Markdown components for consistent styling
const mdComponents = {
  h1: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h1 className={cn('text-xl font-bold mt-3 mb-2 text-white', className)} {...props}>
      {children}
    </h1>
  ),
  h2: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h2 className={cn('text-lg font-bold mt-3 mb-2 text-white', className)} {...props}>
      {children}
    </h2>
  ),
  h3: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h3 className={cn('text-base font-bold mt-2 mb-1 text-white', className)} {...props}>
      {children}
    </h3>
  ),
  p: ({ className, children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p className={cn('mb-2 leading-6 text-neutral-200', className)} {...props}>
      {children}
    </p>
  ),
  ul: ({ className, children, ...props }: React.HTMLAttributes<HTMLUListElement>) => (
    <ul className={cn('list-disc pl-4 mb-2 text-neutral-200', className)} {...props}>
      {children}
    </ul>
  ),
  ol: ({ className, children, ...props }: React.HTMLAttributes<HTMLOListElement>) => (
    <ol className={cn('list-decimal pl-4 mb-2 text-neutral-200', className)} {...props}>
      {children}
    </ol>
  ),
  li: ({ className, children, ...props }: React.HTMLAttributes<HTMLLIElement>) => (
    <li className={cn('mb-1', className)} {...props}>
      {children}
    </li>
  ),
  code: ({ className, children, ...props }: React.HTMLAttributes<HTMLElement>) => (
    <code
      className={cn(
        'bg-neutral-900 rounded px-1 py-0.5 font-mono text-xs text-blue-300',
        className
      )}
      {...props}
    >
      {children}
    </code>
  ),
  pre: ({ className, children, ...props }: React.HTMLAttributes<HTMLPreElement>) => (
    <pre
      className={cn(
        'bg-neutral-900 p-2 rounded-lg overflow-x-auto font-mono text-xs my-2 text-neutral-200',
        className
      )}
      {...props}
    >
      {children}
    </pre>
  ),
};

// ========================================
// TabButton Component
// ========================================

const TabButton: React.FC<TabButtonProps> = React.memo(({
  sequence,
  index,
  isActive,
  status,
  messageCount,
  onClick,
}) => {
  const getStatusIndicator = () => {
    switch (status) {
      case 'typing':
        return <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />;
      case 'paused':
        return <Pause className="w-3 h-3 text-yellow-400" />;
      case 'completed':
        return <CheckCircle className="w-3 h-3 text-green-400" />;
      case 'error':
        return <div className="w-2 h-2 bg-red-400 rounded-full" />;
      default:
        return <div className="w-2 h-2 bg-neutral-500 rounded-full" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'typing': return 'Generating...';
      case 'paused': return 'Paused';
      case 'completed': return 'Complete';
      case 'error': return 'Error';
      default: return 'Starting...';
    }
  };

  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-t-md transition-all min-w-0",
        isActive 
          ? "bg-neutral-800 text-white border-b-2 border-blue-400 shadow-sm"
          : "bg-neutral-900 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800 border-b-2 border-transparent"
      )}
    >
      <Badge variant="outline" className="text-xs flex-shrink-0">
        {index + 1}
      </Badge>
      <span className="truncate max-w-[100px] sm:max-w-[120px] md:max-w-[140px]">
        {sequence.sequence_name}
      </span>
      <div className="flex items-center gap-1 flex-shrink-0">
        {getStatusIndicator()}
        {messageCount > 0 && (
          <Badge variant="secondary" className="text-xs px-1 py-0">
            {messageCount}
          </Badge>
        )}
      </div>
      {/* Status tooltip for mobile */}
      <span className="sr-only sm:not-sr-only text-xs text-neutral-500 hidden lg:block">
        {getStatusText()}
      </span>
    </button>
  );
});

// ========================================
// SequenceTabContent Component
// ========================================

const SequenceTabContent: React.FC<SequenceTabContentProps> = React.memo(({
  sequence,
  messages,
  isActive,
  isLoading,
  index,
}) => {
  if (!isActive) return null;

  // Convert messages to activity events for the timeline
  const activityEvents: ProcessedEvent[] = messages
    .filter(msg => msg.message_type === 'progress' || msg.message_type === 'agent_transition')
    .map((msg, idx) => ({
      title: msg.message_type === 'agent_transition' 
        ? `🔄 ${msg.current_agent || 'Agent'} Transition`
        : `⚡ Processing Step ${idx + 1}`,
      data: typeof msg.content === 'string' 
        ? msg.content.slice(0, 100) + (msg.content.length > 100 ? '...' : '')
        : 'Processing...',
      timestamp: msg.timestamp,
    }));

  // Get content messages for display
  const contentMessages = messages.filter(msg => 
    msg.message_type === 'result' || 
    (msg.message_type === 'progress' && msg.content && typeof msg.content === 'string')
  );

  return (
    <div className="p-4 space-y-4 h-full overflow-y-auto">
      {/* Sequence Header */}
      <div className="border-b border-neutral-700 pb-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-blue-400" />
            {sequence.sequence_name}
          </h3>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              {Math.round(sequence.confidence_score * 100)}% confidence
            </Badge>
            {contentMessages.length > 0 && (
              <Badge variant="secondary" className="text-xs">
                {contentMessages.length} responses
              </Badge>
            )}
          </div>
        </div>
        <p className="text-sm text-neutral-400 mb-2">{sequence.rationale}</p>
        <div className="flex items-center gap-2 text-xs text-neutral-500">
          <span>Focus: {sequence.research_focus}</span>
          {sequence.agent_names.length > 0 && (
            <>
              <span>•</span>
              <span>Agents: {sequence.agent_names.join(', ')}</span>
            </>
          )}
        </div>
      </div>

      {/* Activity Timeline */}
      {activityEvents.length > 0 && (
        <div className="bg-neutral-800 rounded-lg p-3">
          <h4 className="text-sm font-medium text-white mb-2 flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
            Activity Timeline
          </h4>
          <ActivityTimeline
            processedEvents={activityEvents}
            isLoading={isLoading && isActive}
          />
        </div>
      )}

      {/* Messages */}
      <div className="space-y-3">
        {contentMessages.length === 0 ? (
          <div className="bg-neutral-800 rounded-lg p-4 text-center">
            {isLoading ? (
              <div className="flex items-center justify-center gap-2 text-neutral-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Initializing sequence...</span>
              </div>
            ) : (
              <p className="text-neutral-400">No content generated yet</p>
            )}
          </div>
        ) : (
          contentMessages.map((message, msgIndex) => {
            // Calculate typing animation delay based on sequence index and message index
            // This creates simultaneous typing effect across tabs
            const baseDelay = index * 100; // Stagger start times slightly between sequences
            const messageDelay = msgIndex * 200; // Delay between messages in same sequence
            const totalDelay = baseDelay + messageDelay;
            
            return (
              <div key={message.message_id || msgIndex} className="bg-neutral-800 rounded-lg p-3">
                {typeof message.content === 'string' ? (
                  <TypedMarkdown 
                    components={mdComponents} 
                    speed={20} // Slightly faster for better UX
                    delay={totalDelay}
                    hideCursor={!isActive || msgIndex !== contentMessages.length - 1} // Show cursor only on last message in active tab
                  >
                    {message.content}
                  </TypedMarkdown>
                ) : (
                  <ReactMarkdown components={mdComponents}>
                    {JSON.stringify(message.content, null, 2)}
                  </ReactMarkdown>
                )}
                <div className="mt-2 pt-2 border-t border-neutral-700 flex items-center justify-between text-xs text-neutral-500">
                  <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
                  {message.current_agent && (
                    <Badge variant="outline" className="text-xs">
                      {message.current_agent}
                    </Badge>
                  )}
                </div>
              </div>
            );
          })
        )}

        {/* Loading indicator for active sequence */}
        {isLoading && isActive && contentMessages.length > 0 && (
          <div className="bg-neutral-800 rounded-lg p-3">
            <div className="flex items-center gap-2 text-neutral-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Generating more content...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

// ========================================
// Main ParallelTabContainer Component
// ========================================

const ParallelTabContainer: React.FC<ParallelTabContainerProps> = ({
  sequences,
  parallelMessages,
  activeTabId,
  onTabChange,
  isLoading,
  className,
}) => {
  // Ensure we have a valid active tab
  const validActiveTabId = useMemo(() => {
    if (sequences.find(s => s.sequence_id === activeTabId)) {
      return activeTabId;
    }
    return sequences.length > 0 ? sequences[0].sequence_id : '';
  }, [activeTabId, sequences]);

  // Calculate typing status across all tabs for simultaneous animation feedback
  const typingStatus = useMemo(() => {
    const totalTabs = sequences.length;
    const typingTabs = sequences.filter(seq => 
      getTabStatus(seq.sequence_id, parallelMessages) === 'typing'
    ).length;
    const completedTabs = sequences.filter(seq => 
      getTabStatus(seq.sequence_id, parallelMessages) === 'completed'
    ).length;
    
    return { totalTabs, typingTabs, completedTabs };
  }, [sequences, parallelMessages]);

  // Memoize callback to prevent re-renders
  const memoizedOnTabChange = useCallback((tabId: string) => {
    onTabChange(tabId);
  }, [onTabChange]);

  // Update parent if active tab changed
  React.useEffect(() => {
    if (validActiveTabId !== activeTabId && validActiveTabId) {
      memoizedOnTabChange(validActiveTabId);
    }
  }, [validActiveTabId, activeTabId, memoizedOnTabChange]);

  if (sequences.length === 0) {
    return null;
  }

  return (
    <div className={cn("bg-neutral-800 border border-neutral-700 rounded-lg overflow-hidden my-4", className)}>
      {/* Tab Navigation */}
      <div className="border-b border-neutral-700 bg-neutral-900">
        {/* Simultaneous typing indicator */}
        {typingStatus.typingTabs > 1 && (
          <div className="px-3 py-2 bg-green-500/10 border-b border-green-500/20">
            <div className="flex items-center gap-2 text-green-400 text-xs">
              <div className="flex items-center gap-1">
                {Array.from({ length: typingStatus.typingTabs }).map((_, i) => (
                  <div 
                    key={i} 
                    className="w-2 h-2 bg-green-400 rounded-full animate-pulse" 
                    style={{ animationDelay: `${i * 0.2}s` }}
                  />
                ))}
              </div>
              <span>
                {typingStatus.typingTabs} tabs generating simultaneously
              </span>
              <span className="text-green-300">|
                Progress: {typingStatus.completedTabs}/{typingStatus.totalTabs} completed
              </span>
            </div>
          </div>
        )}
        
        <div className="flex items-center gap-1 p-2 overflow-x-auto">
          {sequences.map((sequence, index) => (
            <TabButton
              key={sequence.sequence_id}
              sequence={sequence}
              index={index}
              isActive={validActiveTabId === sequence.sequence_id}
              status={getTabStatus(sequence.sequence_id, parallelMessages)}
              messageCount={parallelMessages[sequence.sequence_id]?.length || 0}
              onClick={() => memoizedOnTabChange(sequence.sequence_id)}
            />
          ))}
          {/* Tab controls */}
          <div className="flex items-center gap-1 ml-auto pl-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 text-neutral-400 hover:text-neutral-200"
              title="Refresh all tabs"
            >
              <RotateCcw className="w-3 h-3" />
            </Button>
          </div>
        </div>
      </div>
      
      {/* Tab Content Area */}
      <div className="min-h-[300px] max-h-[600px] overflow-y-auto">
        {sequences.map((sequence) => (
          <SequenceTabContent
            key={sequence.sequence_id}
            sequence={sequence}
            messages={parallelMessages[sequence.sequence_id] || []}
            isActive={validActiveTabId === sequence.sequence_id}
            isLoading={isLoading}
            index={sequences.findIndex(s => s.sequence_id === sequence.sequence_id)}
          />
        ))}
        
        {/* Empty state */}
        {sequences.length === 0 && (
          <div className="h-[300px] flex items-center justify-center text-neutral-500">
            <div className="text-center">
              <GitBranch className="w-8 h-8 mx-auto mb-2" />
              <p>No parallel sequences available</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ParallelTabContainer;