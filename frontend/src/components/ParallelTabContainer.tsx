import React, { useMemo, useCallback } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { CheckCircle, Loader2, Pause, RotateCcw, GitBranch } from 'lucide-react';
import { cn } from '@/lib/utils';
import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';
import { TypedMarkdown } from '@/components/ui/typed-markdown';
import { ActivityTimeline, ProcessedEvent } from '@/components/ActivityTimeline';
import ReactMarkdown from 'react-markdown';
import { assignStrategyTheme, StrategyTheme } from '@/lib/strategy-themes';

// ========================================
// Component Interfaces
// ========================================


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
  strategyTheme: StrategyTheme;
  onClick: () => void;
}

interface SequenceTabContentProps {
  sequence: LLMGeneratedSequence;
  messages: RoutedMessage[];
  isActive: boolean;
  isLoading: boolean;
  index: number;
  strategyTheme: StrategyTheme;
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
  strategyTheme,
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

  const IconComponent = strategyTheme.icon;
  
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex flex-col items-start gap-1 px-3 py-3 text-sm font-medium rounded-t-md transition-all min-w-0 relative group",
        isActive 
          ? "bg-neutral-800 text-white shadow-sm"
          : "bg-neutral-900 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800"
      )}
      style={{
        borderBottom: isActive ? `2px solid ${strategyTheme.colors.primary}` : '2px solid transparent',
        borderLeft: isActive ? `3px solid ${strategyTheme.colors.primary}` : '3px solid transparent'
      }}
    >
      {/* Strategic Header Row */}
      <div className="flex items-center gap-2 w-full min-w-0">
        <div 
          className="p-1 rounded flex-shrink-0"
          style={{ 
            backgroundColor: isActive ? `${strategyTheme.colors.primary}20` : 'transparent',
            color: isActive ? strategyTheme.colors.primary : strategyTheme.colors.secondary
          }}
        >
          <IconComponent className="w-3 h-3" />
        </div>
        <Badge 
          variant="outline" 
          className="text-xs flex-shrink-0"
          style={{
            borderColor: strategyTheme.colors.primary,
            color: isActive ? strategyTheme.colors.primary : strategyTheme.colors.secondary
          }}
        >
          {index + 1}
        </Badge>
        <div className="flex items-center gap-1 flex-shrink-0 ml-auto">
          {getStatusIndicator()}
          {messageCount > 0 && (
            <Badge 
              variant="secondary" 
              className="text-xs px-1 py-0"
              style={{
                backgroundColor: `${strategyTheme.colors.primary}30`,
                color: strategyTheme.colors.text,
                borderColor: strategyTheme.colors.primary
              }}
            >
              {messageCount}
            </Badge>
          )}
        </div>
      </div>
      
      {/* Strategic Identity Row */}
      <div className="w-full min-w-0 space-y-1">
        <div className="flex items-center gap-2">
          <span 
            className="font-semibold truncate text-xs"
            style={{ color: isActive ? strategyTheme.colors.primary : strategyTheme.colors.text }}
          >
            {strategyTheme.shortName}
          </span>
          <span className="text-xs text-neutral-500 hidden lg:block">
            {getStatusText()}
          </span>
        </div>
        <span className="text-xs text-neutral-400 truncate max-w-[140px] sm:max-w-[160px] md:max-w-[180px] block">
          {sequence.sequence_name}
        </span>
        
        {/* Strategic Characteristics Pills */}
        {isActive && (
          <div className="flex flex-wrap gap-1 mt-1 max-w-[200px]">
            {strategyTheme.characteristics.slice(0, 2).map((char, idx) => (
              <span
                key={idx}
                className="text-xs px-1 py-0.5 rounded"
                style={{
                  backgroundColor: `${strategyTheme.colors.primary}20`,
                  color: strategyTheme.colors.primary,
                  fontSize: '10px'
                }}
              >
                {char}
              </span>
            ))}
          </div>
        )}
      </div>
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
  strategyTheme,
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
      {/* Strategic Context Header */}
      <div 
        className="border-b pb-4 mb-4 rounded-t-lg p-4 -mt-4 -mx-4 mx-4"
        style={{
          borderColor: strategyTheme.colors.border,
          backgroundColor: `${strategyTheme.colors.background}15`
        }}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div 
              className="p-2 rounded-lg"
              style={{ 
                backgroundColor: `${strategyTheme.colors.primary}20`,
                color: strategyTheme.colors.primary
              }}
            >
              <strategyTheme.icon className="w-5 h-5" />
            </div>
            <div>
              <h3 
                className="font-bold text-lg mb-1"
                style={{ color: strategyTheme.colors.primary }}
              >
                {strategyTheme.name}
              </h3>
              <p className="text-sm font-medium text-white">
                {sequence.sequence_name}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge 
              variant="outline" 
              className="text-xs"
              style={{
                borderColor: strategyTheme.colors.primary,
                color: strategyTheme.colors.text
              }}
            >
              {Math.round(sequence.confidence_score * 100)}% confidence
            </Badge>
            {contentMessages.length > 0 && (
              <Badge 
                variant="secondary" 
                className="text-xs"
                style={{
                  backgroundColor: `${strategyTheme.colors.primary}30`,
                  color: strategyTheme.colors.text
                }}
              >
                {contentMessages.length} responses
              </Badge>
            )}
          </div>
        </div>
        
        {/* Strategic Methodology Description */}
        <div className="mb-3">
          <p 
            className="text-sm mb-2 font-medium"
            style={{ color: strategyTheme.colors.text }}
          >
            {strategyTheme.description}
          </p>
          <p className="text-xs text-neutral-400">
            <strong>Methodology:</strong> {strategyTheme.methodology}
          </p>
        </div>
        
        {/* Strategic Characteristics */}
        <div className="mb-3">
          <div className="flex flex-wrap gap-1 mb-2">
            {strategyTheme.characteristics.map((char, idx) => (
              <span
                key={idx}
                className="text-xs px-2 py-1 rounded-full"
                style={{
                  backgroundColor: `${strategyTheme.colors.primary}25`,
                  color: strategyTheme.colors.primary,
                  border: `1px solid ${strategyTheme.colors.primary}50`
                }}
              >
                {char}
              </span>
            ))}
          </div>
        </div>
        
        {/* Research Details */}
        <div className="text-xs text-neutral-400 space-y-1">
          <div>
            <strong style={{ color: strategyTheme.colors.secondary }}>Focus:</strong> {sequence.research_focus}
          </div>
          <div>
            <strong style={{ color: strategyTheme.colors.secondary }}>Approach:</strong> {strategyTheme.approach.focus}
          </div>
          {sequence.agent_names.length > 0 && (
            <div>
              <strong style={{ color: strategyTheme.colors.secondary }}>Agents:</strong> {sequence.agent_names.join(', ')}
            </div>
          )}
        </div>
      </div>

      {/* Activity Timeline */}
      {activityEvents.length > 0 && (
        <div 
          className="rounded-lg p-3 border"
          style={{
            backgroundColor: `${strategyTheme.colors.background}10`,
            borderColor: `${strategyTheme.colors.border}30`
          }}
        >
          <h4 className="text-sm font-medium text-white mb-2 flex items-center gap-2">
            <div 
              className="w-2 h-2 rounded-full animate-pulse" 
              style={{ backgroundColor: strategyTheme.colors.primary }}
            />
            <span style={{ color: strategyTheme.colors.primary }}>Activity Timeline</span>
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
              <div 
                key={message.message_id || msgIndex} 
                className="rounded-lg p-3 border"
                style={{
                  backgroundColor: `${strategyTheme.colors.background}08`,
                  borderColor: `${strategyTheme.colors.border}20`
                }}
              >
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
                <div 
                  className="mt-2 pt-2 border-t flex items-center justify-between text-xs"
                  style={{ 
                    borderColor: `${strategyTheme.colors.border}30`,
                    color: strategyTheme.colors.secondary
                  }}
                >
                  <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
                  {message.current_agent && (
                    <Badge 
                      variant="outline" 
                      className="text-xs"
                      style={{
                        borderColor: strategyTheme.colors.primary,
                        color: strategyTheme.colors.primary
                      }}
                    >
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
          <div 
            className="rounded-lg p-3 border"
            style={{
              backgroundColor: `${strategyTheme.colors.background}10`,
              borderColor: `${strategyTheme.colors.border}30`
            }}
          >
            <div className="flex items-center gap-2 text-sm">
              <Loader2 
                className="w-4 h-4 animate-spin" 
                style={{ color: strategyTheme.colors.primary }}
              />
              <span style={{ color: strategyTheme.colors.text }}>Generating more content...</span>
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
  // Assign strategy themes to sequences
  const sequencesWithThemes = useMemo(() => {
    return sequences.map(sequence => ({
      sequence,
      strategyTheme: assignStrategyTheme(sequence)
    }));
  }, [sequences]);

  // Ensure we have a valid active tab
  const validActiveTabId = useMemo(() => {
    if (sequencesWithThemes.find(s => s.sequence.sequence_id === activeTabId)) {
      return activeTabId;
    }
    return sequencesWithThemes.length > 0 ? sequencesWithThemes[0].sequence.sequence_id : '';
  }, [activeTabId, sequencesWithThemes]);

  // Calculate typing status across all tabs for simultaneous animation feedback
  const typingStatus = useMemo(() => {
    const totalTabs = sequencesWithThemes.length;
    const typingTabs = sequencesWithThemes.filter(({ sequence }) => 
      getTabStatus(sequence.sequence_id, parallelMessages) === 'typing'
    ).length;
    const completedTabs = sequencesWithThemes.filter(({ sequence }) => 
      getTabStatus(sequence.sequence_id, parallelMessages) === 'completed'
    ).length;
    
    return { totalTabs, typingTabs, completedTabs };
  }, [sequencesWithThemes, parallelMessages]);

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

  if (sequencesWithThemes.length === 0) {
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
          {sequencesWithThemes.map(({ sequence, strategyTheme }, index) => (
            <TabButton
              key={sequence.sequence_id}
              sequence={sequence}
              index={index}
              isActive={validActiveTabId === sequence.sequence_id}
              status={getTabStatus(sequence.sequence_id, parallelMessages)}
              messageCount={parallelMessages[sequence.sequence_id]?.length || 0}
              strategyTheme={strategyTheme}
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
        {sequencesWithThemes.map(({ sequence, strategyTheme }) => (
          <SequenceTabContent
            key={sequence.sequence_id}
            sequence={sequence}
            messages={parallelMessages[sequence.sequence_id] || []}
            isActive={validActiveTabId === sequence.sequence_id}
            isLoading={isLoading}
            index={sequencesWithThemes.findIndex(s => s.sequence.sequence_id === sequence.sequence_id)}
            strategyTheme={strategyTheme}
          />
        ))}
        
        {/* Empty state */}
        {sequencesWithThemes.length === 0 && (
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