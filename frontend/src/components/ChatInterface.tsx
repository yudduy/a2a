import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import type { Message } from '@langchain/langgraph-sdk';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Copy, CopyCheck, Activity, Loader2 } from 'lucide-react';
import { InputForm } from '@/components/InputForm';
import { Button } from '@/components/ui/button';
import ReactMarkdown from 'react-markdown';
import { cn } from '@/lib/utils';
import {
  ProcessedEvent,
} from '@/components/ActivityTimeline';
import { ToolMessageDisplay } from '@/components/ToolMessageDisplay';
import {
  extractToolCallsFromMessage,
  findToolMessageForCall,
} from '@/types/messages';
import { ToolCall } from '@/types/tools';
import ParallelTabContainer from '@/components/ParallelTabContainer';
import SupervisorAnnouncementMessage from '@/components/SupervisorAnnouncementMessage';
import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';
import { EnhancedErrorBoundary } from '@/components/ui/enhanced-error-boundary';

// ============================================================================
// INTERFACES AND TYPES - SIMPLIFIED
// ============================================================================

// Unified ChatInterface props - removed unused mode complexity  
export interface ChatInterfaceProps {
  // Core chat functionality
  messages?: Message[];
  isLoading?: boolean;
  onSubmit?: (inputValue: string, enableParallel?: boolean) => void;
  onCancel?: () => void;
  onReset?: () => void;
  
  // Activity and events
  liveActivityEvents?: ProcessedEvent[];
  historicalActivities?: Record<string, ProcessedEvent[]>;
  
  // In-place parallel functionality (unified paradigm)
  parallelTabsState?: ParallelTabsState;
  parallelMessages?: Record<string, RoutedMessage[]>;
  onParallelTabChange?: (tabId: string) => void;
  onTabsInitialized?: () => void;
  
  // Layout and styling
  scrollAreaRef?: React.RefObject<HTMLDivElement | null>;
  className?: string;
}

// Group messages to combine AI responses with their tool calls and results
interface MessageGroup {
  id: string;
  type: 'human' | 'ai_complete' | 'supervisor_announcement';
  messages: Message[];
  primaryMessage: Message;
  toolCalls: ToolCall[];
  toolResults: Message[];
  sequences?: LLMGeneratedSequence[]; // For supervisor announcements
}

// Interface for parallel tabs state
interface ParallelTabsState {
  isActive: boolean;
  activeTabId: string;
  sequences: LLMGeneratedSequence[];
  hasAnnounced: boolean;
}

// ============================================================================
// MESSAGE GROUPING LOGIC
// ============================================================================

// Function to group messages intelligently
const groupMessages = (messages: Message[]): MessageGroup[] => {
  const groups: MessageGroup[] = [];
  let currentGroup: MessageGroup | null = null;

  messages.forEach((message) => {
    const messageId = message.id || `msg_${Date.now()}_${Math.random()}`;

    if (message.type === 'human') {
      // Human messages start new groups
      currentGroup = {
        id: messageId,
        type: 'human',
        messages: [message],
        primaryMessage: message,
        toolCalls: [],
        toolResults: [],
      };
      groups.push(currentGroup);
    } else if (message.type === 'ai') {
      const toolCalls = extractToolCallsFromMessage(message);
      
      // Check if this is a supervisor announcement with sequences
      const messageContent = typeof message.content === 'string' ? message.content : '';
      const hasSequences = messageContent.includes('research sequences') || 
                          messageContent.includes('parallel analysis') ||
                          messageContent.includes('strategic approach');
      
      if (hasSequences) {
        // This is a supervisor announcement
        currentGroup = {
          id: messageId,
          type: 'supervisor_announcement',
          messages: [message],
          primaryMessage: message,
          toolCalls,
          toolResults: [],
          sequences: [], // Will be populated by the app when sequences are generated
        };
        groups.push(currentGroup);
      } else {
        // Regular AI message
        currentGroup = {
          id: messageId,
          type: 'ai_complete',
          messages: [message],
          primaryMessage: message,
          toolCalls,
          toolResults: [],
        };
        groups.push(currentGroup);
      }
    } else if (message.type === 'tool' && currentGroup && currentGroup.type !== 'human') {
      // Tool results belong to the current AI group
      currentGroup.messages.push(message);
      currentGroup.toolResults.push(message);
    }
  });

  return groups;
};

// ============================================================================
// MARKDOWN COMPONENTS
// ============================================================================

const mdComponents = {
  h1: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h1 className={cn('text-2xl font-bold mb-4 text-neutral-100', className)} {...props}>
      {children}
    </h1>
  ),
  h2: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h2 className={cn('text-xl font-semibold mb-3 text-neutral-100', className)} {...props}>
      {children}
    </h2>
  ),
  h3: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h3 className={cn('text-lg font-semibold mb-2 text-neutral-200', className)} {...props}>
      {children}
    </h3>
  ),
  p: ({ className, children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p className={cn('mb-3 leading-relaxed text-neutral-300', className)} {...props}>
      {children}
    </p>
  ),
  ul: ({ className, children, ...props }: React.HTMLAttributes<HTMLUListElement>) => (
    <ul className={cn('list-disc pl-6 mb-3 space-y-1 text-neutral-300', className)} {...props}>
      {children}
    </ul>
  ),
  ol: ({ className, children, ...props }: React.HTMLAttributes<HTMLOListElement>) => (
    <ol className={cn('list-decimal pl-6 mb-3 space-y-1 text-neutral-300', className)} {...props}>
      {children}
    </ol>
  ),
  li: ({ className, children, ...props }: React.HTMLAttributes<HTMLLIElement>) => (
    <li className={cn('text-neutral-300', className)} {...props}>
      {children}
    </li>
  ),
  code: ({ className, children, ...props }: React.HTMLAttributes<HTMLElement>) => (
    <code className={cn('bg-neutral-800 text-blue-300 rounded px-1.5 py-0.5 text-sm font-mono', className)} {...props}>
      {children}
    </code>
  ),
  pre: ({ className, children, ...props }: React.HTMLAttributes<HTMLPreElement>) => (
    <pre className={cn('bg-neutral-800 rounded-md p-4 overflow-x-auto mb-3 border border-neutral-700', className)} {...props}>
      {children}
    </pre>
  ),
  blockquote: ({ className, children, ...props }: React.HTMLAttributes<HTMLQuoteElement>) => (
    <blockquote className={cn('border-l-4 border-neutral-600 pl-4 mb-3 text-neutral-400 italic', className)} {...props}>
      {children}
    </blockquote>
  ),
  a: ({ className, children, href, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a 
      href={href} 
      className={cn('text-blue-400 hover:text-blue-300 underline', className)} 
      target="_blank" 
      rel="noopener noreferrer" 
      {...props}
    >
      {children}
    </a>
  ),
};

// ============================================================================
// MESSAGE BUBBLE COMPONENTS - SIMPLIFIED
// ============================================================================

interface HumanMessageBubbleProps {
  group: MessageGroup;
  mdComponents: any;
}

const HumanMessageBubble: React.FC<HumanMessageBubbleProps> = ({ group, mdComponents }) => {
  const messageContent = typeof group.primaryMessage.content === 'string' 
    ? group.primaryMessage.content 
    : JSON.stringify(group.primaryMessage.content);

  return (
    <div className="bg-blue-600 text-white rounded-2xl px-4 py-3 max-w-2xl shadow-sm">
      <ReactMarkdown components={mdComponents}>
        {messageContent}
      </ReactMarkdown>
    </div>
  );
};

interface AiMessageBubbleProps {
  group: MessageGroup;
  mdComponents: any;
  handleCopy: (text: string, messageId: string) => void;
  copiedMessageId: string | null;
  allMessages: Message[];
}

const AiMessageBubble: React.FC<AiMessageBubbleProps> = ({ 
  group, 
  mdComponents,
  handleCopy,
  copiedMessageId,
  allMessages
}) => {
  const messageId = group.primaryMessage.id || '';
  const messageContent = typeof group.primaryMessage.content === 'string' 
    ? group.primaryMessage.content 
    : JSON.stringify(group.primaryMessage.content);

  return (
    <div className="flex items-start gap-3 w-full max-w-none">
      <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0 mt-1">
        <Activity className="w-4 h-4 text-green-400" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="bg-neutral-800 rounded-2xl p-4 shadow-sm max-w-none">
          {/* Main content */}
          <div className="prose prose-invert max-w-none">
            <ReactMarkdown components={mdComponents}>
              {messageContent}
            </ReactMarkdown>
          </div>
          
          {/* Tool calls and results */}
          {group.toolCalls.length > 0 && (
            <div className="mt-4 space-y-3">
              {group.toolCalls.map((toolCall, index) => {
                const toolResult = findToolMessageForCall(allMessages, toolCall.id || '');
                return (
                  <ToolMessageDisplay
                    key={toolCall.id || index}
                    toolCall={toolCall}
                    toolMessage={toolResult || undefined}
                    isExpanded={false}
                    onToggle={() => {}}
                  />
                );
              })}
            </div>
          )}
          
          {/* Copy button */}
          <div className="flex justify-end mt-3 pt-2 border-t border-neutral-700">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleCopy(messageContent, messageId)}
              className="text-neutral-400 hover:text-neutral-200 h-8 px-2"
            >
              {copiedMessageId === messageId ? (
                <CopyCheck className="w-4 h-4" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// MAIN CHAT INTERFACE COMPONENT - UNIFIED
// ============================================================================

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  // Core functionality
  messages = [],
  isLoading = false,
  onSubmit,
  onCancel,
  onReset,
  
  // Activity tracking
  liveActivityEvents = [],
  
  // Parallel functionality - unified in-place tabs only
  parallelTabsState,
  parallelMessages,
  onParallelTabChange,
  onTabsInitialized,
  
  // Layout
  scrollAreaRef: externalScrollAreaRef,
  className
}) => {
  // Create internal scroll area ref if none provided
  const internalScrollAreaRef = useRef<HTMLDivElement | null>(null);
  const scrollAreaRef = externalScrollAreaRef || internalScrollAreaRef;
  
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const prevMessageCountRef = useRef(messages.length);
  const prevIsLoadingRef = useRef(isLoading);

  // Copy functionality
  const handleCopy = async (text: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  // Auto-scroll functionality
  const scrollToBottom = useCallback((smooth = true) => {
    if (!scrollAreaRef.current) return;
    
    const viewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
    if (!viewport) return;

    viewport.scrollTo({
      top: viewport.scrollHeight,
      behavior: smooth ? 'smooth' : 'auto',
    });
  }, [scrollAreaRef]);

  const isNearBottom = useCallback((): boolean => {
    if (!scrollAreaRef.current) return true;
    
    const viewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
    if (!viewport) return true;

    const { scrollTop, scrollHeight, clientHeight } = viewport;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    
    return distanceFromBottom <= 100;
  }, [scrollAreaRef]);

  // Auto-scroll when messages change
  useEffect(() => {
    const hasNewMessages = messages.length > prevMessageCountRef.current;
    
    if ((hasNewMessages || (!isLoading && prevIsLoadingRef.current)) && isNearBottom()) {
      setTimeout(() => scrollToBottom(true), 50);
    }

    prevMessageCountRef.current = messages.length;
    prevIsLoadingRef.current = isLoading;
  }, [messages.length, isLoading, isNearBottom, scrollToBottom]);

  // Group messages for intelligent display
  const messageGroups = useMemo(() => {
    const groups = groupMessages(messages);
    
    // Enhance supervisor announcement groups with sequences data
    return groups.map(group => {
      if (group.type === 'supervisor_announcement' && parallelTabsState?.sequences) {
        return {
          ...group,
          sequences: parallelTabsState.sequences
        };
      }
      return group;
    });
  }, [messages, parallelTabsState?.sequences]);

  // ============================================================================
  // UNIFIED RENDER - IN-PLACE TABS ONLY
  // ============================================================================
  
  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Main content area with proper height constraints */}
      <div className="flex-1 min-h-0">
        <ScrollArea className="h-full w-full" ref={scrollAreaRef}>
          <div className="p-4 md:p-6 space-y-2 max-w-4xl mx-auto pt-16 pb-4 min-h-full">
              {messageGroups.map((group) => {
                return (
                  <div key={group.id} className="space-y-3">
                    <div
                      className={`flex items-start gap-3 ${
                        group.type === 'human' ? 'justify-end' : ''
                      }`}
                    >
                      {group.type === 'human' ? (
                        <HumanMessageBubble
                          group={group}
                          mdComponents={mdComponents}
                        />
                      ) : group.type === 'supervisor_announcement' ? (
                        <SupervisorAnnouncementMessage
                          sequences={group.sequences || []}
                          onTabsInitialized={onTabsInitialized}
                          isLoading={isLoading}
                          researchQuery={group.primaryMessage?.content?.toString() || "research request"}
                          className="max-w-full"
                        />
                      ) : (
                        <AiMessageBubble
                          group={group}
                          mdComponents={mdComponents}
                          handleCopy={handleCopy}
                          copiedMessageId={copiedMessageId}
                          allMessages={messages}
                        />
                      )}
                    </div>

                    {/* In-place parallel tabs - unified paradigm */}
                    {group.type === 'supervisor_announcement' && 
                     parallelTabsState?.isActive && 
                     parallelTabsState.sequences.length > 0 && (
                      <EnhancedErrorBoundary 
                        level="feature" 
                        resetKeys={[parallelTabsState.sequences.length]}
                      >
                        <ParallelTabContainer
                          sequences={parallelTabsState.sequences}
                          parallelMessages={parallelMessages || {}}
                          onTabChange={onParallelTabChange || (() => {})}
                          activeTabId={parallelTabsState.activeTabId}
                          isLoading={isLoading}
                          className="mt-4"
                        />
                      </EnhancedErrorBoundary>
                    )}
                  </div>
                );
              })}
              
              {/* Activity indicator for live research */}
              {isLoading && liveActivityEvents.length > 0 && (
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0 mt-1">
                    <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="bg-neutral-800 rounded-2xl p-4 shadow-sm">
                      <div className="space-y-2">
                        {liveActivityEvents.slice(-3).map((event, index) => (
                          <div key={index} className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                            <div className="text-sm text-neutral-300">
                              <span className="font-medium">{event.title}</span>
                              <span className="text-neutral-400 ml-2">{typeof event.data === 'string' ? event.data : JSON.stringify(event.data)}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
          </div>
        </ScrollArea>
      </div>

      {/* Input area */}
      <div className="flex-shrink-0 border-t border-neutral-700 p-4">
        <div className="max-w-4xl mx-auto">
          <InputForm
            onSubmit={onSubmit || (() => {})}
            isLoading={isLoading}
            onCancel={onCancel || (() => {})}
            hasHistory={messages.length > 0}
            onReset={onReset}
          />
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;