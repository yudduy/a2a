import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import type { Message } from '@langchain/langgraph-sdk';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Copy, CopyCheck, Activity, TrendingUp, Zap, CheckCircle, Loader2, AlertCircle } from 'lucide-react';
import { InputForm } from '@/components/InputForm';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import ReactMarkdown from 'react-markdown';
import { cn } from '@/lib/utils';
import {
  ProcessedEvent,
} from '@/components/ActivityTimeline';
import { ToolMessageDisplay } from '@/components/ToolMessageDisplay';
import {
  extractToolCallsFromMessage,
  findToolMessageForCall,
  MessageContentParser,
  ParsedMessageContent,
  ThinkingSection,
} from '@/types/messages';
import { ToolCall } from '@/types/tools';
import { TypedMarkdown } from '@/components/ui/typed-markdown';
import { ThinkingSections } from '@/components/ui/collapsible-thinking';
import ParallelTabContainer from '@/components/ParallelTabContainer';
import { LLMGeneratedSequence, RoutedMessage, SequenceState, SequenceStrategy, AgentType } from '@/types/parallel';
import { EnhancedErrorBoundary } from '@/components/ui/enhanced-error-boundary';

// ============================================================================
// INTERFACES AND TYPES
// ============================================================================

// Props interface for the unified ChatInterface
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
  
  // Parallel functionality
  parallelTabsState?: ParallelTabsState;
  parallelMessages?: Record<string, RoutedMessage[]>;
  onParallelTabChange?: (tabId: string) => void;
  onTabsInitialized?: () => void;
  
  // Streaming mode (from StreamingChat)
  streamingMode?: boolean;
  sequence?: SequenceState;
  strategy?: SequenceStrategy;
  isStreamingActive?: boolean;
  
  // Simple mode (from SimpleChatInterface)
  simpleMode?: boolean;
  welcomeTitle?: string;
  welcomeSubtitle?: string;
  placeholder?: string;
  
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
// CONFIGURATION AND CONSTANTS
// ============================================================================

// Strategy configuration for streaming mode styling and icons
const STRATEGY_CONFIG = {
  [SequenceStrategy.THEORY_FIRST]: {
    name: 'Theory First',
    color: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    icon: Activity,
    description: 'Academic → Industry → Technical'
  },
  [SequenceStrategy.MARKET_FIRST]: {
    name: 'Market First', 
    color: 'bg-green-500/20 text-green-300 border-green-500/30',
    icon: TrendingUp,
    description: 'Industry → Academic → Technical'
  },
  [SequenceStrategy.FUTURE_BACK]: {
    name: 'Future Back',
    color: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    icon: Zap,
    description: 'Technical → Academic → Industry'
  }
};

// Agent configuration for styling
const AGENT_CONFIG = {
  [AgentType.ACADEMIC]: {
    name: 'Academic',
    color: 'bg-blue-500/10 text-blue-200 border-blue-500/20',
    emoji: '🎓'
  },
  [AgentType.INDUSTRY]: {
    name: 'Industry',
    color: 'bg-green-500/10 text-green-200 border-green-500/20', 
    emoji: '🏢'
  },
  [AgentType.TECHNICAL_TRENDS]: {
    name: 'Technical',
    color: 'bg-purple-500/10 text-purple-200 border-purple-500/20',
    emoji: '⚡'
  }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Utility function to detect supervisor sequence announcements
const detectSupervisorSequenceAnnouncement = (message: Message): LLMGeneratedSequence[] | null => {
  // First, check for sequences in message metadata (preferred method)
  if ((message as any).sequences && Array.isArray((message as any).sequences)) {
    return (message as any).sequences as LLMGeneratedSequence[];
  }

  // Check for sequences in message data or custom fields
  if ((message as any).data?.sequences && Array.isArray((message as any).data.sequences)) {
    return (message as any).data.sequences as LLMGeneratedSequence[];
  }

  if (!message.content || typeof message.content !== 'string') {
    return null;
  }

  const content = message.content.toLowerCase();
  
  // Enhanced pattern detection for supervisor announcements
  const announcementPatterns = [
    /based on.*research.*plan.*here.*sequences/,
    /research sequences?\s*(generated|created|available)/,
    /parallel sequences?\s*(generated|created|ready)/,
    /sequences?\s*generated.*research/,
    /based on.*research.*sequences?.*\d+/,
    /research plan.*sequences?.*\d+/,
    /here are.*sequences?.*\d+/,
    /generated.*\d+.*sequences?/,
    /subagent registry.*sequences/,
    /here are the sequences/,
  ];
  
  const isAnnouncement = announcementPatterns.some(pattern => pattern.test(content));
  
  if (!isAnnouncement) {
    return null;
  }

  console.log('Detected supervisor announcement in message:', message.content);

  // Try to extract sequence count from content
  const sequenceCountMatch = content.match(/(\d+)\s*sequences?/);
  const sequenceCount = sequenceCountMatch ? parseInt(sequenceCountMatch[1], 10) : 3;

  // Parse sequence information from content if available
  const sequences: LLMGeneratedSequence[] = [];
  
  // Try to extract numbered sequences from content (more flexible patterns)
  const numberedSequencePatterns = [
    /(\d+)[\.)]\s*([^\n\r1-9]+)/g,  // 1) Sequence Name
    /(\d+)[\.]?\s*[-–]\s*([^\n\r1-9]+)/g, // 1 - Sequence Name  
    /Sequence\s*(\d+):\s*([^\n\r]+)/gi, // Sequence 1: Name
  ];
  
  let index = 0;
  for (const pattern of numberedSequencePatterns) {
    let match;
    while ((match = pattern.exec(content)) !== null && index < sequenceCount) {
      const sequenceName = match[2].trim().replace(/["']/g, '').replace(/[:\-–].*/, ''); // Clean quotes and extra text
      
      if (sequenceName && sequenceName.length > 2) {
        sequences.push({
          sequence_id: `seq_${index + 1}_${Date.now()}`,
          sequence_name: sequenceName || `Sequence ${index + 1}`,
          agent_names: ['research_agent', 'analysis_agent', 'synthesis_agent'],
          rationale: `Supervisor-generated sequence: ${sequenceName}`,
          research_focus: `Focus area ${index + 1}`,
          confidence_score: 0.8 - (index * 0.05), // Slightly decreasing confidence
          approach_description: `Research approach for ${sequenceName}`,
          expected_outcomes: ['Research insights', 'Analysis results', 'Synthesis report'],
          created_at: new Date().toISOString(),
        });
        index++;
      }
    }
    pattern.lastIndex = 0; // Reset pattern
  }
  
  // If we couldn't parse specific sequences, generate default ones
  if (sequences.length === 0) {
    const defaultSequenceNames = [
      'Academic Deep Dive',
      'Industry Analysis', 
      'Future Trends',
      'Technical Research',
      'Market Analysis'
    ];
    
    for (let i = 0; i < Math.min(sequenceCount, 5); i++) {
      sequences.push({
        sequence_id: `seq_${i + 1}_${Date.now()}`,
        sequence_name: defaultSequenceNames[i] || `Sequence ${i + 1}`,
        agent_names: ['research_agent', 'analysis_agent', 'synthesis_agent'],
        rationale: `Auto-generated sequence based on supervisor announcement`,
        research_focus: `Research focus area ${i + 1}`,
        confidence_score: 0.8 - (i * 0.05),
        approach_description: `Research approach ${i + 1}`,
        expected_outcomes: ['Research insights', 'Analysis results', 'Synthesis report'],
        created_at: new Date().toISOString(),
      });
    }
  }
  
  console.log(`Detected ${sequences.length} sequences from supervisor announcement:`, sequences);
  return sequences.length > 0 ? sequences : null;
};

// Group messages to combine related AI responses and tool calls
const groupMessages = (messages: Message[]): MessageGroup[] => {
  const groups: MessageGroup[] = [];
  let currentGroup: MessageGroup | null = null;

  for (const message of messages) {
    if (message.type === 'human') {
      // Human messages are always standalone
      groups.push({
        id: message.id || `human-${Date.now()}`,
        type: 'human',
        messages: [message],
        primaryMessage: message,
        toolCalls: [],
        toolResults: [],
      });
      currentGroup = null;
    } else if (message.type === 'ai') {
      // Check if this is a supervisor announcement
      const sequences = detectSupervisorSequenceAnnouncement(message);
      
      if (sequences && sequences.length > 0) {
        // Create supervisor announcement group
        currentGroup = {
          id: message.id || `supervisor-${Date.now()}`,
          type: 'supervisor_announcement',
          messages: [message],
          primaryMessage: message,
          toolCalls: [],
          toolResults: [],
          sequences,
        };
        groups.push(currentGroup);
      } else {
        // Start a new AI group or continue existing one
        if (!currentGroup || currentGroup.type !== 'ai_complete') {
          // Create new AI group
          currentGroup = {
            id: message.id || `ai-${Date.now()}`,
            type: 'ai_complete',
            messages: [message],
            primaryMessage: message,
            toolCalls: [], // Don't accumulate tool calls at group level
            toolResults: [],
          };
          groups.push(currentGroup);
        } else {
          // Add to existing AI group (for cases with multiple AI messages)
          currentGroup.messages.push(message);
          // Don't accumulate tool calls to avoid duplication
          // Update primary message to the latest one with content
          if (
            message.content &&
            typeof message.content === 'string' &&
            message.content.trim()
          ) {
            currentGroup.primaryMessage = message;
          }
        }
      }
    } else if (message.type === 'tool') {
      // Tool results belong to the current AI group
      if (currentGroup && currentGroup.type === 'ai_complete') {
        currentGroup.toolResults.push(message);
        currentGroup.messages.push(message);
      }
    }
  }

  return groups;
};

// ============================================================================
// MARKDOWN COMPONENTS
// ============================================================================

// Shared markdown components for both modes
const createMarkdownComponents = (compact: boolean = false) => {
  const baseClasses = compact ? {
    h1: 'text-lg font-bold mt-2 mb-1 text-white',
    h2: 'text-base font-bold mt-2 mb-1 text-white',
    h3: 'text-sm font-bold mt-2 mb-1 text-white',
    p: 'mb-2 leading-relaxed text-sm text-neutral-200',
    ul: 'list-disc pl-4 mb-2 text-sm space-y-1',
    ol: 'list-decimal pl-4 mb-2 text-sm space-y-1',
    li: 'text-neutral-200',
    code: 'bg-neutral-800 rounded px-1 py-0.5 font-mono text-xs text-neutral-300',
    pre: 'bg-neutral-800 p-2 rounded-md overflow-x-auto font-mono text-xs my-2 text-neutral-300',
    blockquote: 'border-l-2 border-neutral-600 pl-2 italic my-2 text-xs text-neutral-400'
  } : {
    h1: 'text-2xl font-bold mt-4 mb-2',
    h2: 'text-xl font-bold mt-3 mb-2',
    h3: 'text-lg font-bold mt-3 mb-1',
    p: 'mb-3 leading-7',
    ul: 'list-disc pl-6 mb-3',
    ol: 'list-decimal pl-6 mb-3',
    li: 'mb-1',
    code: 'bg-neutral-900 rounded px-1 py-0.5 font-mono text-xs',
    pre: 'bg-neutral-900 p-3 rounded-lg overflow-x-auto font-mono text-xs my-3',
    blockquote: 'border-l-4 border-neutral-600 pl-4 italic my-3 text-sm'
  };

  return {
    h1: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
      <h1 className={cn(baseClasses.h1, className)} {...props}>
        {children}
      </h1>
    ),
    h2: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
      <h2 className={cn(baseClasses.h2, className)} {...props}>
        {children}
      </h2>
    ),
    h3: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
      <h3 className={cn(baseClasses.h3, className)} {...props}>
        {children}
      </h3>
    ),
    p: ({ className, children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
      <p className={cn(baseClasses.p, className)} {...props}>
        {children}
      </p>
    ),
    a: ({ className, children, href, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
      <Badge className="text-xs mx-0.5">
        <a
          className={cn('text-blue-400 hover:text-blue-300 text-xs', className)}
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          {...props}
        >
          {children}
        </a>
      </Badge>
    ),
    ul: ({ className, children, ...props }: React.HTMLAttributes<HTMLUListElement>) => (
      <ul className={cn(baseClasses.ul, className)} {...props}>
        {children}
      </ul>
    ),
    ol: ({ className, children, ...props }: React.HTMLAttributes<HTMLOListElement>) => (
      <ol className={cn(baseClasses.ol, className)} {...props}>
        {children}
      </ol>
    ),
    li: ({ className, children, ...props }: React.HTMLAttributes<HTMLLIElement>) => (
      <li className={cn(baseClasses.li, className)} {...props}>
        {children}
      </li>
    ),
    blockquote: ({ className, children, ...props }: React.HTMLAttributes<HTMLQuoteElement>) => (
      <blockquote className={cn(baseClasses.blockquote, className)} {...props}>
        {children}
      </blockquote>
    ),
    code: ({ className, children, ...props }: React.HTMLAttributes<HTMLElement>) => (
      <code className={cn(baseClasses.code, className)} {...props}>
        {children}
      </code>
    ),
    pre: ({ className, children, ...props }: React.HTMLAttributes<HTMLPreElement>) => (
      <pre className={cn(baseClasses.pre, className)} {...props}>
        {children}
      </pre>
    ),
    hr: ({ className, ...props }: React.HTMLAttributes<HTMLHRElement>) => (
      <hr className={cn('border-neutral-600 my-4', className)} {...props} />
    ),
    table: ({ className, children, ...props }: React.TableHTMLAttributes<HTMLTableElement>) => (
      <div className="my-3 overflow-x-auto">
        <table className={cn('border-collapse w-full', className)} {...props}>
          {children}
        </table>
      </div>
    ),
    th: ({ className, children, ...props }: React.ThHTMLAttributes<HTMLTableHeaderCellElement>) => (
      <th className={cn('border border-neutral-600 px-3 py-2 text-left font-bold', className)} {...props}>
        {children}
      </th>
    ),
    td: ({ className, children, ...props }: React.TdHTMLAttributes<HTMLTableDataCellElement>) => (
      <td className={cn('border border-neutral-600 px-3 py-2', className)} {...props}>
        {children}
      </td>
    ),
  };
};

// ============================================================================
// SUBCOMPONENTS
// ============================================================================

// Progress indicator component for streaming mode
interface ProgressIndicatorProps {
  progress: number;
  status: string;
  isActive: boolean;
}

const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ 
  progress, 
  status, 
  isActive 
}) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      case 'active':
        return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />;
      default:
        return <div className="w-4 h-4 rounded-full bg-neutral-600" />;
    }
  };

  return (
    <div className="flex items-center gap-2 px-3 py-2">
      {getStatusIcon()}
      <div className="flex-1">
        <div className="w-full bg-neutral-700 rounded-full h-1.5">
          <div 
            className={cn(
              "h-1.5 rounded-full transition-all duration-300",
              status === 'completed' ? 'bg-green-500' : 
              status === 'failed' ? 'bg-red-500' :
              isActive ? 'bg-blue-500' : 'bg-neutral-500'
            )}
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
      </div>
      <span className="text-xs text-neutral-400">
        {Math.round(progress)}%
      </span>
    </div>
  );
};

// Message bubble component for streaming mode
interface MessageBubbleProps {
  message: RoutedMessage;
  isLatest: boolean;
  currentAgent: AgentType | null;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ 
  message, 
  isLatest, 
  currentAgent 
}) => {
  const isSystemMessage = message.message_type === 'progress' || 
                         message.message_type === 'agent_transition';
  
  const isError = message.message_type === 'error';
  
  if (isSystemMessage) {
    return (
      <div className="flex items-center gap-2 py-1">
        <div className="w-1 h-1 rounded-full bg-neutral-500" />
        <div className="text-xs text-neutral-400 flex-1">
          {message.agent_type && (
            <span className={cn(
              'inline-block px-1.5 py-0.5 rounded text-xs mr-2',
              AGENT_CONFIG[message.agent_type].color
            )}>
              {AGENT_CONFIG[message.agent_type].emoji} {AGENT_CONFIG[message.agent_type].name}
            </span>
          )}
          {typeof message.content === 'string' ? message.content : 'System update'}
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 rounded-lg p-2 my-1">
        <AlertCircle className="w-3 h-3 text-red-400 mt-0.5 flex-shrink-0" />
        <div className="text-xs text-red-300">
          {typeof message.content === 'string' ? message.content : 'An error occurred'}
        </div>
      </div>
    );
  }

  // AI message bubble
  return (
    <div className={cn(
      'bg-neutral-800 rounded-lg p-3 my-2 border-l-2 transition-colors',
      isLatest && 'animate-pulse',
      currentAgent && AGENT_CONFIG[currentAgent] ? 
        `border-l-${AGENT_CONFIG[currentAgent].color.split(' ')[0].replace('bg-', '').replace('/10', '/50')}` : 
        'border-l-neutral-600'
    )}>
      {message.agent_type && (
        <div className="flex items-center gap-1 mb-2">
          <span className={cn(
            'inline-block px-1.5 py-0.5 rounded text-xs',
            AGENT_CONFIG[message.agent_type].color
          )}>
            {AGENT_CONFIG[message.agent_type].emoji} {AGENT_CONFIG[message.agent_type].name}
          </span>
        </div>
      )}
      <div className="prose prose-sm max-w-none text-neutral-200">
        <ReactMarkdown components={createMarkdownComponents(true)}>
          {typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}
        </ReactMarkdown>
      </div>
    </div>
  );
};

// Human message bubble
interface HumanMessageBubbleProps {
  group: MessageGroup;
  mdComponents: ReturnType<typeof createMarkdownComponents>;
}

const HumanMessageBubble: React.FC<HumanMessageBubbleProps> = ({
  group,
  mdComponents,
}) => {
  const message = group.primaryMessage;
  return (
    <div className="text-white rounded-3xl break-words min-h-7 bg-neutral-700 max-w-[100%] sm:max-w-[90%] px-4 pt-3 rounded-br-lg overflow-hidden">
      <ReactMarkdown components={mdComponents}>
        {typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content)}
      </ReactMarkdown>
    </div>
  );
};

// Component to handle proper sequential rendering of thinking sections
interface MessageWithThinkingProps {
  parsedContent: ParsedMessageContent;
  expandedThinking: Set<string>;
  onToggleThinking: (sectionId: string) => void;
  shouldShowTyping: boolean;
  mdComponents: ReturnType<typeof createMarkdownComponents>;
  messageIndex: number;
}

const MessageWithThinking: React.FC<MessageWithThinkingProps> = ({
  parsedContent,
  expandedThinking,
  onToggleThinking,
  shouldShowTyping,
  mdComponents,
  messageIndex
}) => {
  const [currentSection, setCurrentSection] = useState(0);
  const [sectionsCompleted, setSectionsCompleted] = useState<Set<number>>(new Set());
  
  // Calculate total sections: pre-thinking + thinking sections + post-thinking
  const allSections: Array<{type: string, content: string, section?: ThinkingSection}> = [
    ...(parsedContent.preThinking ? [{ type: 'pre', content: parsedContent.preThinking }] : []),
    ...parsedContent.thinkingSections.map(section => ({ type: 'thinking', content: section.content, section })),
    ...(parsedContent.postThinking ? [{ type: 'post', content: parsedContent.postThinking }] : [])
  ];

  const handleSectionComplete = (sectionIndex: number) => {
    setSectionsCompleted(prev => new Set([...prev, sectionIndex]));
    // Move to next section after a brief delay
    if (sectionIndex < allSections.length - 1) {
      setTimeout(() => {
        setCurrentSection(prev => Math.max(prev, sectionIndex + 1));
      }, 100);
    }
  };

  // Reset state when shouldShowTyping changes
  useEffect(() => {
    if (shouldShowTyping) {
      setCurrentSection(0);
      setSectionsCompleted(new Set());
    }
  }, [shouldShowTyping]);

  return (
    <div className="space-y-3">
      {allSections.map((section, index) => {
        const isCurrentOrCompleted = !shouldShowTyping || currentSection >= index || sectionsCompleted.has(index);
        const isCurrentlyTyping = shouldShowTyping && currentSection === index && !sectionsCompleted.has(index);
        
        if (!isCurrentOrCompleted) {
          return null; // Don't render future sections yet
        }

        if (section.type === 'thinking') {
          // Thinking sections appear immediately with proper state
          return (
            <div key={`thinking-${section.section?.id || index}`}>
              <ThinkingSections
                sections={[section.section!]}
                expandedSections={expandedThinking}
                onToggleSection={onToggleThinking}
                hasTypingAnimation={isCurrentlyTyping}
                typingSpeed={20}
                onTypingComplete={() => handleSectionComplete(index)}
              />
            </div>
          );
        } else {
          // Pre-thinking and post-thinking content
          return (
            <div key={`${section.type}-${index}`}>
              <TypedMarkdown
                components={mdComponents}
                speed={20}
                delay={index === 0 ? messageIndex * 200 : 0}
                hideCursor={true}
                verticalMode={true}
                onTypingComplete={() => handleSectionComplete(index)}
                enableTyping={isCurrentlyTyping || !shouldShowTyping}
              >
                {section.content}
              </TypedMarkdown>
            </div>
          );
        }
      })}
    </div>
  );
};

// AI message bubble for regular chat mode
interface AiMessageBubbleProps {
  group: MessageGroup;
  historicalActivity: ProcessedEvent[] | undefined;
  liveActivity: ProcessedEvent[] | undefined;
  isLastGroup: boolean;
  isOverallLoading: boolean;
  mdComponents: ReturnType<typeof createMarkdownComponents>;
  handleCopy: (text: string, messageId: string) => void;
  copiedMessageId: string | null;
  allMessages: Message[];
}

const AiMessageBubble: React.FC<AiMessageBubbleProps> = ({
  group,
  historicalActivity,
  liveActivity,
  isLastGroup,
  isOverallLoading,
  mdComponents,
  handleCopy,
  copiedMessageId,
  allMessages,
}) => {
  // Tool message state
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());
  // Thinking sections state
  const [expandedThinking, setExpandedThinking] = useState<Set<string>>(new Set());

  const toggleTool = (toolId: string) => {
    setExpandedTools((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(toolId)) {
        newSet.delete(toolId);
      } else {
        newSet.add(toolId);
      }
      return newSet;
    });
  };

  const toggleThinking = (thinkingId: string) => {
    setExpandedThinking((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(thinkingId)) {
        newSet.delete(thinkingId);
      } else {
        newSet.add(thinkingId);
      }
      return newSet;
    });
  };

  // Determine which activity events to show and if it's for a live loading message
  const activityForThisBubble =
    isLastGroup && isOverallLoading ? liveActivity : historicalActivity;
  const isLiveActivityForThisBubble = isLastGroup && isOverallLoading;

  // Always show activity timeline for research interface
  const shouldShowActivity = 
    isLiveActivityForThisBubble ||
    (activityForThisBubble && activityForThisBubble.length > 0);

  // Show tool messages to provide visibility into backend tool interactions
  const shouldHideToolMessages = false;

  // Check if we should hide copy button (when still loading for this message group)
  const shouldHideCopyButton = isLastGroup && isOverallLoading;

  // Combine all text content for copy functionality
  const combinedTextContent = group.messages
    .filter((msg) => msg.type === 'ai' && msg.content)
    .map((msg) =>
      typeof msg.content === 'string'
        ? msg.content
        : JSON.stringify(msg.content)
    )
    .filter((content) => content.trim())
    .join('\n\n');

  return (
    <div
      className={`relative break-words flex flex-col group max-w-[85%] md:max-w-[80%] w-full rounded-xl p-3 shadow-sm bg-neutral-800 text-neutral-100 min-h-[56px] overflow-hidden ${
        shouldShowActivity
          ? 'rounded-bl-none'
          : 'rounded-bl-xl'
      }`}
    >
      {/* Render messages in chronological order */}
      {group.messages.map((message, index) => {
        if (message.type === 'ai') {
          const toolCalls = extractToolCallsFromMessage(message);
          const hasContent =
            message.content &&
            typeof message.content === 'string' &&
            message.content.trim();

          // Parse message content for thinking sections
          let parsedContent: ParsedMessageContent | null = null;
          if (hasContent) {
            try {
              parsedContent = MessageContentParser.parse(message);
              
              // Only log in development mode
              if (import.meta.env.DEV && parsedContent?.hasThinking) {
                console.log(`Parsed message with ${parsedContent.thinkingSections.length} thinking sections`);
              }
            } catch (error) {
              console.warn('Failed to parse message content:', error);
              // Create fallback parsed content to prevent rendering issues
              parsedContent = {
                preThinking: typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
                thinkingSections: [],
                postThinking: undefined,
                toolCalls: [],
                toolResults: [],
                hasThinking: false,
                totalCharacters: (typeof message.content === 'string' ? message.content : JSON.stringify(message.content)).length,
                renderSections: [],
              };
            }
          }

          const shouldShowTyping = isLastGroup && index === group.messages.length - 1;

          return (
            <div key={message.id || `ai-${index}`} className="space-y-3">
              {/* Render AI content if present */}
              {hasContent && (
                <div className="space-y-3">
                  {parsedContent && parsedContent.hasThinking ? (
                    // Render parsed content with thinking sections in proper sequence
                    <EnhancedErrorBoundary 
                      level="component" 
                      resetKeys={[parsedContent.thinkingSections.length]}
                    >
                      <MessageWithThinking
                        parsedContent={parsedContent}
                        expandedThinking={expandedThinking}
                        onToggleThinking={toggleThinking}
                        shouldShowTyping={shouldShowTyping}
                        mdComponents={mdComponents}
                        messageIndex={index}
                      />
                    </EnhancedErrorBoundary>
                  ) : (
                    // Render normal content without thinking sections
                    <TypedMarkdown 
                      components={mdComponents} 
                      speed={20} 
                      delay={index * 200}
                      hideCursor={true}
                      verticalMode={true}
                    >
                      {typeof message.content === 'string'
                        ? message.content
                        : JSON.stringify(message.content)}
                    </TypedMarkdown>
                  )}
                </div>
              )}

              {/* Render tool calls immediately after the AI message that triggered them */}
              {!shouldHideToolMessages && toolCalls.length > 0 && (
                <div className="space-y-2">
                  {toolCalls.map((toolCall) => (
                    <ToolMessageDisplay
                      key={toolCall.id}
                      toolCall={toolCall}
                      toolMessage={findToolMessageForCall(
                        allMessages,
                        toolCall.id
                      )}
                      isExpanded={expandedTools.has(toolCall.id)}
                      onToggle={() => toggleTool(toolCall.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        }
        // Skip tool messages as they're handled by ToolMessageDisplay above
        return null;
      })}

      {!shouldHideCopyButton && (
        <Button
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0 self-start mt-2 hover:bg-neutral-600/50 text-neutral-400 hover:text-neutral-200"
          onClick={() => {
            if (group.primaryMessage.id) {
              handleCopy(combinedTextContent, group.primaryMessage.id);
            }
          }}
        >
          {group.primaryMessage.id && copiedMessageId === group.primaryMessage.id ? (
            <CopyCheck className="h-3 w-3" />
          ) : (
            <Copy className="h-3 w-3" />
          )}
        </Button>
      )}
    </div>
  );
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  // Core props
  messages = [],
  isLoading = false,
  onSubmit = () => {},
  onCancel = () => {},
  onReset,
  
  // Activity and events
  liveActivityEvents = [],
  historicalActivities = {},
  
  // Parallel functionality
  parallelTabsState,
  parallelMessages,
  onParallelTabChange,
  onTabsInitialized,
  
  // Streaming mode
  streamingMode = false,
  sequence,
  strategy = SequenceStrategy.THEORY_FIRST,
  isStreamingActive = false,
  
  // Simple mode
  simpleMode = false,
  welcomeTitle = "💬 Research Assistant",
  welcomeSubtitle = "What would you like to research?",
  placeholder = "What would you like to research?",
  
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
      setTimeout(() => setCopiedMessageId(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  // Auto-scroll functionality
  const scrollToBottom = useCallback((smooth = true) => {
    if (!scrollAreaRef.current) return;
    
    // Get the viewport element from the ScrollArea
    const viewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
    if (!viewport) return;

    viewport.scrollTo({
      top: viewport.scrollHeight,
      behavior: smooth ? 'smooth' : 'auto',
    });
  }, [scrollAreaRef]);

  const isNearBottom = useCallback((): boolean => {
    if (!scrollAreaRef.current) return true;
    
    // Get the viewport element from the ScrollArea
    const viewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
    if (!viewport) return true;

    const { scrollTop, scrollHeight, clientHeight } = viewport;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    
    // Consider "near bottom" if within 100px
    return distanceFromBottom <= 100;
  }, [scrollAreaRef]);

  // Auto-scroll when messages change or loading state changes
  useEffect(() => {
    const hasNewMessages = messages.length > prevMessageCountRef.current;
    
    // Auto-scroll if:
    // 1. New messages were added and user is near bottom, OR
    // 2. Loading state changed to false (response completed) and user is near bottom
    if ((hasNewMessages || (!isLoading && prevIsLoadingRef.current)) && isNearBottom()) {
      // Small delay to ensure DOM is updated before scrolling
      setTimeout(() => scrollToBottom(true), 50);
    }

    // Update refs for next comparison
    prevMessageCountRef.current = messages.length;
    prevIsLoadingRef.current = isLoading;
  }, [messages.length, isLoading, isNearBottom, scrollToBottom]);

  // ============================================================================
  // RENDER SIMPLE MODE
  // ============================================================================
  
  if (simpleMode) {
    return (
      <div className={cn("flex flex-col items-center justify-center text-center px-4 flex-1 mb-16 w-full max-w-3xl mx-auto gap-4", className)}>
        <div className="flex flex-col items-center gap-6">
          <img
            src="./logo-icon.svg"
            alt="Research Assistant Logo"
            className="h-24 w-24 text-primary"
          />
          <div>
            <h1 className="text-5xl md:text-6xl font-semibold text-neutral-100 mb-3">
              {welcomeTitle}
            </h1>
            <p className="text-xl md:text-2xl text-neutral-400">
              {welcomeSubtitle}
            </p>
          </div>
        </div>
        <div className="w-full mt-4">
          <InputForm
            onSubmit={onSubmit}
            isLoading={isLoading}
            onCancel={onCancel}
            hasHistory={false}
            placeholder={placeholder}
          />
        </div>
      </div>
    );
  }

  // ============================================================================
  // RENDER STREAMING MODE
  // ============================================================================
  
  if (streamingMode && sequence && strategy) {
    const strategyConfig = STRATEGY_CONFIG[strategy];
    const StrategyIcon = strategyConfig.icon;

    // Auto-scroll for streaming mode
    useEffect(() => {
      if (scrollAreaRef.current) {
        const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
        if (scrollContainer) {
          scrollContainer.scrollTop = scrollContainer.scrollHeight;
        }
      }
    }, [sequence?.messages]);

    // Process messages for display
    const displayMessages = useMemo(() => {
      if (!sequence?.messages) return [];
      
      // Sort messages by timestamp and filter out duplicates
      return sequence.messages
        .sort((a, b) => a.timestamp - b.timestamp)
        .filter((msg, index, arr) => 
          index === 0 || msg.message_id !== arr[index - 1].message_id
        );
    }, [sequence?.messages]);

    // Determine status
    const status = sequence?.progress.status || 'pending';
    const progress = sequence?.progress.completion_percentage || 0;
    const currentAgent = sequence?.current_agent || null;

    return (
      <Card className={cn(
        'flex flex-col h-full bg-neutral-900 border-neutral-700',
        className
      )}>
        {/* Header */}
        <div className="flex-shrink-0 border-b border-neutral-700 p-3">
          <div className="flex items-center justify-between mb-2">
            <Badge className={cn(
              'text-xs font-medium border',
              strategyConfig.color
            )}>
              <StrategyIcon className="w-3 h-3 mr-1" />
              {strategyConfig.name}
            </Badge>
            
            <div className="flex items-center gap-1">
              {isStreamingActive && <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />}
              {status === 'completed' && <CheckCircle className="w-4 h-4 text-green-400" />}
              {status === 'failed' && <AlertCircle className="w-4 h-4 text-red-400" />}
            </div>
          </div>
          
          <div className="text-xs text-neutral-400 mb-2">
            {strategyConfig.description}
          </div>
          
          <ProgressIndicator 
            progress={progress}
            status={status}
            isActive={isStreamingActive}
          />
        </div>

        {/* Messages */}
        <CardContent className="flex-1 p-0 min-h-0">
          <ScrollArea className="h-full" ref={scrollAreaRef}>
            <div className="p-3 space-y-1">
              {displayMessages.length === 0 && !isLoading ? (
                <div className="flex items-center justify-center h-24 text-neutral-500">
                  <div className="text-center">
                    <div className="text-xs">Waiting to start...</div>
                  </div>
                </div>
              ) : (
                displayMessages.map((message, index) => (
                  <MessageBubble
                    key={message.message_id}
                    message={message}
                    isLatest={index === displayMessages.length - 1 && isStreamingActive}
                    currentAgent={currentAgent}
                  />
                ))
              )}
              
              {isLoading && displayMessages.length === 0 && (
                <div className="flex items-center justify-center h-24">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                    <span className="text-xs text-neutral-400">Starting sequence...</span>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
        
        {/* Footer - Current Agent */}
        {currentAgent && (
          <div className="flex-shrink-0 border-t border-neutral-700 p-2">
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
              <span className={cn(
                'text-xs px-2 py-1 rounded',
                AGENT_CONFIG[currentAgent].color
              )}>
                {AGENT_CONFIG[currentAgent].emoji} {AGENT_CONFIG[currentAgent].name} Active
              </span>
            </div>
          </div>
        )}
      </Card>
    );
  }

  // ============================================================================
  // RENDER REGULAR CHAT MODE  
  // ============================================================================
  
  // Group messages to combine related AI responses and tool calls
  const messageGroups = groupMessages(messages);
  const mdComponents = createMarkdownComponents(false);

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Main content area with proper height constraints */}
      <div className="flex-1 min-h-0">
        {/* Sequential chat view with fixed height */}
        <ScrollArea className="h-full w-full" ref={scrollAreaRef}>
          <div className="p-4 md:p-6 space-y-2 max-w-4xl mx-auto pt-16 pb-4 min-h-full">
              {messageGroups.map((group, index) => {
                const isLast = index === messageGroups.length - 1;
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
                        <div className="relative break-words flex flex-col group max-w-[90%] md:max-w-[85%] w-full rounded-xl p-3 shadow-sm bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-blue-500/10 border border-blue-500/20 text-neutral-100 min-h-[56px] overflow-hidden">
                          <div className="text-center p-4">
                            <p className="text-blue-400 font-semibold mb-2">Research Sequences Detected</p>
                            <p className="text-sm text-neutral-300 mb-3">
                              The supervisor has generated {group.sequences?.length || 0} parallel research sequences.
                            </p>
                            <button 
                              onClick={onTabsInitialized}
                              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded text-sm"
                            >
                              Launch Parallel Research
                            </button>
                          </div>
                        </div>
                      ) : (
                        <AiMessageBubble
                          group={group}
                          historicalActivity={
                            historicalActivities[group.primaryMessage.id!]
                          }
                          liveActivity={liveActivityEvents}
                          isLastGroup={isLast}
                          isOverallLoading={isLoading}
                          mdComponents={mdComponents}
                          handleCopy={handleCopy}
                          copiedMessageId={copiedMessageId}
                          allMessages={messages}
                        />
                      )}
                    </div>

                    {/* Parallel tabs integration - show after supervisor announcement */}
                    {group.type === 'supervisor_announcement' && 
                     parallelTabsState?.isActive && 
                     parallelTabsState.sequences.length > 0 && (
                      <EnhancedErrorBoundary 
                        level="feature" 
                        resetKeys={[parallelTabsState.sequences.length, parallelTabsState.activeTabId]}
                      >
                        <ParallelTabContainer
                          sequences={parallelTabsState.sequences}
                          parallelMessages={parallelMessages || {}}
                          activeTabId={parallelTabsState.activeTabId}
                          onTabChange={onParallelTabChange || (() => {})}
                          isLoading={isLoading}
                          className="animate-in slide-in-from-top-2 duration-300"
                        />
                      </EnhancedErrorBoundary>
                    )}
                  </div>
                );
              })}
              {isLoading &&
                (messageGroups.length === 0 ||
                  messageGroups[messageGroups.length - 1].type === 'human') && (
                  <div className="flex items-start gap-3 mt-3">
                    <div className="relative group max-w-[85%] md:max-w-[80%] rounded-xl p-3 shadow-sm break-words bg-neutral-800 text-neutral-100 w-full min-h-[56px] flex items-center justify-start">
                      <div className="flex justify-center items-center gap-1">
                        <div className="w-2 h-2 bg-neutral-100 rounded-full animate-bounce [animation-delay:-0.32s]"></div>
                        <div className="w-2 h-2 bg-neutral-100 rounded-full animate-bounce [animation-delay:-0.16s]"></div>
                        <div className="w-2 h-2 bg-neutral-100 rounded-full animate-bounce"></div>
                      </div>
                    </div>
                  </div>
                )}
          </div>
        </ScrollArea>
      </div>

      {/* Input form at bottom */}
      <div className="flex-shrink-0">
        <InputForm
          onSubmit={(inputValue: string) => onSubmit(inputValue)}
          isLoading={isLoading}
          onCancel={onCancel}
          hasHistory={messages.length > 0}
          onReset={onReset}
        />
      </div>
    </div>
  );
};

export default ChatInterface;