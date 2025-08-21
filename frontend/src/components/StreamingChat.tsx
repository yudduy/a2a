import { useMemo, useEffect, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Activity, TrendingUp, Zap, CheckCircle, Loader2, AlertCircle } from 'lucide-react';
import { SequenceState, SequenceStrategy, AgentType, RoutedMessage } from '@/types/parallel';
import ReactMarkdown from 'react-markdown';
import { cn } from '@/lib/utils';

interface StreamingChatProps {
  sequence: SequenceState | undefined;
  strategy: SequenceStrategy;
  isActive: boolean;
  isLoading: boolean;
  className?: string;
}

// Strategy configuration for styling and icons
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

// Markdown components optimized for narrow columns
const mdComponents = {
  h1: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h1 className="text-lg font-bold mt-2 mb-1 text-white" {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h2 className="text-base font-bold mt-2 mb-1 text-white" {...props}>
      {children}
    </h2>
  ),
  h3: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h3 className="text-sm font-bold mt-2 mb-1 text-white" {...props}>
      {children}
    </h3>
  ),
  p: ({ children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p className="mb-2 leading-relaxed text-sm text-neutral-200" {...props}>
      {children}
    </p>
  ),
  ul: ({ children, ...props }: React.HTMLAttributes<HTMLUListElement>) => (
    <ul className="list-disc pl-4 mb-2 text-sm space-y-1" {...props}>
      {children}
    </ul>
  ),
  ol: ({ children, ...props }: React.HTMLAttributes<HTMLOListElement>) => (
    <ol className="list-decimal pl-4 mb-2 text-sm space-y-1" {...props}>
      {children}
    </ol>
  ),
  li: ({ children, ...props }: React.HTMLAttributes<HTMLLIElement>) => (
    <li className="text-neutral-200" {...props}>
      {children}
    </li>
  ),
  code: ({ children, ...props }: React.HTMLAttributes<HTMLElement>) => (
    <code 
      className="bg-neutral-800 rounded px-1 py-0.5 font-mono text-xs text-neutral-300"
      {...props}
    >
      {children}
    </code>
  ),
  pre: ({ children, ...props }: React.HTMLAttributes<HTMLPreElement>) => (
    <pre 
      className="bg-neutral-800 p-2 rounded-md overflow-x-auto font-mono text-xs my-2 text-neutral-300"
      {...props}
    >
      {children}
    </pre>
  ),
  blockquote: ({ children, ...props }: React.HTMLAttributes<HTMLQuoteElement>) => (
    <blockquote 
      className="border-l-2 border-neutral-600 pl-2 italic my-2 text-xs text-neutral-400"
      {...props}
    >
      {children}
    </blockquote>
  )
};

// Message component for individual messages
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
        <ReactMarkdown components={mdComponents}>
          {typeof message.content === 'string' ? message.content : JSON.stringify(message.content)}
        </ReactMarkdown>
      </div>
    </div>
  );
};

// Progress indicator component
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

export const StreamingChat: React.FC<StreamingChatProps> = ({
  sequence,
  strategy,
  isActive,
  isLoading,
  className
}) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const strategyConfig = STRATEGY_CONFIG[strategy];
  const StrategyIcon = strategyConfig.icon;

  // Auto-scroll to bottom when new messages arrive
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
            {isActive && <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />}
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
          isActive={isActive}
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
                  isLatest={index === displayMessages.length - 1 && isActive}
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
};

export default StreamingChat;