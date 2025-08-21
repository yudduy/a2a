import { useState, useCallback, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { GitCompare, Home, Activity, TrendingUp, Zap, Loader2 } from 'lucide-react';
import { useParallelSequences } from '@/hooks/useParallelSequences';
import { SequenceStrategy, SequenceState, RoutedMessage } from '@/types/parallel';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';

interface ParallelSequenceViewProps {
  query: string;
  onNewQuery: () => void;
}

// Strategy configurations with clean styling
const STRATEGY_CONFIG = {
  [SequenceStrategy.THEORY_FIRST]: {
    label: 'Theory First',
    icon: Activity,
    color: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    description: 'Academic research first, then practical applications',
  },
  [SequenceStrategy.MARKET_FIRST]: {
    label: 'Market First',
    icon: TrendingUp,
    color: 'bg-green-500/10 text-green-400 border-green-500/20',
    description: 'Market analysis first, then underlying theory',
  },
  [SequenceStrategy.FUTURE_BACK]: {
    label: 'Future Back',
    icon: Zap,
    color: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    description: 'Future trends first, working backward to current state',
  },
};

// Clean markdown components for message rendering
const mdComponents = {
  h1: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h1 className={cn('text-lg font-bold mt-3 mb-2', className)} {...props}>
      {children}
    </h1>
  ),
  h2: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h2 className={cn('text-base font-bold mt-2 mb-1', className)} {...props}>
      {children}
    </h2>
  ),
  h3: ({ className, children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h3 className={cn('text-sm font-bold mt-2 mb-1', className)} {...props}>
      {children}
    </h3>
  ),
  p: ({ className, children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p className={cn('mb-2 text-sm leading-relaxed', className)} {...props}>
      {children}
    </p>
  ),
  ul: ({ className, children, ...props }: React.HTMLAttributes<HTMLUListElement>) => (
    <ul className={cn('list-disc pl-4 mb-2 text-sm', className)} {...props}>
      {children}
    </ul>
  ),
  ol: ({ className, children, ...props }: React.HTMLAttributes<HTMLOListElement>) => (
    <ol className={cn('list-decimal pl-4 mb-2 text-sm', className)} {...props}>
      {children}
    </ol>
  ),
  li: ({ className, children, ...props }: React.HTMLAttributes<HTMLLIElement>) => (
    <li className={cn('mb-1', className)} {...props}>
      {children}
    </li>
  ),
  code: ({ className, children, ...props }: React.HTMLAttributes<HTMLElement>) => (
    <code className={cn('bg-neutral-800 rounded px-1 py-0.5 font-mono text-xs', className)} {...props}>
      {children}
    </code>
  ),
  pre: ({ className, children, ...props }: React.HTMLAttributes<HTMLPreElement>) => (
    <pre className={cn('bg-neutral-800 p-2 rounded overflow-x-auto font-mono text-xs my-2', className)} {...props}>
      {children}
    </pre>
  ),
};

// Individual sequence column component
interface SequenceColumnProps {
  sequence: SequenceState;
  messages: RoutedMessage[];
}

function SequenceColumn({ sequence, messages }: SequenceColumnProps) {
  const config = STRATEGY_CONFIG[sequence.strategy];
  const IconComponent = config.icon;
  const progress = sequence.progress.completion_percentage;
  
  return (
    <Card className="h-full flex flex-col bg-neutral-900 border-neutral-700">
      {/* Header */}
      <div className="border-b border-neutral-700 p-4">
        <div className="flex items-center gap-3 mb-2">
          <IconComponent className="w-5 h-5 text-neutral-400" />
          <Badge variant="outline" className={cn('text-xs', config.color)}>
            {config.label}
          </Badge>
        </div>
        <p className="text-xs text-neutral-500 leading-relaxed">
          {config.description}
        </p>
        
        {/* Progress bar */}
        <div className="mt-3">
          <div className="flex items-center justify-between text-xs text-neutral-400 mb-1">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-neutral-800 rounded-full h-1.5">
            <div 
              className="h-1.5 rounded-full bg-blue-500 transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>
      
      {/* Messages */}
      <CardContent className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-3">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-32 text-neutral-500">
              <div className="text-center">
                <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                <p className="text-sm">Initializing sequence...</p>
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <div 
                key={message.message_id || index}
                className="bg-neutral-800 rounded-lg p-3 text-sm"
              >
                {/* Agent badge if available */}
                {message.agent_type && (
                  <Badge variant="outline" className="mb-2 text-xs">
                    {message.agent_type.replace('_', ' ').toUpperCase()}
                  </Badge>
                )}
                
                {/* Message content */}
                <ReactMarkdown components={mdComponents}>
                  {typeof message.content === 'string' 
                    ? message.content 
                    : JSON.stringify(message.content, null, 2)}
                </ReactMarkdown>
              </div>
            ))
          )}
          
          {/* Loading indicator if sequence is active */}
          {sequence.progress.status === 'active' && (
            <div className="flex items-center gap-2 text-neutral-500 text-sm py-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Processing...</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export function ParallelSequenceView({ query, onNewQuery }: ParallelSequenceViewProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const [hasStarted, setHasStarted] = useState(false);
  
  const {
    sequences,
    isLoading,
    error,
    progress,
    start,
    getSequenceMessages,
    connectionState,
  } = useParallelSequences();

  // Start the parallel sequences when component mounts
  useEffect(() => {
    if (!hasStarted && query) {
      setHasStarted(true);
      start(query).catch((err) => {
        console.error('Failed to start parallel sequences:', err);
        setIsAnalyzing(false);
      });
    }
  }, [query, hasStarted, start]);

  // Transition from analyzing to columns view
  useEffect(() => {
    if (isAnalyzing) {
      const timer = setTimeout(() => {
        setIsAnalyzing(false);
      }, 2500); // 2.5 seconds analyzing phase
      
      return () => clearTimeout(timer);
    }
  }, [isAnalyzing]);

  const handleNewQuery = useCallback(() => {
    setIsAnalyzing(true);
    setHasStarted(false);
    onNewQuery();
  }, [onNewQuery]);

  // Show analyzing state
  if (isAnalyzing) {
    return (
      <div className="h-full flex flex-col bg-neutral-950">
        {/* Header */}
        <div className="border-b border-neutral-800 p-6">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-4">
              <GitCompare className="w-6 h-6 text-blue-400" />
              <div>
                <h1 className="text-lg font-semibold text-white">
                  Analyzing your query
                </h1>
                <p className="text-sm text-neutral-400 mt-1">
                  "{query}"
                </p>
              </div>
            </div>
            
            <Button
              onClick={handleNewQuery}
              variant="outline"
              size="sm"
              className="gap-2"
            >
              <Home className="w-4 h-4" />
              New Query
            </Button>
          </div>
        </div>
        
        {/* Analyzing content */}
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <Loader2 className="w-12 h-12 animate-spin mx-auto mb-6 text-blue-400" />
            <h2 className="text-xl font-semibold text-white mb-3">
              Processing Your Research Query
            </h2>
            <p className="text-neutral-400 leading-relaxed">
              We're setting up three specialized research sequences to analyze your query 
              from different perspectives. This will take just a moment.
            </p>
            
            {/* Progress dots */}
            <div className="flex justify-center gap-2 mt-6">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.32s]" />
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.16s]" />
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="h-full flex flex-col bg-neutral-950">
        <div className="border-b border-neutral-800 p-6">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-4">
              <GitCompare className="w-6 h-6 text-red-400" />
              <div>
                <h1 className="text-lg font-semibold text-white">
                  Research Error
                </h1>
                <p className="text-sm text-neutral-400 mt-1">
                  Failed to start parallel sequences
                </p>
              </div>
            </div>
            
            <Button
              onClick={handleNewQuery}
              variant="outline"
              size="sm"
              className="gap-2"
            >
              <Home className="w-4 h-4" />
              New Query
            </Button>
          </div>
        </div>
        
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="w-12 h-12 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
              <Activity className="w-6 h-6 text-red-400" />
            </div>
            <h2 className="text-xl font-semibold text-white mb-3">
              Connection Error
            </h2>
            <p className="text-neutral-400 leading-relaxed mb-4">
              {error.message || 'Failed to establish connection with research servers'}
            </p>
            <Button onClick={handleNewQuery} variant="default">
              Try Again
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Main 3-column view
  return (
    <div className="h-full flex flex-col bg-neutral-950">
      {/* Header */}
      <div className="border-b border-neutral-800 p-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <GitCompare className="w-6 h-6 text-blue-400" />
            <div>
              <h1 className="text-lg font-semibold text-white">
                Parallel Research Analysis
              </h1>
              <p className="text-sm text-neutral-400 mt-1 max-w-2xl">
                "{query}"
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Overall progress indicator */}
            <div className="hidden sm:flex items-center gap-2 text-sm text-neutral-400">
              <div className="w-20 bg-neutral-800 rounded-full h-2">
                <div 
                  className="h-2 rounded-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${progress.overall_progress}%` }}
                />
              </div>
              <span>{Math.round(progress.overall_progress)}%</span>
            </div>
            
            <Button
              onClick={handleNewQuery}
              variant="outline"
              size="sm"
              className="gap-2"
            >
              <Home className="w-4 h-4" />
              New Query
            </Button>
          </div>
        </div>
      </div>
      
      {/* 3-column grid */}
      <div className="flex-1 p-6 overflow-hidden">
        <div className="max-w-7xl mx-auto h-full">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
            {sequences.map((sequence) => (
              <SequenceColumn
                key={sequence.sequence_id}
                sequence={sequence}
                messages={getSequenceMessages(sequence.sequence_id)}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ParallelSequenceView;