/**
 * Claude-style message components with progressive disclosure
 * 
 * Features:
 * - Collapsible thinking sections
 * - Tool usage blocks
 * - Search result expansions
 * - Agent reasoning display
 * - Clean ChatGPT-style visual design
 */

import React, { useState, useCallback } from 'react';
import { Badge } from './badge';
import { Button } from './button';
import { Separator } from './separator';
import { cn } from '@/lib/utils';
import {
  ChevronRight,
  Brain,
  Code,
  Search,
  FileText,
  CheckCircle,
  Loader2,
  Copy,
  CopyCheck,
  ExternalLink,
  Eye,
  EyeOff
} from 'lucide-react';

// ============================================================================
// INTERFACES
// ============================================================================

interface ThinkingSection {
  id: string;
  title: string;
  content: string;
  charLength: number;
}

interface ToolCall {
  id: string;
  name: string;
  input: any;
  output?: any;
  status: 'running' | 'completed' | 'error';
  duration?: number;
}

interface SearchResult {
  id: string;
  title: string;
  url: string;
  snippet: string;
  relevanceScore?: number;
}

interface AgentTransition {
  from: string;
  to: string;
  reason: string;
  timestamp: number;
}

interface ClaudeStyleMessageProps {
  content: string;
  thinkingSections?: ThinkingSection[];
  toolCalls?: ToolCall[];
  searchResults?: SearchResult[];
  agentTransitions?: AgentTransition[];
  currentAgent?: string;
  timestamp: number;
  onCopy?: (content: string) => void;
  copiedMessageId?: string | null;
  messageId: string;
  className?: string;
}

// ============================================================================
// THINKING SECTION COMPONENT
// ============================================================================

interface ThinkingSectionProps {
  section: ThinkingSection;
  isExpanded: boolean;
  onToggle: () => void;
}

const ClaudeThinkingSection: React.FC<ThinkingSectionProps> = ({
  section,
  isExpanded,
  onToggle
}) => {
  return (
    <div className="border border-blue-500/20 bg-blue-50/[0.02] rounded-lg overflow-hidden mb-3">
      <button
        onClick={onToggle}
        className="w-full px-3 py-2.5 hover:bg-blue-50/[0.03] transition-colors text-left focus:outline-none focus:ring-1 focus:ring-blue-500/50"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded bg-blue-500/20 flex-shrink-0">
              <Brain className="h-3.5 w-3.5 text-blue-400" />
            </div>
            <span className="text-sm font-medium text-blue-300">
              {section.title}
            </span>
            <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/30 px-1.5 py-0.5">
              {section.charLength} chars
            </Badge>
          </div>
          <ChevronRight
            className={cn(
              'h-3.5 w-3.5 text-blue-400 transition-transform duration-200 flex-shrink-0',
              isExpanded && 'rotate-90'
            )}
          />
        </div>
      </button>
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-blue-500/20 bg-blue-950/[0.05]">
          <div className="pt-3">
            <div className="font-mono text-xs text-blue-100 leading-relaxed whitespace-pre-wrap break-words">
              {section.content}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// TOOL USAGE BLOCK COMPONENT
// ============================================================================

interface ToolUsageBlockProps {
  toolCall: ToolCall;
  isExpanded: boolean;
  onToggle: () => void;
}

const ClaudeToolUsageBlock: React.FC<ToolUsageBlockProps> = ({
  toolCall,
  isExpanded,
  onToggle
}) => {
  const getStatusIcon = () => {
    switch (toolCall.status) {
      case 'running':
        return <Loader2 className="h-3.5 w-3.5 text-amber-400 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-3.5 w-3.5 text-green-400" />;
      case 'error':
        return <div className="h-3.5 w-3.5 rounded-full bg-red-400" />;
      default:
        return <Code className="h-3.5 w-3.5 text-neutral-400" />;
    }
  };

  const getStatusColor = () => {
    switch (toolCall.status) {
      case 'running':
        return 'border-amber-500/30 bg-amber-50/[0.02]';
      case 'completed':
        return 'border-green-500/30 bg-green-50/[0.02]';
      case 'error':
        return 'border-red-500/30 bg-red-50/[0.02]';
      default:
        return 'border-neutral-500/30 bg-neutral-50/[0.02]';
    }
  };

  const getStatusText = () => {
    switch (toolCall.status) {
      case 'running':
        return 'Running...';
      case 'completed':
        return `Completed${toolCall.duration ? ` in ${toolCall.duration}ms` : ''}`;
      case 'error':
        return 'Error';
      default:
        return 'Pending';
    }
  };

  return (
    <div className={cn('border rounded-lg overflow-hidden mb-3', getStatusColor())}>
      <button
        onClick={onToggle}
        className="w-full px-3 py-2.5 hover:bg-white/[0.02] transition-colors text-left focus:outline-none focus:ring-1 focus:ring-blue-500/50"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex-shrink-0">
              {getStatusIcon()}
            </div>
            <div className="min-w-0">
              <span className="text-sm font-medium text-neutral-200">
                {toolCall.name}
              </span>
              <div className="text-xs text-neutral-400 mt-0.5">
                {getStatusText()}
              </div>
            </div>
            <Badge variant="outline" className="text-xs bg-neutral-500/10 text-neutral-400 border-neutral-500/30 flex-shrink-0">
              Tool
            </Badge>
          </div>
          <ChevronRight
            className={cn(
              'h-3.5 w-3.5 text-neutral-400 transition-transform duration-200 flex-shrink-0',
              isExpanded && 'rotate-90'
            )}
          />
        </div>
      </button>
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-neutral-500/20 bg-neutral-950/20">
          <div className="pt-3 space-y-3">
            <div>
              <div className="text-xs font-medium text-neutral-400 mb-1.5">Input:</div>
              <div className="bg-neutral-900 rounded-md p-2.5 border border-neutral-800">
                <pre className="font-mono text-xs text-neutral-200 whitespace-pre-wrap break-all">
                  {JSON.stringify(toolCall.input, null, 2)}
                </pre>
              </div>
            </div>
            {toolCall.output && (
              <div>
                <div className="text-xs font-medium text-neutral-400 mb-1.5">Output:</div>
                <div className="bg-neutral-900 rounded-md p-2.5 border border-neutral-800">
                  <pre className="font-mono text-xs text-neutral-200 whitespace-pre-wrap break-all">
                    {typeof toolCall.output === 'string' ? toolCall.output : JSON.stringify(toolCall.output, null, 2)}
                  </pre>
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
// SEARCH RESULTS COMPONENT
// ============================================================================

interface SearchResultsProps {
  results: SearchResult[];
  isExpanded: boolean;
  onToggle: () => void;
}

const ClaudeSearchResults: React.FC<SearchResultsProps> = ({
  results,
  isExpanded,
  onToggle
}) => {
  return (
    <div className="border border-emerald-500/30 bg-emerald-50/[0.02] rounded-lg overflow-hidden mb-3">
      <button
        onClick={onToggle}
        className="w-full px-3 py-2.5 hover:bg-emerald-50/[0.03] transition-colors text-left focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded bg-emerald-500/20 flex-shrink-0">
              <Search className="h-3.5 w-3.5 text-emerald-400" />
            </div>
            <span className="text-sm font-medium text-emerald-300">
              Search Results
            </span>
            <Badge variant="outline" className="text-xs bg-emerald-500/10 text-emerald-400 border-emerald-500/30 px-1.5 py-0.5">
              {results.length} found
            </Badge>
          </div>
          <ChevronRight
            className={cn(
              'h-3.5 w-3.5 text-emerald-400 transition-transform duration-200 flex-shrink-0',
              isExpanded && 'rotate-90'
            )}
          />
        </div>
      </button>
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-emerald-500/20 bg-emerald-950/[0.05]">
          <div className="pt-3 space-y-3">
            {results.map((result, index) => (
              <div key={result.id} className="p-3 bg-neutral-900/50 rounded-md border border-neutral-800">
                <div className="flex items-start gap-2">
                  <div className="flex-shrink-0 mt-0.5">
                    <FileText className="h-3.5 w-3.5 text-neutral-400" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <a 
                        href={result.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-sm font-medium text-blue-400 hover:text-blue-300 truncate"
                      >
                        {result.title}
                      </a>
                      <ExternalLink className="h-3 w-3 text-neutral-500 flex-shrink-0" />
                      {result.relevanceScore && (
                        <Badge variant="outline" className="text-xs bg-emerald-500/10 text-emerald-400 border-emerald-500/30 ml-auto">
                          {Math.round(result.relevanceScore * 100)}%
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-neutral-400 leading-relaxed">
                      {result.snippet}
                    </p>
                    <div className="text-xs text-neutral-500 mt-1 truncate">
                      {result.url}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// AGENT TRANSITION COMPONENT
// ============================================================================

interface AgentTransitionProps {
  transition: AgentTransition;
}

const ClaudeAgentTransition: React.FC<AgentTransitionProps> = ({ transition }) => {
  return (
    <div className="border border-purple-500/30 bg-purple-50/[0.02] rounded-lg p-3 mb-3">
      <div className="flex items-center gap-2.5 mb-2">
        <div className="p-1 rounded bg-purple-500/20">
          <ChevronRight className="h-3 w-3 text-purple-400" />
        </div>
        <span className="text-sm font-medium text-purple-300">
          Agent Transition
        </span>
        <Badge variant="outline" className="text-xs bg-purple-500/10 text-purple-400 border-purple-500/30">
          {new Date(transition.timestamp).toLocaleTimeString()}
        </Badge>
      </div>
      <div className="text-xs text-neutral-300 leading-relaxed">
        Switched from <span className="font-medium text-purple-300">{transition.from}</span> to{' '}
        <span className="font-medium text-purple-300">{transition.to}</span>
        <div className="mt-1 text-neutral-400">
          Reason: {transition.reason}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// MAIN CLAUDE-STYLE MESSAGE COMPONENT
// ============================================================================

const ClaudeStyleMessage: React.FC<ClaudeStyleMessageProps> = ({
  content,
  thinkingSections = [],
  toolCalls = [],
  searchResults = [],
  agentTransitions = [],
  currentAgent,
  timestamp,
  onCopy,
  copiedMessageId,
  messageId,
  className
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set());
  const [showAllSections, setShowAllSections] = useState(false);

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

  const toggleResults = useCallback((resultsId: string) => {
    setExpandedResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(resultsId)) {
        newSet.delete(resultsId);
      } else {
        newSet.add(resultsId);
      }
      return newSet;
    });
  }, []);

  const handleCopy = useCallback(async () => {
    if (onCopy) {
      onCopy(content);
    }
  }, [content, onCopy]);

  const toggleAllSections = useCallback(() => {
    const newShowAll = !showAllSections;
    setShowAllSections(newShowAll);
    
    if (newShowAll) {
      // Expand all
      const allSectionIds = thinkingSections.map(s => s.id);
      const allToolIds = toolCalls.map(t => t.id);
      const allResultIds = searchResults.length > 0 ? ['search-results'] : [];
      
      setExpandedSections(new Set(allSectionIds));
      setExpandedTools(new Set(allToolIds));
      setExpandedResults(new Set(allResultIds));
    } else {
      // Collapse all
      setExpandedSections(new Set());
      setExpandedTools(new Set());
      setExpandedResults(new Set());
    }
  }, [showAllSections, thinkingSections, toolCalls, searchResults]);

  const totalSections = thinkingSections.length + toolCalls.length + (searchResults.length > 0 ? 1 : 0);

  return (
    <div className={cn('mb-6', className)}>
      <div className="flex items-start gap-3">
        <div className="w-7 h-7 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0 mt-1">
          <Brain className="w-4 h-4 text-green-400" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="bg-neutral-50/[0.02] rounded-2xl p-4 shadow-sm border border-neutral-700/30">
            {/* Agent badge */}
            {currentAgent && (
              <div className="mb-3">
                <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/30">
                  {currentAgent}
                </Badge>
              </div>
            )}

            {/* Progressive disclosure controls */}
            {totalSections > 0 && (
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-neutral-500">
                    {totalSections} expandable section{totalSections !== 1 ? 's' : ''}
                  </span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleAllSections}
                  className="text-neutral-400 hover:text-neutral-200 h-6 px-2"
                >
                  {showAllSections ? (
                    <>
                      <EyeOff className="w-3 h-3 mr-1" />
                      <span className="text-xs">Collapse All</span>
                    </>
                  ) : (
                    <>
                      <Eye className="w-3 h-3 mr-1" />
                      <span className="text-xs">Expand All</span>
                    </>
                  )}
                </Button>
              </div>
            )}

            {/* Agent transitions */}
            {agentTransitions.map((transition, index) => (
              <ClaudeAgentTransition key={index} transition={transition} />
            ))}

            {/* Thinking sections */}
            {thinkingSections.map(section => (
              <ClaudeThinkingSection
                key={section.id}
                section={section}
                isExpanded={expandedSections.has(section.id)}
                onToggle={() => toggleSection(section.id)}
              />
            ))}

            {/* Tool usage blocks */}
            {toolCalls.map(toolCall => (
              <ClaudeToolUsageBlock
                key={toolCall.id}
                toolCall={toolCall}
                isExpanded={expandedTools.has(toolCall.id)}
                onToggle={() => toggleTool(toolCall.id)}
              />
            ))}

            {/* Search results */}
            {searchResults.length > 0 && (
              <ClaudeSearchResults
                results={searchResults}
                isExpanded={expandedResults.has('search-results')}
                onToggle={() => toggleResults('search-results')}
              />
            )}

            {/* Main content */}
            <div className="prose prose-invert max-w-none text-sm text-neutral-200 leading-relaxed">
              {content.split('\n').map((paragraph, index) => (
                <p key={index} className="mb-3 last:mb-0">
                  {paragraph}
                </p>
              ))}
            </div>

            {/* Metadata and copy button */}
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-neutral-700/30">
              <div className="text-xs text-neutral-500">
                {new Date(timestamp).toLocaleTimeString()}
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                className="text-neutral-400 hover:text-neutral-200 h-7 px-2"
              >
                {copiedMessageId === messageId ? (
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

export { ClaudeStyleMessage };
export default ClaudeStyleMessage;