/**
 * ParallelResearchDemo - Demonstration of the sophisticated side-by-side interface
 * 
 * Shows off:
 * - 3-column side-by-side layout
 * - ChatGPT-style clean message bubbles
 * - Claude-style progressive disclosure
 * - Tool usage blocks
 * - Independent scrolling
 */

import React, { useState, useEffect } from 'react';
import ParallelResearchInterface from '@/components/ParallelResearchInterface';
import ClaudeStyleMessage from '@/components/ui/claude-style-message';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Play, 
  Square, 
  RotateCcw, 
  Sparkles,
  GitBranch,
  Activity 
} from 'lucide-react';

import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';

// Mock data for demonstration
const mockSequences: LLMGeneratedSequence[] = [
  {
    sequence_id: 'theory-first-sequence',
    sequence_name: 'Theory-First Analysis',
    agent_names: ['research_agent', 'analysis_agent', 'synthesis_agent'],
    rationale: 'Starting with academic foundations to build comprehensive theoretical understanding before practical application.',
    research_focus: 'Academic literature review and theoretical frameworks',
    confidence_score: 0.92,
    approach_description: 'Academic → Industry → Technical progression for theoretical depth',
    expected_outcomes: ['Literature review', 'Theoretical framework', 'Academic insights'],
    created_at: new Date().toISOString(),
  },
  {
    sequence_id: 'market-first-sequence', 
    sequence_name: 'Market-First Strategy',
    agent_names: ['market_agent', 'research_agent', 'technical_agent'],
    rationale: 'Focusing on current market dynamics and industry trends to understand practical applications and commercial viability.',
    research_focus: 'Market analysis and commercial applications',
    confidence_score: 0.88,
    approach_description: 'Market → Academic → Technical for practical insights',
    expected_outcomes: ['Market analysis', 'Industry trends', 'Commercial viability'],
    created_at: new Date().toISOString(),
  },
  {
    sequence_id: 'future-back-sequence',
    sequence_name: 'Future-Back Planning',
    agent_names: ['technical_agent', 'market_agent', 'synthesis_agent'],
    rationale: 'Working backwards from emerging technologies and future possibilities to identify current opportunities and research gaps.',
    research_focus: 'Emerging technologies and future trends',
    confidence_score: 0.85,
    approach_description: 'Technical → Future → Present for innovative perspectives',
    expected_outcomes: ['Technology roadmap', 'Future scenarios', 'Innovation opportunities'],
    created_at: new Date().toISOString(),
  },
];

const generateMockMessages = (sequenceId: string, count: number): RoutedMessage[] => {
  const messages: RoutedMessage[] = [];
  const agents = ['research_agent', 'analysis_agent', 'synthesis_agent', 'market_agent', 'technical_agent'];
  
  for (let i = 0; i < count; i++) {
    const agent = agents[i % agents.length];
    const messageTypes: Array<'progress' | 'result' | 'error' | 'completion' | 'agent_transition'> = ['progress', 'agent_transition', 'result', 'completion'];
    const messageType = messageTypes[i % messageTypes.length];
    
    messages.push({
      message_id: `${sequenceId}-msg-${i}`,
      sequence_id: sequenceId,
      sequence_name: mockSequences.find(s => s.sequence_id === sequenceId)?.sequence_name || 'Unknown',
      message_type: messageType,
      timestamp: Date.now() - (count - i) * 30000, // Stagger timestamps
      content: generateMockContent(messageType, agent, i),
      sequence_index: 0,
      routing_timestamp: Date.now(),
      current_agent: agent,
      agent_type: agent as any,
    });
  }
  
  return messages;
};

const generateMockContent = (messageType: string, agent: string, index: number): string => {
  const contents = {
    progress: [
      `Analyzing the research landscape for ${agent.replace('_', ' ')}. Found ${5 + index} relevant sources and ${10 + index * 2} related papers.`,
      `Deep diving into the theoretical frameworks. Current focus: examining methodological approaches and their practical implications.`,
      `Synthesizing findings from multiple perspectives. Identifying key themes and patterns that emerge across different research domains.`,
      `Conducting comprehensive literature review. Processing academic sources and extracting relevant insights for synthesis.`,
    ],
    agent_transition: [
      `Transitioning from initial research phase to analysis. Handing off ${15 + index} sources to the analysis agent for deeper examination.`,
      `Moving from broad exploration to focused synthesis. Preparing comprehensive briefing for the synthesis agent.`,
      `Shifting focus from individual analysis to collaborative integration. Coordinating with other agents for holistic insights.`,
    ],
    result: [
      `# Research Results\n\nCompleted comprehensive analysis of the research domain. Key findings include:\n\n- **Theoretical Framework**: Identified 3 major theoretical approaches\n- **Methodological Insights**: 5 proven methodologies with strong empirical support\n- **Gap Analysis**: 4 significant research gaps requiring further investigation\n- **Practical Applications**: 7 potential real-world applications\n\nThe analysis reveals strong convergence around core concepts while highlighting areas for innovation.`,
      `# Analysis Summary\n\nDeep analysis completed with following insights:\n\n## Key Themes\n1. **Convergence Patterns**: Strong alignment across multiple research streams\n2. **Innovation Opportunities**: 3 high-potential areas for breakthrough research\n3. **Methodological Advances**: Novel approaches showing promising results\n\n## Recommendations\n- Pursue interdisciplinary collaboration\n- Focus on practical implementation\n- Address identified research gaps`,
      `# Synthesis Report\n\nIntegrated findings from multiple research sequences:\n\n## Core Insights\n- **Cross-Domain Patterns**: Consistent themes across all research areas\n- **Emerging Trends**: 5 key trends shaping the future landscape\n- **Strategic Implications**: Clear pathways for practical application\n\n## Next Steps\nRecommend moving to implementation phase with focused pilot studies.`,
    ],
    tool_usage: [
      `Executed web search and found 23 highly relevant academic sources. Processing citations and abstracting key concepts for integration.`,
      `Completed database query across 5 research databases. Retrieved 156 papers with relevance scores above 0.8 threshold.`,
      `Ran analysis tool on collected data. Generated statistical summaries and identified significant patterns in the research landscape.`,
    ],
  };
  
  const typeContents = contents[messageType as keyof typeof contents];
  return typeContents[index % typeContents.length];
};

const ParallelResearchDemo: React.FC = () => {
  const [activeSequenceId, setActiveSequenceId] = useState(mockSequences[0].sequence_id);
  const [isRunning, setIsRunning] = useState(false);
  const [parallelMessages, setParallelMessages] = useState<Record<string, RoutedMessage[]>>({});
  const [messageCount, setMessageCount] = useState(3);

  // Initialize with some demo messages
  useEffect(() => {
    const initialMessages: Record<string, RoutedMessage[]> = {};
    mockSequences.forEach(sequence => {
      initialMessages[sequence.sequence_id] = generateMockMessages(sequence.sequence_id, 2);
    });
    setParallelMessages(initialMessages);
  }, []);

  const handleStart = () => {
    setIsRunning(true);
    // Simulate adding messages over time
    const interval = setInterval(() => {
      setParallelMessages(prev => {
        const updated = { ...prev };
        mockSequences.forEach(sequence => {
          if ((updated[sequence.sequence_id]?.length || 0) < 8) {
            const currentMessages = updated[sequence.sequence_id] || [];
            const newMessage = generateMockMessages(sequence.sequence_id, 1)[0];
            newMessage.message_id = `${sequence.sequence_id}-msg-${currentMessages.length}`;
            newMessage.timestamp = Date.now();
            updated[sequence.sequence_id] = [...currentMessages, newMessage];
          }
        });
        return updated;
      });
      
      setMessageCount(prev => prev + 1);
      
      // Stop after adding several messages
      if (messageCount >= 8) {
        setIsRunning(false);
        clearInterval(interval);
      }
    }, 2000);

    // Clean up interval
    setTimeout(() => {
      setIsRunning(false);
      clearInterval(interval);
    }, 15000);
  };

  const handleStop = () => {
    setIsRunning(false);
  };

  const handleReset = () => {
    setIsRunning(false);
    setMessageCount(3);
    const resetMessages: Record<string, RoutedMessage[]> = {};
    mockSequences.forEach(sequence => {
      resetMessages[sequence.sequence_id] = generateMockMessages(sequence.sequence_id, 1);
    });
    setParallelMessages(resetMessages);
  };

  const totalMessages = Object.values(parallelMessages).reduce((sum, msgs) => sum + msgs.length, 0);

  return (
    <div className="h-screen bg-neutral-900 text-neutral-100 flex flex-col">
      {/* Header */}
      <div className="border-b border-neutral-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <GitBranch className="h-6 w-6 text-blue-400" />
            <div>
              <h1 className="text-xl font-bold text-neutral-100">
                Parallel Research Interface Demo
              </h1>
              <p className="text-sm text-neutral-400">
                Sophisticated side-by-side research with ChatGPT + Claude design patterns
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/30">
              <Sparkles className="h-3 w-3 mr-1" />
              {totalMessages} messages
            </Badge>
            
            <div className="flex items-center gap-2">
              {!isRunning ? (
                <Button
                  onClick={handleStart}
                  className="bg-green-600 hover:bg-green-700 text-white"
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start Demo
                </Button>
              ) : (
                <Button
                  onClick={handleStop}
                  variant="outline"
                  className="border-red-600 text-red-400 hover:bg-red-600/10"
                >
                  <Square className="h-4 w-4 mr-2" />
                  Stop
                </Button>
              )}
              
              <Button
                onClick={handleReset}
                variant="outline"
                className="border-blue-600 text-blue-400 hover:bg-blue-600/10"
              >
                <RotateCcw className="h-4 w-4 mr-2" />
                Reset
              </Button>
            </div>
          </div>
        </div>

        {/* Status indicators */}
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {mockSequences.map(sequence => {
              const messageCount = parallelMessages[sequence.sequence_id]?.length || 0;
              const isActive = activeSequenceId === sequence.sequence_id;
              
              return (
                <div
                  key={sequence.sequence_id}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors ${
                    isActive 
                      ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30' 
                      : 'bg-neutral-800 text-neutral-400'
                  }`}
                >
                  <Activity className="h-3 w-3" />
                  <span className="text-sm font-medium">{sequence.sequence_name}</span>
                  <Badge variant="secondary" className="text-xs bg-neutral-700 text-neutral-300">
                    {messageCount}
                  </Badge>
                  {isRunning && (
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  )}
                </div>
              );
            })}
          </div>
          
          <div className="text-sm text-neutral-500">
            {isRunning ? 'Generating parallel research...' : 'Ready to start parallel research'}
          </div>
        </div>
      </div>

      {/* Main Interface */}
      <div className="flex-1 min-h-0 p-4">
        <div className="h-full border border-neutral-700/50 rounded-lg overflow-hidden bg-neutral-900/50">
          <ParallelResearchInterface
            sequences={mockSequences}
            parallelMessages={parallelMessages}
            activeTabId={activeSequenceId}
            onTabChange={setActiveSequenceId}
            isLoading={isRunning}
          />
        </div>
      </div>

      {/* Footer info */}
      <div className="border-t border-neutral-700 p-3">
        <div className="flex items-center justify-between text-xs text-neutral-500">
          <div>
            Features: Side-by-side columns • Independent scrolling • Progressive disclosure • Tool blocks
          </div>
          <div>
            Design: ChatGPT clean bubbles + Claude thinking sections
          </div>
        </div>
      </div>
    </div>
  );
};

// Demo component showing Claude-style message features
export const ClaudeStyleMessageDemo: React.FC = () => {
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  const handleCopy = (content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedMessageId('demo-message');
    setTimeout(() => setCopiedMessageId(null), 2000);
  };

  const mockThinkingSections = [
    {
      id: 'thinking-1',
      title: 'Initial Analysis',
      content: `I need to approach this research query systematically. Let me break down the key components:

1. Understanding the scope and objectives
2. Identifying relevant research domains
3. Mapping methodological approaches
4. Considering interdisciplinary connections

The user is asking for a comprehensive analysis, so I should ensure I cover both theoretical foundations and practical applications.`,
      charLength: 356,
    },
    {
      id: 'thinking-2', 
      title: 'Strategic Planning',
      content: `Now I'll develop a research strategy:

Primary focus areas:
- Academic literature review
- Industry applications
- Emerging trends and future directions

I should prioritize high-impact sources and ensure I'm capturing diverse perspectives across the research landscape.`,
      charLength: 267,
    },
  ];

  const mockToolCalls = [
    {
      id: 'search-1',
      name: 'web_search',
      input: { 
        query: 'academic research methodology systematic review', 
        max_results: 10,
        time_range: 'recent'
      },
      output: 'Found 47 relevant academic sources with high citation counts. Retrieved abstracts and key findings from top 15 papers.',
      status: 'completed' as const,
      duration: 1247,
    },
    {
      id: 'analysis-1',
      name: 'citation_analysis',
      input: { 
        papers: ['paper1', 'paper2', 'paper3'],
        analysis_type: 'thematic'
      },
      output: 'Identified 5 major themes and 12 sub-themes across the analyzed papers. Generated thematic map with relevance scores.',
      status: 'completed' as const,
      duration: 892,
    },
  ];

  const mockSearchResults = [
    {
      id: 'result-1',
      title: 'Systematic Review Methodologies in Contemporary Research',
      url: 'https://example.com/systematic-review-methods',
      snippet: 'Comprehensive overview of modern systematic review approaches, including PRISMA guidelines and meta-analysis techniques for evidence synthesis.',
      relevanceScore: 0.94,
    },
    {
      id: 'result-2',
      title: 'Interdisciplinary Research Frameworks: A Methodological Guide',
      url: 'https://example.com/interdisciplinary-frameworks',
      snippet: 'Practical guide to conducting research across disciplinary boundaries, with emphasis on methodological integration and collaborative approaches.',
      relevanceScore: 0.89,
    },
    {
      id: 'result-3',
      title: 'Evidence-Based Research Design in Academic Settings',
      url: 'https://example.com/evidence-based-design',
      snippet: 'Best practices for designing robust research studies with strong methodological foundations and reproducible results.',
      relevanceScore: 0.87,
    },
  ];

  const mockAgentTransitions = [
    {
      from: 'research_agent',
      to: 'analysis_agent',
      reason: 'Completed initial literature review and source gathering. Ready for deeper thematic analysis.',
      timestamp: Date.now() - 300000, // 5 minutes ago
    },
  ];

  return (
    <div className="max-w-4xl mx-auto p-6 bg-neutral-900 min-h-screen">
      <Card className="mb-6 border-neutral-700 bg-neutral-800">
        <CardHeader>
          <CardTitle className="text-neutral-100">Claude-Style Message Demo</CardTitle>
          <CardDescription className="text-neutral-400">
            Showcasing progressive disclosure with thinking sections, tool usage, and search results
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="space-y-6">
        <ClaudeStyleMessage
          messageId="demo-message"
          content="I've completed a comprehensive analysis of the research landscape. The methodology I employed involved systematic literature review, thematic analysis, and cross-domain synthesis. Here are the key findings and insights from this research process."
          thinkingSections={mockThinkingSections}
          toolCalls={mockToolCalls}
          searchResults={mockSearchResults}
          agentTransitions={mockAgentTransitions}
          currentAgent="research_agent"
          timestamp={Date.now()}
          onCopy={handleCopy}
          copiedMessageId={copiedMessageId}
        />
      </div>
    </div>
  );
};

export default ParallelResearchDemo;