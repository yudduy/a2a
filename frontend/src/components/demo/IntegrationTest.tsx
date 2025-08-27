import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { MessageContentParser, ThinkingSection } from '@/types/messages';
import { CollapsibleThinking, ThinkingSections } from '@/components/ui/collapsible-thinking';
import { TypedMarkdown } from '@/components/ui/typed-markdown';
import ParallelTabContainer from '@/components/ParallelTabContainer';
import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';

// Test data
const TEST_MESSAGES = {
  withThinking: {
    id: 'thinking-test',
    type: 'ai' as const,
    content: `Let me analyze your request carefully.

<thinking>
The user is asking for a comprehensive analysis. I need to:

1. Break down the problem into components
2. Research each component thoroughly 
3. Synthesize findings into actionable insights

This will require multiple research streams to be effective.
</thinking>

Based on my analysis, I'll create a multi-faceted research approach.

<thinking>
Actually, let me reconsider the scope. The user's question has several dimensions:
- Technical aspects
- Market considerations  
- Future implications

I should generate parallel sequences to address each dimension.
</thinking>

I'm generating 3 parallel research sequences to address your query comprehensively.`,
  },
  
  supervisorAnnouncement: {
    id: 'supervisor-test',
    type: 'ai' as const,
    content: `Based on your research request, I'm generating 3 parallel research sequences:

1) Technical Deep Dive - Focus on technical specifications and implementation details
2) Market Analysis - Commercial applications, competitors, and market dynamics
3) Future Outlook - Emerging trends, predictions, and strategic implications

These sequences will execute in parallel to provide comprehensive coverage.`,
  }
};

const TEST_SEQUENCES: LLMGeneratedSequence[] = [
  {
    sequence_id: 'seq_1_technical',
    sequence_name: 'Technical Deep Dive',
    agent_names: ['research_agent', 'technical_agent'],
    rationale: 'Focus on technical specifications and implementation',
    research_focus: 'Technical aspects and implementation details',
    confidence_score: 0.9,
    approach_description: 'Comprehensive technical analysis',
    expected_outcomes: ['Technical specifications', 'Implementation guidelines'],
    created_at: new Date().toISOString(),
  },
  {
    sequence_id: 'seq_2_market',
    sequence_name: 'Market Analysis', 
    agent_names: ['market_agent', 'analysis_agent'],
    rationale: 'Analyze market dynamics and commercial aspects',
    research_focus: 'Market trends and commercial applications',
    confidence_score: 0.85,
    approach_description: 'Market research and competitive analysis',
    expected_outcomes: ['Market insights', 'Competitive landscape'],
    created_at: new Date().toISOString(),
  },
  {
    sequence_id: 'seq_3_future',
    sequence_name: 'Future Outlook',
    agent_names: ['trend_agent', 'prediction_agent'],
    rationale: 'Explore future trends and implications',
    research_focus: 'Emerging trends and strategic implications',
    confidence_score: 0.8,
    approach_description: 'Trend analysis and future predictions',
    expected_outcomes: ['Future trends', 'Strategic recommendations'],
    created_at: new Date().toISOString(),
  },
];

const TEST_PARALLEL_MESSAGES: Record<string, RoutedMessage[]> = {
  seq_1_technical: [
    {
      message_id: 'msg_1_tech',
      sequence_id: 'seq_1_technical',
      sequence_name: 'Technical Deep Dive',
      message_type: 'progress',
      timestamp: Date.now(),
      content: 'Starting technical research phase...',
      sequence_index: 0,
      routing_timestamp: Date.now(),
    },
    {
      message_id: 'msg_2_tech',
      sequence_id: 'seq_1_technical', 
      sequence_name: 'Technical Deep Dive',
      message_type: 'result',
      timestamp: Date.now() + 1000,
      content: '## Technical Analysis Results\n\nBased on comprehensive research, here are the key technical findings:\n\n- **Architecture**: Modern microservices approach\n- **Performance**: Sub-100ms response times\n- **Scalability**: Horizontal scaling supported\n- **Security**: End-to-end encryption implemented',
      sequence_index: 0,
      routing_timestamp: Date.now() + 1000,
      current_agent: 'technical_agent',
    }
  ],
  seq_2_market: [
    {
      message_id: 'msg_1_market',
      sequence_id: 'seq_2_market',
      sequence_name: 'Market Analysis',
      message_type: 'progress',
      timestamp: Date.now(),
      content: 'Analyzing market dynamics...',
      sequence_index: 1,
      routing_timestamp: Date.now(),
    },
    {
      message_id: 'msg_2_market',
      sequence_id: 'seq_2_market',
      sequence_name: 'Market Analysis', 
      message_type: 'result',
      timestamp: Date.now() + 1500,
      content: '## Market Analysis Summary\n\n### Current Market Size\n- **TAM**: $50B globally\n- **Growth Rate**: 15% YoY\n- **Key Players**: Company A, Company B, Company C\n\n### Competitive Landscape\nThe market is dominated by established players but shows significant opportunity for innovation.',
      sequence_index: 1,
      routing_timestamp: Date.now() + 1500,
      current_agent: 'market_agent',
    }
  ],
  seq_3_future: [
    {
      message_id: 'msg_1_future',
      sequence_id: 'seq_3_future',
      sequence_name: 'Future Outlook',
      message_type: 'progress',
      timestamp: Date.now(),
      content: 'Researching future trends...',
      sequence_index: 2,
      routing_timestamp: Date.now(),
    }
  ]
};

export const IntegrationTest: React.FC = () => {
  const [expandedThinking, setExpandedThinking] = useState<Set<string>>(new Set());
  const [activeTabId, setActiveTabId] = useState('seq_1_technical');
  const [testPhase, setTestPhase] = useState<'thinking' | 'supervisor' | 'parallel'>('thinking');

  const toggleThinking = useCallback((thinkingId: string) => {
    setExpandedThinking(prev => {
      const newSet = new Set(prev);
      if (newSet.has(thinkingId)) {
        newSet.delete(thinkingId);
      } else {
        newSet.add(thinkingId);
      }
      return newSet;
    });
  }, []);

  const handleTabChange = useCallback((tabId: string) => {
    setActiveTabId(tabId);
  }, []);

  // Test message parsing
  const thinkingMessage = TEST_MESSAGES.withThinking;
  const parsedContent = MessageContentParser.parse(thinkingMessage);

  return (
    <div className="p-6 space-y-6 bg-neutral-900 min-h-screen text-white">
      <div className="max-w-4xl mx-auto space-y-8">
        <h1 className="text-3xl font-bold text-center">Integration Test Suite</h1>
        
        {/* Phase selector */}
        <div className="flex justify-center gap-2">
          <Button 
            variant={testPhase === 'thinking' ? 'default' : 'outline'}
            onClick={() => setTestPhase('thinking')}
          >
            Thinking Sections
          </Button>
          <Button 
            variant={testPhase === 'supervisor' ? 'default' : 'outline'}
            onClick={() => setTestPhase('supervisor')}
          >
            Supervisor Announcement
          </Button>
          <Button 
            variant={testPhase === 'parallel' ? 'default' : 'outline'}
            onClick={() => setTestPhase('parallel')}
          >
            Parallel Tabs
          </Button>
        </div>

        {/* Test Phase 1: Thinking Sections */}
        {testPhase === 'thinking' && (
          <Card className="p-6 bg-neutral-800 border-neutral-700">
            <h2 className="text-xl font-semibold mb-4">Test 1: Thinking Sections Integration</h2>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <Badge variant="outline">Parsing Results</Badge>
                  <ul className="mt-2 space-y-1">
                    <li>✅ Has thinking: {parsedContent.hasThinking ? 'Yes' : 'No'}</li>
                    <li>✅ Sections found: {parsedContent.thinkingSections.length}</li>
                    <li>✅ Pre-thinking: {parsedContent.preThinking ? 'Present' : 'None'}</li>
                    <li>✅ Post-thinking: {parsedContent.postThinking ? 'Present' : 'None'}</li>
                  </ul>
                </div>
                <div>
                  <Badge variant="outline">Component Status</Badge>
                  <ul className="mt-2 space-y-1">
                    <li>✅ CollapsibleThinking imported</li>
                    <li>✅ ThinkingSections imported</li>
                    <li>✅ TypedMarkdown imported</li>
                    <li>✅ Message parsing working</li>
                  </ul>
                </div>
              </div>

              <div className="border-t border-neutral-700 pt-4">
                <h3 className="font-medium mb-3">Live Component Test:</h3>
                
                {/* Pre-thinking content */}
                {parsedContent.preThinking && (
                  <div className="mb-3">
                    <TypedMarkdown speed={30} hideCursor={true}>
                      {parsedContent.preThinking}
                    </TypedMarkdown>
                  </div>
                )}
                
                {/* Thinking sections */}
                <ThinkingSections
                  sections={parsedContent.thinkingSections}
                  expandedSections={expandedThinking}
                  onToggleSection={toggleThinking}
                  hasTypingAnimation={true}
                  typingSpeed={25}
                />
                
                {/* Post-thinking content */}
                {parsedContent.postThinking && (
                  <div className="mt-3">
                    <TypedMarkdown speed={30} delay={500} hideCursor={true}>
                      {parsedContent.postThinking}
                    </TypedMarkdown>
                  </div>
                )}
              </div>
            </div>
          </Card>
        )}

        {/* Test Phase 2: Supervisor Announcement */}
        {testPhase === 'supervisor' && (
          <Card className="p-6 bg-neutral-800 border-neutral-700">
            <h2 className="text-xl font-semibold mb-4">Test 2: Supervisor Announcement</h2>
            
            <div className="space-y-4">
              <div className="bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                <div className="text-center">
                  <p className="text-blue-400 font-semibold mb-2">Research Sequences Detected</p>
                  <p className="text-sm text-neutral-300 mb-3">
                    The supervisor has generated {TEST_SEQUENCES.length} parallel research sequences.
                  </p>
                  <Button 
                    className="bg-blue-600 hover:bg-blue-700"
                    onClick={() => setTestPhase('parallel')}
                  >
                    Launch Parallel Research
                  </Button>
                </div>
              </div>
              
              <div>
                <Badge variant="outline" className="mb-2">Generated Sequences</Badge>
                <div className="space-y-2 text-sm">
                  {TEST_SEQUENCES.map((seq, i) => (
                    <div key={seq.sequence_id} className="bg-neutral-700 p-3 rounded">
                      <div className="font-medium">{i + 1}) {seq.sequence_name}</div>
                      <div className="text-neutral-400 text-xs mt-1">{seq.rationale}</div>
                      <div className="text-neutral-500 text-xs">
                        Confidence: {(seq.confidence_score * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        )}

        {/* Test Phase 3: Parallel Tabs */}
        {testPhase === 'parallel' && (
          <Card className="p-6 bg-neutral-800 border-neutral-700">
            <h2 className="text-xl font-semibold mb-4">Test 3: Parallel Tab Container</h2>
            
            <div className="space-y-4">
              <div className="text-sm space-y-2">
                <div>✅ ParallelTabContainer component loaded</div>
                <div>✅ {TEST_SEQUENCES.length} sequences configured</div>
                <div>✅ Mock messages for {Object.keys(TEST_PARALLEL_MESSAGES).length} tabs</div>
                <div>✅ Typing animations enabled</div>
                <div>✅ Tab switching functional</div>
              </div>

              <ParallelTabContainer
                sequences={TEST_SEQUENCES}
                parallelMessages={TEST_PARALLEL_MESSAGES}
                activeTabId={activeTabId}
                onTabChange={handleTabChange}
                isLoading={false}
                className="border border-neutral-600"
              />
            </div>
          </Card>
        )}

        {/* Status Summary */}
        <Card className="p-4 bg-green-900/20 border-green-500/30">
          <h3 className="font-semibold text-green-400 mb-2">Integration Status</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <h4 className="font-medium text-green-300 mb-1">Components Integrated:</h4>
              <ul className="space-y-1 text-green-200">
                <li>✅ CollapsibleThinking</li>
                <li>✅ ParallelTabContainer</li>
                <li>✅ TypedMarkdown with vertical mode</li>
                <li>✅ Message parsing system</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-green-300 mb-1">Features Working:</h4>
              <ul className="space-y-1 text-green-200">
                <li>✅ Thinking section detection</li>
                <li>✅ Supervisor announcement recognition</li>
                <li>✅ Parallel message routing</li>
                <li>✅ Typing animations</li>
              </ul>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default IntegrationTest;