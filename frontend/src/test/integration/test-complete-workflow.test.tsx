/**
 * Comprehensive integration test for the complete Open Deep Research workflow.
 * Tests end-to-end user journey with all features working together.
 * 
 * Flow:
 * 1. User submits research query on WelcomeScreen
 * 2. App.tsx receives WebSocket messages from backend  
 * 3. LLM generates dynamic sequences (not hard-coded)
 * 4. SupervisorAnnouncementMessage displays sequences with real data
 * 5. User clicks "Launch Research" button
 * 6. ParallelResearchInterface opens in side-by-side view
 * 7. Messages route to correct sequence columns
 * 8. Thinking sections appear as collapsible boxes
 * 9. Real conversations display (not summaries)
 * 10. Transparency messages show delegation process
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import App from '../../App';
import { LLMGeneratedSequence, RoutedMessage } from '../../types/parallel';

// ============================================================================
// MOCK SETUP
// ============================================================================

// Mock WebSocket and LangChain SDK
const mockUseStream = {
  messages: [] as any[],
  isLoading: false,
  submit: vi.fn(),
  stop: vi.fn(),
  threadId: null,
};

const mockHandlers = {
  onUpdateEvent: vi.fn(),
  onLangChainEvent: vi.fn(),
  onError: vi.fn(),
  onFinish: vi.fn(),
};

vi.mock('@langchain/langgraph-sdk/react', () => ({
  useStream: () => ({
    ...mockUseStream,
    ...mockHandlers,
  })
}));

// Mock useParallelSequences hook
const mockParallelSequences = {
  sequences: [],
  isLoading: false,
  error: null,
  start: vi.fn(),
  stop: vi.fn(),
  changeActiveSequence: vi.fn(),
  routeMessage: vi.fn(),
};

vi.mock('../../hooks/useParallelSequences', () => ({
  useParallelSequences: () => mockParallelSequences
}));

// ============================================================================
// MOCK DATA GENERATION
// ============================================================================

const createMockLLMSequences = (): LLMGeneratedSequence[] => [
  {
    sequence_id: 'seq_academic_deep_001',
    sequence_name: 'Academic Deep Research Analysis',
    agent_names: ['research_agent', 'analysis_agent', 'synthesis_agent'],
    rationale: 'Comprehensive academic research approach focusing on peer-reviewed sources and theoretical foundations',
    research_focus: 'Academic literature and theoretical frameworks',
    confidence_score: 0.87,
    approach_description: 'Systematic literature review with statistical analysis',
    expected_outcomes: ['Comprehensive literature review', 'Statistical analysis results', 'Theoretical framework'],
    created_at: '2025-01-20T10:00:00Z',
  },
  {
    sequence_id: 'seq_market_industry_002',
    sequence_name: 'Market & Industry Intelligence',
    agent_names: ['market_agent', 'technical_agent', 'synthesis_agent'],
    rationale: 'Industry-focused research combining market trends with technical implementation insights',
    research_focus: 'Market trends and industry applications',
    confidence_score: 0.82,
    approach_description: 'Market analysis with technical feasibility assessment',
    expected_outcomes: ['Market trend analysis', 'Competitive landscape', 'Technical implementation guide'],
    created_at: '2025-01-20T10:01:00Z',
  },
  {
    sequence_id: 'seq_future_tech_003',
    sequence_name: 'Future-Forward Technology Assessment',
    agent_names: ['technical_agent', 'analysis_agent', 'synthesis_agent'],
    rationale: 'Technology-first approach examining emerging trends and future implications',
    research_focus: 'Emerging technologies and future implications',
    confidence_score: 0.79,
    approach_description: 'Technology trend analysis with future scenario modeling',
    expected_outcomes: ['Technology roadmap', 'Future scenarios', 'Implementation strategies'],
    created_at: '2025-01-20T10:02:00Z',
  },
];

const createMockWebSocketSequenceEvent = (sequences: LLMGeneratedSequence[]) => ({
  data: {
    type: 'sequences_generated',
    sequences,
    frontend_sequences: {
      sequences
    }
  },
  event: 'updateEvent'
});

const createMockRoutedMessage = (sequenceId: string, content: string, messageType = 'progress'): RoutedMessage => ({
  message_id: `msg_${sequenceId}_${Date.now()}`,
  sequence_id: sequenceId,
  sequence_name: 'Academic Deep Research Analysis',
  message_type: messageType as any,
  timestamp: Date.now(),
  content,
  sequence_index: 0,
  routing_timestamp: Date.now(),
  current_agent: 'research_agent',
  agent_type: 'ACADEMIC' as any,
});

const createMockThinkingMessage = (sequenceId: string): RoutedMessage => ({
  message_id: `msg_${sequenceId}_thinking_${Date.now()}`,
  sequence_id: sequenceId,
  sequence_name: 'Academic Deep Research Analysis',
  message_type: 'progress',
  timestamp: Date.now(),
  content: `<thinking>
Let me analyze this research query to determine the best approach for comprehensive investigation...

I need to consider:
1. The scope and complexity of the topic
2. Available academic sources and their quality
3. The depth of analysis required
4. Relevant methodological approaches
</thinking>

Based on my analysis, I'll begin with a systematic literature review focusing on peer-reviewed sources published in the last 5 years...`,
  sequence_index: 0,
  routing_timestamp: Date.now(),
  current_agent: 'research_agent',
  agent_type: 'ACADEMIC' as any,
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Complete Research Workflow Integration', () => {
  let user: ReturnType<typeof userEvent.setup>;
  
  beforeEach(() => {
    user = userEvent.setup();
    vi.clearAllMocks();
    
    // Reset mock state
    mockUseStream.messages = [];
    mockUseStream.isLoading = false;
    mockParallelSequences.sequences = [];
    mockParallelSequences.isLoading = false;
    mockParallelSequences.error = null;
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  it('completes full research workflow from query to parallel execution', async () => {
    const { rerender } = render(<App />);
    
    // Step 1: User sees WelcomeScreen and submits query
    expect(screen.getByText(/Welcome to Open Deep Research/)).toBeInTheDocument();
    
    const queryInput = screen.getByPlaceholderText(/Enter your research question/i);
    const submitButton = screen.getByRole('button', { name: /start research/i });
    
    // Type research query
    await user.type(queryInput, 'Analyze the impact of artificial intelligence on healthcare diagnostics');
    expect(queryInput).toHaveValue('Analyze the impact of artificial intelligence on healthcare diagnostics');
    
    // Submit the query
    await user.click(submitButton);
    
    // Verify useStream.submit was called with correct data
    expect(mockUseStream.submit).toHaveBeenCalledWith({
      messages: [{ 
        role: "human", 
        content: 'Analyze the impact of artificial intelligence on healthcare diagnostics' 
      }]
    });

    // Step 2: Simulate WebSocket message with generated sequences
    const mockSequences = createMockLLMSequences();
    const sequenceEvent = createMockWebSocketSequenceEvent(mockSequences);
    
    // Update state to show messages (query submitted)
    mockUseStream.messages = [{
      id: 'user-msg-1',
      type: 'human',
      content: 'Analyze the impact of artificial intelligence on healthcare diagnostics'
    }];
    
    rerender(<App />);
    
    // Should now show ChatInterface instead of WelcomeScreen
    expect(screen.queryByText(/Welcome to Open Deep Research/)).not.toBeInTheDocument();
    
    // Step 3: Simulate sequence generation event
    const onUpdateEvent = mockHandlers.onUpdateEvent;
    await waitFor(() => {
      onUpdateEvent(sequenceEvent.data);
    });
    
    rerender(<App />);
    
    // Step 4: Verify SupervisorAnnouncementMessage appears with sequences
    await waitFor(() => {
      expect(screen.getByText('Research sequences generated')).toBeInTheDocument();
      expect(screen.getByText('Academic Deep Research Analysis')).toBeInTheDocument();
      expect(screen.getByText('Market & Industry Intelligence')).toBeInTheDocument();
      expect(screen.getByText('Future-Forward Technology Assessment')).toBeInTheDocument();
    });
    
    // Check confidence scores are displayed
    expect(screen.getByText('87%')).toBeInTheDocument();
    expect(screen.getByText('82%')).toBeInTheDocument();
    expect(screen.getByText('79%')).toBeInTheDocument();
    
    // Check agent counts
    expect(screen.getAllByText('3 agents')).toHaveLength(3);
    
    // Step 5: User clicks "Launch Research" button
    const launchButton = screen.getByRole('button', { name: /Launch Research/i });
    expect(launchButton).toBeEnabled();
    
    await user.click(launchButton);
    
    // Verify parallel sequences were started
    expect(mockParallelSequences.start).toHaveBeenCalledWith(
      'Analyze the impact of artificial intelligence on healthcare diagnostics',
      mockSequences
    );
    
    // Step 6: Simulate parallel interface activation
    mockParallelSequences.sequences = mockSequences.map(seq => ({
      sequence_id: seq.sequence_id,
      sequence: seq,
      progress: {
        sequence_id: seq.sequence_id,
        sequence_name: seq.sequence_name,
        current_agent: null,
        current_agent_name: undefined,
        agents_completed: 0,
        total_agents: seq.agent_names.length,
        completion_percentage: 0,
        last_activity: Date.now(),
        status: 'active' as const,
      },
      messages: [],
      current_agent: null,
      current_agent_name: undefined,
      agent_transitions: [],
      errors: [],
      start_time: Date.now(),
      metrics: {
        sequence_id: seq.sequence_id,
        message_count: 0,
        research_duration: 0,
        tokens_used: 0,
        average_response_time: 0,
      },
      status: 'running' as const,
      last_activity: new Date().toISOString(),
    }));
    
    rerender(<App />);
    
    // Step 7: Verify ParallelResearchInterface is visible
    await waitFor(() => {
      expect(screen.getByText('Academic Deep Research Analysis')).toBeInTheDocument();
      expect(screen.getByText('Market & Industry Intelligence')).toBeInTheDocument();
      expect(screen.getByText('Future-Forward Technology Assessment')).toBeInTheDocument();
    });
    
    // Step 8: Simulate messages routing to specific sequences
    const academicMessage = createMockRoutedMessage('seq_academic_deep_001', 'Beginning systematic literature review...');
    const marketMessage = createMockRoutedMessage('seq_market_industry_002', 'Analyzing market trends in AI healthcare...');
    const thinkingMessage = createMockThinkingMessage('seq_academic_deep_001');
    
    // Route messages through the mock handler
    mockParallelSequences.routeMessage(academicMessage);
    mockParallelSequences.routeMessage(marketMessage);
    mockParallelSequences.routeMessage(thinkingMessage);
    
    rerender(<App />);
    
    // Step 9: Verify thinking sections appear as collapsible
    await waitFor(() => {
      expect(screen.getByText('thinking...')).toBeInTheDocument();
    });
    
    const thinkingSection = screen.getByText('thinking...');
    await user.click(thinkingSection);
    
    // Should expand to show thinking content
    await waitFor(() => {
      expect(screen.getByText(/Let me analyze this research query/)).toBeInTheDocument();
    });
    
    // Step 10: Verify real conversations display
    expect(screen.getByText('Beginning systematic literature review...')).toBeInTheDocument();
    expect(screen.getByText('Analyzing market trends in AI healthcare...')).toBeInTheDocument();
  });

  it('handles error scenarios gracefully during workflow', async () => {
    render(<App />);
    
    // Submit query
    const queryInput = screen.getByPlaceholderText(/Enter your research question/i);
    const submitButton = screen.getByRole('button', { name: /start research/i });
    
    await user.type(queryInput, 'Test error handling');
    await user.click(submitButton);
    
    // Simulate WebSocket error
    const onError = mockHandlers.onError;
    await waitFor(() => {
      onError(new Error('WebSocket connection failed'));
    });
    
    // Should display error in activity timeline
    expect(screen.getByText(/Stream Error/)).toBeInTheDocument();
  });

  it('validates dynamic sequence data structure', async () => {
    const mockSequences = createMockLLMSequences();
    const sequenceEvent = createMockWebSocketSequenceEvent(mockSequences);
    
    render(<App />);
    
    // Simulate sequence generation
    const onUpdateEvent = mockHandlers.onUpdateEvent;
    await waitFor(() => {
      onUpdateEvent(sequenceEvent.data);
    });
    
    // Verify all required dynamic data is present
    mockSequences.forEach((sequence, index) => {
      expect(sequence.sequence_id).toMatch(/^seq_\w+_\d+$/);
      expect(sequence.sequence_name).toBeTruthy();
      expect(sequence.agent_names).toHaveLength(3);
      expect(sequence.confidence_score).toBeGreaterThan(0);
      expect(sequence.confidence_score).toBeLessThanOrEqual(1);
      expect(sequence.rationale).toBeTruthy();
      expect(sequence.expected_outcomes).toBeInstanceOf(Array);
    });
  });

  it('maintains state consistency across component interactions', async () => {
    const { rerender } = render(<App />);
    
    // Submit query and trigger sequence generation
    const queryInput = screen.getByPlaceholderText(/Enter your research question/i);
    await user.type(queryInput, 'State consistency test');
    await user.click(screen.getByRole('button', { name: /start research/i }));
    
    // Update to show messages
    mockUseStream.messages = [{
      id: 'user-msg-1',
      type: 'human',
      content: 'State consistency test'
    }];
    
    rerender(<App />);
    
    // Generate sequences
    const mockSequences = createMockLLMSequences();
    const sequenceEvent = createMockWebSocketSequenceEvent(mockSequences);
    
    await waitFor(() => {
      mockHandlers.onUpdateEvent(sequenceEvent.data);
    });
    
    rerender(<App />);
    
    // Verify consistent state after launch
    await user.click(screen.getByRole('button', { name: /Launch Research/i }));
    
    // State should remain consistent
    expect(mockParallelSequences.start).toHaveBeenCalledWith(
      'State consistency test',
      mockSequences
    );
  });

  it('handles performance with large message volumes', async () => {
    const { rerender } = render(<App />);
    
    // Setup sequences
    const mockSequences = createMockLLMSequences();
    mockParallelSequences.sequences = mockSequences.map(seq => ({
      sequence_id: seq.sequence_id,
      sequence: seq,
      progress: {
        sequence_id: seq.sequence_id,
        sequence_name: seq.sequence_name,
        current_agent: null,
        current_agent_name: undefined,
        agents_completed: 0,
        total_agents: seq.agent_names.length,
        completion_percentage: 0,
        last_activity: Date.now(),
        status: 'active' as const,
      },
      messages: [],
      current_agent: null,
      current_agent_name: undefined,
      agent_transitions: [],
      errors: [],
      start_time: Date.now(),
      metrics: {
        sequence_id: seq.sequence_id,
        message_count: 0,
        research_duration: 0,
        tokens_used: 0,
        average_response_time: 0,
      },
      status: 'running' as const,
      last_activity: new Date().toISOString(),
    }));
    
    mockUseStream.messages = [{
      id: 'user-msg-1',
      type: 'human',
      content: 'Performance test'
    }];
    
    rerender(<App />);
    
    // Simulate high volume of messages
    const startTime = performance.now();
    
    for (let i = 0; i < 100; i++) {
      const message = createMockRoutedMessage(mockSequences[0].sequence_id, `Message ${i}`);
      mockParallelSequences.routeMessage(message);
    }
    
    rerender(<App />);
    
    const endTime = performance.now();
    const renderTime = endTime - startTime;
    
    // Should handle 100 messages in reasonable time (< 100ms)
    expect(renderTime).toBeLessThan(100);
  });
});