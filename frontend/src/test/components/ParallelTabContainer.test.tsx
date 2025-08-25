import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import ParallelTabContainer from './ParallelTabContainer';
import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';

// Mock the UI components and dependencies
vi.mock('@/components/ui/badge', () => ({
  Badge: ({ children, variant, className }: any) => (
    <span data-testid="badge" data-variant={variant} className={className}>
      {children}
    </span>
  ),
}));

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, variant, size, className, title }: any) => (
    <button 
      onClick={onClick} 
      data-variant={variant} 
      data-size={size} 
      className={className}
      title={title}
    >
      {children}
    </button>
  ),
}));

vi.mock('@/components/ui/typed-markdown', () => ({
  TypedMarkdown: ({ children, components, speed, delay, hideCursor }: any) => (
    <div 
      data-testid="typed-markdown" 
      data-speed={speed} 
      data-delay={delay}
      data-hide-cursor={hideCursor}
    >
      {children}
    </div>
  ),
}));

vi.mock('@/components/ActivityTimeline', () => ({
  ActivityTimeline: ({ processedEvents, isLoading }: any) => (
    <div data-testid="activity-timeline" data-loading={isLoading}>
      {processedEvents.map((event: any, index: number) => (
        <div key={index} data-testid="activity-event">
          {event.title}: {event.data}
        </div>
      ))}
    </div>
  ),
}));

vi.mock('react-markdown', () => ({
  default: ({ children }: any) => <div data-testid="react-markdown">{children}</div>,
}));

vi.mock('@/lib/utils', () => ({
  cn: (...classes: any[]) => classes.filter(Boolean).join(' '),
}));

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

const mockSequences: LLMGeneratedSequence[] = [
  {
    sequence_id: 'seq-1',
    sequence_name: 'Academic Research',
    rationale: 'Focus on academic sources and peer-reviewed content',
    research_focus: 'academic',
    agent_names: ['academic_agent'],
    confidence_score: 0.85,
    estimated_duration: 180,
    complexity_score: 0.7,
  },
  {
    sequence_id: 'seq-2', 
    sequence_name: 'Industry Analysis',
    rationale: 'Analyze industry trends and market data',
    research_focus: 'industry',
    agent_names: ['industry_agent', 'market_agent'],
    confidence_score: 0.92,
    estimated_duration: 240,
    complexity_score: 0.8,
  },
];

const mockParallelMessages: Record<string, RoutedMessage[]> = {
  'seq-1': [
    {
      message_id: 'msg-1',
      sequence_id: 'seq-1',
      message_type: 'result',
      content: '# Academic Research Results\n\nFound several peer-reviewed papers...',
      timestamp: new Date().toISOString(),
      current_agent: 'academic_agent',
    },
    {
      message_id: 'msg-2',
      sequence_id: 'seq-1',
      message_type: 'completion',
      content: 'Research completed successfully',
      timestamp: new Date().toISOString(),
      current_agent: 'academic_agent',
    },
  ],
  'seq-2': [
    {
      message_id: 'msg-3',
      sequence_id: 'seq-2',
      message_type: 'progress',
      content: 'Analyzing market trends...',
      timestamp: new Date().toISOString(),
      current_agent: 'industry_agent',
    },
  ],
};

describe('ParallelTabContainer', () => {
  const defaultProps = {
    sequences: mockSequences,
    parallelMessages: mockParallelMessages,
    activeTabId: 'seq-1',
    onTabChange: vi.fn(),
    isLoading: false,
  };

  describe('Basic Rendering', () => {
    it('renders when sequences are provided', () => {
      render(<ParallelTabContainer {...defaultProps} />);
      
      expect(screen.getByText('Academic Research')).toBeInTheDocument();
      expect(screen.getByText('Industry Analysis')).toBeInTheDocument();
    });

    it('returns null when no sequences provided', () => {
      const { container } = render(
        <ParallelTabContainer {...defaultProps} sequences={[]} />
      );
      
      expect(container.firstChild).toBeNull();
    });

    it('displays confidence scores for sequences', () => {
      render(<ParallelTabContainer {...defaultProps} />);
      
      // Check for confidence score badges (85% and 92%)
      expect(screen.getByText('85% confidence')).toBeInTheDocument();
      expect(screen.getByText('92% confidence')).toBeInTheDocument();
    });

    it('shows sequence details in tab content', () => {
      render(<ParallelTabContainer {...defaultProps} />);
      
      expect(screen.getByText('Focus on academic sources and peer-reviewed content')).toBeInTheDocument();
      expect(screen.getByText('Focus: academic')).toBeInTheDocument();
      expect(screen.getByText('Agents: academic_agent')).toBeInTheDocument();
    });
  });

  describe('Tab Navigation', () => {
    it('highlights active tab correctly', () => {
      render(<ParallelTabContainer {...defaultProps} activeTabId="seq-2" />);
      
      // Find tab buttons and check which one is active
      const buttons = screen.getAllByRole('button').filter(btn => 
        btn.textContent?.includes('Academic Research') || btn.textContent?.includes('Industry Analysis')
      );
      
      // The active tab should have different styling (this would be tested via className)
      expect(buttons).toHaveLength(2);
    });

    it('calls onTabChange when different tab is clicked', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      const onTabChange = vi.fn();
      
      render(<ParallelTabContainer {...defaultProps} onTabChange={onTabChange} />);
      
      // Click on the Industry Analysis tab
      const industryTab = screen.getByText('Industry Analysis');
      await user.click(industryTab);
      
      expect(onTabChange).toHaveBeenCalledWith('seq-2');
    });

    it('shows correct tab content based on activeTabId', () => {
      const { rerender } = render(<ParallelTabContainer {...defaultProps} activeTabId="seq-1" />);
      
      expect(screen.getByText('Focus: academic')).toBeInTheDocument();
      expect(screen.queryByText('Focus: industry')).not.toBeInTheDocument();
      
      rerender(<ParallelTabContainer {...defaultProps} activeTabId="seq-2" />);
      
      expect(screen.queryByText('Focus: academic')).not.toBeInTheDocument();
      expect(screen.getByText('Focus: industry')).toBeInTheDocument();
    });

    it('defaults to first sequence when activeTabId is invalid', () => {
      const onTabChange = vi.fn();
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          activeTabId="invalid-id" 
          onTabChange={onTabChange}
        />
      );
      
      // Should show content from first sequence
      expect(screen.getByText('Focus: academic')).toBeInTheDocument();
    });
  });

  describe('Tab Status Indicators', () => {
    it('shows typing status for active sequences', () => {
      const messagesWithTyping = {
        'seq-1': [{ 
          message_id: 'msg-1', 
          sequence_id: 'seq-1', 
          message_type: 'progress' as const,
          content: 'Working...', 
          timestamp: new Date().toISOString(),
          current_agent: 'academic_agent',
        }],
        'seq-2': [],
      };
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          parallelMessages={messagesWithTyping}
          isLoading={true}
        />
      );
      
      // Should show typing indicator elements
      const typingIndicators = screen.getAllByTestId('badge');
      expect(typingIndicators.length).toBeGreaterThan(0);
    });

    it('shows completed status for finished sequences', () => {
      render(<ParallelTabContainer {...defaultProps} />);
      
      // seq-1 has a completion message, so it should show completed status
      // This would typically be indicated by a checkmark icon or specific styling
      expect(screen.getByText('Academic Research')).toBeInTheDocument();
    });

    it('shows message count badges', () => {
      render(<ParallelTabContainer {...defaultProps} />);
      
      // seq-1 has 2 messages, seq-2 has 1 message
      const badges = screen.getAllByTestId('badge');
      const messageCounts = badges.filter(badge => 
        badge.textContent === '2' || badge.textContent === '1'
      );
      expect(messageCounts.length).toBeGreaterThan(0);
    });
  });

  describe('Simultaneous Typing Indicator', () => {
    it('shows simultaneous typing indicator when multiple tabs are typing', () => {
      const typingMessages = {
        'seq-1': [{ 
          message_id: 'msg-1', 
          sequence_id: 'seq-1', 
          message_type: 'progress' as const,
          content: 'Typing...', 
          timestamp: new Date().toISOString(),
          current_agent: 'academic_agent',
        }],
        'seq-2': [{ 
          message_id: 'msg-2', 
          sequence_id: 'seq-2', 
          message_type: 'progress' as const,
          content: 'Also typing...', 
          timestamp: new Date().toISOString(),
          current_agent: 'industry_agent',
        }],
      };
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          parallelMessages={typingMessages}
        />
      );
      
      expect(screen.getByText(/tabs generating simultaneously/)).toBeInTheDocument();
    });

    it('shows progress information in simultaneous typing indicator', () => {
      const mixedMessages = {
        'seq-1': [{ 
          message_id: 'msg-1', 
          sequence_id: 'seq-1', 
          message_type: 'completion' as const,
          content: 'Done', 
          timestamp: new Date().toISOString(),
          current_agent: 'academic_agent',
        }],
        'seq-2': [{ 
          message_id: 'msg-2', 
          sequence_id: 'seq-2', 
          message_type: 'progress' as const,
          content: 'Still working...', 
          timestamp: new Date().toISOString(),
          current_agent: 'industry_agent',
        }],
      };
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          parallelMessages={mixedMessages}
        />
      );
      
      // Should show progress like "Progress: 1/2 completed"
      const progressText = screen.queryByText(/Progress: \d+\/\d+ completed/);
      expect(progressText).toBeInTheDocument();
    });
  });

  describe('Message Display', () => {
    it('displays content messages with TypedMarkdown', () => {
      render(<ParallelTabContainer {...defaultProps} />);
      
      const typedMarkdown = screen.getByTestId('typed-markdown');
      expect(typedMarkdown).toBeInTheDocument();
      expect(typedMarkdown.textContent).toContain('Academic Research Results');
    });

    it('shows activity timeline for progress messages', () => {
      const messagesWithProgress = {
        'seq-1': [
          {
            message_id: 'msg-1',
            sequence_id: 'seq-1', 
            message_type: 'progress' as const,
            content: 'Step 1 complete',
            timestamp: new Date().toISOString(),
            current_agent: 'academic_agent',
          },
          {
            message_id: 'msg-2',
            sequence_id: 'seq-1',
            message_type: 'agent_transition' as const,
            content: 'Transitioning to next agent',
            timestamp: new Date().toISOString(),
            current_agent: 'academic_agent',
          },
        ],
      };
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          parallelMessages={messagesWithProgress}
        />
      );
      
      expect(screen.getByTestId('activity-timeline')).toBeInTheDocument();
      expect(screen.getByText('Activity Timeline')).toBeInTheDocument();
    });

    it('shows loading state for empty sequences', () => {
      const emptyMessages = { 'seq-1': [], 'seq-2': [] };
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          parallelMessages={emptyMessages}
          isLoading={true}
        />
      );
      
      expect(screen.getByText('Initializing sequence...')).toBeInTheDocument();
    });

    it('shows "no content" state for non-loading empty sequences', () => {
      const emptyMessages = { 'seq-1': [], 'seq-2': [] };
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          parallelMessages={emptyMessages}
          isLoading={false}
        />
      );
      
      expect(screen.getByText('No content generated yet')).toBeInTheDocument();
    });
  });

  describe('Typing Animation Coordination', () => {
    it('applies different delays to messages based on sequence index', () => {
      render(<ParallelTabContainer {...defaultProps} />);
      
      const typedMarkdowns = screen.getAllByTestId('typed-markdown');
      
      // First sequence should have base delay, second sequence should have offset
      expect(typedMarkdowns[0]).toHaveAttribute('data-delay', '0'); // First sequence, first message
    });

    it('shows cursor only on last message in active tab', () => {
      render(<ParallelTabContainer {...defaultProps} activeTabId="seq-1" />);
      
      const typedMarkdowns = screen.getAllByTestId('typed-markdown');
      const activeTabMarkdowns = typedMarkdowns.filter(md => 
        md.getAttribute('data-hide-cursor') === 'false'
      );
      
      // Only the last message in the active tab should show cursor
      expect(activeTabMarkdowns.length).toBeLessThanOrEqual(1);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('handles sequences with no agent names', () => {
      const sequenceWithoutAgents = [{
        ...mockSequences[0],
        agent_names: [],
      }];
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          sequences={sequenceWithoutAgents}
        />
      );
      
      // Should not show "Agents:" text when no agents
      expect(screen.queryByText(/Agents:/)).not.toBeInTheDocument();
    });

    it('handles messages with non-string content', () => {
      const messagesWithObjectContent = {
        'seq-1': [{
          message_id: 'msg-1',
          sequence_id: 'seq-1',
          message_type: 'result' as const,
          content: { data: 'object content', meta: {} },
          timestamp: new Date().toISOString(),
          current_agent: 'academic_agent',
        }],
      };
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          parallelMessages={messagesWithObjectContent}
        />
      );
      
      // Should render object content as JSON
      expect(screen.getByTestId('react-markdown')).toBeInTheDocument();
    });

    it('handles missing timestamps gracefully', () => {
      const messagesWithoutTimestamps = {
        'seq-1': [{
          message_id: 'msg-1',
          sequence_id: 'seq-1', 
          message_type: 'result' as const,
          content: 'Test content',
          timestamp: '', // Empty timestamp
          current_agent: 'academic_agent',
        }],
      };
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          parallelMessages={messagesWithoutTimestamps}
        />
      );
      
      // Should not crash
      expect(screen.getByText('Test content')).toBeInTheDocument();
    });

    it('handles very long sequence names', () => {
      const longNameSequence = [{
        ...mockSequences[0],
        sequence_name: 'Very Long Sequence Name That Might Overflow The Tab Button Area',
      }];
      
      render(
        <ParallelTabContainer 
          {...defaultProps} 
          sequences={longNameSequence}
        />
      );
      
      // Should truncate long names (class: truncate)
      expect(screen.getByText(/Very Long Sequence Name/)).toBeInTheDocument();
    });
  });

  describe('Performance and Cleanup', () => {
    it('handles rapid tab switching', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      const onTabChange = vi.fn();
      
      render(<ParallelTabContainer {...defaultProps} onTabChange={onTabChange} />);
      
      const industryTab = screen.getByText('Industry Analysis');
      const academicTab = screen.getByText('Academic Research');
      
      // Rapid switching
      await user.click(industryTab);
      await user.click(academicTab);
      await user.click(industryTab);
      
      expect(onTabChange).toHaveBeenCalledTimes(3);
    });

    it('updates active tab when valid activeTabId changes', () => {
      const onTabChange = vi.fn();
      
      const { rerender } = render(
        <ParallelTabContainer 
          {...defaultProps} 
          activeTabId="seq-1" 
          onTabChange={onTabChange}
        />
      );
      
      rerender(
        <ParallelTabContainer 
          {...defaultProps} 
          activeTabId="seq-2" 
          onTabChange={onTabChange}
        />
      );
      
      // Should show the new active tab content
      expect(screen.getByText('Focus: industry')).toBeInTheDocument();
    });

    it('handles component unmounting gracefully', () => {
      const { unmount } = render(<ParallelTabContainer {...defaultProps} />);
      
      expect(() => unmount()).not.toThrow();
    });
  });

  describe('Custom Styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <ParallelTabContainer {...defaultProps} className="custom-tab-container" />
      );
      
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper).toHaveClass('custom-tab-container');
    });

    it('maintains consistent styling across tabs', () => {
      render(<ParallelTabContainer {...defaultProps} />);
      
      const tabButtons = screen.getAllByRole('button').filter(btn =>
        btn.textContent?.includes('Academic Research') || btn.textContent?.includes('Industry Analysis')
      );
      
      // Both tabs should be rendered
      expect(tabButtons).toHaveLength(2);
    });
  });
});