import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MessageContentParser } from '@/types/messages';
import { CollapsibleThinking } from '@/components/ui/collapsible-thinking';
import { TypedMarkdown } from '@/components/ui/typed-markdown';
import { TypedText } from '@/components/ui/typed-text';
import ParallelTabContainer from '@/components/ParallelTabContainer';
import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';

// Mock dependencies for integration tests
vi.mock('@/components/ui/badge', () => ({
  Badge: ({ children, variant, className }: any) => (
    <span data-testid="badge" data-variant={variant} className={className}>
      {children}
    </span>
  ),
}));

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, variant, size, className, title }: any) => (
    <button onClick={onClick} className={className} title={title}>
      {children}
    </button>
  ),
}));

vi.mock('@/components/ui/collapsible', () => ({
  Collapsible: ({ children, open, onOpenChange }: any) => (
    <div data-testid="collapsible" data-open={open} onClick={() => onOpenChange?.(!open)}>
      {children}
    </div>
  ),
  CollapsibleContent: ({ children }: any) => <div data-testid="collapsible-content">{children}</div>,
  CollapsibleTrigger: ({ children, asChild }: any) => 
    asChild ? children : <div data-testid="collapsible-trigger">{children}</div>,
}));

vi.mock('@/components/ActivityTimeline', () => ({
  ActivityTimeline: ({ processedEvents, isLoading }: any) => (
    <div data-testid="activity-timeline" data-loading={isLoading}>
      Timeline with {processedEvents.length} events
    </div>
  ),
}));

vi.mock('react-markdown', () => ({
  default: ({ children, components }: any) => (
    <div data-testid="react-markdown" data-components={!!components}>
      {children}
    </div>
  ),
}));

vi.mock('@/lib/utils', () => ({
  cn: (...classes: any[]) => classes.filter(Boolean).join(' '),
}));

beforeEach(() => {
  vi.useFakeTimers();
  // Reset parser counter for consistent tests
  (MessageContentParser as any).thinkingCounter = 0;
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('Integration Tests: Complete User Flows', () => {
  describe('Flow 1: Thinking Section → Supervisor Announcement → Parallel Tabs', () => {
    it('handles complete flow from thinking to parallel execution', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      
      // Step 1: Parse message with thinking section
      const messageWithThinking = {
        type: 'ai',
        content: `I need to research this topic comprehensively.

<thinking>
This is a complex research question that would benefit from multiple perspectives:
1. Academic research from scholarly sources
2. Industry analysis from market data
3. Technical trends from recent developments

I should generate parallel research sequences to cover all these angles simultaneously.
</thinking>

Based on my analysis, I'll create multiple research sequences to explore this topic from different angles.`
      };

      const parsedContent = MessageContentParser.parse(messageWithThinking);
      
      // Verify thinking section was parsed correctly
      expect(parsedContent.hasThinking).toBe(true);
      expect(parsedContent.thinkingSections).toHaveLength(1);
      expect(parsedContent.thinkingSections[0].content).toContain('This is a complex research question');
      
      // Step 2: Render thinking section with typing animation
      const expandedSections = new Set<string>();
      const onToggleSection = vi.fn();
      
      render(
        <CollapsibleThinking
          section={parsedContent.thinkingSections[0]}
          isExpanded={false}
          onToggle={onToggleSection}
          hasTypingAnimation={true}
          typingSpeed={10}
        />
      );

      // Verify thinking section renders
      expect(screen.getByText('thinking...')).toBeInTheDocument();
      expect(screen.getByTestId('collapsible')).toHaveAttribute('data-open', 'false');

      // Step 3: Expand thinking section
      await user.click(screen.getByTestId('collapsible'));
      expect(onToggleSection).toHaveBeenCalled();
      
      // Step 4: Simulate supervisor generating sequences
      const supervisorSequences: LLMGeneratedSequence[] = [
        {
          sequence_id: 'academic-seq',
          sequence_name: 'Academic Research',
          rationale: 'Focus on peer-reviewed sources and scholarly articles',
          research_focus: 'academic',
          agent_names: ['academic_agent'],
          confidence_score: 0.88,
          estimated_duration: 180,
          complexity_score: 0.7,
        },
        {
          sequence_id: 'industry-seq',
          sequence_name: 'Industry Analysis',
          rationale: 'Analyze market trends and industry reports',
          research_focus: 'industry',
          agent_names: ['industry_agent', 'market_agent'],
          confidence_score: 0.91,
          estimated_duration: 240,
          complexity_score: 0.8,
        },
        {
          sequence_id: 'technical-seq',
          sequence_name: 'Technical Trends',
          rationale: 'Focus on recent technical developments and innovations',
          research_focus: 'technical',
          agent_names: ['technical_agent'],
          confidence_score: 0.84,
          estimated_duration: 200,
          complexity_score: 0.75,
        },
      ];

      // Step 5: Initialize parallel execution with empty messages
      const initialMessages: Record<string, RoutedMessage[]> = {
        'academic-seq': [],
        'industry-seq': [],
        'technical-seq': [],
      };

      const onTabChange = vi.fn();

      render(
        <ParallelTabContainer
          sequences={supervisorSequences}
          parallelMessages={initialMessages}
          activeTabId="academic-seq"
          onTabChange={onTabChange}
          isLoading={true}
        />
      );

      // Verify tabs are rendered
      expect(screen.getByText('Academic Research')).toBeInTheDocument();
      expect(screen.getByText('Industry Analysis')).toBeInTheDocument();
      expect(screen.getByText('Technical Trends')).toBeInTheDocument();
      
      // Verify loading state
      expect(screen.getByText('Initializing sequence...')).toBeInTheDocument();

      // Step 6: Simulate messages arriving for parallel sequences
      const messagesWithContent: Record<string, RoutedMessage[]> = {
        'academic-seq': [{
          message_id: 'academic-msg-1',
          sequence_id: 'academic-seq',
          message_type: 'result',
          content: '# Academic Research Results\n\nFound 15 peer-reviewed papers on this topic...',
          timestamp: new Date().toISOString(),
          current_agent: 'academic_agent',
        }],
        'industry-seq': [{
          message_id: 'industry-msg-1',
          sequence_id: 'industry-seq',
          message_type: 'progress',
          content: 'Analyzing market data from last 5 years...',
          timestamp: new Date().toISOString(),
          current_agent: 'industry_agent',
        }],
        'technical-seq': [{
          message_id: 'technical-msg-1',
          sequence_id: 'technical-seq',
          message_type: 'result',
          content: '# Technical Trends Analysis\n\n**Key Developments:**\n- AI/ML advances\n- Cloud computing evolution',
          timestamp: new Date().toISOString(),
          current_agent: 'technical_agent',
        }],
      };

      // Re-render with content
      const { rerender } = render(
        <ParallelTabContainer
          sequences={supervisorSequences}
          parallelMessages={messagesWithContent}
          activeTabId="academic-seq"
          onTabChange={onTabChange}
          isLoading={false}
        />
      );

      // Verify content appears with typing animation
      expect(screen.getByText(/Academic Research Results/)).toBeInTheDocument();
      expect(screen.getByText('88% confidence')).toBeInTheDocument();

      // Step 7: Switch between tabs
      await user.click(screen.getByText('Industry Analysis'));
      expect(onTabChange).toHaveBeenCalledWith('industry-seq');

      // Step 8: Verify simultaneous typing indicator appears
      const updatedMessagesWithTyping: Record<string, RoutedMessage[]> = {
        'academic-seq': [...messagesWithContent['academic-seq']],
        'industry-seq': [...messagesWithContent['industry-seq']],
        'technical-seq': [...messagesWithContent['technical-seq']],
      };

      // Add progress messages to simulate multiple tabs typing
      updatedMessagesWithTyping['academic-seq'].push({
        message_id: 'academic-msg-2',
        sequence_id: 'academic-seq', 
        message_type: 'progress',
        content: 'Analyzing citations...',
        timestamp: new Date().toISOString(),
        current_agent: 'academic_agent',
      });

      rerender(
        <ParallelTabContainer
          sequences={supervisorSequences}
          parallelMessages={updatedMessagesWithTyping}
          activeTabId="industry-seq"
          onTabChange={onTabChange}
          isLoading={false}
        />
      );

      // Should show simultaneous typing indicator
      expect(screen.getByText(/tabs generating simultaneously/)).toBeInTheDocument();
    });
  });

  describe('Flow 2: Message Parsing → Thinking Expansion → Content Display', () => {
    it('handles message parsing with multiple thinking sections and content', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

      // Complex message with multiple thinking sections
      const complexMessage = {
        type: 'ai',
        content: `Let me analyze your research question step by step.

<thinking>
First, I need to understand the scope of this question. The user is asking about market trends in renewable energy, which is a broad topic that spans:
- Solar energy developments
- Wind power innovations  
- Energy storage solutions
- Government policies and regulations

I should break this down into focused research areas.
</thinking>

Based on my initial analysis, I can see several key areas to explore.

<thinking>
Now I need to prioritize these areas. Solar and wind are the most mature technologies with substantial data available. Energy storage is rapidly evolving. Policy analysis will require recent regulatory documents.

I should create separate research sequences for:
1. Solar/wind technology trends
2. Energy storage innovations
3. Policy and regulatory landscape
</thinking>

I'll create a comprehensive research plan targeting these three critical areas.`
      };

      const parsedContent = MessageContentParser.parse(complexMessage);

      // Verify parsing results
      expect(parsedContent.hasThinking).toBe(true);
      expect(parsedContent.thinkingSections).toHaveLength(2);
      expect(parsedContent.renderSections).toHaveLength(4); // pre + thinking1 + middle + thinking2 + post

      // Render the complete message with thinking sections
      const expandedSections = new Set<string>();
      const onToggleSection = vi.fn();

      const { rerender } = render(
        <div data-testid="complete-message">
          {/* Pre-thinking content */}
          {parsedContent.preThinking && (
            <TypedText 
              text={parsedContent.preThinking} 
              speed={20}
              hideCursor={true}
            />
          )}
          
          {/* Thinking sections */}
          {parsedContent.thinkingSections.map((section, index) => (
            <CollapsibleThinking
              key={section.id}
              section={section}
              isExpanded={expandedSections.has(section.id)}
              onToggle={() => onToggleSection(section.id)}
              hasTypingAnimation={index === 0} // Only first gets typing animation
            />
          ))}
          
          {/* Post-thinking content */}
          {parsedContent.postThinking && (
            <TypedMarkdown speed={25} hideCursor={true}>
              {parsedContent.postThinking}
            </TypedMarkdown>
          )}
        </div>
      );

      // Verify initial render
      expect(screen.getByTestId('complete-message')).toBeInTheDocument();
      expect(screen.getByText(/Let me analyze your research question/)).toBeInTheDocument();
      expect(screen.getAllByText('thinking...')).toHaveLength(2);

      // Fast-forward typing animations
      vi.advanceTimersByTime(1000);
      await waitFor(() => {
        expect(screen.getByText(/Based on my initial analysis/)).toBeInTheDocument();
      });

      // Expand first thinking section
      const thinkingSections = screen.getAllByTestId('collapsible');
      await user.click(thinkingSections[0]);
      
      expect(onToggleSection).toHaveBeenCalledWith(parsedContent.thinkingSections[0].id);

      // Simulate expansion
      expandedSections.add(parsedContent.thinkingSections[0].id);
      
      rerender(
        <div data-testid="complete-message">
          {parsedContent.preThinking && (
            <TypedText 
              text={parsedContent.preThinking} 
              speed={20}
              hideCursor={true}
            />
          )}
          
          {parsedContent.thinkingSections.map((section, index) => (
            <CollapsibleThinking
              key={section.id}
              section={section}
              isExpanded={expandedSections.has(section.id)}
              onToggle={() => onToggleSection(section.id)}
              hasTypingAnimation={index === 0}
            />
          ))}
          
          {parsedContent.postThinking && (
            <TypedMarkdown speed={25} hideCursor={true}>
              {parsedContent.postThinking}
            </TypedMarkdown>
          )}
        </div>
      );

      // Verify expanded thinking section shows content
      const expandedCollapsible = screen.getAllByTestId('collapsible')[0];
      expect(expandedCollapsible).toHaveAttribute('data-open', 'true');

      // Expand second thinking section
      await user.click(thinkingSections[1]);
      expect(onToggleSection).toHaveBeenCalledWith(parsedContent.thinkingSections[1].id);
    });
  });

  describe('Flow 3: Error Handling and Recovery', () => {
    it('handles malformed content and recovers gracefully', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

      // Message with malformed thinking tags
      const malformedMessage = {
        type: 'ai',
        content: `Let me think about this <thinking>This thinking section is never closed...

And here's some content after the unclosed thinking section.`
      };

      const parsedContent = MessageContentParser.parse(malformedMessage);

      // Should handle unclosed thinking tag
      expect(parsedContent.hasThinking).toBe(true);
      expect(parsedContent.thinkingSections).toHaveLength(1);
      expect(parsedContent.thinkingSections[0].id).toMatch(/unclosed/);

      // Render with error boundary handling
      const onToggleSection = vi.fn();
      const expandedSections = new Set<string>();

      render(
        <div data-testid="error-handling-test">
          <CollapsibleThinking
            section={parsedContent.thinkingSections[0]}
            isExpanded={false}
            onToggle={onToggleSection}
            hasTypingAnimation={true}
          />
        </div>
      );

      // Should render without crashing
      expect(screen.getByTestId('error-handling-test')).toBeInTheDocument();
      expect(screen.getByText('thinking...')).toBeInTheDocument();

      // Should be able to expand malformed section
      await user.click(screen.getByTestId('collapsible'));
      expect(onToggleSection).toHaveBeenCalled();
    });

    it('handles empty sequences and messages gracefully', async () => {
      // Test with empty sequences
      const { rerender } = render(
        <ParallelTabContainer
          sequences={[]}
          parallelMessages={{}}
          activeTabId=""
          onTabChange={vi.fn()}
          isLoading={false}
        />
      );

      // Should return null for empty sequences
      expect(screen.queryByText('Academic Research')).not.toBeInTheDocument();

      // Test with sequences but no messages
      const emptySequences: LLMGeneratedSequence[] = [{
        sequence_id: 'test-seq',
        sequence_name: 'Test Sequence',
        rationale: 'Test rationale',
        research_focus: 'test',
        agent_names: ['test_agent'],
        confidence_score: 0.8,
        estimated_duration: 100,
        complexity_score: 0.5,
      }];

      rerender(
        <ParallelTabContainer
          sequences={emptySequences}
          parallelMessages={{ 'test-seq': [] }}
          activeTabId="test-seq"
          onTabChange={vi.fn()}
          isLoading={false}
        />
      );

      // Should show no content message
      expect(screen.getByText('Test Sequence')).toBeInTheDocument();
      expect(screen.getByText('No content generated yet')).toBeInTheDocument();
    });
  });

  describe('Flow 4: Performance Under Load', () => {
    it('handles large number of thinking sections efficiently', async () => {
      // Create message with many thinking sections
      const manyThinkingSections = Array.from({ length: 10 }, (_, i) => 
        `<thinking>Thinking section ${i + 1} with some content to analyze</thinking>`
      ).join(' Some text between sections. ');

      const messageWithManyThinking = {
        type: 'ai',
        content: `Start of message. ${manyThinkingSections} End of message.`
      };

      const parsedContent = MessageContentParser.parse(messageWithManyThinking);

      expect(parsedContent.thinkingSections).toHaveLength(10);

      // Render all thinking sections
      const expandedSections = new Set<string>();
      const onToggleSection = vi.fn();

      render(
        <div data-testid="many-thinking-sections">
          {parsedContent.thinkingSections.map((section, index) => (
            <CollapsibleThinking
              key={section.id}
              section={section}
              isExpanded={expandedSections.has(section.id)}
              onToggle={() => onToggleSection(section.id)}
              hasTypingAnimation={false} // Disable for performance
            />
          ))}
        </div>
      );

      // Should render all sections
      const thinkingSections = screen.getAllByText('thinking...');
      expect(thinkingSections).toHaveLength(10);

      // Performance: should render quickly without blocking
      expect(screen.getByTestId('many-thinking-sections')).toBeInTheDocument();
    });

    it('handles rapid tab switching without memory leaks', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

      const sequences: LLMGeneratedSequence[] = Array.from({ length: 5 }, (_, i) => ({
        sequence_id: `seq-${i}`,
        sequence_name: `Sequence ${i}`,
        rationale: `Test sequence ${i}`,
        research_focus: 'test',
        agent_names: ['test_agent'],
        confidence_score: 0.8,
        estimated_duration: 100,
        complexity_score: 0.5,
      }));

      const messages: Record<string, RoutedMessage[]> = {};
      sequences.forEach((seq, i) => {
        messages[seq.sequence_id] = [{
          message_id: `msg-${i}`,
          sequence_id: seq.sequence_id,
          message_type: 'result',
          content: `Content for sequence ${i}`,
          timestamp: new Date().toISOString(),
          current_agent: 'test_agent',
        }];
      });

      const onTabChange = vi.fn();

      render(
        <ParallelTabContainer
          sequences={sequences}
          parallelMessages={messages}
          activeTabId="seq-0"
          onTabChange={onTabChange}
          isLoading={false}
        />
      );

      // Rapidly switch between all tabs
      for (let i = 1; i < sequences.length; i++) {
        await user.click(screen.getByText(`Sequence ${i}`));
        expect(onTabChange).toHaveBeenCalledWith(`seq-${i}`);
      }

      // Should handle rapid switching without issues
      expect(onTabChange).toHaveBeenCalledTimes(4);
    });
  });

  describe('Flow 5: Accessibility and Keyboard Navigation', () => {
    it('supports keyboard navigation through thinking sections', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

      const messageWithThinking = {
        type: 'ai',
        content: 'Content <thinking>First section</thinking> more <thinking>Second section</thinking> end'
      };

      const parsedContent = MessageContentParser.parse(messageWithThinking);
      const expandedSections = new Set<string>();
      const onToggleSection = vi.fn();

      const { container } = render(
        <div>
          {parsedContent.thinkingSections.map(section => (
            <CollapsibleThinking
              key={section.id}
              section={section}
              isExpanded={expandedSections.has(section.id)}
              onToggle={() => onToggleSection(section.id)}
            />
          ))}
        </div>
      );

      // Find focusable elements (buttons in thinking sections)
      const buttons = container.querySelectorAll('button');
      expect(buttons).toHaveLength(2);

      // Tab to first button and activate with Enter
      buttons[0].focus();
      await user.keyboard('{Enter}');
      expect(onToggleSection).toHaveBeenCalledWith(parsedContent.thinkingSections[0].id);

      // Tab to second button and activate with Space
      buttons[1].focus();
      await user.keyboard(' ');
      expect(onToggleSection).toHaveBeenCalledWith(parsedContent.thinkingSections[1].id);
    });

    it('provides proper ARIA labels and roles', () => {
      const mockSequence: LLMGeneratedSequence = {
        sequence_id: 'test-seq',
        sequence_name: 'Test Sequence', 
        rationale: 'Test rationale',
        research_focus: 'test',
        agent_names: ['test_agent'],
        confidence_score: 0.8,
        estimated_duration: 100,
        complexity_score: 0.5,
      };

      render(
        <ParallelTabContainer
          sequences={[mockSequence]}
          parallelMessages={{ 'test-seq': [] }}
          activeTabId="test-seq"
          onTabChange={vi.fn()}
          isLoading={false}
        />
      );

      // Check for proper button roles
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);

      // Each tab button should be focusable
      buttons.forEach(button => {
        expect(button.tagName).toBe('BUTTON');
      });
    });
  });
});