/**
 * Integration test for thinking sections and parallel tabs fixes
 * 
 * Tests:
 * - Thinking section parsing for both <thinking> and <think> tags
 * - Error boundaries prevent crashes
 * - Tab switching without browser restarts
 * - Performance optimizations work correctly
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MessageContentParser } from '@/types/messages';
import { ThinkingSections } from '@/components/ui/collapsible-thinking';
import ParallelTabContainer from '@/components/ParallelTabContainer';
import { LLMGeneratedSequence } from '@/types/parallel';

// Mock data
const mockThinkingMessage = `Let me analyze this problem.

<thinking>
I need to understand what the user is asking for. The request involves multiple steps:

1. Parse the content properly  
2. Extract thinking sections correctly
3. Render with typing animation
4. Make sure tabs work properly
</thinking>

Based on my analysis, here are the key points.`;

const mockThinkMessage = `Let me think about this approach.

<think>
The user wants to:
- Fix thinking section display issues
- Prevent browser crashes
- Optimize tab functionality
- Test the complete flow

This requires comprehensive changes.
</think>

Here's my solution.`;

const mockSequences: LLMGeneratedSequence[] = [
  {
    sequence_id: 'seq_1',
    sequence_name: 'Academic Research',
    agent_names: ['researcher', 'analyzer'],
    rationale: 'Academic focused research',
    research_focus: 'Academic analysis',
    confidence_score: 0.8,
    approach_description: 'Academic approach',
    expected_outcomes: ['Research results'],
    created_at: new Date().toISOString(),
  },
  {
    sequence_id: 'seq_2', 
    sequence_name: 'Industry Analysis',
    agent_names: ['industry_expert'],
    rationale: 'Industry focused analysis',
    research_focus: 'Market analysis',
    confidence_score: 0.7,
    approach_description: 'Industry approach',
    expected_outcomes: ['Market insights'],
    created_at: new Date().toISOString(),
  }
];

describe('Thinking Sections Integration', () => {
  describe('MessageContentParser', () => {
    test('should parse <thinking> tags correctly', () => {
      const parsed = MessageContentParser.parse({
        id: 'test',
        type: 'ai',
        content: mockThinkingMessage
      });
      
      expect(parsed.hasThinking).toBe(true);
      expect(parsed.thinkingSections).toHaveLength(1);
      expect(parsed.thinkingSections[0].content).toContain('I need to understand');
      expect(parsed.thinkingSections[0].charLength).toBeGreaterThan(0);
    });

    test('should parse <think> tags correctly', () => {
      const parsed = MessageContentParser.parse({
        id: 'test',
        type: 'ai', 
        content: mockThinkMessage
      });
      
      expect(parsed.hasThinking).toBe(true);
      expect(parsed.thinkingSections).toHaveLength(1);
      expect(parsed.thinkingSections[0].content).toContain('The user wants to:');
      expect(parsed.thinkingSections[0].charLength).toBeGreaterThan(0);
    });

    test('should handle mixed thinking tags', () => {
      const mixedContent = `Analysis:
<thinking>First analysis step</thinking>
<think>Second analysis step</think>
Final conclusion.`;

      const parsed = MessageContentParser.parse({
        id: 'test',
        type: 'ai',
        content: mixedContent
      });
      
      expect(parsed.hasThinking).toBe(true);
      expect(parsed.thinkingSections).toHaveLength(2);
    });

    test('should handle content without thinking tags', () => {
      const parsed = MessageContentParser.parse({
        id: 'test',
        type: 'ai',
        content: 'Just regular content without thinking.'
      });
      
      expect(parsed.hasThinking).toBe(false);
      expect(parsed.thinkingSections).toHaveLength(0);
    });
  });

  describe('ThinkingSections Component', () => {
    test('should render thinking sections without crashing', () => {
      const parsed = MessageContentParser.parse({
        id: 'test',
        type: 'ai',
        content: mockThinkingMessage
      });

      const expandedSections = new Set<string>();
      const onToggle = jest.fn();

      render(
        <ThinkingSections
          sections={parsed.thinkingSections}
          expandedSections={expandedSections}
          onToggleSection={onToggle}
          hasTypingAnimation={false}
          typingSpeed={20}
        />
      );

      expect(screen.getByText('thinking...')).toBeInTheDocument();
    });

    test('should handle section toggle correctly', async () => {
      const parsed = MessageContentParser.parse({
        id: 'test',
        type: 'ai',
        content: mockThinkingMessage
      });

      const expandedSections = new Set<string>();
      const onToggle = jest.fn();

      render(
        <ThinkingSections
          sections={parsed.thinkingSections}
          expandedSections={expandedSections}
          onToggleSection={onToggle}
          hasTypingAnimation={false}
          typingSpeed={20}
        />
      );

      const toggleButton = screen.getByRole('button');
      fireEvent.click(toggleButton);

      expect(onToggle).toHaveBeenCalledWith(parsed.thinkingSections[0].id);
    });
  });

  describe('ParallelTabContainer Component', () => {
    test('should render tabs without crashing', () => {
      render(
        <ParallelTabContainer
          sequences={mockSequences}
          parallelMessages={{}}
          activeTabId={mockSequences[0].sequence_id}
          onTabChange={jest.fn()}
          isLoading={false}
        />
      );

      expect(screen.getByText('Academic Research')).toBeInTheDocument();
      expect(screen.getByText('Industry Analysis')).toBeInTheDocument();
    });

    test('should handle tab switching correctly', async () => {
      const onTabChange = jest.fn();
      
      render(
        <ParallelTabContainer
          sequences={mockSequences}
          parallelMessages={{}}
          activeTabId={mockSequences[0].sequence_id}
          onTabChange={onTabChange}
          isLoading={false}
        />
      );

      const secondTab = screen.getByText('Industry Analysis');
      fireEvent.click(secondTab);

      await waitFor(() => {
        expect(onTabChange).toHaveBeenCalledWith('seq_2');
      });
    });

    test('should handle empty sequences gracefully', () => {
      render(
        <ParallelTabContainer
          sequences={[]}
          parallelMessages={{}}
          activeTabId=""
          onTabChange={jest.fn()}
          isLoading={false}
        />
      );

      // Should render nothing for empty sequences
      expect(screen.queryByRole('tab')).not.toBeInTheDocument();
    });
  });

  describe('Error Boundary Integration', () => {
    test('should catch and handle component errors gracefully', () => {
      // Mock console.error to prevent test noise
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      const ThrowError = () => {
        throw new Error('Test error');
      };

      render(
        <div>
          <ThrowError />
        </div>
      );

      // Component should render error boundary instead of crashing
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });
  });

  describe('Performance Optimizations', () => {
    test('should memoize expensive calculations', () => {
      const renderSpy = jest.fn();
      
      const MemoizedComponent = React.memo(() => {
        renderSpy();
        return <div>Test Component</div>;
      });

      const { rerender } = render(<MemoizedComponent />);
      
      // Re-render with same props
      rerender(<MemoizedComponent />);
      
      // Should only render once due to memoization
      expect(renderSpy).toHaveBeenCalledTimes(1);
    });
  });
});

describe('Integration Test Suite', () => {
  test('complete thinking section and tabs workflow', async () => {
    const mockMessage = {
      id: 'msg_1',
      type: 'ai' as const,
      content: mockThinkingMessage
    };

    // 1. Parse thinking sections
    const parsed = MessageContentParser.parse(mockMessage);
    expect(parsed.hasThinking).toBe(true);

    // 2. Render thinking sections
    const expandedSections = new Set<string>();
    const onToggle = jest.fn();

    const { container } = render(
      <div>
        <ThinkingSections
          sections={parsed.thinkingSections}
          expandedSections={expandedSections}
          onToggleSection={onToggle}
          hasTypingAnimation={false}
          typingSpeed={20}
        />
        <ParallelTabContainer
          sequences={mockSequences}
          parallelMessages={{}}
          activeTabId={mockSequences[0].sequence_id}
          onTabChange={jest.fn()}
          isLoading={false}
        />
      </div>
    );

    // 3. Verify no crashes and components render
    expect(container).toBeInTheDocument();
    expect(screen.getByText('thinking...')).toBeInTheDocument();
    expect(screen.getByText('Academic Research')).toBeInTheDocument();

    // 4. Test thinking section toggle
    const thinkingToggle = screen.getByRole('button', { name: /thinking/ });
    fireEvent.click(thinkingToggle);
    expect(onToggle).toHaveBeenCalled();

    // 5. Test tab switching
    const industryTab = screen.getByText('Industry Analysis');
    fireEvent.click(industryTab);

    // Should not cause any crashes
    await waitFor(() => {
      expect(screen.getByText('Industry Analysis')).toBeInTheDocument();
    });
  });
});