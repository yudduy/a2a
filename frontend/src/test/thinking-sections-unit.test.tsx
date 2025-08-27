/**
 * Unit Tests for Thinking Sections Feature
 * 
 * Tests the MessageContentParser and ThinkingSections components
 * to ensure Claude-style thinking boxes work correctly.
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { vi, describe, test, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { MessageContentParser, ThinkingSection } from '@/types/messages';
import { ThinkingSections, CollapsibleThinking } from '@/components/ui/collapsible-thinking';

describe('MessageContentParser', () => {
  describe('thinking section extraction', () => {
    test('extracts closed thinking tags', () => {
      const content = `Here's my analysis:

<thinking>
I need to analyze this request carefully.

Key considerations:
1. Research scope
2. Methodology
3. Expected outcomes

This looks like a complex research task.
</thinking>

Based on this analysis, I'll proceed with the research.`;

      const parsed = MessageContentParser.parse({
        id: 'test-msg',
        type: 'ai',
        content
      });

      expect(parsed.hasThinking).toBe(true);
      expect(parsed.thinkingSections).toHaveLength(1);
      expect(parsed.thinkingSections[0].content).toContain('I need to analyze this request');
      expect(parsed.thinkingSections[0].charLength).toBeGreaterThan(0);
      expect(parsed.preThinking).toBe("Here's my analysis:");
      expect(parsed.postThinking).toBe("Based on this analysis, I'll proceed with the research.");
    });

    test('extracts closed think tags', () => {
      const content = `Starting research:

<think>
The user wants comprehensive analysis. I should:
- Gather multiple sources
- Analyze from different angles  
- Synthesize findings
</think>

I'll begin the research process.`;

      const parsed = MessageContentParser.parse({
        id: 'test-msg',
        type: 'ai',
        content
      });

      expect(parsed.hasThinking).toBe(true);
      expect(parsed.thinkingSections).toHaveLength(1);
      expect(parsed.thinkingSections[0].content).toContain('The user wants comprehensive analysis');
    });

    test('handles unclosed thinking tags', () => {
      const content = `I'll help with this research.

<thinking>
This is an interesting request that requires careful consideration.

I should approach this systematically by:
1. Understanding the core requirements
2. Identifying key research areas
3. Planning the methodology`;

      const parsed = MessageContentParser.parse({
        id: 'test-msg',
        type: 'ai',
        content
      });

      expect(parsed.hasThinking).toBe(true);
      expect(parsed.thinkingSections).toHaveLength(1);
      expect(parsed.thinkingSections[0].content).toContain('This is an interesting request');
      expect(parsed.preThinking).toBe("I'll help with this research.");
      expect(parsed.postThinking).toBeUndefined();
    });

    test('handles multiple thinking sections', () => {
      const content = `Initial analysis:

<thinking>
First, I need to understand the scope.
</thinking>

Intermediate step:

<think>
Now I'll consider the methodology.
</think>

Final conclusion based on analysis.`;

      const parsed = MessageContentParser.parse({
        id: 'test-msg',
        type: 'ai',
        content
      });

      expect(parsed.hasThinking).toBe(true);
      expect(parsed.thinkingSections).toHaveLength(2);
      expect(parsed.thinkingSections[0].content).toContain('understand the scope');
      expect(parsed.thinkingSections[1].content).toContain('consider the methodology');
    });

    test('handles content without thinking tags', () => {
      const content = 'This is a regular message without any thinking sections.';

      const parsed = MessageContentParser.parse({
        id: 'test-msg',
        type: 'ai',
        content
      });

      expect(parsed.hasThinking).toBe(false);
      expect(parsed.thinkingSections).toHaveLength(0);
      expect(parsed.preThinking).toBe(content);
      expect(parsed.postThinking).toBeUndefined();
    });

    test('removes thinking tags from clean content', () => {
      const content = `Analysis:

<thinking>
Internal reasoning here.
</thinking>

This is the clean output.

<think>
More internal thoughts.
</think>

Final conclusion.`;

      const cleanContent = MessageContentParser.getCleanContent(content);
      
      expect(cleanContent).not.toContain('<thinking>');
      expect(cleanContent).not.toContain('</thinking>');
      expect(cleanContent).not.toContain('<think>');
      expect(cleanContent).not.toContain('</think>');
      expect(cleanContent).toContain('Analysis:');
      expect(cleanContent).toContain('This is the clean output.');
      expect(cleanContent).toContain('Final conclusion.');
    });

    test('counts thinking sections correctly', () => {
      const contentWithTwo = `<thinking>First</thinking> Some text <think>Second</think>`;
      const contentWithNone = `Just regular content here.`;
      
      expect(MessageContentParser.countThinkingSections(contentWithTwo)).toBe(2);
      expect(MessageContentParser.countThinkingSections(contentWithNone)).toBe(0);
    });

    test('detects thinking content correctly', () => {
      expect(MessageContentParser.hasThinkingContent('<thinking>content</thinking>')).toBe(true);
      expect(MessageContentParser.hasThinkingContent('<think>content</think>')).toBe(true);
      expect(MessageContentParser.hasThinkingContent('<thinking>unclosed')).toBe(true);
      expect(MessageContentParser.hasThinkingContent('no thinking here')).toBe(false);
    });
  });

  describe('render sections creation', () => {
    test('creates proper render sections', () => {
      const content = `Pre-thinking content.

<thinking>
Internal analysis here.
</thinking>

Post-thinking content.`;

      const parsed = MessageContentParser.parse({
        id: 'test-msg',
        type: 'ai',
        content
      });

      expect(parsed.renderSections).toHaveLength(3);
      expect(parsed.renderSections[0].type).toBe('text');
      expect(parsed.renderSections[1].type).toBe('thinking');
      expect(parsed.renderSections[2].type).toBe('text');
    });
  });
});

describe('CollapsibleThinking Component', () => {
  const mockSection: ThinkingSection = {
    id: 'thinking-test-1',
    content: `I need to analyze this request carefully.

The user is asking for research on AI systems. This requires:
1. Understanding current state of AI
2. Identifying key challenges
3. Proposing solutions

I should approach this systematically.`,
    startIndex: 0,
    endIndex: 100,
    isCollapsed: true,
    charLength: 150
  };

  let user: ReturnType<typeof userEvent.setup>;

  beforeEach(() => {
    user = userEvent.setup();
  });

  test('renders thinking section with proper styling', () => {
    const mockToggle = vi.fn();
    
    render(
      <CollapsibleThinking
        section={mockSection}
        isExpanded={false}
        onToggle={mockToggle}
      />
    );

    // Check for Claude-style blue theming
    expect(screen.getByText('thinking...')).toBeInTheDocument();
    expect(screen.getByText('150 chars')).toBeInTheDocument();
    
    // Check for proper styling classes
    const container = screen.getByText('thinking...').closest('.border-blue-500\\/30');
    expect(container).toBeInTheDocument();
    
    const brainIcon = container?.querySelector('svg');
    expect(brainIcon).toBeInTheDocument();
  });

  test('toggles expansion on click', async () => {
    const mockToggle = vi.fn();
    
    render(
      <CollapsibleThinking
        section={mockSection}
        isExpanded={false}
        onToggle={mockToggle}
      />
    );

    const toggleButton = screen.getByRole('button');
    await user.click(toggleButton);
    
    expect(mockToggle).toHaveBeenCalledTimes(1);
  });

  test('shows content when expanded', () => {
    render(
      <CollapsibleThinking
        section={mockSection}
        isExpanded={true}
        onToggle={vi.fn()}
      />
    );

    expect(screen.getByText(/I need to analyze this request/)).toBeInTheDocument();
    expect(screen.getByText(/Understanding current state of AI/)).toBeInTheDocument();
  });

  test('hides content when collapsed', () => {
    render(
      <CollapsibleThinking
        section={mockSection}
        isExpanded={false}
        onToggle={vi.fn()}
      />
    );

    expect(screen.queryByText(/I need to analyze this request/)).not.toBeInTheDocument();
  });

  test('displays correct character count', () => {
    const longSection = {
      ...mockSection,
      content: 'A'.repeat(500),
      charLength: 500
    };

    render(
      <CollapsibleThinking
        section={longSection}
        isExpanded={false}
        onToggle={vi.fn()}
      />
    );

    expect(screen.getByText('500 chars')).toBeInTheDocument();
  });

  test('handles typing animation when enabled', () => {
    render(
      <CollapsibleThinking
        section={mockSection}
        isExpanded={true}
        onToggle={vi.fn()}
        hasTypingAnimation={true}
        typingSpeed={10}
      />
    );

    // Component should render with typing animation setup
    // (Full animation testing would require more complex mocking)
    expect(screen.getByText('thinking...')).toBeInTheDocument();
  });
});

describe('ThinkingSections Component', () => {
  const mockSections: ThinkingSection[] = [
    {
      id: 'thinking-1',
      content: 'First thinking section content.',
      startIndex: 0,
      endIndex: 50,
      isCollapsed: true,
      charLength: 35
    },
    {
      id: 'thinking-2', 
      content: 'Second thinking section with more detailed analysis and reasoning.',
      startIndex: 60,
      endIndex: 120,
      isCollapsed: true,
      charLength: 68
    }
  ];

  test('renders multiple thinking sections', () => {
    const mockToggle = vi.fn();
    
    render(
      <ThinkingSections
        sections={mockSections}
        expandedSections={new Set()}
        onToggleSection={mockToggle}
      />
    );

    expect(screen.getAllByText('thinking...')).toHaveLength(2);
    expect(screen.getByText('35 chars')).toBeInTheDocument();
    expect(screen.getByText('68 chars')).toBeInTheDocument();
  });

  test('handles section toggle correctly', async () => {
    const mockToggle = vi.fn();
    const user = userEvent.setup();
    
    render(
      <ThinkingSections
        sections={mockSections}
        expandedSections={new Set()}
        onToggleSection={mockToggle}
      />
    );

    const firstSection = screen.getAllByRole('button')[0];
    await user.click(firstSection);
    
    expect(mockToggle).toHaveBeenCalledWith('thinking-1');
  });

  test('shows expanded content for expanded sections', () => {
    render(
      <ThinkingSections
        sections={mockSections}
        expandedSections={new Set(['thinking-1'])}
        onToggleSection={vi.fn()}
      />
    );

    expect(screen.getByText('First thinking section content.')).toBeInTheDocument();
    expect(screen.queryByText(/Second thinking section with more/)).not.toBeInTheDocument();
  });

  test('renders nothing when no sections provided', () => {
    const { container } = render(
      <ThinkingSections
        sections={[]}
        expandedSections={new Set()}
        onToggleSection={vi.fn()}
      />
    );

    expect(container.firstChild).toBeNull();
  });

  test('applies typing animation only to first section', () => {
    render(
      <ThinkingSections
        sections={mockSections}
        expandedSections={new Set(['thinking-1', 'thinking-2'])}
        onToggleSection={vi.fn()}
        hasTypingAnimation={true}
      />
    );

    // Both sections should be rendered but only first gets animation
    expect(screen.getByText('First thinking section content.')).toBeInTheDocument();
    expect(screen.getByText(/Second thinking section with more/)).toBeInTheDocument();
  });

  test('handles keyboard navigation', async () => {
    const mockToggle = vi.fn();
    const user = userEvent.setup();
    
    render(
      <ThinkingSections
        sections={mockSections}
        expandedSections={new Set()}
        onToggleSection={mockToggle}
      />
    );

    const firstSection = screen.getAllByRole('button')[0];
    firstSection.focus();
    await user.keyboard('{Enter}');
    
    expect(mockToggle).toHaveBeenCalledWith('thinking-1');
  });
});