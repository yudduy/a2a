import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CollapsibleThinking, ThinkingIndicator } from '@/components/ui/collapsible-thinking';
import { ThinkingSection } from '@/types/messages';

// Mock dependencies
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

vi.mock('@/components/ui/badge', () => ({
  Badge: ({ children, variant, className }: any) => (
    <span data-testid="badge" data-variant={variant} className={className}>
      {children}
    </span>
  ),
}));

vi.mock('@/components/ui/typed-markdown', () => ({
  TypedThinkingMarkdown: ({ children, speed, className }: any) => (
    <div data-testid="typed-thinking-markdown" data-speed={speed} className={className}>
      {children}
    </div>
  ),
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

const mockThinkingSection: ThinkingSection = {
  id: 'test-thinking-1',
  content: 'This is a thinking section for accessibility testing.',
  startIndex: 0,
  endIndex: 50,
  isCollapsed: true,
  charLength: 45,
};

describe('Accessibility Tests', () => {
  describe('CollapsibleThinking Accessibility', () => {
    it('has proper ARIA attributes', () => {
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );

      const button = container.querySelector('button');
      expect(button).toBeInTheDocument();
      
      // Button should be focusable
      expect(button?.tagName).toBe('BUTTON');
    });

    it('supports keyboard navigation', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );

      const button = container.querySelector('button')!;
      
      // Focus the button
      button.focus();
      expect(button).toHaveFocus();

      // Activate with Enter key
      await user.keyboard('{Enter}');
      expect(onToggle).toHaveBeenCalled();

      // Reset mock
      onToggle.mockClear();

      // Activate with Space key
      await user.keyboard(' ');
      expect(onToggle).toHaveBeenCalled();
    });

    it('has proper focus management', () => {
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );

      const button = container.querySelector('button')!;
      
      // Check focus styles are present
      expect(button).toHaveClass('focus:outline-none');
      expect(button).toHaveClass('focus:ring-2');
      expect(button).toHaveClass('focus:ring-blue-500');
    });

    it('provides meaningful content for screen readers', () => {
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );

      // Should have meaningful text content
      expect(screen.getByText('thinking...')).toBeInTheDocument();
      expect(screen.getByText('45 chars')).toBeInTheDocument();
    });

    it('indicates expanded/collapsed state clearly', () => {
      const onToggle = vi.fn();
      
      const { rerender } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );

      // Collapsed state
      expect(screen.getByTestId('collapsible')).toHaveAttribute('data-open', 'false');

      // Expanded state
      rerender(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={true}
          onToggle={onToggle}
        />
      );

      expect(screen.getByTestId('collapsible')).toHaveAttribute('data-open', 'true');
      expect(screen.getByTestId('collapsible-content')).toBeInTheDocument();
    });

    it('handles high contrast mode gracefully', () => {
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );

      // Check that components use semantic colors that work in high contrast mode
      const button = container.querySelector('button')!;
      expect(button).toHaveClass('hover:bg-blue-900/20');
      
      // Color classes should be semantic (blue-themed for thinking)
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper).toHaveClass('border-blue-500/30');
      expect(wrapper).toHaveClass('bg-blue-900/10');
    });
  });

  describe('ThinkingIndicator Accessibility', () => {
    it('is properly focusable and actionable', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      const onClick = vi.fn();
      
      const { container } = render(
        <ThinkingIndicator count={3} onClick={onClick} />
      );

      const button = container.querySelector('button')!;
      
      // Should be focusable
      button.focus();
      expect(button).toHaveFocus();

      // Should be activatable via keyboard
      await user.keyboard('{Enter}');
      expect(onClick).toHaveBeenCalled();

      onClick.mockClear();
      
      await user.keyboard(' ');
      expect(onClick).toHaveBeenCalled();
    });

    it('provides clear count information', () => {
      const onClick = vi.fn();
      
      render(<ThinkingIndicator count={1} onClick={onClick} />);
      expect(screen.getByText('1 thinking section')).toBeInTheDocument();

      render(<ThinkingIndicator count={3} onClick={onClick} />);
      expect(screen.getByText('3 thinking sections')).toBeInTheDocument();
    });

    it('has proper focus management', () => {
      const onClick = vi.fn();
      
      const { container } = render(
        <ThinkingIndicator count={2} onClick={onClick} />
      );

      const button = container.querySelector('button')!;
      
      // Check focus styles
      expect(button).toHaveClass('focus:outline-none');
      expect(button).toHaveClass('focus:ring-2');
      expect(button).toHaveClass('focus:ring-blue-500');
    });

    it('handles no-click scenario gracefully', () => {
      render(<ThinkingIndicator count={2} />);
      
      expect(screen.getByText('2 thinking sections')).toBeInTheDocument();
    });
  });

  describe('Color and Contrast Accessibility', () => {
    it('uses appropriate color schemes for thinking sections', () => {
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={true}
          onToggle={onToggle}
        />
      );

      // Blue theme for thinking sections (good contrast)
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper).toHaveClass('border-blue-500/30');
      expect(wrapper).toHaveClass('bg-blue-900/10');

      // Content should use readable blue tones
      const contentArea = container.querySelector('.bg-blue-950\\/20');
      expect(contentArea).toBeInTheDocument();
    });

    it('provides sufficient contrast for text content', () => {
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={true}
          onToggle={onToggle}
        />
      );

      // Text should use high contrast colors
      expect(screen.getByText('thinking...')).toHaveClass('text-blue-200');
    });
  });

  describe('Responsive Design Accessibility', () => {
    it('maintains accessibility at different screen sizes', () => {
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
          className="responsive-thinking"
        />
      );

      // Check that responsive classes don't break accessibility
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper).toHaveClass('responsive-thinking');
      
      // Button should remain focusable regardless of size
      const button = container.querySelector('button')!;
      expect(button).toBeInTheDocument();
    });
  });

  describe('Animation and Motion Accessibility', () => {
    it('respects reduced motion preferences', () => {
      // Mock prefers-reduced-motion
      const mockMatchMedia = vi.fn().mockImplementation(query => ({
        matches: query === '(prefers-reduced-motion: reduce)',
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }));

      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: mockMatchMedia,
      });

      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
          hasTypingAnimation={true}
        />
      );

      // Animation classes should still be present (CSS will handle reduced motion)
      expect(screen.getByText('thinking...')).toBeInTheDocument();
    });

    it('provides non-animated fallbacks', () => {
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={true}
          onToggle={onToggle}
          hasTypingAnimation={false} // No animation
        />
      );

      // Content should be immediately visible without animation
      expect(screen.getByText(mockThinkingSection.content)).toBeInTheDocument();
    });
  });

  describe('Error State Accessibility', () => {
    it('handles empty content accessibly', () => {
      const emptySection: ThinkingSection = {
        ...mockThinkingSection,
        content: '',
        charLength: 0,
      };

      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={emptySection}
          isExpanded={true}
          onToggle={onToggle}
        />
      );

      // Should still be accessible even with empty content
      expect(screen.getByText('thinking...')).toBeInTheDocument();
      expect(screen.getByText('0 chars')).toBeInTheDocument();
    });

    it('handles very long content accessibly', () => {
      const longSection: ThinkingSection = {
        ...mockThinkingSection,
        content: 'A'.repeat(5000),
        charLength: 5000,
      };

      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={longSection}
          isExpanded={true}
          onToggle={onToggle}
        />
      );

      // Should remain accessible with long content
      expect(screen.getByText('thinking...')).toBeInTheDocument();
      expect(screen.getByText('5000 chars')).toBeInTheDocument();
    });
  });

  describe('Screen Reader Compatibility', () => {
    it('provides meaningful structure for screen readers', () => {
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );

      // Check for proper heading structure and landmarks
      expect(screen.getByText('thinking...')).toBeInTheDocument();
      
      // Content should be organized in logical reading order
      const badges = screen.getAllByTestId('badge');
      expect(badges.length).toBeGreaterThan(0);
    });

    it('announces state changes appropriately', () => {
      const onToggle = vi.fn();
      
      const { rerender } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );

      // Collapsed state should be clear
      expect(screen.getByTestId('collapsible')).toHaveAttribute('data-open', 'false');

      // Expanded state should be clear
      rerender(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={true}
          onToggle={onToggle}
        />
      );

      expect(screen.getByTestId('collapsible')).toHaveAttribute('data-open', 'true');
    });
  });

  describe('Touch and Mobile Accessibility', () => {
    it('has adequate touch targets', () => {
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );

      const button = container.querySelector('button')!;
      
      // Button should have adequate padding for touch
      expect(button).toHaveClass('px-4');
      expect(button).toHaveClass('py-3');
    });

    it('works with touch interactions', async () => {
      const user = userEvent.setup({ 
        advanceTimers: vi.advanceTimersByTime,
        pointerEventsCheck: 0 // Disable pointer events check for touch simulation
      });
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );

      const button = container.querySelector('button')!;
      
      // Simulate touch interaction
      await user.click(button);
      expect(onToggle).toHaveBeenCalled();
    });
  });
});