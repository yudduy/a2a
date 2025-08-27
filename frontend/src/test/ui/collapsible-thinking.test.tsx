import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { 
  CollapsibleThinking, 
  ThinkingSections, 
  ThinkingIndicator 
} from '@/components/ui/collapsible-thinking';
import { ThinkingSection } from '@/types/messages';

// Mock the ui components
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

vi.mock('./badge', () => ({
  Badge: ({ children, className, variant }: any) => (
    <span data-testid="badge" data-variant={variant} className={className}>
      {children}
    </span>
  ),
}));

vi.mock('./typed-markdown', () => ({
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
  content: 'This is a test thinking section with some content to think about.',
  startIndex: 0,
  endIndex: 100,
  isCollapsed: true,
  charLength: 63,
};

describe('CollapsibleThinking', () => {
  describe('Basic Rendering', () => {
    it('renders thinking section with correct structure', () => {
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );
      
      expect(screen.getByTestId('collapsible')).toBeInTheDocument();
      expect(screen.getByText('thinking...')).toBeInTheDocument();
      expect(screen.getByText('63 chars')).toBeInTheDocument();
    });

    it('displays character count in badge', () => {
      const onToggle = vi.fn();
      const section = { ...mockThinkingSection, charLength: 150 };
      
      render(
        <CollapsibleThinking
          section={section}
          isExpanded={false}
          onToggle={onToggle}
        />
      );
      
      expect(screen.getByText('150 chars')).toBeInTheDocument();
    });

    it('applies custom className', () => {
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
          className="custom-thinking-class"
        />
      );
      
      const wrapper = container.firstChild as HTMLElement;
      expect(wrapper).toHaveClass('custom-thinking-class');
    });
  });

  describe('Expansion/Collapse Functionality', () => {
    it('shows as collapsed when isExpanded is false', () => {
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );
      
      const collapsible = screen.getByTestId('collapsible');
      expect(collapsible).toHaveAttribute('data-open', 'false');
    });

    it('shows as expanded when isExpanded is true', () => {
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={true}
          onToggle={onToggle}
        />
      );
      
      const collapsible = screen.getByTestId('collapsible');
      expect(collapsible).toHaveAttribute('data-open', 'true');
    });

    it('calls onToggle when trigger is clicked', async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );
      
      const collapsible = screen.getByTestId('collapsible');
      await user.click(collapsible);
      
      expect(onToggle).toHaveBeenCalled();
    });

    it('rotates chevron when expanded', () => {
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={true}
          onToggle={onToggle}
        />
      );
      
      // Look for rotated chevron (should have rotate-90 class)
      const chevron = container.querySelector('.rotate-90');
      expect(chevron).toBeInTheDocument();
    });

    it('does not rotate chevron when collapsed', () => {
      const onToggle = vi.fn();
      
      const { container } = render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={false}
          onToggle={onToggle}
        />
      );
      
      const chevron = container.querySelector('.rotate-90');
      expect(chevron).not.toBeInTheDocument();
    });
  });

  describe('Content Display', () => {
    it('shows content when expanded', () => {
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={true}
          onToggle={onToggle}
        />
      );
      
      expect(screen.getByTestId('collapsible-content')).toBeInTheDocument();
      expect(screen.getByText(mockThinkingSection.content)).toBeInTheDocument();
    });

    it('uses typed animation when hasTypingAnimation is true', () => {
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={true}
          onToggle={onToggle}
          hasTypingAnimation={true}
          typingSpeed={15}
        />
      );
      
      const typedMarkdown = screen.getByTestId('typed-thinking-markdown');
      expect(typedMarkdown).toBeInTheDocument();
      expect(typedMarkdown).toHaveAttribute('data-speed', '15');
    });

    it('displays static content when hasTypingAnimation is false', () => {
      const onToggle = vi.fn();
      
      render(
        <CollapsibleThinking
          section={mockThinkingSection}
          isExpanded={true}
          onToggle={onToggle}
          hasTypingAnimation={false}
        />
      );
      
      expect(screen.queryByTestId('typed-thinking-markdown')).not.toBeInTheDocument();
      expect(screen.getByText(mockThinkingSection.content)).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper focus handling', () => {
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
      expect(button).toHaveClass('focus:outline-none');
      expect(button).toHaveClass('focus:ring-2');
      expect(button).toHaveClass('focus:ring-blue-500');
    });

    it('has proper keyboard navigation', async () => {
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
      button.focus();
      
      await user.keyboard('{Enter}');
      expect(onToggle).toHaveBeenCalled();
    });
  });
});

describe('ThinkingSections', () => {
  const mockSections: ThinkingSection[] = [
    {
      id: 'thinking-1',
      content: 'First thinking section',
      startIndex: 0,
      endIndex: 50,
      isCollapsed: true,
      charLength: 22,
    },
    {
      id: 'thinking-2',
      content: 'Second thinking section',
      startIndex: 50,
      endIndex: 100,
      isCollapsed: true,
      charLength: 23,
    },
  ];

  it('renders multiple thinking sections', () => {
    const expandedSections = new Set<string>();
    const onToggleSection = vi.fn();
    
    render(
      <ThinkingSections
        sections={mockSections}
        expandedSections={expandedSections}
        onToggleSection={onToggleSection}
      />
    );
    
    expect(screen.getAllByTestId('collapsible')).toHaveLength(2);
    expect(screen.getByText('22 chars')).toBeInTheDocument();
    expect(screen.getByText('23 chars')).toBeInTheDocument();
  });

  it('returns null when sections array is empty', () => {
    const expandedSections = new Set<string>();
    const onToggleSection = vi.fn();
    
    const { container } = render(
      <ThinkingSections
        sections={[]}
        expandedSections={expandedSections}
        onToggleSection={onToggleSection}
      />
    );
    
    expect(container.firstChild).toBeNull();
  });

  it('handles expanded sections correctly', () => {
    const expandedSections = new Set(['thinking-1']);
    const onToggleSection = vi.fn();
    
    render(
      <ThinkingSections
        sections={mockSections}
        expandedSections={expandedSections}
        onToggleSection={onToggleSection}
      />
    );
    
    const collapsibles = screen.getAllByTestId('collapsible');
    expect(collapsibles[0]).toHaveAttribute('data-open', 'true');
    expect(collapsibles[1]).toHaveAttribute('data-open', 'false');
  });

  it('calls onToggleSection with correct section ID', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const expandedSections = new Set<string>();
    const onToggleSection = vi.fn();
    
    render(
      <ThinkingSections
        sections={mockSections}
        expandedSections={expandedSections}
        onToggleSection={onToggleSection}
      />
    );
    
    const collapsibles = screen.getAllByTestId('collapsible');
    await user.click(collapsibles[1]);
    
    expect(onToggleSection).toHaveBeenCalledWith('thinking-2');
  });

  it('applies typing animation only to first section', () => {
    const expandedSections = new Set(['thinking-1', 'thinking-2']);
    const onToggleSection = vi.fn();
    
    render(
      <ThinkingSections
        sections={mockSections}
        expandedSections={expandedSections}
        onToggleSection={onToggleSection}
        hasTypingAnimation={true}
      />
    );
    
    const typedMarkdowns = screen.getAllByTestId('typed-thinking-markdown');
    expect(typedMarkdowns).toHaveLength(1); // Only first section gets typing
  });

  it('applies custom className to container', () => {
    const expandedSections = new Set<string>();
    const onToggleSection = vi.fn();
    
    const { container } = render(
      <ThinkingSections
        sections={mockSections}
        expandedSections={expandedSections}
        onToggleSection={onToggleSection}
        className="custom-sections-class"
      />
    );
    
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper).toHaveClass('custom-sections-class');
  });
});

describe('ThinkingIndicator', () => {
  it('renders indicator with correct count', () => {
    const onClick = vi.fn();
    
    render(
      <ThinkingIndicator count={3} onClick={onClick} />
    );
    
    expect(screen.getByText('3 thinking sections')).toBeInTheDocument();
  });

  it('uses singular form for count of 1', () => {
    const onClick = vi.fn();
    
    render(
      <ThinkingIndicator count={1} onClick={onClick} />
    );
    
    expect(screen.getByText('1 thinking section')).toBeInTheDocument();
  });

  it('returns null when count is 0', () => {
    const onClick = vi.fn();
    
    const { container } = render(
      <ThinkingIndicator count={0} onClick={onClick} />
    );
    
    expect(container.firstChild).toBeNull();
  });

  it('calls onClick when clicked', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const onClick = vi.fn();
    
    render(
      <ThinkingIndicator count={2} onClick={onClick} />
    );
    
    const button = screen.getByRole('button');
    await user.click(button);
    
    expect(onClick).toHaveBeenCalled();
  });

  it('applies custom className', () => {
    const onClick = vi.fn();
    
    const { container } = render(
      <ThinkingIndicator 
        count={1} 
        onClick={onClick} 
        className="custom-indicator-class" 
      />
    );
    
    const button = container.querySelector('button');
    expect(button).toHaveClass('custom-indicator-class');
  });

  it('has proper accessibility attributes', () => {
    const onClick = vi.fn();
    
    const { container } = render(
      <ThinkingIndicator count={2} onClick={onClick} />
    );
    
    const button = container.querySelector('button');
    expect(button).toHaveClass('focus:outline-none');
    expect(button).toHaveClass('focus:ring-2');
    expect(button).toHaveClass('focus:ring-blue-500');
  });

  it('renders without onClick handler', () => {
    render(
      <ThinkingIndicator count={1} />
    );
    
    expect(screen.getByText('1 thinking section')).toBeInTheDocument();
  });
});

describe('Edge Cases and Error Handling', () => {
  it('handles section with very long content', () => {
    const longSection: ThinkingSection = {
      ...mockThinkingSection,
      content: 'A'.repeat(10000),
      charLength: 10000,
    };
    
    const onToggle = vi.fn();
    
    render(
      <CollapsibleThinking
        section={longSection}
        isExpanded={true}
        onToggle={onToggle}
      />
    );
    
    expect(screen.getByText('10000 chars')).toBeInTheDocument();
    expect(screen.getByText('A'.repeat(10000))).toBeInTheDocument();
  });

  it('handles section with empty content', () => {
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
    
    expect(screen.getByText('0 chars')).toBeInTheDocument();
  });

  it('handles special characters in content', () => {
    const specialSection: ThinkingSection = {
      ...mockThinkingSection,
      content: 'Special chars: <>&"\'`{}[]()🚀💡',
      charLength: 30,
    };
    
    const onToggle = vi.fn();
    
    render(
      <CollapsibleThinking
        section={specialSection}
        isExpanded={true}
        onToggle={onToggle}
      />
    );
    
    expect(screen.getByText('Special chars: <>&"\'`{}[]()🚀💡')).toBeInTheDocument();
  });

  it('handles sections with markdown content', () => {
    const markdownSection: ThinkingSection = {
      ...mockThinkingSection,
      content: '# Heading\n**Bold** *italic* `code`',
      charLength: 32,
    };
    
    const onToggle = vi.fn();
    
    render(
      <CollapsibleThinking
        section={markdownSection}
        isExpanded={true}
        onToggle={onToggle}
        hasTypingAnimation={true}
      />
    );
    
    const typedMarkdown = screen.getByTestId('typed-thinking-markdown');
    expect(typedMarkdown).toHaveTextContent('# Heading\n**Bold** *italic* `code`');
  });

  it('handles rapid toggle operations', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const onToggle = vi.fn();
    
    render(
      <CollapsibleThinking
        section={mockThinkingSection}
        isExpanded={false}
        onToggle={onToggle}
      />
    );
    
    const collapsible = screen.getByTestId('collapsible');
    
    // Rapid clicks
    await user.click(collapsible);
    await user.click(collapsible);
    await user.click(collapsible);
    
    expect(onToggle).toHaveBeenCalledTimes(3);
  });
});