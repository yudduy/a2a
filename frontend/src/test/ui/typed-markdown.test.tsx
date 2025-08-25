import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { TypedMarkdown, TypedCodeMarkdown, TypedThinkingMarkdown } from './typed-markdown';

// Mock react-markdown
vi.mock('react-markdown', () => ({
  default: ({ children, components }: any) => (
    <div data-testid="markdown-content" data-components={JSON.stringify(components)}>
      {children}
    </div>
  ),
}));

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('TypedMarkdown', () => {
  describe('Basic Functionality', () => {
    it('renders empty content', () => {
      render(<TypedMarkdown>{""}</TypedMarkdown>);
      expect(screen.queryByTestId('markdown-content')).not.toBeInTheDocument();
    });

    it('renders markdown with typing animation', async () => {
      const markdownContent = '# Hello World\nThis is **bold** text.';
      
      render(
        <TypedMarkdown speed={10} delay={0}>
          {markdownContent}
        </TypedMarkdown>
      );
      
      // Initially should show partial content
      vi.advanceTimersByTime(50);
      
      // Should have markdown container but with partial content
      const container = screen.queryByTestId('markdown-content');
      expect(container).toBeInTheDocument();
      
      // Complete typing
      vi.advanceTimersByTime(500);
      
      await waitFor(() => {
        const finalContainer = screen.getByTestId('markdown-content');
        expect(finalContainer).toHaveTextContent(markdownContent);
      });
    });

    it('switches from typing to final markdown render on completion', async () => {
      const content = '# Title\nContent here';
      
      render(
        <TypedMarkdown speed={10}>
          {content}
        </TypedMarkdown>
      );
      
      // Fast-forward to completion
      vi.advanceTimersByTime(500);
      
      await waitFor(() => {
        // Should have the final markdown rendered
        expect(screen.getByTestId('markdown-content')).toHaveTextContent(content);
      });
    });
  });

  describe('Typing Animation', () => {
    it('respects typing speed', async () => {
      const content = 'Short';
      
      render(<TypedMarkdown speed={100}>{content}</TypedMarkdown>);
      
      // Should not complete immediately
      vi.advanceTimersByTime(50);
      let container = screen.queryByTestId('markdown-content');
      expect(container?.textContent).not.toBe(content);
      
      // Should complete after sufficient time
      vi.advanceTimersByTime(600);
      
      await waitFor(() => {
        container = screen.getByTestId('markdown-content');
        expect(container.textContent).toBe(content);
      });
    });

    it('respects delay before starting', async () => {
      const content = 'Test content';
      
      render(<TypedMarkdown speed={10} delay={100}>{content}</TypedMarkdown>);
      
      // Should not start typing immediately
      vi.advanceTimersByTime(50);
      expect(screen.queryByTestId('markdown-content')).toBeInTheDocument();
      
      // Should start after delay
      vi.advanceTimersByTime(200);
      
      await waitFor(() => {
        const container = screen.getByTestId('markdown-content');
        expect(container).toBeInTheDocument();
      });
    });

    it('handles line breaks faster in vertical mode', async () => {
      const content = 'Line 1\nLine 2\nLine 3';
      
      render(
        <TypedMarkdown speed={100} verticalMode={true}>
          {content}
        </TypedMarkdown>
      );
      
      // With faster line breaks, should complete sooner
      vi.advanceTimersByTime(800);
      
      await waitFor(() => {
        const container = screen.getByTestId('markdown-content');
        expect(container.textContent).toBe(content);
      });
    });
  });

  describe('Cursor Behavior', () => {
    it('hides cursor by default', () => {
      const { container } = render(
        <TypedMarkdown>{'Test content'}</TypedMarkdown>
      );
      
      // Should not show cursor (hideCursor defaults to true)
      const cursor = container.querySelector('.animate-pulse');
      expect(cursor).not.toBeInTheDocument();
    });

    it('shows cursor when hideCursor is false', () => {
      const { container } = render(
        <TypedMarkdown hideCursor={false}>{'Test content'}</TypedMarkdown>
      );
      
      const cursor = container.querySelector('.animate-pulse');
      expect(cursor).toBeInTheDocument();
    });
  });

  describe('Styling and Layout', () => {
    it('applies custom className', () => {
      const { container } = render(
        <TypedMarkdown className="custom-class">{'Test'}</TypedMarkdown>
      );
      
      const div = container.querySelector('div');
      expect(div).toHaveClass('custom-class');
    });

    it('applies whitespace-pre-wrap in vertical mode', () => {
      const { container } = render(
        <TypedMarkdown verticalMode={true}>{'Test\nContent'}</TypedMarkdown>
      );
      
      const preWrapDiv = container.querySelector('.whitespace-pre-wrap');
      expect(preWrapDiv).toBeInTheDocument();
    });

    it('does not apply whitespace-pre-wrap when vertical mode is false', () => {
      const { container } = render(
        <TypedMarkdown verticalMode={false}>{'Test\nContent'}</TypedMarkdown>
      );
      
      const preWrapDiv = container.querySelector('.whitespace-pre-wrap');
      expect(preWrapDiv).not.toBeInTheDocument();
    });
  });

  describe('Custom Components', () => {
    it('passes custom components to ReactMarkdown', async () => {
      const customComponents = {
        h1: ({ children }: any) => <h1 data-testid="custom-h1">{children}</h1>,
      };
      
      render(
        <TypedMarkdown components={customComponents} speed={10}>
          {'# Custom Header'}
        </TypedMarkdown>
      );
      
      vi.advanceTimersByTime(300);
      
      await waitFor(() => {
        const container = screen.getByTestId('markdown-content');
        const componentsAttr = container.getAttribute('data-components');
        expect(componentsAttr).toBeDefined();
      });
    });
  });
});

describe('TypedCodeMarkdown', () => {
  it('uses faster typing speed by default', async () => {
    const content = 'const x = 1;';
    
    render(<TypedCodeMarkdown>{content}</TypedCodeMarkdown>);
    
    // Should complete faster than regular TypedMarkdown (15ms vs 25ms per char)
    vi.advanceTimersByTime(200);
    
    await waitFor(() => {
      const container = screen.getByTestId('markdown-content');
      expect(container.textContent).toBe(content);
    });
  });

  it('hides cursor by default', () => {
    const { container } = render(
      <TypedCodeMarkdown>{'console.log("test");'}</TypedCodeMarkdown>
    );
    
    const cursor = container.querySelector('.animate-pulse');
    expect(cursor).not.toBeInTheDocument();
  });

  it('enables vertical mode by default', () => {
    const { container } = render(
      <TypedCodeMarkdown>{'line1\nline2'}</TypedCodeMarkdown>
    );
    
    const preWrapDiv = container.querySelector('.whitespace-pre-wrap');
    expect(preWrapDiv).toBeInTheDocument();
  });

  it('allows speed override', async () => {
    const content = 'fast';
    
    render(<TypedCodeMarkdown speed={5}>{content}</TypedCodeMarkdown>);
    
    // Should complete very quickly with speed=5
    vi.advanceTimersByTime(50);
    
    await waitFor(() => {
      const container = screen.getByTestId('markdown-content');
      expect(container.textContent).toBe(content);
    });
  });
});

describe('TypedThinkingMarkdown', () => {
  it('uses appropriate typing speed for thinking content', async () => {
    const content = 'Thinking process...';
    
    render(<TypedThinkingMarkdown>{content}</TypedThinkingMarkdown>);
    
    // Should complete with thinking speed (20ms per char)
    vi.advanceTimersByTime(400);
    
    await waitFor(() => {
      const container = screen.getByTestId('markdown-content');
      expect(container.textContent).toBe(content);
    });
  });

  it('applies thinking-specific styling', () => {
    const { container } = render(
      <TypedThinkingMarkdown>{'Thinking...'}</TypedThinkingMarkdown>
    );
    
    const div = container.querySelector('div');
    expect(div).toHaveClass('font-mono');
    expect(div).toHaveClass('text-sm');
    expect(div).toHaveClass('leading-relaxed');
  });

  it('hides cursor by default', () => {
    const { container } = render(
      <TypedThinkingMarkdown>{'Deep thought...'}</TypedThinkingMarkdown>
    );
    
    const cursor = container.querySelector('.animate-pulse');
    expect(cursor).not.toBeInTheDocument();
  });

  it('enables vertical mode by default', () => {
    const { container } = render(
      <TypedThinkingMarkdown>{'Line 1\nLine 2'}</TypedThinkingMarkdown>
    );
    
    const preWrapDiv = container.querySelector('.whitespace-pre-wrap');
    expect(preWrapDiv).toBeInTheDocument();
  });

  it('combines custom className with default styling', () => {
    const { container } = render(
      <TypedThinkingMarkdown className="extra-class">{'Test'}</TypedThinkingMarkdown>
    );
    
    const div = container.querySelector('div');
    expect(div).toHaveClass('font-mono');
    expect(div).toHaveClass('extra-class');
  });
});

describe('Edge Cases', () => {
  it('handles null or undefined content', () => {
    render(<TypedMarkdown>{null}</TypedMarkdown>);
    expect(screen.queryByTestId('markdown-content')).not.toBeInTheDocument();
  });

  it('handles very long markdown content', async () => {
    const longContent = '# '.repeat(500) + 'Long content';
    
    render(<TypedMarkdown speed={1}>{longContent}</TypedMarkdown>);
    
    vi.advanceTimersByTime(2000);
    
    await waitFor(() => {
      const container = screen.getByTestId('markdown-content');
      expect(container.textContent).toBe(longContent);
    });
  });

  it('handles complex markdown with multiple elements', async () => {
    const complexMarkdown = `# Title
    
## Subtitle

- List item 1
- List item 2

**Bold text** and *italic text*.

\`\`\`javascript
console.log('code block');
\`\`\`

[Link](https://example.com)`;
    
    render(<TypedMarkdown speed={5}>{complexMarkdown}</TypedMarkdown>);
    
    vi.advanceTimersByTime(1000);
    
    await waitFor(() => {
      const container = screen.getByTestId('markdown-content');
      expect(container.textContent).toBe(complexMarkdown);
    });
  });

  it('handles component unmounting during typing', () => {
    const { unmount } = render(
      <TypedMarkdown speed={100}>{'Long content...'}</TypedMarkdown>
    );
    
    vi.advanceTimersByTime(50);
    
    expect(() => unmount()).not.toThrow();
  });

  it('handles special markdown characters', async () => {
    const specialContent = '# Test\n**Bold** *italic* `code` [link](url) ![image](src)';
    
    render(<TypedMarkdown speed={10}>{specialContent}</TypedMarkdown>);
    
    vi.advanceTimersByTime(800);
    
    await waitFor(() => {
      const container = screen.getByTestId('markdown-content');
      expect(container.textContent).toBe(specialContent);
    });
  });
});

describe('Performance', () => {
  it('cleans up timers on unmount', () => {
    const clearTimeoutSpy = vi.spyOn(global, 'clearTimeout');
    
    const { unmount } = render(
      <TypedMarkdown speed={100}>{'Test content'}</TypedMarkdown>
    );
    
    unmount();
    
    expect(clearTimeoutSpy).toHaveBeenCalled();
  });

  it('handles rapid content changes', () => {
    const { rerender } = render(<TypedMarkdown speed={10}>{'First'}</TypedMarkdown>);
    
    rerender(<TypedMarkdown speed={10}>{'Second'}</TypedMarkdown>);
    rerender(<TypedMarkdown speed={10}>{'Third'}</TypedMarkdown>);
    
    vi.advanceTimersByTime(100);
    
    expect(screen.getByTestId('markdown-content')).toHaveTextContent('Third');
  });

  it('optimizes rendering for static content after typing completes', async () => {
    const content = '# Static Content';
    
    render(<TypedMarkdown speed={10}>{content}</TypedMarkdown>);
    
    // Complete typing
    vi.advanceTimersByTime(300);
    
    await waitFor(() => {
      // After completion, should render static markdown (not the typing version)
      const container = screen.getByTestId('markdown-content');
      expect(container).toBeInTheDocument();
      expect(container.textContent).toBe(content);
    });
  });
});