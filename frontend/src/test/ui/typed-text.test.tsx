import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { TypedText, TypedParagraph, TypedCode, TypedList } from './typed-text';

// Mock timers for predictable testing
beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('TypedText', () => {
  describe('Basic Functionality', () => {
    it('renders empty text correctly', () => {
      render(<TypedText text="" />);
      expect(screen.queryByText('')).toBeInTheDocument();
    });

    it('renders text with typing animation', async () => {
      const onComplete = vi.fn();
      render(
        <TypedText 
          text="Hello World" 
          speed={10} 
          delay={0}
          onComplete={onComplete} 
        />
      );
      
      // Initially should show no text
      expect(screen.queryByText('Hello World')).not.toBeInTheDocument();
      
      // Fast-forward time to complete typing
      vi.advanceTimersByTime(200); // More than enough time for "Hello World"
      
      await waitFor(() => {
        expect(screen.getByText('Hello World')).toBeInTheDocument();
        expect(onComplete).toHaveBeenCalled();
      });
    });

    it('respects typing speed', async () => {
      render(<TypedText text="Hi" speed={100} delay={0} />);
      
      // After 50ms, should not be complete
      vi.advanceTimersByTime(50);
      expect(screen.queryByText('Hi')).not.toBeInTheDocument();
      
      // After 250ms, should be complete (100ms * 2 chars + buffer)
      vi.advanceTimersByTime(200);
      await waitFor(() => {
        expect(screen.getByText('Hi')).toBeInTheDocument();
      });
    });

    it('respects delay before starting', async () => {
      render(<TypedText text="Test" speed={10} delay={100} />);
      
      // Should not start typing immediately
      vi.advanceTimersByTime(50);
      expect(screen.queryByText('Test')).not.toBeInTheDocument();
      
      // Should start after delay
      vi.advanceTimersByTime(200);
      await waitFor(() => {
        expect(screen.getByText('Test')).toBeInTheDocument();
      });
    });
  });

  describe('Vertical Mode', () => {
    it('applies whitespace-pre-wrap class in vertical mode', () => {
      const { container } = render(
        <TypedText text="Line 1\nLine 2" verticalMode={true} />
      );
      
      const span = container.querySelector('span');
      expect(span).toHaveClass('whitespace-pre-wrap');
    });

    it('does not apply whitespace-pre-wrap class when vertical mode is false', () => {
      const { container } = render(
        <TypedText text="Line 1\nLine 2" verticalMode={false} />
      );
      
      const span = container.querySelector('span');
      expect(span).not.toHaveClass('whitespace-pre-wrap');
    });

    it('handles line breaks faster in vertical mode', async () => {
      const onComplete = vi.fn();
      render(
        <TypedText 
          text="Line1\nLine2" 
          speed={100} 
          verticalMode={true}
          onComplete={onComplete}
        />
      );
      
      vi.advanceTimersByTime(1000); // Should be enough with faster line breaks
      
      await waitFor(() => {
        expect(onComplete).toHaveBeenCalled();
      });
    });
  });

  describe('Cursor Behavior', () => {
    it('shows cursor by default', () => {
      const { container } = render(<TypedText text="Test" hideCursor={false} />);
      
      // Look for cursor element (animated pulse element)
      const cursor = container.querySelector('.animate-pulse');
      expect(cursor).toBeInTheDocument();
    });

    it('hides cursor when hideCursor is true', () => {
      const { container } = render(<TypedText text="Test" hideCursor={true} />);
      
      const cursor = container.querySelector('.animate-pulse');
      expect(cursor).not.toBeInTheDocument();
    });

    it('hides cursor after completion when hideCursor is true', async () => {
      const { container } = render(
        <TypedText text="Test" speed={10} hideCursor={true} />
      );
      
      vi.advanceTimersByTime(100);
      
      await waitFor(() => {
        const cursor = container.querySelector('.animate-pulse');
        expect(cursor).not.toBeInTheDocument();
      });
    });
  });

  describe('Children Rendering', () => {
    it('renders children after completion', async () => {
      render(
        <TypedText text="Hello" speed={10}>
          <span data-testid="child">Child Component</span>
        </TypedText>
      );
      
      // Children should not be visible initially
      expect(screen.queryByTestId('child')).not.toBeInTheDocument();
      
      // Complete typing
      vi.advanceTimersByTime(100);
      
      await waitFor(() => {
        expect(screen.getByTestId('child')).toBeInTheDocument();
      });
    });
  });

  describe('Custom Styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <TypedText text="Test" className="custom-class" />
      );
      
      const span = container.querySelector('span');
      expect(span).toHaveClass('custom-class');
    });
  });
});

describe('TypedParagraph', () => {
  it('renders text within a div', () => {
    const { container } = render(<TypedParagraph text="Paragraph text" />);
    
    const div = container.querySelector('div');
    expect(div).toBeInTheDocument();
  });

  it('applies custom className to div', () => {
    const { container } = render(
      <TypedParagraph text="Test" className="paragraph-class" />
    );
    
    const div = container.querySelector('div');
    expect(div).toHaveClass('paragraph-class');
  });
});

describe('TypedCode', () => {
  it('applies font-mono class', () => {
    const { container } = render(<TypedCode text="console.log('hello');" />);
    
    const span = container.querySelector('span');
    expect(span).toHaveClass('font-mono');
  });

  it('uses faster default speed', async () => {
    const onComplete = vi.fn();
    render(
      <TypedCode 
        text="code" 
        onComplete={onComplete}
      />
    );
    
    // Should complete faster than regular TypedText (15ms per char vs 30ms)
    vi.advanceTimersByTime(80);
    
    await waitFor(() => {
      expect(onComplete).toHaveBeenCalled();
    });
  });
});

describe('TypedList', () => {
  it('renders list items sequentially', async () => {
    const items = ['Item 1', 'Item 2', 'Item 3'];
    const onComplete = vi.fn();
    
    render(
      <TypedList 
        items={items}
        itemSpeed={10}
        itemDelay={50}
        onComplete={onComplete}
      />
    );
    
    // Initially, only first item should start typing
    vi.advanceTimersByTime(20);
    await waitFor(() => {
      expect(screen.getByText('Item 1')).toBeInTheDocument();
    });
    
    // After item delay, second item should start
    vi.advanceTimersByTime(70);
    await waitFor(() => {
      expect(screen.getByText('Item 2')).toBeInTheDocument();
    });
    
    // Complete all items
    vi.advanceTimersByTime(300);
    await waitFor(() => {
      expect(screen.getByText('Item 3')).toBeInTheDocument();
      expect(onComplete).toHaveBeenCalled();
    });
  });

  it('handles empty items array', () => {
    render(<TypedList items={[]} />);
    
    // Should render without errors
    expect(screen.queryByText('Item')).not.toBeInTheDocument();
  });

  it('applies custom className to container', () => {
    const { container } = render(
      <TypedList items={['Test']} className="list-class" />
    );
    
    const div = container.querySelector('div');
    expect(div).toHaveClass('list-class');
  });
});

describe('Edge Cases', () => {
  it('handles special characters', async () => {
    const specialText = 'Hello! @#$%^&*()_+ 🚀';
    render(<TypedText text={specialText} speed={10} />);
    
    vi.advanceTimersByTime(500);
    
    await waitFor(() => {
      expect(screen.getByText(specialText)).toBeInTheDocument();
    });
  });

  it('handles multiline text', async () => {
    const multilineText = 'Line 1\nLine 2\nLine 3';
    render(<TypedText text={multilineText} speed={10} verticalMode={true} />);
    
    vi.advanceTimersByTime(500);
    
    await waitFor(() => {
      expect(screen.getByText(multilineText)).toBeInTheDocument();
    });
  });

  it('handles very long text', async () => {
    const longText = 'A'.repeat(1000);
    const onComplete = vi.fn();
    
    render(
      <TypedText 
        text={longText} 
        speed={1} 
        onComplete={onComplete}
      />
    );
    
    vi.advanceTimersByTime(1200); // Should be enough time
    
    await waitFor(() => {
      expect(onComplete).toHaveBeenCalled();
    });
  });

  it('handles component unmounting during typing', () => {
    const { unmount } = render(<TypedText text="Test" speed={100} />);
    
    // Start typing
    vi.advanceTimersByTime(50);
    
    // Unmount component
    expect(() => unmount()).not.toThrow();
  });
});

describe('Performance', () => {
  it('cleans up timers on unmount', () => {
    const clearTimeoutSpy = vi.spyOn(global, 'clearTimeout');
    
    const { unmount } = render(<TypedText text="Test" speed={100} />);
    
    unmount();
    
    expect(clearTimeoutSpy).toHaveBeenCalled();
  });

  it('handles rapid prop changes', () => {
    const { rerender } = render(<TypedText text="First" speed={10} />);
    
    // Change text rapidly
    rerender(<TypedText text="Second" speed={10} />);
    rerender(<TypedText text="Third" speed={10} />);
    
    // Should not throw errors
    vi.advanceTimersByTime(100);
    
    expect(screen.getByText('Third')).toBeInTheDocument();
  });
});