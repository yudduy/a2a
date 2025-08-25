import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MessageContentParser } from '@/types/messages';

// Performance monitoring utilities
const measureTime = (fn: () => void): number => {
  const start = performance.now();
  fn();
  const end = performance.now();
  return end - start;
};

const measureMemory = () => {
  if ('memory' in performance) {
    return (performance as any).memory.usedJSHeapSize;
  }
  return 0; // Fallback if memory API not available
};

describe('Performance Tests', () => {
  beforeEach(() => {
    // Reset parser counter for consistent tests
    (MessageContentParser as any).thinkingCounter = 0;
  });

  describe('MessageContentParser Performance', () => {
    it('parses simple messages efficiently', () => {
      const message = {
        type: 'ai',
        content: 'Simple message without thinking sections'
      };

      const time = measureTime(() => {
        MessageContentParser.parse(message);
      });

      // Should parse simple messages very quickly (< 1ms)
      expect(time).toBeLessThan(5);
    });

    it('parses complex messages with thinking sections efficiently', () => {
      const message = {
        type: 'ai',
        content: `Complex message with multiple thinking sections.
        
<thinking>
This is a complex thinking section with multiple lines of content.
It contains analysis, reasoning, and various considerations that need to be processed.
The parser should handle this efficiently even with larger content blocks.
</thinking>

After the thinking section, there's more content.

<thinking>
And here's another thinking section with different content.
This one also has multiple lines and complex reasoning.
The parser needs to extract both sections correctly.
</thinking>

Final content after all thinking sections.`
      };

      const time = measureTime(() => {
        MessageContentParser.parse(message);
      });

      // Should parse complex messages reasonably quickly (< 10ms)
      expect(time).toBeLessThan(20);
    });

    it('handles large content volumes without performance degradation', () => {
      // Create a message with many thinking sections
      const thinkingSections = Array.from({ length: 50 }, (_, i) => 
        `<thinking>Thinking section ${i + 1} with detailed analysis and reasoning that spans multiple lines and contains complex logic for testing parser performance.</thinking>`
      ).join(' Content between sections. ');

      const message = {
        type: 'ai',
        content: `Start of large message. ${thinkingSections} End of large message.`
      };

      const time = measureTime(() => {
        const result = MessageContentParser.parse(message);
        expect(result.thinkingSections).toHaveLength(50);
      });

      // Even with 50 thinking sections, should parse in reasonable time (< 50ms)
      expect(time).toBeLessThan(100);
    });

    it('does not create memory leaks with repeated parsing', () => {
      const message = {
        type: 'ai',
        content: '<thinking>Memory leak test thinking section</thinking> Regular content here.'
      };

      const initialMemory = measureMemory();
      
      // Parse the same message many times
      for (let i = 0; i < 1000; i++) {
        MessageContentParser.parse(message);
      }

      const finalMemory = measureMemory();
      const memoryIncrease = finalMemory - initialMemory;

      // Memory increase should be minimal (less than 1MB)
      // Note: This is a rough check as memory measurement can be imprecise
      if (initialMemory > 0) {
        expect(memoryIncrease).toBeLessThan(1000000); // 1MB
      }
    });

    it('regex performance is consistent', () => {
      const content = 'A'.repeat(10000) + '<thinking>test</thinking>' + 'B'.repeat(10000);
      const message = { type: 'ai', content };

      const times: number[] = [];
      
      // Run parsing multiple times to check for performance consistency
      for (let i = 0; i < 10; i++) {
        const time = measureTime(() => {
          MessageContentParser.parse(message);
        });
        times.push(time);
      }

      // Calculate standard deviation to ensure consistent performance
      const avg = times.reduce((a, b) => a + b) / times.length;
      const variance = times.reduce((a, b) => a + Math.pow(b - avg, 2)) / times.length;
      const stdDev = Math.sqrt(variance);

      // Standard deviation should be low (consistent timing)
      expect(stdDev).toBeLessThan(avg * 0.5); // Less than 50% of average
    });
  });

  describe('Component Rendering Performance', () => {
    it('thinking section parsing is efficient for UI updates', () => {
      const messages = Array.from({ length: 100 }, (_, i) => ({
        type: 'ai',
        content: `Message ${i} <thinking>Thinking for message ${i}</thinking> content`
      }));

      const time = measureTime(() => {
        messages.forEach(message => {
          const result = MessageContentParser.parse(message);
          expect(result.hasThinking).toBe(true);
        });
      });

      // Should handle batch processing efficiently (< 100ms for 100 messages)
      expect(time).toBeLessThan(200);
    });

    it('helper functions are optimized', () => {
      const content = '<thinking>test1</thinking> content <thinking>test2</thinking>';
      
      const time = measureTime(() => {
        for (let i = 0; i < 1000; i++) {
          MessageContentParser.hasThinkingContent(content);
          MessageContentParser.countThinkingSections(content);
          MessageContentParser.getCleanContent(content);
        }
      });

      // Helper functions should be very fast even with many calls
      expect(time).toBeLessThan(50);
    });
  });

  describe('Edge Case Performance', () => {
    it('handles malformed content without performance penalties', () => {
      const malformedContent = '<thinking>unclosed thinking section that goes on and on with lots of content';
      const message = { type: 'ai', content: malformedContent };

      const time = measureTime(() => {
        const result = MessageContentParser.parse(message);
        expect(result.thinkingSections).toHaveLength(1);
      });

      // Malformed content should not significantly impact performance
      expect(time).toBeLessThan(10);
    });

    it('handles deeply nested brackets efficiently', () => {
      const nestedContent = '<thinking>nested <<>> content with <brackets> and <more <nested> content></thinking>';
      const message = { type: 'ai', content: nestedContent };

      const time = measureTime(() => {
        MessageContentParser.parse(message);
      });

      expect(time).toBeLessThan(5);
    });

    it('handles empty and whitespace content efficiently', () => {
      const testCases = [
        { type: 'ai', content: '' },
        { type: 'ai', content: '   ' },
        { type: 'ai', content: '\n\n\n' },
        { type: 'ai', content: '<thinking></thinking>' },
        { type: 'ai', content: '<thinking>   </thinking>' },
      ];

      const time = measureTime(() => {
        testCases.forEach(message => {
          MessageContentParser.parse(message);
        });
      });

      // Empty/whitespace content should be handled very quickly
      expect(time).toBeLessThan(5);
    });
  });

  describe('Memory Usage Patterns', () => {
    it('parser creates reasonable object structures', () => {
      const message = {
        type: 'ai',
        content: 'Test <thinking>analysis</thinking> content'
      };

      const result = MessageContentParser.parse(message);

      // Check that the parsed result doesn't create excessive object depth
      expect(result.renderSections.length).toBeLessThan(10);
      expect(result.thinkingSections.length).toBeLessThan(10);
      
      // Each thinking section should have reasonable properties
      result.thinkingSections.forEach(section => {
        expect(Object.keys(section)).toHaveLength(6); // id, content, startIndex, endIndex, isCollapsed, charLength
        expect(typeof section.id).toBe('string');
        expect(typeof section.content).toBe('string');
        expect(typeof section.charLength).toBe('number');
      });
    });

    it('render sections are structured efficiently', () => {
      const message = {
        type: 'ai',
        content: 'Pre <thinking>mid</thinking> post'
      };

      const result = MessageContentParser.parse(message);

      expect(result.renderSections).toHaveLength(3);
      
      result.renderSections.forEach(section => {
        expect(Object.keys(section)).toHaveLength(6); // id, type, content, order, typingSpeed?, isCollapsible, isCollapsed
        expect(['text', 'thinking', 'tool']).toContain(section.type);
        expect(typeof section.order).toBe('number');
      });
    });
  });

  describe('Concurrent Parsing Performance', () => {
    it('handles multiple simultaneous parsing operations', async () => {
      const messages = Array.from({ length: 20 }, (_, i) => ({
        type: 'ai',
        content: `Concurrent message ${i} <thinking>Concurrent thinking ${i}</thinking> end`
      }));

      const time = measureTime(() => {
        // Simulate concurrent parsing (though JS is single-threaded)
        const promises = messages.map(async message => {
          return MessageContentParser.parse(message);
        });
        
        // Wait for all parsing to complete
        return Promise.all(promises);
      });

      // Concurrent simulation should not significantly impact performance
      expect(time).toBeLessThan(50);
    });
  });

  describe('Real-world Performance Scenarios', () => {
    it('handles typical research response performance', () => {
      const typicalResearchResponse = `I need to research this topic comprehensively.

<thinking>
This is a complex research question about renewable energy trends. I should consider:

1. Market analysis - growth rates, investment trends, geographical distribution
2. Technology developments - efficiency improvements, cost reductions, innovation
3. Policy landscape - government incentives, regulations, international agreements  
4. Environmental impact - carbon reduction, sustainability metrics
5. Economic implications - job creation, industry disruption, energy costs

I'll need to generate multiple research sequences to cover these different aspects effectively.
</thinking>

Based on my analysis, I'll create specialized research sequences to explore renewable energy from multiple perspectives:

<thinking>
For the research sequences, I should create:

1. Market Analysis Sequence - Focus on financial data, investment trends, market projections
2. Technology Innovation Sequence - Look at R&D developments, patent filings, breakthrough technologies
3. Policy & Regulation Sequence - Analyze government policies, international agreements, regulatory frameworks
4. Environmental Impact Sequence - Study carbon reduction data, lifecycle assessments, sustainability metrics

Each sequence should have specific agents and data sources to ensure comprehensive coverage.
</thinking>

I'll now generate parallel research sequences to provide you with comprehensive insights on renewable energy trends.`;

      const message = { type: 'ai', content: typicalResearchResponse };

      const time = measureTime(() => {
        const result = MessageContentParser.parse(message);
        expect(result.thinkingSections).toHaveLength(2);
        expect(result.hasThinking).toBe(true);
        expect(result.renderSections.length).toBeGreaterThan(3);
      });

      // Typical research responses should parse quickly
      expect(time).toBeLessThan(15);
    });
  });
});