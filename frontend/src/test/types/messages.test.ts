import { describe, it, expect, beforeEach } from 'vitest';
import { 
  MessageContentParser,
  ThinkingSection,
  ParsedMessageContent,
  ExtendedMessage,
  isToolMessage,
  isAIMessageWithToolCalls,
  extractToolCallsFromMessage,
  findToolMessageForCall,
  getToolMessages,
  getAIMessagesWithToolCalls,
  getToolCallResultsForMessage,
  areAllToolCallsCompleted,
  getPendingToolCalls,
  getFailedToolCalls,
  countToolCallsByStatus
} from '@/types/messages';

describe('MessageContentParser', () => {
  beforeEach(() => {
    // Reset the thinking counter for consistent test results
    (MessageContentParser as any).thinkingCounter = 0;
  });

  describe('Basic Parsing', () => {
    it('parses message with no thinking sections', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'This is a regular message without thinking.',
      };

      const result = MessageContentParser.parse(message);

      expect(result.hasThinking).toBe(false);
      expect(result.thinkingSections).toHaveLength(0);
      expect(result.preThinking).toBe('This is a regular message without thinking.');
      expect(result.postThinking).toBeUndefined();
      expect(result.renderSections).toHaveLength(1);
      expect(result.renderSections[0].type).toBe('text');
    });

    it('parses message with single thinking section', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'Before thinking <thinking>I need to analyze this carefully</thinking> after thinking.',
      };

      const result = MessageContentParser.parse(message);

      expect(result.hasThinking).toBe(true);
      expect(result.thinkingSections).toHaveLength(1);
      expect(result.thinkingSections[0].content).toBe('I need to analyze this carefully');
      expect(result.preThinking).toBe('Before thinking');
      expect(result.postThinking).toBe('after thinking.');
      expect(result.renderSections).toHaveLength(3); // pre + thinking + post
    });

    it('parses message with multiple thinking sections', () => {
      const content = 'Start <thinking>First thought</thinking> middle <thinking>Second thought</thinking> end';
      const message: ExtendedMessage = { type: 'ai', content };

      const result = MessageContentParser.parse(message);

      expect(result.hasThinking).toBe(true);
      expect(result.thinkingSections).toHaveLength(2);
      expect(result.thinkingSections[0].content).toBe('First thought');
      expect(result.thinkingSections[1].content).toBe('Second thought');
    });

    it('handles thinking section at the beginning', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: '<thinking>Initial thought</thinking> This comes after.',
      };

      const result = MessageContentParser.parse(message);

      expect(result.preThinking).toBeUndefined();
      expect(result.postThinking).toBe('This comes after.');
      expect(result.thinkingSections[0].content).toBe('Initial thought');
    });

    it('handles thinking section at the end', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'This comes first <thinking>Final thought</thinking>',
      };

      const result = MessageContentParser.parse(message);

      expect(result.preThinking).toBe('This comes first');
      expect(result.postThinking).toBeUndefined();
      expect(result.thinkingSections[0].content).toBe('Final thought');
    });

    it('handles only thinking section', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: '<thinking>Only thinking here</thinking>',
      };

      const result = MessageContentParser.parse(message);

      expect(result.preThinking).toBeUndefined();
      expect(result.postThinking).toBeUndefined();
      expect(result.thinkingSections[0].content).toBe('Only thinking here');
      expect(result.renderSections).toHaveLength(1);
      expect(result.renderSections[0].type).toBe('thinking');
    });
  });

  describe('Thinking Section Properties', () => {
    it('sets correct properties for thinking sections', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'Test <thinking>Detailed analysis of the problem</thinking> done',
      };

      const result = MessageContentParser.parse(message);
      const section = result.thinkingSections[0];

      expect(section.id).toMatch(/^thinking-\d+$/);
      expect(section.isCollapsed).toBe(true);
      expect(section.charLength).toBe('Detailed analysis of the problem'.length);
      expect(section.startIndex).toBeGreaterThanOrEqual(0);
      expect(section.endIndex).toBeGreaterThan(section.startIndex);
    });

    it('generates unique IDs for multiple sections', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: '<thinking>First</thinking> and <thinking>Second</thinking>',
      };

      const result = MessageContentParser.parse(message);

      expect(result.thinkingSections[0].id).not.toBe(result.thinkingSections[1].id);
      expect(result.thinkingSections[0].id).toMatch(/^thinking-\d+$/);
      expect(result.thinkingSections[1].id).toMatch(/^thinking-\d+$/);
    });
  });

  describe('Edge Cases', () => {
    it('handles nested angle brackets in thinking content', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: '<thinking>Consider this: if x < y and y < z, then x < z</thinking>',
      };

      const result = MessageContentParser.parse(message);

      expect(result.thinkingSections[0].content).toBe('Consider this: if x < y and y < z, then x < z');
    });

    it('handles unclosed thinking tags', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'Start <thinking>This thinking never closes',
      };

      const result = MessageContentParser.parse(message);

      expect(result.thinkingSections).toHaveLength(1);
      expect(result.thinkingSections[0].content).toBe('This thinking never closes');
      expect(result.thinkingSections[0].id).toMatch(/^thinking-unclosed-\d+$/);
    });

    it('handles empty thinking sections', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'Before <thinking></thinking> after',
      };

      const result = MessageContentParser.parse(message);

      expect(result.thinkingSections).toHaveLength(1);
      expect(result.thinkingSections[0].content).toBe('');
      expect(result.thinkingSections[0].charLength).toBe(0);
    });

    it('handles thinking sections with only whitespace', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: '<thinking>   \n\t  </thinking>',
      };

      const result = MessageContentParser.parse(message);

      expect(result.thinkingSections[0].content).toBe('');
    });

    it('handles malformed thinking tags', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'Text with <thinking> and </thinking> and <THINKING>uppercase</THINKING>',
      };

      const result = MessageContentParser.parse(message);

      // Should only match properly formatted lowercase tags
      expect(result.thinkingSections).toHaveLength(1);
      expect(result.thinkingSections[0].content).toBe('');
    });

    it('handles multiline thinking content', () => {
      const thinkingContent = `First line of thinking
      Second line with indentation
      Third line`;
      
      const message: ExtendedMessage = {
        type: 'ai',
        content: `Before <thinking>${thinkingContent}</thinking> after`,
      };

      const result = MessageContentParser.parse(message);

      expect(result.thinkingSections[0].content).toBe(thinkingContent.trim());
    });
  });

  describe('Tool Integration', () => {
    it('includes tool calls in parsed result', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'I will use a tool now',
        tool_calls: [{
          id: 'tool-1',
          type: 'function',
          function: { name: 'test_tool', arguments: '{}' }
        }]
      };

      const result = MessageContentParser.parse(message);

      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].id).toBe('tool-1');
      expect(result.renderSections.some(section => section.type === 'tool')).toBe(true);
    });

    it('combines thinking sections and tool calls', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: '<thinking>I should use a tool</thinking> Using tool now',
        tool_calls: [{
          id: 'tool-1',
          type: 'function',
          function: { name: 'search', arguments: '{"query": "test"}' }
        }]
      };

      const result = MessageContentParser.parse(message);

      expect(result.hasThinking).toBe(true);
      expect(result.toolCalls).toHaveLength(1);
      expect(result.renderSections).toHaveLength(3); // thinking + text + tool
    });
  });

  describe('Render Sections', () => {
    it('creates render sections in correct order', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'Pre <thinking>thinking</thinking> post',
      };

      const result = MessageContentParser.parse(message);
      const sections = result.renderSections;

      expect(sections).toHaveLength(3);
      expect(sections[0].type).toBe('text');
      expect(sections[0].content).toBe('Pre');
      expect(sections[0].order).toBe(0);
      
      expect(sections[1].type).toBe('thinking');
      expect(sections[1].content).toBe('thinking');
      expect(sections[1].order).toBe(1);
      
      expect(sections[2].type).toBe('text');
      expect(sections[2].content).toBe('post');
      expect(sections[2].order).toBe(2);
    });

    it('sets appropriate typing speeds for different section types', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'Text <thinking>thought</thinking> more text',
      };

      const result = MessageContentParser.parse(message);

      const textSection = result.renderSections.find(s => s.type === 'text');
      const thinkingSection = result.renderSections.find(s => s.type === 'thinking');

      expect(textSection?.typingSpeed).toBe(25);
      expect(thinkingSection?.typingSpeed).toBe(20);
    });

    it('sets collapsible properties correctly', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'Text <thinking>thought</thinking>',
      };

      const result = MessageContentParser.parse(message);

      const textSection = result.renderSections.find(s => s.type === 'text');
      const thinkingSection = result.renderSections.find(s => s.type === 'thinking');

      expect(textSection?.isCollapsible).toBe(false);
      expect(textSection?.isCollapsed).toBe(false);
      
      expect(thinkingSection?.isCollapsible).toBe(true);
      expect(thinkingSection?.isCollapsed).toBe(true);
    });
  });

  describe('Helper Methods', () => {
    it('detects thinking content correctly', () => {
      expect(MessageContentParser.hasThinkingContent('<thinking>test</thinking>')).toBe(true);
      expect(MessageContentParser.hasThinkingContent('regular text')).toBe(false);
      expect(MessageContentParser.hasThinkingContent('<THINKING>uppercase</THINKING>')).toBe(false);
    });

    it('counts thinking sections correctly', () => {
      expect(MessageContentParser.countThinkingSections('<thinking>one</thinking>')).toBe(1);
      expect(MessageContentParser.countThinkingSections('<thinking>one</thinking> and <thinking>two</thinking>')).toBe(2);
      expect(MessageContentParser.countThinkingSections('no thinking here')).toBe(0);
    });

    it('extracts clean content without thinking tags', () => {
      const content = 'Before <thinking>remove this</thinking> after';
      const clean = MessageContentParser.getCleanContent(content);
      expect(clean).toBe('Before  after');
    });
  });

  describe('Non-string Content', () => {
    it('handles non-string message content', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: { text: 'object content', metadata: {} },
      };

      const result = MessageContentParser.parse(message);

      expect(result.preThinking).toContain('object content');
      expect(result.hasThinking).toBe(false);
    });

    it('handles null content', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: null,
      };

      const result = MessageContentParser.parse(message);

      expect(result.preThinking).toBe('null');
      expect(result.hasThinking).toBe(false);
    });
  });
});

describe('Type Guards and Utility Functions', () => {
  describe('isToolMessage', () => {
    it('identifies tool messages correctly', () => {
      const toolMessage = { type: 'tool', tool_call_id: 'test-id', content: 'result' };
      const aiMessage = { type: 'ai', content: 'hello' };
      
      expect(isToolMessage(toolMessage)).toBe(true);
      expect(isToolMessage(aiMessage)).toBe(false);
    });
  });

  describe('isAIMessageWithToolCalls', () => {
    it('identifies AI messages with tool calls correctly', () => {
      const aiWithTools: ExtendedMessage = {
        type: 'ai',
        content: 'using tools',
        tool_calls: [{ id: 'test', type: 'function', function: { name: 'test', arguments: '{}' } }]
      };
      
      const aiWithoutTools = { type: 'ai', content: 'no tools' };
      
      expect(isAIMessageWithToolCalls(aiWithTools)).toBe(true);
      expect(isAIMessageWithToolCalls(aiWithoutTools)).toBe(false);
    });
  });

  describe('extractToolCallsFromMessage', () => {
    it('extracts tool calls from AI messages', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'test',
        tool_calls: [
          { id: 'tool-1', type: 'function', function: { name: 'test1', arguments: '{}' } },
          { id: 'tool-2', type: 'function', function: { name: 'test2', arguments: '{}' } }
        ]
      };
      
      const toolCalls = extractToolCallsFromMessage(message);
      expect(toolCalls).toHaveLength(2);
      expect(toolCalls[0].id).toBe('tool-1');
      expect(toolCalls[1].id).toBe('tool-2');
    });

    it('returns empty array for messages without tool calls', () => {
      const message = { type: 'ai', content: 'no tools' };
      const toolCalls = extractToolCallsFromMessage(message);
      expect(toolCalls).toHaveLength(0);
    });
  });

  describe('getToolMessages', () => {
    it('filters tool messages from message list', () => {
      const messages = [
        { type: 'ai', content: 'hello' },
        { type: 'tool', tool_call_id: 'test-1', content: 'result1' },
        { type: 'human', content: 'hi' },
        { type: 'tool', tool_call_id: 'test-2', content: 'result2' }
      ];
      
      const toolMessages = getToolMessages(messages);
      expect(toolMessages).toHaveLength(2);
      expect(toolMessages.every(msg => msg.type === 'tool')).toBe(true);
    });
  });

  describe('getAIMessagesWithToolCalls', () => {
    it('filters AI messages that have tool calls', () => {
      const messages: ExtendedMessage[] = [
        { type: 'ai', content: 'no tools' },
        { type: 'ai', content: 'with tools', tool_calls: [{ id: 'test', type: 'function', function: { name: 'test', arguments: '{}' } }] },
        { type: 'human', content: 'human message' }
      ];
      
      const aiWithTools = getAIMessagesWithToolCalls(messages);
      expect(aiWithTools).toHaveLength(1);
      expect(aiWithTools[0].content).toBe('with tools');
    });
  });

  describe('Error Handling', () => {
    it('handles malformed tool calls gracefully', () => {
      const message: ExtendedMessage = {
        type: 'ai',
        content: 'test',
        tool_calls: null as any
      };
      
      const toolCalls = extractToolCallsFromMessage(message);
      expect(toolCalls).toHaveLength(0);
    });

    it('handles missing properties gracefully', () => {
      const message = { type: 'ai' } as any;
      const toolCalls = extractToolCallsFromMessage(message);
      expect(toolCalls).toHaveLength(0);
    });
  });
});