import type { Message as LangGraphMessage } from '@langchain/langgraph-sdk';
import { ToolCall, ToolMessage, ToolCallResult, ToolCallUtils, ToolExecutionStatus } from './tools';

export interface ExtendedMessage {
  id?: string;
  type: string;
  content: unknown;
  tool_calls?: ToolCall[];
  tool_call_chunks?: unknown[];
  tool_call_id?: string;
}

// New interfaces for thinking sections and parsed content
export interface ThinkingSection {
  id: string;
  content: string;
  startIndex: number;
  endIndex: number;
  isCollapsed: boolean;
  charLength: number;
}

export interface RenderSection {
  id: string;
  type: 'text' | 'thinking' | 'tool';
  content: string;
  order: number;
  typingSpeed?: number;
  isCollapsible: boolean;
  isCollapsed: boolean;
}

export interface ParsedMessageContent {
  // Main content sections
  preThinking?: string;
  thinkingSections: ThinkingSection[];
  postThinking?: string;
  
  // Tool-related content
  toolCalls: ToolCall[];
  toolResults: ToolMessage[];
  
  // Metadata
  hasThinking: boolean;
  totalCharacters: number;
  renderSections: RenderSection[];
}

// Type guard to check if a message is a tool message
export function isToolMessage(message: ExtendedMessage | LangGraphMessage): message is ToolMessage {
  return message.type === 'tool' && 'tool_call_id' in message;
}

// Type guard to check if a message is an AI message with tool calls
export function isAIMessageWithToolCalls(message: ExtendedMessage | LangGraphMessage): boolean {
  return message.type === 'ai' && 'tool_calls' in message && Array.isArray((message as ExtendedMessage).tool_calls);
}

// Enhanced tool call extraction with type safety
export function extractToolCallsFromMessage(
  message: ExtendedMessage | LangGraphMessage
): ToolCall[] {
  if (!isAIMessageWithToolCalls(message)) {
    return [];
  }
  return (message as ExtendedMessage).tool_calls || [];
}

// Find tool message for a specific tool call ID
export function findToolMessageForCall(
  messages: (ExtendedMessage | LangGraphMessage)[],
  toolCallId: string
): ToolMessage | undefined {
  return messages.find(
    (msg) =>
      isToolMessage(msg) &&
      (msg as ToolMessage).tool_call_id === toolCallId
  ) as ToolMessage | undefined;
}

// Get all tool messages from a list of messages
export function getToolMessages(
  messages: (ExtendedMessage | LangGraphMessage)[]
): ToolMessage[] {
  return messages.filter(isToolMessage) as ToolMessage[];
}

// Get all AI messages with tool calls
export function getAIMessagesWithToolCalls(
  messages: (ExtendedMessage | LangGraphMessage)[]
): ExtendedMessage[] {
  return messages.filter(isAIMessageWithToolCalls) as ExtendedMessage[];
}

// Group tool calls with their results for a specific message
export function getToolCallResultsForMessage(
  message: ExtendedMessage | LangGraphMessage,
  allMessages: (ExtendedMessage | LangGraphMessage)[]
): ToolCallResult[] {
  const toolCalls = extractToolCallsFromMessage(message);
  const toolMessages = getToolMessages(allMessages);
  
  return ToolCallUtils.groupToolCallsWithResults(toolCalls, toolMessages);
}

// Check if all tool calls in a message have completed
export function areAllToolCallsCompleted(
  message: ExtendedMessage | LangGraphMessage,
  allMessages: (ExtendedMessage | LangGraphMessage)[]
): boolean {
  const toolCalls = extractToolCallsFromMessage(message);
  if (toolCalls.length === 0) return true;
  
  const toolMessages = getToolMessages(allMessages);
  return toolCalls.every(toolCall => 
    ToolCallUtils.isToolCallCompleted(toolCall, toolMessages)
  );
}

// Get the execution status for all tool calls in a message
export function getMessageToolCallsStatus(
  message: ExtendedMessage | LangGraphMessage,
  allMessages: (ExtendedMessage | LangGraphMessage)[]
): ToolExecutionStatus {
  const toolCallResults = getToolCallResultsForMessage(message, allMessages);
  return ToolCallUtils.getGroupStatus(toolCallResults);
}

// Find all related messages for a tool call (the AI message that triggered it and the tool result)
export function findRelatedMessagesForToolCall(
  toolCallId: string,
  messages: (ExtendedMessage | LangGraphMessage)[]
): {
  aiMessage?: ExtendedMessage;
  toolMessage?: ToolMessage;
} {
  let aiMessage: ExtendedMessage | undefined;
  let toolMessage: ToolMessage | undefined;

  for (const message of messages) {
    // Check if this AI message contains the tool call
    if (isAIMessageWithToolCalls(message)) {
      const toolCalls = extractToolCallsFromMessage(message);
      if (toolCalls.some(tc => tc.id === toolCallId)) {
        aiMessage = message as ExtendedMessage;
      }
    }
    
    // Check if this is the tool result message
    if (isToolMessage(message) && (message as ToolMessage).tool_call_id === toolCallId) {
      toolMessage = message as ToolMessage;
    }
  }

  return { aiMessage, toolMessage };
}

// Get tool calls that are still pending (no result message found)
export function getPendingToolCalls(
  messages: (ExtendedMessage | LangGraphMessage)[]
): ToolCall[] {
  const allToolCalls: ToolCall[] = [];
  const toolMessages = getToolMessages(messages);
  
  // Collect all tool calls from AI messages
  for (const message of messages) {
    if (isAIMessageWithToolCalls(message)) {
      allToolCalls.push(...extractToolCallsFromMessage(message));
    }
  }
  
  // Filter out tool calls that have results
  return allToolCalls.filter(toolCall => 
    !ToolCallUtils.isToolCallCompleted(toolCall, toolMessages)
  );
}

// Get tool calls that have errors
export function getFailedToolCalls(
  messages: (ExtendedMessage | LangGraphMessage)[]
): Array<{ toolCall: ToolCall; toolMessage: ToolMessage }> {
  const allToolCalls: ToolCall[] = [];
  const toolMessages = getToolMessages(messages);
  const failedCalls: Array<{ toolCall: ToolCall; toolMessage: ToolMessage }> = [];
  
  // Collect all tool calls from AI messages
  for (const message of messages) {
    if (isAIMessageWithToolCalls(message)) {
      allToolCalls.push(...extractToolCallsFromMessage(message));
    }
  }
  
  // Find tool calls with error results
  for (const toolCall of allToolCalls) {
    const toolMessage = toolMessages.find(tm => 
      tm.tool_call_id === toolCall.id && tm.is_error
    );
    if (toolMessage) {
      failedCalls.push({ toolCall, toolMessage });
    }
  }
  
  return failedCalls;
}

// Count tool calls by status
export function countToolCallsByStatus(
  messages: (ExtendedMessage | LangGraphMessage)[]
): Record<ToolExecutionStatus, number> {
  const counts: Record<ToolExecutionStatus, number> = {
    pending: 0,
    running: 0,
    completed: 0,
    error: 0,
  };
  
  const allToolCalls: ToolCall[] = [];
  const toolMessages = getToolMessages(messages);
  
  // Collect all tool calls
  for (const message of messages) {
    if (isAIMessageWithToolCalls(message)) {
      allToolCalls.push(...extractToolCallsFromMessage(message));
    }
  }
  
  // Count by status
  const toolCallResults = ToolCallUtils.groupToolCallsWithResults(allToolCalls, toolMessages);
  for (const result of toolCallResults) {
    counts[result.status]++;
  }
  
  return counts;
}

// Message Content Parser for thinking sections
export class MessageContentParser {
  private static thinkingCounter = 0;
  
  static parse(message: ExtendedMessage | LangGraphMessage): ParsedMessageContent {
    const content = typeof message.content === 'string' ? message.content : JSON.stringify(message.content || '');
    
    // Extract thinking sections
    const thinkingSections = this.extractThinkingSections(content);
    
    // Extract tool calls
    const toolCalls = extractToolCallsFromMessage(message);
    
    // Split clean content into pre/post thinking
    const { preThinking, postThinking } = this.splitAroundThinking(content, thinkingSections);
    
    // Create render sections
    const renderSections = this.createRenderSections(preThinking, thinkingSections, postThinking, toolCalls);
    
    return {
      preThinking,
      thinkingSections,
      postThinking,
      toolCalls,
      toolResults: [], // Will be populated by caller with tool results
      hasThinking: thinkingSections.length > 0,
      totalCharacters: content.length,
      renderSections,
    };
  }
  
  private static extractThinkingSections(content: string): ThinkingSection[] {
    const sections: ThinkingSection[] = [];
    const allMatches: Array<{match: RegExpExecArray, type: string}> = [];
    
    // Support both <thinking> and <think> tags with improved patterns
    const thinkingPatterns = [
      {
        // Closed thinking tags
        pattern: /<thinking>([\s\S]*?)<\/thinking>/gm,
        type: 'thinking'
      },
      {
        // Closed think tags  
        pattern: /<think>([\s\S]*?)<\/think>/gm,
        type: 'think'
      },
      {
        // Unclosed thinking tags (only if no closing tag found)
        pattern: /<thinking>([\s\S]*)$/gm,
        type: 'thinking-unclosed'
      },
      {
        // Unclosed think tags (only if no closing tag found)
        pattern: /<think>([\s\S]*)$/gm,
        type: 'think-unclosed'
      }
    ];
    
    // First, collect all matches
    for (const { pattern, type } of thinkingPatterns) {
      let match;
      pattern.lastIndex = 0;
      
      while ((match = pattern.exec(content)) !== null) {
        allMatches.push({ match, type });
      }
    }
    
    // Process matches in order of appearance
    const processedRanges = new Set<string>();
    
    for (const { match, type } of allMatches) {
      const thinkingContent = match[1]?.trim() || '';
      const startIndex = match.index;
      const endIndex = match.index + match[0].length;
      const rangeKey = `${startIndex}-${endIndex}`;
      
      // Skip if we've already processed this range
      if (processedRanges.has(rangeKey)) {
        continue;
      }
      
      // Skip empty content
      if (thinkingContent.length === 0) {
        continue;
      }
      
      // Skip unclosed tags if we have closed tags covering the same content
      if (type.includes('unclosed')) {
        const hasClosedVersion = allMatches.some(other => 
          !other.type.includes('unclosed') && 
          Math.abs(other.match.index - startIndex) < 50
        );
        if (hasClosedVersion) {
          continue;
        }
      }
      
      sections.push({
        id: `${type.replace('-unclosed', '')}-${++this.thinkingCounter}`,
        content: thinkingContent,
        startIndex,
        endIndex,
        isCollapsed: true, // Default to collapsed for better UX
        charLength: thinkingContent.length,
      });
      
      processedRanges.add(rangeKey);
    }
    
    // Sort sections by start index to maintain document order
    return sections.sort((a, b) => a.startIndex - b.startIndex);
  }
  
  private static removeThinkingTags(content: string): string {
    // More comprehensive tag removal that handles all edge cases
    let cleanContent = content;
    
    // Remove closed thinking tags first
    cleanContent = cleanContent.replace(/<thinking>[\s\S]*?<\/thinking>/gm, '');
    cleanContent = cleanContent.replace(/<think>[\s\S]*?<\/think>/gm, '');
    
    // Remove unclosed thinking tags (from opening tag to end of content)
    cleanContent = cleanContent.replace(/<thinking>[\s\S]*$/gm, '');
    cleanContent = cleanContent.replace(/<think>[\s\S]*$/gm, '');
    
    // Clean up any remaining orphaned closing tags
    cleanContent = cleanContent.replace(/<\/thinking>/gm, '');
    cleanContent = cleanContent.replace(/<\/think>/gm, '');
    
    // Normalize whitespace and return
    return cleanContent.replace(/\n\s*\n\s*\n/g, '\n\n').trim();
  }
  
  private static splitAroundThinking(
    content: string,
    thinkingSections: ThinkingSection[]
  ): { preThinking?: string; postThinking?: string } {
    if (thinkingSections.length === 0) {
      // No thinking sections - return all content as pre-thinking
      const cleanedContent = this.removeThinkingTags(content);
      return { preThinking: cleanedContent || undefined };
    }
    
    const firstThinking = thinkingSections[0];
    const lastThinking = thinkingSections[thinkingSections.length - 1];
    
    // Extract content before first thinking section
    let preThinking = content.slice(0, firstThinking.startIndex).trim();
    // Remove any thinking tags from pre-content (in case of malformed content)
    preThinking = this.removeThinkingTags(preThinking);
    
    // Extract content after last thinking section
    let postThinking = content.slice(lastThinking.endIndex).trim();
    // Remove any thinking tags from post-content (in case of malformed content)
    postThinking = this.removeThinkingTags(postThinking);
    
    return {
      preThinking: preThinking || undefined,
      postThinking: postThinking || undefined,
    };
  }
  
  private static createRenderSections(
    preThinking: string | undefined,
    thinkingSections: ThinkingSection[],
    postThinking: string | undefined,
    toolCalls: ToolCall[]
  ): RenderSection[] {
    const sections: RenderSection[] = [];
    let order = 0;
    
    // Add pre-thinking text
    if (preThinking) {
      sections.push({
        id: 'pre-thinking',
        type: 'text',
        content: preThinking,
        order: order++,
        typingSpeed: 25,
        isCollapsible: false,
        isCollapsed: false,
      });
    }
    
    // Add thinking sections
    for (const thinking of thinkingSections) {
      sections.push({
        id: thinking.id,
        type: 'thinking',
        content: thinking.content,
        order: order++,
        typingSpeed: 20,
        isCollapsible: true,
        isCollapsed: thinking.isCollapsed,
      });
    }
    
    // Add post-thinking text
    if (postThinking) {
      sections.push({
        id: 'post-thinking',
        type: 'text',
        content: postThinking,
        order: order++,
        typingSpeed: 25,
        isCollapsible: false,
        isCollapsed: false,
      });
    }
    
    // Add tool sections
    for (const toolCall of toolCalls) {
      sections.push({
        id: `tool-${toolCall.id}`,
        type: 'tool',
        content: JSON.stringify(toolCall),
        order: order++,
        isCollapsible: true,
        isCollapsed: false, // Tools start expanded
      });
    }
    
    return sections.sort((a, b) => a.order - b.order);
  }
  
  // Helper to check if content has thinking sections
  static hasThinkingContent(content: string): boolean {
    return /<think(?:ing)?>([\s\S]*?)(<\/think(?:ing)?>|$)/i.test(content);
  }
  
  // Helper to get thinking sections count
  static countThinkingSections(content: string): number {
    const sections = this.extractThinkingSections(content);
    return sections.length;
  }
  
  // Helper to extract clean content without thinking tags
  static getCleanContent(content: string): string {
    return this.removeThinkingTags(content);
  }
}
