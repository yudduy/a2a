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
