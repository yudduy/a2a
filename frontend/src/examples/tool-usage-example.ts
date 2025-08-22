/**
 * Example usage of enhanced tool call support
 * This file demonstrates how to use the new tool types and utilities
 */

import type { Message as LangGraphMessage } from '@langchain/langgraph-sdk';
import {
  ToolCall,
  ToolMessage,
  ToolCallUtils,
  ToolCallResult,
  EnhancedToolCall,
  ToolExecutionStatus,
} from '../types/tools';
import {
  extractToolCallsFromMessage,
  findToolMessageForCall,
  getToolCallResultsForMessage,
  areAllToolCallsCompleted,
  getMessageToolCallsStatus,
  getPendingToolCalls,
  getFailedToolCalls,
  countToolCallsByStatus,
} from '../types/messages';

// Example: Processing messages with tool calls
export function processMessagesWithTools(messages: LangGraphMessage[]) {
  // Extract all tool calls from messages
  const allToolCalls: ToolCall[] = [];
  for (const message of messages) {
    const toolCalls = extractToolCallsFromMessage(message);
    allToolCalls.push(...toolCalls);
  }

  // Get tool call results for each message
  const messageResults = messages.map(message => ({
    message,
    toolCallResults: getToolCallResultsForMessage(message, messages),
    allCompleted: areAllToolCallsCompleted(message, messages),
    status: getMessageToolCallsStatus(message, messages),
  }));

  // Get pending and failed tool calls
  const pendingToolCalls = getPendingToolCalls(messages);
  const failedToolCalls = getFailedToolCalls(messages);

  // Count tool calls by status
  const statusCounts = countToolCallsByStatus(messages);

  return {
    allToolCalls,
    messageResults,
    pendingToolCalls,
    failedToolCalls,
    statusCounts,
  };
}

// Example: Working with enhanced tool calls
export function createEnhancedToolCallExample() {
  const basicToolCall: ToolCall = {
    id: 'tool_call_1',
    name: 'search_web',
    args: { query: 'TypeScript best practices', limit: 10 },
    type: 'tool_call',
  };

  // Create enhanced tool call with status tracking
  let enhancedToolCall = ToolCallUtils.createEnhancedToolCall(basicToolCall);
  
  // Update to running status
  enhancedToolCall = ToolCallUtils.updateToolCallStatus(enhancedToolCall, 'running');
  
  // Simulate completion with result
  const toolMessage: ToolMessage = {
    id: 'tool_msg_1',
    type: 'tool',
    tool_call_id: 'tool_call_1',
    name: 'search_web',
    content: 'Found 10 results for TypeScript best practices...',
    is_error: false,
  };
  
  enhancedToolCall = ToolCallUtils.updateToolCallStatus(
    enhancedToolCall,
    'completed',
    toolMessage
  );

  return enhancedToolCall;
}

// Example: Grouping tool calls with results
export function groupToolCallsExample() {
  const toolCalls: ToolCall[] = [
    {
      id: 'tool_1',
      name: 'fetch_data',
      args: { url: 'https://api.example.com/data' },
      type: 'tool_call',
    },
    {
      id: 'tool_2',
      name: 'process_data',
      args: { input: 'raw_data' },
      type: 'tool_call',
    },
  ];

  const toolMessages: ToolMessage[] = [
    {
      id: 'msg_1',
      type: 'tool',
      tool_call_id: 'tool_1',
      name: 'fetch_data',
      content: '{"status": "success", "data": [...]}',
      is_error: false,
    },
    {
      id: 'msg_2',
      type: 'tool',
      tool_call_id: 'tool_2',
      name: 'process_data',
      content: 'Processing failed: Invalid input format',
      is_error: true,
      error_type: 'validation',
    },
  ];

  const results = ToolCallUtils.groupToolCallsWithResults(toolCalls, toolMessages);
  const overallStatus = ToolCallUtils.getGroupStatus(results);

  return {
    results,
    overallStatus,
    completedCalls: ToolCallUtils.filterToolCallsByStatus(results, 'completed'),
    errorCalls: ToolCallUtils.filterToolCallsByStatus(results, 'error'),
  };
}

// Example: Validating tool arguments
export function validateToolArgsExample() {
  const args = {
    query: 'search term',
    limit: 10,
    // missing required 'api_key' argument
  };

  const validation = ToolCallUtils.validateToolArgs(args, ['query', 'api_key']);
  
  if (!validation.valid) {
    console.log('Validation errors:', validation.errors);
    // Output: ['Missing required argument: api_key']
  }

  return validation;
}

// Example: Formatting tool data for display
export function formatToolDataExample() {
  const complexArgs = {
    filters: {
      category: 'technology',
      date_range: {
        start: '2024-01-01',
        end: '2024-12-31',
      },
    },
    options: ['include_metadata', 'sort_by_relevance'],
  };

  const formattedArgs = ToolCallUtils.formatArgsForDisplay(complexArgs);
  console.log('Formatted arguments:', formattedArgs);

  const resultContent = {
    results: [
      { title: 'Result 1', score: 0.95 },
      { title: 'Result 2', score: 0.87 },
    ],
    metadata: { total_count: 42, execution_time: '250ms' },
  };

  const formattedResult = ToolCallUtils.formatResultForDisplay(resultContent);
  console.log('Formatted result:', formattedResult);

  return { formattedArgs, formattedResult };
}

// Example: Error handling
export function toolErrorHandlingExample() {
  const toolCall: ToolCall = {
    id: 'error_tool',
    name: 'risky_operation',
    args: { unsafe_param: 'dangerous_value' },
    type: 'tool_call',
  };

  const error = ToolCallUtils.createExecutionError(
    'validation',
    'Unsafe parameter detected',
    toolCall,
    { suggested_fix: 'Use safe parameter values' }
  );

  return error;
}