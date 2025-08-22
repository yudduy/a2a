// Core tool call interface representing a request to execute a tool
export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  type: 'tool_call';
}

// Tool message interface representing the result of a tool execution
export interface ToolMessage {
  id: string;
  type: 'tool';
  tool_call_id: string;
  name: string;
  content: string | Record<string, unknown>;
  is_error?: boolean;
  error_type?: string;
  timestamp?: string;
}

// Extended tool call with execution metadata
export interface EnhancedToolCall extends ToolCall {
  status: 'pending' | 'running' | 'completed' | 'error';
  startTime?: Date;
  endTime?: Date;
  duration?: number;
  result?: ToolMessage;
}

// Tool execution status
export type ToolExecutionStatus = 'pending' | 'running' | 'completed' | 'error';

// Tool call group with related messages
export interface ToolCallGroup {
  id: string;
  toolCalls: ToolCall[];
  toolMessages: ToolMessage[];
  status: ToolExecutionStatus;
  startTime?: Date;
  endTime?: Date;
}

// Tool call result with metadata
export interface ToolCallResult {
  toolCall: ToolCall;
  toolMessage?: ToolMessage;
  status: ToolExecutionStatus;
  error?: string;
  duration?: number;
}

// Utility type for tool argument validation
export type ToolArgs = Record<string, unknown>;

// Tool execution error types
export interface ToolExecutionError {
  type: 'timeout' | 'validation' | 'execution' | 'network' | 'permission' | 'unknown';
  message: string;
  details?: Record<string, unknown>;
  toolCall: ToolCall;
}

// Tool call utilities
export class ToolCallUtils {
  /**
   * Create an enhanced tool call with status tracking
   */
  static createEnhancedToolCall(toolCall: ToolCall): EnhancedToolCall {
    return {
      ...toolCall,
      status: 'pending',
      startTime: new Date(),
    };
  }

  /**
   * Update tool call status and metadata
   */
  static updateToolCallStatus(
    enhancedToolCall: EnhancedToolCall,
    status: ToolExecutionStatus,
    result?: ToolMessage
  ): EnhancedToolCall {
    const updated = {
      ...enhancedToolCall,
      status,
    };

    if (status === 'running' && !updated.startTime) {
      updated.startTime = new Date();
    }

    if (status === 'completed' || status === 'error') {
      updated.endTime = new Date();
      if (updated.startTime) {
        updated.duration = updated.endTime.getTime() - updated.startTime.getTime();
      }
      if (result) {
        updated.result = result;
      }
    }

    return updated;
  }

  /**
   * Group tool calls with their corresponding results
   */
  static groupToolCallsWithResults(
    toolCalls: ToolCall[],
    toolMessages: ToolMessage[]
  ): ToolCallResult[] {
    return toolCalls.map((toolCall) => {
      const toolMessage = toolMessages.find(
        (msg) => msg.tool_call_id === toolCall.id
      );

      let status: ToolExecutionStatus = 'pending';
      if (toolMessage) {
        status = toolMessage.is_error ? 'error' : 'completed';
      }

      return {
        toolCall,
        toolMessage,
        status,
        error: toolMessage?.is_error ? toolMessage.content as string : undefined,
      };
    });
  }

  /**
   * Check if tool call has completed (either success or error)
   */
  static isToolCallCompleted(toolCall: ToolCall, toolMessages: ToolMessage[]): boolean {
    return toolMessages.some((msg) => msg.tool_call_id === toolCall.id);
  }

  /**
   * Get tool call execution duration
   */
  static getToolCallDuration(
    _toolCall: ToolCall,
    toolMessage?: ToolMessage
  ): number | undefined {
    if (!toolMessage?.timestamp) return undefined;
    
    // This would need to be implemented based on how timestamps are stored
    // For now, return undefined as we don't have start timestamps
    return undefined;
  }

  /**
   * Validate tool arguments
   */
  static validateToolArgs(
    args: ToolArgs,
    requiredArgs: string[] = []
  ): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    for (const required of requiredArgs) {
      if (!(required in args) || args[required] === undefined || args[required] === null) {
        errors.push(`Missing required argument: ${required}`);
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Format tool arguments for display
   */
  static formatArgsForDisplay(args: ToolArgs): string {
    try {
      return JSON.stringify(args, null, 2);
    } catch {
      return String(args);
    }
  }

  /**
   * Format tool result for display
   */
  static formatResultForDisplay(content: string | Record<string, unknown>): string {
    if (typeof content === 'string') {
      return content;
    }
    try {
      return JSON.stringify(content, null, 2);
    } catch {
      return String(content);
    }
  }

  /**
   * Create a tool execution error
   */
  static createExecutionError(
    type: ToolExecutionError['type'],
    message: string,
    toolCall: ToolCall,
    details?: Record<string, unknown>
  ): ToolExecutionError {
    return {
      type,
      message,
      toolCall,
      details,
    };
  }

  /**
   * Check if a tool message represents an error
   */
  static isErrorMessage(toolMessage: ToolMessage): boolean {
    return Boolean(toolMessage.is_error);
  }

  /**
   * Get human-readable status for tool call
   */
  static getStatusLabel(status: ToolExecutionStatus): string {
    switch (status) {
      case 'pending':
        return 'Pending';
      case 'running':
        return 'Running';
      case 'completed':
        return 'Completed';
      case 'error':
        return 'Error';
      default:
        return 'Unknown';
    }
  }

  /**
   * Filter tool calls by status
   */
  static filterToolCallsByStatus(
    toolCallResults: ToolCallResult[],
    status: ToolExecutionStatus
  ): ToolCallResult[] {
    return toolCallResults.filter((result) => result.status === status);
  }

  /**
   * Get overall status for a group of tool calls
   */
  static getGroupStatus(toolCallResults: ToolCallResult[]): ToolExecutionStatus {
    if (toolCallResults.length === 0) return 'pending';
    
    const hasError = toolCallResults.some((result) => result.status === 'error');
    const hasRunning = toolCallResults.some((result) => result.status === 'running');
    const hasPending = toolCallResults.some((result) => result.status === 'pending');
    const allCompleted = toolCallResults.every((result) => result.status === 'completed');

    if (hasError) return 'error';
    if (hasRunning || hasPending) return 'running';
    if (allCompleted) return 'completed';
    
    return 'pending';
  }
}
