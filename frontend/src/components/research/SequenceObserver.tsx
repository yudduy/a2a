/**
 * SequenceObserver - Tab-based interface for observing parallel sequence execution
 * 
 * This component provides a real-time dashboard for monitoring 3 concurrent
 * LLM-generated research sequences with background execution and message caching.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { 
  SequenceState, 
  LLMGeneratedSequence, 
  RoutedMessage 
} from '@/types/parallel';
import { 
  Activity, 
  CheckCircle, 
  Loader2, 
  AlertTriangle,
  Clock,
  Users,
  Target,
  TrendingUp,
  MessageSquare,
  Zap
} from 'lucide-react';
import { TypedText } from '@/components/ui/typed-text';

interface SequenceObserverProps {
  sequences: SequenceState[];
  activeSequenceId: string;
  onSequenceChange: (sequenceId: string) => void;
  className?: string;
}

export function SequenceObserver({ 
  sequences, 
  activeSequenceId, 
  onSequenceChange,
  className = ""
}: SequenceObserverProps) {
  
  // Generate dynamic grid class based on number of sequences
  const getGridClass = (count: number) => {
    if (count === 1) return 'grid-cols-1';
    if (count === 2) return 'grid-cols-1 md:grid-cols-2';
    if (count === 3) return 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3';
    return 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'; // fallback for more than 3
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Loader2 className="w-4 h-4 animate-spin text-blue-500" />;
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'initializing': return <Activity className="w-4 h-4 text-yellow-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'completed': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'failed': return 'bg-red-500/10 text-red-400 border-red-500/20';
      case 'initializing': return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  // Ensure we have a valid active sequence
  const validActiveSequenceId = useMemo(() => {
    if (sequences.length === 0) return '';
    
    const hasActiveSequence = sequences.some(seq => seq.sequence_id === activeSequenceId);
    return hasActiveSequence ? activeSequenceId : sequences[0].sequence_id;
  }, [sequences, activeSequenceId]);

  // Update active sequence if it's invalid
  useEffect(() => {
    if (validActiveSequenceId && validActiveSequenceId !== activeSequenceId) {
      onSequenceChange(validActiveSequenceId);
    }
  }, [validActiveSequenceId, activeSequenceId, onSequenceChange]);

  if (sequences.length === 0) {
    return (
      <div className={`w-full h-full flex items-center justify-center ${className}`}>
        <div className="text-center space-y-4">
          <Activity className="w-8 h-8 text-gray-400 mx-auto animate-pulse" />
          <div className="text-gray-500">
            <div className="font-medium">Initializing Sequences</div>
            <div className="text-sm">Waiting for LLM-generated research sequences...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`w-full h-full ${className}`}>
      <Tabs value={validActiveSequenceId} onValueChange={onSequenceChange} className="h-full flex flex-col tabs-smooth">
        {/* Tab Headers - Dark Theme */}
        <TabsList className={`grid w-full ${getGridClass(sequences.length)} mb-4 transition-all duration-300 ease-in-out bg-neutral-700 border-neutral-600`}>
          {sequences.map((seq) => (
            <TabsTrigger 
              key={seq.sequence_id} 
              value={seq.sequence_id}
              className="flex items-center gap-2 px-3 py-2 content-transition data-[state=active]:bg-neutral-600 data-[state=active]:text-white text-neutral-300 border-neutral-600"
            >
              {getStatusIcon(seq.status)}
              <span className="truncate max-w-[120px] font-medium">
                {seq.sequence?.sequence_name || `Sequence ${sequences.indexOf(seq) + 1}`}
              </span>
              {seq.status === 'running' && (
                <Badge variant="secondary" className="ml-1 text-xs bg-neutral-600 text-neutral-200 border-neutral-500">
                  {seq.progress.agents_completed}/{seq.progress.total_agents}
                </Badge>
              )}
            </TabsTrigger>
          ))}
        </TabsList>

        {/* Tab Contents */}
        <div className="flex-1 overflow-hidden">
          {sequences.map((seq) => (
            <TabsContent 
              key={seq.sequence_id} 
              value={seq.sequence_id} 
              className="h-full flex flex-col mt-0 animate-scaleIn"
            >
              <SequenceDetailView sequence={seq} />
            </TabsContent>
          ))}
        </div>
      </Tabs>
    </div>
  );
}

// Individual sequence detail component - Chat-focused design
function SequenceDetailView({ sequence }: { sequence: SequenceState }) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'completed': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'failed': return 'bg-red-500/10 text-red-400 border-red-500/20';
      case 'initializing': return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  const getChatMessages = () => {
    const chatMessages = [];
    
    // Add initial sequence setup message
    if (sequence.sequence) {
      chatMessages.push({
        id: 'setup',
        type: 'system' as const,
        sender: 'Research Director',
        content: `Starting ${sequence.sequence.sequence_name} with strategy: ${sequence.sequence.rationale}`,
        timestamp: sequence.start_time,
        metadata: {
          agents: sequence.sequence.agent_names,
          confidence: sequence.sequence.confidence_score,
          focus: sequence.sequence.research_focus
        }
      });
    }

    // Convert sequence messages to chat format
    sequence.messages.forEach((msg, idx) => {
      chatMessages.push({
        id: msg.message_id || `msg-${idx}`,
        type: msg.message_type === 'error' ? 'error' : 'agent' as const,
        sender: msg.current_agent || msg.agent_type || 'Research Agent',
        content: formatMessageContent(msg.content),
        timestamp: msg.timestamp
      });
    });

    return chatMessages;
  };

  const formatMessageContent = (content: any): string => {
    if (typeof content === 'string') return content;
    if (typeof content === 'object') {
      // Handle LangGraph message format
      if (content.data) return JSON.stringify(content.data, null, 2);
      if (content.messages) return content.messages.map((m: any) => m.content).join('\n');
      return JSON.stringify(content, null, 2);
    }
    return String(content);
  };

  const chatMessages = getChatMessages();

  return (
    <div className="h-full flex flex-col bg-neutral-800 min-h-0">
      {/* Minimal Header */}
      <div className="bg-neutral-700 border-b border-neutral-600 px-4 py-2 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="font-medium text-neutral-100 text-sm">
              {sequence.sequence?.sequence_name || `Sequence ${sequence.sequence_id}`}
            </h3>
            <Badge className={`${getStatusColor(sequence.status)} text-xs`}>
              {sequence.status}
            </Badge>
          </div>
          <div className="text-xs text-neutral-400">
            {sequence.progress.completion_percentage.toFixed(0)}% • {sequence.sequence?.agent_names?.length || 0} agents
          </div>
        </div>
      </div>

      {/* Chat Messages Area - Like main chat */}
      <div className="flex-1 overflow-hidden min-h-0">
        <ScrollArea className="h-full">
          <div className="p-4 space-y-4">
            {chatMessages.length > 0 ? (
              chatMessages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))
            ) : (
              <div className="flex items-center justify-center h-full text-center text-neutral-500">
                <div className="space-y-3">
                  {sequence.status === 'initializing' && (
                    <>
                      <Activity className="w-6 w-6 mx-auto animate-pulse" />
                      <div className="text-sm">Initializing research sequence...</div>
                    </>
                  )}
                  {sequence.status === 'running' && (
                    <>
                      <Loader2 className="w-6 h-6 mx-auto animate-spin" />
                      <div className="text-sm">Research agents working...</div>
                    </>
                  )}
                  {sequence.status === 'completed' && (
                    <>
                      <CheckCircle className="w-6 h-6 mx-auto text-green-500" />
                      <div className="text-sm">Research sequence completed</div>
                    </>
                  )}
                  {sequence.status === 'failed' && (
                    <>
                      <AlertTriangle className="w-6 h-6 mx-auto text-red-500" />
                      <div className="text-sm">Research sequence failed</div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}

// Individual chat message component
function ChatMessage({ message }: { message: any }) {
  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getMessageIcon = (type: string, sender: string) => {
    if (type === 'system') return <Target className="w-4 h-4 text-blue-400" />;
    if (type === 'error') return <AlertTriangle className="w-4 h-4 text-red-400" />;
    if (sender.toLowerCase().includes('director')) return <Users className="w-4 h-4 text-purple-400" />;
    return <MessageSquare className="w-4 h-4 text-green-400" />;
  };

  return (
    <div className="space-y-2">
      {/* Message Header */}
      <div className="flex items-center gap-2">
        {getMessageIcon(message.type, message.sender)}
        <span className="font-medium text-sm text-neutral-200">{message.sender}</span>
        <span className="text-xs text-neutral-500">{formatTimestamp(message.timestamp)}</span>
      </div>
      
      {/* Message Content */}
      <div className="bg-neutral-700 rounded-lg p-3 ml-6">
        {message.type === 'system' && message.metadata && (
          <div className="mb-3 p-2 bg-neutral-600/50 rounded text-xs text-neutral-300">
            <div>Strategy: {message.content}</div>
            <div>Agents: {message.metadata.agents?.join(', ')}</div>
            <div>Focus: {message.metadata.focus}</div>
            <div>Confidence: {(message.metadata.confidence * 100).toFixed(1)}%</div>
          </div>
        )}
        <div className="text-sm text-neutral-300 whitespace-pre-wrap break-words">
          {message.content}
        </div>
      </div>
    </div>
  );
}

// Legacy MessageCard component - Chat-like format with dark theme
function MessageCard({ message, index }: { message: RoutedMessage; index: number }) {
  // This is now a legacy component - new ChatMessage component is used above
  return null;
}

export default SequenceObserver;