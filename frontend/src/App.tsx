import { useState, useCallback, useRef, useEffect } from 'react';
import { useStream } from '@langchain/langgraph-sdk/react';
import { WelcomeScreen } from '@/components/WelcomeScreen';
import { ChatMessagesView } from '@/components/ChatMessagesView';
import { useParallelSequences } from '@/hooks/useParallelSequences';
import { Message } from '@langchain/langgraph-sdk';
import { ProcessedEvent } from '@/components/ActivityTimeline';

export default function App() {
  // Chat state
  const [localMessages, setLocalMessages] = useState<(Message & { _locallyAdded?: boolean })[]>([]);
  const [liveActivityEvents, setLiveActivityEvents] = useState<ProcessedEvent[]>([]);
  const [historicalActivities, setHistoricalActivities] = useState<Record<string, ProcessedEvent[]>>({});
  const [threadId, setThreadId] = useState<string | null>(null);
  
  // Refs
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  
  // Parallel sequences for research streams
  const { 
    stop: stopParallelResearch,
    isLoading: isParallelLoading 
  } = useParallelSequences();

  // Event handlers for useStream
  const handleUpdateEvent = useCallback((data: any) => {
    try {
      const activityEvent = mapBackendEventToUIState({ data, event: 'updates' });
      if (activityEvent) {
        setLiveActivityEvents(prev => {
          // Prevent duplicate events by checking recent entries
          const isDuplicate = prev.length > 0 && 
            prev[prev.length - 1].title === activityEvent.title &&
            prev[prev.length - 1].data === activityEvent.data;
          
          if (isDuplicate) {
            return prev;
          }
          
          // Limit to last 50 events for performance
          const newEvents = [...prev, activityEvent];
          return newEvents.length > 50 ? newEvents.slice(-50) : newEvents;
        });
      }
    } catch (error) {
      console.warn('Error processing update event:', error);
    }
  }, [mapBackendEventToUIState]);

  const handleLangChainEvent = useCallback((data: any) => {
    try {
      // Handle tool calls and detailed backend events
      if (data && typeof data === 'object') {
        const eventType = data.event || data.name;
        let activityEvent: ProcessedEvent | null = null;

        // Map specific LangChain events to UI events with enhanced details
        if (eventType === 'on_tool_start') {
          const toolName = data.name || 'Unknown Tool';
          const toolInput = data.data?.input ? JSON.stringify(data.data.input).slice(0, 100) : '';
          activityEvent = {
            title: `🔧 ${toolName}`,
            data: toolInput ? `Starting ${toolName} with: ${toolInput}...` : `Starting ${toolName}...`
          };
        } else if (eventType === 'on_tool_end') {
          const toolName = data.name || 'Unknown Tool';
          const success = data.data?.output ? true : false;
          activityEvent = {
            title: `✅ ${toolName}`,
            data: success ? `${toolName} completed successfully` : `${toolName} completed`
          };
        } else if (eventType === 'on_tool_error') {
          const toolName = data.name || 'Unknown Tool';
          const error = data.data?.error?.message || 'Unknown error';
          activityEvent = {
            title: `❌ ${toolName}`,
            data: `Error in ${toolName}: ${error.slice(0, 100)}${error.length > 100 ? '...' : ''}`
          };
        } else if (eventType === 'on_chain_start') {
          const chainName = data.name || 'Processing';
          activityEvent = {
            title: `⚡ ${chainName}`,
            data: `Starting ${chainName.replace(/_/g, ' ')}...`
          };
        } else if (eventType === 'on_chain_end') {
          const chainName = data.name || 'Processing';
          activityEvent = {
            title: `✨ ${chainName}`,
            data: `${chainName.replace(/_/g, ' ')} completed`
          };
        } else if (eventType === 'on_llm_start') {
          activityEvent = {
            title: `🤖 LLM Processing`,
            data: `Model generating response...`
          };
        } else if (eventType === 'on_llm_end') {
          const tokenUsage = data.data?.llm_output?.token_usage;
          const usageInfo = tokenUsage ? ` (${tokenUsage.total_tokens} tokens)` : '';
          activityEvent = {
            title: `🎯 LLM Complete`,
            data: `Response generated${usageInfo}`
          };
        }

        if (activityEvent) {
          setLiveActivityEvents(prev => {
            // Enhanced duplicate detection for LangChain events
            const isDuplicate = prev.some(event => 
              event.title === activityEvent!.title && 
              Math.abs(Date.now() - (event as any).timestamp || 0) < 1000
            );
            
            if (isDuplicate) {
              return prev;
            }
            
            // Add timestamp for duplicate detection
            const timestampedEvent = { ...activityEvent!, timestamp: Date.now() };
            
            // Limit to last 50 events for performance
            const newEvents = [...prev, timestampedEvent];
            return newEvents.length > 50 ? newEvents.slice(-50) : newEvents;
          });
        }
      }
    } catch (error) {
      console.warn('Error processing LangChain event:', error);
      // Add error event to timeline
      const errorEvent: ProcessedEvent = {
        title: '❌ Event Processing Error',
        data: `Failed to process event: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
      setLiveActivityEvents(prev => [...prev, errorEvent]);
    }
  }, []);

  const handleStreamError = useCallback((error: unknown) => {
    console.error('Stream error:', error);
    const errorEvent: ProcessedEvent = {
      title: '🚨 Stream Error',
      data: `Connection error: ${error instanceof Error ? error.message : 'Unknown error'}`
    };
    setLiveActivityEvents(prev => {
      // Avoid duplicate error messages
      const hasRecentError = prev.some(event => 
        event.title.includes('Error') && 
        Math.abs(Date.now() - (event as any).timestamp || 0) < 5000
      );
      
      if (hasRecentError) {
        return prev;
      }
      
      return [...prev, { ...errorEvent, timestamp: Date.now() }];
    });
  }, []);

  const handleStreamFinish = useCallback((state: any) => {
    console.log('Stream finished with state:', state);
    const finishEvent: ProcessedEvent = {
      title: '🎉 Research Complete',
      data: 'All research processing completed successfully'
    };
    setLiveActivityEvents(prev => {
      // Only add completion event if not already present
      const hasCompletionEvent = prev.some(event => 
        event.title.includes('Complete') || event.title.includes('🎉')
      );
      
      if (hasCompletionEvent) {
        return prev;
      }
      
      return [...prev, { ...finishEvent, timestamp: Date.now() }];
    });
  }, []);

  // Enhanced streaming with useStream hook
  const {
    messages: streamMessages,
    isLoading: isStreamLoading,
    submit: streamSubmit,
    stop: streamStop,
  } = useStream({
    assistantId: 'Deep Researcher',
    apiUrl: import.meta.env.DEV ? 'http://localhost:2024' : 'http://localhost:8123',
    threadId,
    onThreadId: setThreadId,
    onUpdateEvent: handleUpdateEvent,
    onLangChainEvent: handleLangChainEvent,
    onError: handleStreamError,
    onFinish: handleStreamFinish,
  });

  // Sync stream messages with local message state
  useEffect(() => {
    if (streamMessages && streamMessages.length > 0) {
      setLocalMessages(prev => {
        const existing = new Set(prev.map(m => m.id));
        const existingLocalUserContents = new Set(
          prev
            .filter(m => m.type === 'human' && m._locallyAdded)
            .map(m => m.content)
        );
        
        const filtered = streamMessages.filter((m: Message) => {
          // Skip if we already have this message ID
          if (m.id && existing.has(m.id)) {
            return false;
          }
          
          // Skip user messages if we already have locally added message with same content
          if (m.type === 'human' && existingLocalUserContents.has(m.content)) {
            return false;
          }
          
          // Only include human, ai, or tool messages
          return m.type === 'human' || m.type === 'ai' || m.type === 'tool';
        });
        
        // Convert Message to extended message with _locallyAdded flag
        const localFiltered = filtered.map(m => ({ ...m, _locallyAdded: false }));
        
        return [...prev, ...localFiltered];
      });
    }
  }, [streamMessages]);

  // Combined messages for UI (prioritize stream messages, fall back to local)
  const messages = streamMessages && streamMessages.length > 0 ? 
    streamMessages.map(m => ({ ...m, _locallyAdded: false })) : 
    localMessages;

  // Handle activity cleanup when streaming finishes
  useEffect(() => {
    if (!isStreamLoading && liveActivityEvents.length > 0) {
      const timeoutId = setTimeout(() => {
        const currentEvents = liveActivityEvents;
        
        if (currentEvents.length > 0) {
          // Get the latest AI message to associate with activities
          const lastAiMessage = messages.filter(m => m.type === 'ai').pop();
          if (lastAiMessage?.id) {
            setHistoricalActivities(prev => {
              // Limit historical activities to last 10 conversations for memory management
              const entries = Object.entries(prev);
              const newEntry = [lastAiMessage.id!, [...currentEvents]];
              
              if (entries.length >= 10) {
                // Keep only the 9 most recent entries plus the new one
                const recentEntries = entries.slice(-9);
                return Object.fromEntries([...recentEntries, newEntry]);
              } else {
                return {
                  ...prev,
                  [lastAiMessage.id!]: [...currentEvents],
                };
              }
            });
          }
        }
        
        // Clear live activity events
        setLiveActivityEvents([]);
      }, 500);

      return () => clearTimeout(timeoutId);
    }
  }, [isStreamLoading, liveActivityEvents, messages]);

  // Map backend events to meaningful UI states
  const mapBackendEventToUIState = useCallback((chunk: any): ProcessedEvent | null => {
    try {
      const data = chunk.data || {};
      
      // Extract node name from chunk data to determine current phase
      let currentNode = '';
      let eventDescription = '';
      
      // Handle different event structures with enhanced detection
      if (typeof data === 'object') {
        // Check for node information in various formats
        if (data.node) {
          currentNode = data.node;
        } else if (data.metadata?.langgraph_node) {
          currentNode = data.metadata.langgraph_node;
        } else if (Array.isArray(data) && data.length > 0 && data[0].metadata?.langgraph_node) {
          currentNode = data[0].metadata.langgraph_node;
        } else if (chunk.name) {
          currentNode = chunk.name;
        } else if (chunk.event) {
          // Extract node from event name (e.g., "on_chain_start:research_supervisor")
          const eventParts = chunk.event.split(':');
          if (eventParts.length > 1) {
            currentNode = eventParts[1];
          }
        }
        
        // Extract meaningful description from event data with better parsing
        if (data.messages && Array.isArray(data.messages)) {
          const lastMessage = data.messages[data.messages.length - 1];
          if (lastMessage?.content && typeof lastMessage.content === 'string') {
            // Clean up common patterns in research messages
            let content = lastMessage.content;
            // Remove markdown formatting for cleaner display
            content = content.replace(/#{1,6}\s/g, '').replace(/\*\*/g, '').replace(/\*/g, '');
            eventDescription = content.slice(0, 120) + (content.length > 120 ? '...' : '');
          }
        } else if (data.content && typeof data.content === 'string') {
          let content = data.content;
          content = content.replace(/#{1,6}\s/g, '').replace(/\*\*/g, '').replace(/\*/g, '');
          eventDescription = content.slice(0, 120) + (content.length > 120 ? '...' : '');
        } else if (typeof data === 'string') {
          eventDescription = data.slice(0, 120) + (data.length > 120 ? '...' : '');
        }
      } else if (typeof data === 'string') {
        eventDescription = data.slice(0, 120) + (data.length > 120 ? '...' : '');
      }
      
      // Enhanced node to phase mapping with emojis and better descriptions
      const nodeToPhaseMap: Record<string, { title: string; description: string }> = {
        'clarify_with_user': {
          title: '❓ Analyzing Request',
          description: 'Reviewing your request and determining if clarification is needed...'
        },
        'write_research_brief': {
          title: '📋 Planning Research',
          description: 'Creating structured research brief and planning approach...'
        },
        'sequence_optimization_router': {
          title: '🎯 Optimizing Strategy',
          description: 'Selecting optimal research sequence for your topic...'
        },
        'research_supervisor': {
          title: '🔍 Conducting Research',
          description: 'Coordinating focused research with specialized agents...'
        },
        'sequence_research_supervisor': {
          title: '🚀 Advanced Research',
          description: 'Executing optimized research sequence with domain experts...'
        },
        'final_report_generation': {
          title: '📝 Writing Report',
          description: 'Synthesizing findings into comprehensive research report...'
        },
        'researcher': {
          title: '📚 Deep Research',
          description: 'Gathering detailed information from multiple sources...'
        },
        'researcher_tools': {
          title: '🛠️ Using Research Tools',
          description: 'Executing search and analysis tools...'
        },
        'supervisor': {
          title: '👥 Research Coordination',
          description: 'Coordinating research activities and managing workflow...'
        },
        'supervisor_tools': {
          title: '⚙️ Supervisor Tools',
          description: 'Executing coordination and management tools...'
        },
        'compress_research': {
          title: '📦 Compressing Findings',
          description: 'Summarizing and organizing research findings...'
        }
      };
      
      // Determine the appropriate phase
      let phaseInfo = nodeToPhaseMap[currentNode];
      
      // Enhanced content-based inference
      if (!phaseInfo && eventDescription) {
        const lowerDesc = eventDescription.toLowerCase();
        if (lowerDesc.includes('clarif') || lowerDesc.includes('question') || lowerDesc.includes('understand')) {
          phaseInfo = nodeToPhaseMap['clarify_with_user'];
        } else if (lowerDesc.includes('research') || lowerDesc.includes('search') || lowerDesc.includes('investigating')) {
          phaseInfo = nodeToPhaseMap['research_supervisor'];
        } else if (lowerDesc.includes('report') || lowerDesc.includes('brief') || lowerDesc.includes('writing') || lowerDesc.includes('summariz')) {
          phaseInfo = nodeToPhaseMap['final_report_generation'];
        } else if (lowerDesc.includes('tool') || lowerDesc.includes('executing') || lowerDesc.includes('running')) {
          phaseInfo = { title: '🔧 Tool Execution', description: 'Running specialized tools and utilities...' };
        } else if (lowerDesc.includes('planning') || lowerDesc.includes('organizing') || lowerDesc.includes('structur')) {
          phaseInfo = nodeToPhaseMap['write_research_brief'];
        }
      }
      
      // Return mapped event or enhanced fallback
      if (phaseInfo) {
        return {
          title: phaseInfo.title,
          data: eventDescription || phaseInfo.description
        };
      } else if (currentNode) {
        // Enhanced fallback with emoji and better formatting
        const cleanTitle = currentNode
          .replace(/_/g, ' ')
          .replace(/\b\w/g, l => l.toUpperCase())
          .replace(/^/, '⚡ '); // Add lightning emoji for unknown phases
        
        return {
          title: cleanTitle,
          data: eventDescription || 'Processing in progress...'
        };
      }
      
      return null; // Don't show unmappable events
      
    } catch (error) {
      console.warn('Error mapping backend event:', error);
      // Return error event instead of null for better debugging
      return {
        title: '⚠️ Event Processing',
        data: 'Error processing event data'
      };
    }
  }, []);

  const handleSubmit = useCallback(async (query: string) => {
    if (!query.trim()) return;
    
    try {
      // Add user message immediately with local flag
      const userMessage = {
        id: `user-${Date.now()}`,
        type: 'human' as const,
        content: query,
        _locallyAdded: true,
      };
      setLocalMessages(prev => [...prev, userMessage]);
      
      // Add activity event for starting research
      const startEvent: ProcessedEvent = {
        title: '🚀 Initializing Research',
        data: 'Connecting to research agents and preparing analysis...',
        timestamp: Date.now()
      };
      setLiveActivityEvents([startEvent]);
      
      // Use the enhanced useStream submit method
      streamSubmit({ 
        messages: [{ role: "human", content: query }]
      });
      
    } catch (error) {
      console.error('Chat submission error:', error);
      
      // Add error message
      const errorMessage = {
        id: `error-${Date.now()}`,
        type: 'ai' as const,
        content: `Error: ${error instanceof Error ? error.message : 'An unexpected error occurred'}`,
        _locallyAdded: true,
      };
      setLocalMessages(prev => [...prev, errorMessage]);
    }
  }, [streamSubmit]);

  const handleCancel = useCallback(() => {
    streamStop();
    stopParallelResearch();
    setLiveActivityEvents([]);
  }, [streamStop, stopParallelResearch]);

  const handleReset = useCallback(() => {
    setLocalMessages([]);
    setLiveActivityEvents([]);
    setHistoricalActivities({});
    setThreadId(null);
    streamStop();
    stopParallelResearch();
  }, [streamStop, stopParallelResearch]);

  return (
    <div className="flex h-screen bg-neutral-800">
      <main className="flex-1 flex flex-col max-w-5xl mx-auto">
        {messages.length === 0 ? (
          <WelcomeScreen 
            handleSubmit={handleSubmit}
            isLoading={isStreamLoading || isParallelLoading}
            onCancel={handleCancel}
          />
        ) : (
          <ChatMessagesView
            messages={messages as Message[]}
            isLoading={isStreamLoading || isParallelLoading}
            scrollAreaRef={scrollAreaRef}
            onSubmit={handleSubmit}
            onCancel={handleCancel}
            liveActivityEvents={liveActivityEvents}
            historicalActivities={historicalActivities}
            onReset={handleReset}
          />
        )}
      </main>
    </div>
  );
}
