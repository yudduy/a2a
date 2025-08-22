import { useState, useCallback, useRef, useMemo } from 'react';
import { Client } from '@langchain/langgraph-sdk';
import { WelcomeScreen } from '@/components/WelcomeScreen';
import { ChatMessagesView } from '@/components/ChatMessagesView';
import { useParallelSequences } from '@/hooks/useParallelSequences';
import { Message } from '@langchain/langgraph-sdk';
import { ProcessedEvent } from '@/components/ActivityTimeline';

export default function App() {
  // Chat state
  const [messages, setMessages] = useState<(Message & { _locallyAdded?: boolean })[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [liveActivityEvents, setLiveActivityEvents] = useState<ProcessedEvent[]>([]);
  const [historicalActivities, setHistoricalActivities] = useState<Record<string, ProcessedEvent[]>>({});
  
  // Refs
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  const threadIdRef = useRef<string | null>(null);
  const clientRef = useRef<Client | null>(null);
  
  // Initialize LangGraph client
  const client = useMemo(() => {
    if (!clientRef.current) {
      clientRef.current = new Client({
        apiUrl: import.meta.env.DEV ? 'http://localhost:2024' : 'http://localhost:8123',
      });
    }
    return clientRef.current;
  }, []);
  
  // Parallel sequences for research streams
  const { 
    start: startParallelResearch, 
    stop: stopParallelResearch,
    isLoading: isParallelLoading 
  } = useParallelSequences();

  // Map backend events to meaningful UI states
  const mapBackendEventToUIState = useCallback((chunk: any): ProcessedEvent | null => {
    try {
      const data = chunk.data || {};
      
      // Extract node name from chunk data to determine current phase
      let currentNode = '';
      let eventDescription = '';
      
      // Handle different event structures
      if (typeof data === 'object') {
        // Check for node information in various formats
        if (data.node) {
          currentNode = data.node;
        } else if (data.metadata?.langgraph_node) {
          currentNode = data.metadata.langgraph_node;
        } else if (Array.isArray(data) && data.length > 0 && data[0].metadata?.langgraph_node) {
          currentNode = data[0].metadata.langgraph_node;
        } else if (chunk.name) {
          // Sometimes node name is in chunk.name
          currentNode = chunk.name;
        }
        
        // Extract meaningful description from event data
        if (data.messages && Array.isArray(data.messages)) {
          const lastMessage = data.messages[data.messages.length - 1];
          if (lastMessage?.content && typeof lastMessage.content === 'string') {
            eventDescription = lastMessage.content.slice(0, 100) + (lastMessage.content.length > 100 ? '...' : '');
          }
        } else if (data.content && typeof data.content === 'string') {
          eventDescription = data.content.slice(0, 100) + (data.content.length > 100 ? '...' : '');
        } else if (typeof data === 'string') {
          eventDescription = data.slice(0, 100) + (data.length > 100 ? '...' : '');
        }
      } else if (typeof data === 'string') {
        eventDescription = data.slice(0, 100) + (data.length > 100 ? '...' : '');
      }
      
      // Map node names to user-friendly phase names and descriptions
      const nodeToPhaseMap: Record<string, { title: string; description: string }> = {
        'clarify_with_user': {
          title: 'Asking Clarification',
          description: 'Analyzing your request and determining if clarification is needed...'
        },
        'write_research_brief': {
          title: 'Planning Research',
          description: 'Creating structured research brief and planning approach...'
        },
        'sequence_optimization_router': {
          title: 'Optimizing Strategy',
          description: 'Selecting optimal research sequence for your topic...'
        },
        'research_supervisor': {
          title: 'Researching',
          description: 'Conducting focused research with specialized agents...'
        },
        'sequence_research_supervisor': {
          title: 'Advanced Research',
          description: 'Executing optimized research sequence with domain experts...'
        },
        'final_report_generation': {
          title: 'Writing Brief',
          description: 'Synthesizing findings into comprehensive research report...'
        },
        // Handle researcher subgraph nodes
        'researcher': {
          title: 'Deep Research',
          description: 'Gathering detailed information from multiple sources...'
        }
      };
      
      // Determine the appropriate phase
      let phaseInfo = nodeToPhaseMap[currentNode];
      
      // If no specific node mapping, try to infer from content
      if (!phaseInfo && eventDescription) {
        if (eventDescription.toLowerCase().includes('clarif') || eventDescription.toLowerCase().includes('question')) {
          phaseInfo = nodeToPhaseMap['clarify_with_user'];
        } else if (eventDescription.toLowerCase().includes('research') || eventDescription.toLowerCase().includes('search')) {
          phaseInfo = nodeToPhaseMap['research_supervisor'];
        } else if (eventDescription.toLowerCase().includes('report') || eventDescription.toLowerCase().includes('brief')) {
          phaseInfo = nodeToPhaseMap['final_report_generation'];
        }
      }
      
      // Return mapped event or fallback
      if (phaseInfo) {
        return {
          title: phaseInfo.title,
          data: eventDescription || phaseInfo.description
        };
      } else if (currentNode) {
        // Fallback with cleaned node name
        const cleanTitle = currentNode.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        return {
          title: cleanTitle,
          data: eventDescription || 'Processing...'
        };
      }
      
      return null; // Don't show unmappable events
      
    } catch (error) {
      console.warn('Error mapping backend event:', error);
      return null;
    }
  }, []);

  const handleSubmit = useCallback(async (query: string) => {
    if (!query.trim()) return;
    
    try {
      setIsLoading(true);
      
      // Add user message immediately with local flag
      const userMessage = {
        id: `user-${Date.now()}`,
        type: 'human' as const,
        content: query,
        _locallyAdded: true,
      };
      setMessages(prev => [...prev, userMessage]);
      
      // Note: Parallel research disabled to ensure single supervisor flow
      // await startParallelResearch(query);
      
      // Create or get thread for main chat
      let threadId = threadIdRef.current;
      if (!threadId) {
        const thread = await client.threads.create();
        threadId = thread.thread_id;
        threadIdRef.current = threadId;
      }
      
      // Add activity event for starting research
      const startEvent: ProcessedEvent = {
        title: 'Initializing Research',
        data: 'Connecting to research agents and preparing analysis...',
      };
      setLiveActivityEvents([startEvent]);
      
      // Start main chat stream
      const stream = client.runs.stream(
        threadId,
        'Deep Researcher', // Use the exact registered graph name
        {
          input: { 
            messages: [{ role: "human", content: query }]
          },
        }
      );
      
      // Process stream
      for await (const chunk of stream) {
        // Handle both 'messages' and 'values' events
        if ((chunk.event === 'messages' || chunk.event === 'values') && chunk.data) {
          // For 'values' events, extract messages from the data
          let messagesToProcess: Message[] = [];
          
          if (chunk.event === 'values') {
            // For values events, check if data has messages property
            const valuesData = chunk.data as any;
            if (valuesData && typeof valuesData === 'object' && valuesData.messages) {
              messagesToProcess = Array.isArray(valuesData.messages) ? valuesData.messages : [valuesData.messages];
            }
          } else if (chunk.event === 'messages') {
            // For messages events, handle the data directly
            const messageData = chunk.data as any;
            if (Array.isArray(messageData)) {
              // Handle messages/complete and messages/partial events
              messagesToProcess = messageData;
            } else if (messageData && typeof messageData === 'object') {
              // Handle other message event formats
              if ('message' in messageData) {
                messagesToProcess = [messageData.message];
              }
            }
          }
          
          if (messagesToProcess.length > 0) {
            setMessages(prev => {
              const existing = new Set(prev.map(m => m.id));
              const existingLocalUserContents = new Set(
                prev
                  .filter(m => m.type === 'human' && m._locallyAdded)
                  .map(m => m.content)
              );
              
              const filtered = messagesToProcess.filter((m: Message) => {
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
        }
        
        // Handle activity events with meaningful mapping
        if (chunk.event === 'events' || chunk.event === 'updates' || chunk.event === 'debug') {
          const activityEvent = mapBackendEventToUIState(chunk);
          if (activityEvent) {
            setLiveActivityEvents(prev => [...prev, activityEvent]);
          }
        }
        
      }
      
    } catch (error) {
      console.error('Chat submission error:', error);
      
      // Add error message
      const errorMessage = {
        id: `error-${Date.now()}`,
        type: 'ai' as const,
        content: `Error: ${error instanceof Error ? error.message : 'An unexpected error occurred'}`,
        _locallyAdded: true,
      };
      setMessages(prev => [...prev, errorMessage]);
      
    } finally {
      setIsLoading(false);
      // Stop parallel research when main stream completes
      // This is especially important for clarification scenarios
      stopParallelResearch();
      
      // Clean up activity state with proper timing
      setTimeout(() => {
        const currentEvents = liveActivityEvents;
        
        if (currentEvents.length > 0) {
          // Get the latest state to find the last AI message
          setMessages(currentMessages => {
            const lastAiMessage = currentMessages.filter(m => m.type === 'ai').pop();
            if (lastAiMessage?.id) {
              setHistoricalActivities(prev => ({
                ...prev,
                [lastAiMessage.id!]: [...currentEvents],
              }));
            }
            return currentMessages;
          });
        }
        
        // Clear live activity events
        setLiveActivityEvents([]);
      }, 500); // Increased delay to ensure state is settled
    }
  }, [client, startParallelResearch, stopParallelResearch, liveActivityEvents, messages]);

  const handleCancel = useCallback(() => {
    setIsLoading(false);
    stopParallelResearch();
    setLiveActivityEvents([]);
  }, [stopParallelResearch]);

  const handleReset = useCallback(() => {
    setMessages([]);
    setLiveActivityEvents([]);
    setHistoricalActivities({});
    threadIdRef.current = null;
    stopParallelResearch();
  }, [stopParallelResearch]);

  return (
    <div className="flex h-screen bg-neutral-800">
      <main className="flex-1 flex flex-col max-w-5xl mx-auto">
        {messages.length === 0 ? (
          <WelcomeScreen 
            handleSubmit={handleSubmit}
            isLoading={isLoading || isParallelLoading}
            onCancel={handleCancel}
          />
        ) : (
          <ChatMessagesView
            messages={messages as Message[]}
            isLoading={isLoading || isParallelLoading}
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
