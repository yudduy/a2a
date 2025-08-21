import { useState, useCallback, useRef, useMemo } from 'react';
import { Client } from '@langchain/langgraph-sdk';
import { WelcomeScreen } from '@/components/WelcomeScreen';
import { ChatMessagesView } from '@/components/ChatMessagesView';
import { useParallelSequences } from '@/hooks/useParallelSequences';
import { Message } from '@langchain/langgraph-sdk';
import { ProcessedEvent } from '@/components/ActivityTimeline';

export default function App() {
  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
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
    sequences, 
    start: startParallelResearch, 
    stop: stopParallelResearch,
    isLoading: isParallelLoading 
  } = useParallelSequences();

  const handleSubmit = useCallback(async (query: string) => {
    if (!query.trim()) return;
    
    try {
      setIsLoading(true);
      
      // Add user message immediately
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        type: 'human',
        content: query,
      };
      setMessages(prev => [...prev, userMessage]);
      
      // Start parallel research sequences
      await startParallelResearch(query);
      
      // Create or get thread for main chat
      let threadId = threadIdRef.current;
      if (!threadId) {
        const thread = await client.threads.create();
        threadId = thread.thread_id;
        threadIdRef.current = threadId;
      }
      
      // Add activity event for starting research
      const startEvent: ProcessedEvent = {
        title: 'Starting Research',
        data: 'Initializing deep research process...',
      };
      setLiveActivityEvents([startEvent]);
      
      // Start main chat stream
      const stream = client.runs.stream(
        threadId,
        'deep_researcher',
        {
          input: { messages: query },
          configurable: { thread_id: threadId },
        }
      );
      
      // Process stream
      for await (const chunk of stream) {
        // Handle messages
        if (chunk.event === 'messages' && chunk.data) {
          const newMessages = Array.isArray(chunk.data) ? chunk.data : [chunk.data];
          setMessages(prev => {
            const existing = new Set(prev.map(m => m.id));
            const filtered = newMessages.filter((m: Message) => m.id && !existing.has(m.id));
            return [...prev, ...filtered];
          });
        }
        
        // Handle activity events - simplified
        if (chunk.event === 'events' || chunk.event === 'updates') {
          const activityEvent: ProcessedEvent = {
            title: `Processing: ${chunk.event}`,
            data: JSON.stringify(chunk.data || {}),
          };
          setLiveActivityEvents(prev => [...prev, activityEvent]);
        }
      }
      
    } catch (error) {
      console.error('Chat submission error:', error);
      
      // Add error message
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        type: 'ai',
        content: `Error: ${error instanceof Error ? error.message : 'An unexpected error occurred'}`,
      };
      setMessages(prev => [...prev, errorMessage]);
      
    } finally {
      setIsLoading(false);
      
      // Move live activity to historical for the last AI message
      if (liveActivityEvents.length > 0) {
        const lastAiMessage = messages.filter(m => m.type === 'ai').pop();
        if (lastAiMessage?.id) {
          setHistoricalActivities(prev => ({
            ...prev,
            [lastAiMessage.id!]: [...liveActivityEvents],
          }));
          setLiveActivityEvents([]);
        }
      }
    }
  }, [client, startParallelResearch, liveActivityEvents, messages]);

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
            messages={messages}
            isLoading={isLoading || isParallelLoading}
            scrollAreaRef={scrollAreaRef}
            onSubmit={handleSubmit}
            onCancel={handleCancel}
            liveActivityEvents={liveActivityEvents}
            historicalActivities={historicalActivities}
            onReset={handleReset}
            isParallelResearch={true}
            sequences={sequences}
          />
        )}
      </main>
    </div>
  );
}
