import { useState, useCallback, useRef, useEffect } from 'react';
import { useStream } from '@langchain/langgraph-sdk/react';
import { WelcomeScreen } from '@/components/WelcomeScreen';
import { ChatInterface } from '@/components/ChatInterface';
import { useParallelSequences } from '@/hooks/useParallelSequences';
import { Message } from '@langchain/langgraph-sdk';
import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';
import { EnhancedErrorBoundary } from '@/components/ui/enhanced-error-boundary';
// Removed unused imports: SequenceObserver, Button, Tabs, MessageSquare, BarChart3

// Interface for parallel tabs state
interface ParallelTabsState {
  isActive: boolean;
  activeTabId: string;
  sequences: LLMGeneratedSequence[];
  hasAnnounced: boolean;
}

export default function App() {
  
  // Chat state
  const [localMessages, setLocalMessages] = useState<(Message & { _locallyAdded?: boolean })[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  
  // Parallel tabs state for in-place tabs
  const [parallelTabsState, setParallelTabsState] = useState<ParallelTabsState>({
    isActive: false,
    activeTabId: '',
    sequences: [],
    hasAnnounced: false,
  });
  
  // Messages routed to specific tabs
  const [parallelMessages, setParallelMessages] = useState<Record<string, RoutedMessage[]>>({});
  
  // Refs
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  
  // Handle parallel message routing (defined early to avoid dependency issues)
  const handleParallelMessage = useCallback((data: any) => {
    if (data.sequence_id && parallelTabsState.isActive) {
      const routedMessage: RoutedMessage = {
        message_id: `msg_${data.sequence_id}_${Date.now()}`,
        sequence_id: data.sequence_id,
        sequence_name: data.sequence_name || 'Unknown Sequence',
        message_type: data.message_type || 'progress',
        timestamp: Date.now(),
        content: data.content || data,
        sequence_index: parallelTabsState.sequences.findIndex(s => s.sequence_id === data.sequence_id),
        routing_timestamp: Date.now(),
        current_agent: data.current_agent,
        agent_type: data.agent_type,
      };

      setParallelMessages(prev => ({
        ...prev,
        [data.sequence_id]: [...(prev[data.sequence_id] || []), routedMessage],
      }));
    }
  }, [parallelTabsState.isActive, parallelTabsState.sequences]);
  
  // Parallel sequences for research streams - simplified hook
  const parallelSequences = useParallelSequences();
  const { 
    start: startParallelResearch,
    stop: stopParallelResearch,
    isLoading: isParallelLoading,
    changeActiveSequence,
    routeMessage: routeParallelMessage
  } = parallelSequences;

  // Simplified event processing - no phase mapping
  const processBackendEvent = useCallback((chunk: any): void => {
    // Only log in development mode
    if (import.meta.env.DEV) {
      console.log('Backend event received:', chunk);
    }
    // No UI state mapping - let messages flow naturally
  }, []);

  // Event handlers for useStream
  const handleUpdateEvent = useCallback((data: any) => {
    try {
      // Only log in development mode
      if (import.meta.env.DEV) {
        console.log('UpdateEvent received:', JSON.stringify(data, null, 2));
      }
      // Check if this event contains generated sequences from the supervisor
      if (data && typeof data === 'object' && 
          ((data.sequences && Array.isArray(data.sequences)) || 
           (data.type === 'sequences_generated' && data.sequences && Array.isArray(data.sequences)) ||
           (data.frontend_sequences && data.frontend_sequences.sequences && Array.isArray(data.frontend_sequences.sequences)))) {
        
        // Extract sequences from different possible locations
        let sequences = data.sequences;
        if (data.frontend_sequences && data.frontend_sequences.sequences) {
          sequences = data.frontend_sequences.sequences;
        }
        if (import.meta.env.DEV) {
          console.log('Received sequences from backend supervisor:', sequences);
        }
        
        // Convert backend sequences to LLMGeneratedSequence format if needed
        const llmSequences: LLMGeneratedSequence[] = sequences.map((seq: any, index: number) => ({
          sequence_id: seq.sequence_id || `seq_${Date.now()}_${index}`,
          sequence_name: seq.sequence_name || `Sequence ${index + 1}`,
          agent_names: seq.agent_names || ['research_agent', 'analysis_agent', 'synthesis_agent'],
          rationale: seq.rationale || `Backend-generated sequence ${index + 1}`,
          research_focus: seq.research_focus || `Sequence ${index + 1} focus`,
          confidence_score: seq.confidence_score || 0.8,
          approach_description: seq.approach_description || `Approach for sequence ${index + 1}`,
          expected_outcomes: seq.expected_outcomes || ['Research results', 'Analysis output'],
          created_at: seq.created_at || new Date().toISOString(),
        }));
        
        // Initialize parallel tabs state with sequences
        setParallelTabsState({
          isActive: false, // Will be activated when user clicks "Launch"
          activeTabId: llmSequences[0]?.sequence_id || '',
          sequences: llmSequences,
          hasAnnounced: true,
        });

        // Debug logging
        if (import.meta.env.DEV) {
          console.log('Frontend: Parallel tabs state updated with sequences:', {
            sequences: llmSequences,
            hasAnnounced: true,
            sequenceCount: llmSequences.length
          });
        }
        
        return; // Don't process as regular activity event
      }
      
      // Check if this is a message for parallel tabs - route through simplified hook
      if (data && typeof data === 'object' && data.sequence_id && parallelTabsState.isActive) {
        handleParallelMessage(data);
        // Also route to the parallel sequences hook for state management
        routeParallelMessage({
          message_id: `msg_${data.sequence_id}_${Date.now()}`,
          sequence_id: data.sequence_id,
          sequence_name: data.sequence_name || 'Unknown Sequence',
          message_type: data.message_type || 'progress',
          timestamp: Date.now(),
          content: data.content || data,
          sequence_index: parallelTabsState.sequences.findIndex(s => s.sequence_id === data.sequence_id),
          routing_timestamp: Date.now(),
          current_agent: data.current_agent,
          agent_type: data.agent_type,
        });
        return; // Don't process as regular activity event when routing to tabs
      }

      // Process backend events without UI mapping
      processBackendEvent({ data, event: 'updates' });
    } catch (error) {
      console.warn('Error processing update event:', error);
    }
  }, [localMessages, startParallelResearch, parallelTabsState.isActive, handleParallelMessage, routeParallelMessage]);

  const handleLangChainEvent = useCallback((data: any) => {
    try {
      if (import.meta.env.DEV) {
        console.log('LangChainEvent received:', data);
      }
      // Check if this LangChain event contains generated sequences
      if (data && typeof data === 'object') {
        // Check for sequences in the data payload
        if (data.data?.sequences && Array.isArray(data.data.sequences)) {
          if (import.meta.env.DEV) {
            console.log('Received sequences from LangChain event:', data.data.sequences);
          }
          
          const llmSequences: LLMGeneratedSequence[] = data.data.sequences.map((seq: any, index: number) => ({
            sequence_id: seq.sequence_id || `seq_${Date.now()}_${index}`,
            sequence_name: seq.sequence_name || `Sequence ${index + 1}`,
            agent_names: seq.agent_names || ['research_agent', 'analysis_agent', 'synthesis_agent'],
            rationale: seq.rationale || `Backend-generated sequence ${index + 1}`,
            research_focus: seq.research_focus || `Sequence ${index + 1} focus`,
            confidence_score: seq.confidence_score || 0.8,
            approach_description: seq.approach_description || `Approach for sequence ${index + 1}`,
            expected_outcomes: seq.expected_outcomes || ['Research results', 'Analysis output'],
            created_at: seq.created_at || new Date().toISOString(),
          }));
          
          // Initialize parallel tabs state with sequences
          setParallelTabsState({
            isActive: false, // Will be activated when user clicks "Launch"
            activeTabId: llmSequences[0]?.sequence_id || '',
            sequences: llmSequences,
            hasAnnounced: true,
          });

          // Debug logging
          if (import.meta.env.DEV) {
            console.log('Frontend: Parallel tabs state updated from LangChain event with sequences:', {
              sequences: llmSequences,
              hasAnnounced: true,
              sequenceCount: llmSequences.length
            });
          }
          
          return;
        }
        
        // Check if this is a message for parallel tabs - route through simplified hook  
        if (data.data && data.data.sequence_id && parallelTabsState.isActive) {
          handleParallelMessage(data.data);
          // Also route to the parallel sequences hook for state management
          routeParallelMessage({
            message_id: `msg_${data.data.sequence_id}_${Date.now()}`,
            sequence_id: data.data.sequence_id,
            sequence_name: data.data.sequence_name || 'Unknown Sequence',
            message_type: data.data.message_type || 'progress',
            timestamp: Date.now(),
            content: data.data.content || data.data,
            sequence_index: parallelTabsState.sequences.findIndex(s => s.sequence_id === data.data.sequence_id),
            routing_timestamp: Date.now(),
            current_agent: data.data.current_agent,
            agent_type: data.data.agent_type,
          });
          return; // Don't process as regular activity event when routing to tabs
        }

        // Simplified LangChain event processing
        if (import.meta.env.DEV) {
          console.log('LangChain event processed:', data.event || data.name);
        }
      }
    } catch (error) {
      console.warn('Error processing LangChain event:', error);
    }
  }, [localMessages, startParallelResearch, parallelTabsState.isActive, handleParallelMessage, routeParallelMessage]);

  const handleStreamError = useCallback((error: unknown) => {
    console.error('Stream error:', error);
  }, []);

  const handleStreamFinish = useCallback((state: any) => {
    if (import.meta.env.DEV) {
      console.log('Stream finished with state:', state);
    }
  }, []);

  // Enhanced streaming with useStream hook
  const {
    messages: streamMessages,
    isLoading: isStreamLoading,
    submit: streamSubmit,
    stop: streamStop,
  } = useStream({
    assistantId: 'Deep Researcher',
    apiUrl: import.meta.env.DEV ? 'http://localhost:2024' : (import.meta.env.VITE_API_URL || 'https://your-production-api-host.com'),
    threadId,
    // streamMode: 'messages' as const, // Commented out as this property doesn't exist in the hook
    onThreadId: setThreadId,
    onUpdateEvent: handleUpdateEvent,
    onCustomEvent: handleUpdateEvent,  // Handle custom events from get_stream_writer() the same way as UpdateEvents
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
      
      
      // Always submit to backend supervisor - it automatically decides on single vs parallel sequences
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
  }, [streamStop, stopParallelResearch]);

  // Handle parallel tabs initialization
  const handleTabsInitialized = useCallback(() => {
    if (import.meta.env.DEV) {
      console.log('handleTabsInitialized called:', {
        sequencesLength: parallelTabsState.sequences.length,
        currentIsActive: parallelTabsState.isActive,
        sequences: parallelTabsState.sequences.map(s => s.sequence_name)
      });
    }
    
    if (parallelTabsState.sequences.length > 0) {
      setParallelTabsState(prev => {
        const newState = {
          ...prev,
          isActive: true,
        };
        if (import.meta.env.DEV) {
          console.log('Setting parallelTabsState.isActive to true:', newState);
        }
        return newState;
      });
      
      // Extract the original query from recent messages
      const lastHumanMessage = localMessages.filter(m => m.type === 'human').pop();
      const researchQuery = typeof lastHumanMessage?.content === 'string' 
        ? lastHumanMessage.content 
        : 'Parallel research request';
      
      if (import.meta.env.DEV) {
        console.log('Starting parallel research with query:', researchQuery);
      }
      
      // Start parallel research with the announced sequences
      startParallelResearch(researchQuery, parallelTabsState.sequences);
    } else {
      if (import.meta.env.DEV) {
        console.log('No sequences available for parallel research');
      }
    }
  }, [parallelTabsState.sequences, parallelTabsState.isActive, localMessages, startParallelResearch]);

  // Handle parallel tab change
  const handleParallelTabChange = useCallback((tabId: string) => {
    setParallelTabsState(prev => ({
      ...prev,
      activeTabId: tabId,
    }));
    changeActiveSequence(tabId);
  }, [changeActiveSequence]);


  const handleReset = useCallback(() => {
    setLocalMessages([]);
    setThreadId(null);
    setParallelTabsState({
      isActive: false,
      activeTabId: '',
      sequences: [],
      hasAnnounced: false,
    });
    setParallelMessages({});
    streamStop();
    stopParallelResearch();
  }, [streamStop, stopParallelResearch]);


  return (
    <EnhancedErrorBoundary 
      level="page" 
      resetKeys={[messages.length, parallelTabsState.isActive ? 1 : 0]}
      onError={(error, errorInfo) => {
        console.error('App-level error:', error, errorInfo);
      }}
    >
      <div className="flex h-screen bg-neutral-800 text-neutral-100 overflow-hidden">
        <main className="flex-1 flex flex-col max-w-full xl:max-w-7xl mx-auto w-full min-h-0">
          {messages.length === 0 ? (
            <EnhancedErrorBoundary level="component" resetKeys={[isStreamLoading ? 1 : 0]}>
              <WelcomeScreen 
                handleSubmit={handleSubmit}
                isLoading={isStreamLoading || isParallelLoading}
                onCancel={handleCancel}
              />
            </EnhancedErrorBoundary>
          ) : (
            // Unified chat interface with in-place parallel tabs
            <EnhancedErrorBoundary 
              level="feature" 
              resetKeys={[messages.length, parallelTabsState.sequences.length]}
            >
              <ChatInterface
                messages={messages as Message[]}
                isLoading={isStreamLoading || isParallelLoading}
                scrollAreaRef={scrollAreaRef}
                onSubmit={handleSubmit}
                onCancel={handleCancel}
                onReset={handleReset}
                // Parallel tabs props
                parallelTabsState={parallelTabsState}
                parallelMessages={parallelMessages}
                onParallelTabChange={handleParallelTabChange}
                onTabsInitialized={handleTabsInitialized}
              />
            </EnhancedErrorBoundary>
          )}
        </main>
      </div>
    </EnhancedErrorBoundary>
  );
}
