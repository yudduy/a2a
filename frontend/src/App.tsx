import { useState, useCallback, useRef, useEffect } from 'react';
import { useStream } from '@langchain/langgraph-sdk/react';
import { WelcomeScreen } from '@/components/WelcomeScreen';
import { ChatInterface } from '@/components/ChatInterface';
import { useParallelSequences } from '@/hooks/useParallelSequences';
import { Message } from '@langchain/langgraph-sdk';
import { ProcessedEvent } from '@/components/ActivityTimeline';
import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';
import IntegrationTest from '@/test/components/IntegrationTest';
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
  // Test mode detection - simple URL-based routing
  const isIntegrationTestMode = window.location.search.includes('test=integration');
  
  // Chat state
  const [localMessages, setLocalMessages] = useState<(Message & { _locallyAdded?: boolean })[]>([]);
  const [liveActivityEvents, setLiveActivityEvents] = useState<ProcessedEvent[]>([]);
  const [historicalActivities, setHistoricalActivities] = useState<Record<string, ProcessedEvent[]>>({});
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
  
  // Parallel sequences for research streams
  const { 
    start: startParallelResearch,
    stop: stopParallelResearch,
    isLoading: isParallelLoading,
    changeActiveSequence
  } = useParallelSequences();

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
        
        // Add activity event for sequence generation
        const sequenceEvent: ProcessedEvent = {
          title: '🎯 Sequences Generated',
          data: `Supervisor generated ${llmSequences.length} research sequences - ready for parallel execution...`,
          timestamp: Date.now()
        };
        setLiveActivityEvents(prev => [...prev, sequenceEvent]);
        
        return; // Don't process as regular activity event
      }
      
      // Check if this is a message for parallel tabs
      if (data && typeof data === 'object' && data.sequence_id && parallelTabsState.isActive) {
        handleParallelMessage(data);
        return; // Don't process as regular activity event when routing to tabs
      }

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
  }, [mapBackendEventToUIState, localMessages, startParallelResearch, parallelTabsState.isActive, handleParallelMessage]);

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
          
          const sequenceEvent: ProcessedEvent = {
            title: '🎯 Sequences Generated',
            data: `LLM generated ${llmSequences.length} research sequences - ready for parallel execution...`,
            timestamp: Date.now()
          };
          setLiveActivityEvents(prev => [...prev, sequenceEvent]);
          
          return;
        }
        
        // Check if this is a message for parallel tabs
        if (data.data && data.data.sequence_id && parallelTabsState.isActive) {
          handleParallelMessage(data.data);
          return; // Don't process as regular activity event when routing to tabs
        }

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
  }, [localMessages, startParallelResearch, parallelTabsState.isActive, handleParallelMessage]);

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
    if (import.meta.env.DEV) {
      console.log('Stream finished with state:', state);
    }
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
      
      // Add activity event for starting research - backend supervisor will automatically decide processing type
      const startEvent: ProcessedEvent = {
        title: '🚀 Initializing Research',
        data: 'Submitting query to research supervisor - will automatically generate parallel sequences if appropriate...',
        timestamp: Date.now()
      };
      setLiveActivityEvents([startEvent]);
      
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
    setLiveActivityEvents([]);
  }, [streamStop, stopParallelResearch]);

  // Handle parallel tabs initialization
  const handleTabsInitialized = useCallback(() => {
    if (parallelTabsState.sequences.length > 0) {
      setParallelTabsState(prev => ({
        ...prev,
        isActive: true,
      }));
      
      // Extract the original query from recent messages
      const lastHumanMessage = localMessages.filter(m => m.type === 'human').pop();
      const researchQuery = typeof lastHumanMessage?.content === 'string' 
        ? lastHumanMessage.content 
        : 'Parallel research request';
      
      // Start parallel research with the announced sequences
      startParallelResearch(researchQuery, parallelTabsState.sequences);
    }
  }, [parallelTabsState.sequences, localMessages, startParallelResearch]);

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
    setLiveActivityEvents([]);
    setHistoricalActivities({});
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

  // Render test mode if requested
  if (isIntegrationTestMode) {
    return <IntegrationTest />;
  }

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
                liveActivityEvents={liveActivityEvents}
                historicalActivities={historicalActivities}
                onReset={handleReset}
                // New parallel tabs props
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
