# WebSocket Flows and Component Mapping

## Current WebSocket Implementation Analysis

### Main WebSocket Flow (App.tsx)

#### Stream Configuration
```typescript
const thread = useStream<{
  messages: Message[];
  initial_search_query_count: number;
  max_research_loops: number;
  reasoning_model: string;
}>({
  apiUrl: import.meta.env.DEV ? 'http://localhost:2024' : 'http://localhost:8123',
  assistantId: selectedAgentId,
  messagesKey: 'messages',
  onFinish: (event: unknown) => { console.log(event); },
  onUpdateEvent: (event: Record<string, unknown>) => { /* Complex event processing */ }
});
```

#### Event Processing Pipeline
**Current Event Types Handled**:

1. **Deep Researcher Events**:
   - `generate_query`: Search query generation
   - `web_research`: Source gathering with metadata
   - `reflection`: Research sufficiency assessment
   - `finalize_answer`: Final synthesis trigger

2. **Delegation Pattern Events**:
   - Processed via `processDelegationEvent(event, selectedAgentId)`
   - Generates `DelegationProcessedEvent[]` with sequence tracking

3. **Universal Events**:
   - `tool_call_chunks`: Real-time tool execution display

#### State Management Flow
```
Raw WebSocket Event → Event Type Detection → Agent-Specific Processing → State Update → UI Refresh
```

**State Variables**:
- `processedEventsTimeline`: Live activity events
- `historicalActivities`: Completed research activities by message ID
- `delegationEvents`: Delegation-specific events
- `selectedAgentId`: Current agent context

### Multi-Sequence Transformation Plan

#### Target Architecture
```typescript
// Multiple concurrent streams
const sequences = {
  'theory-first': useStream({ assistantId: 'theory_first_sequence', ... }),
  'market-first': useStream({ assistantId: 'market_first_sequence', ... }),
  'future-back': useStream({ assistantId: 'future_back_sequence', ... })
};

// Event routing by sequence
const routeEvent = (event: Record<string, unknown>, sequenceId: string) => {
  // Route to appropriate sequence state container
};
```

#### Event Processing Transformation
**Current**:
```typescript
onUpdateEvent: (event: Record<string, unknown>) => {
  // Single agent processing
  if (selectedAgentId === AgentId.DEEP_RESEARCHER) { /* ... */ }
  else if (isDelegationAgent) { /* ... */ }
}
```

**Target**:
```typescript
onUpdateEvent: (event: Record<string, unknown>, sequenceId: SequenceStrategy) => {
  // Multi-sequence event routing
  const sequenceProcessor = getSequenceProcessor(sequenceId);
  const processedEvent = sequenceProcessor.processEvent(event);
  updateSequenceState(sequenceId, processedEvent);
}
```

## Component Removal and Preservation Strategy

### Components to Remove Completely

#### 1. WelcomeScreen.tsx
**Location**: `/frontend/src/components/WelcomeScreen.tsx`
**Reason**: Agent dropdown paradigm being replaced
**Current Function**: 
- Agent selection dropdown
- Welcome message display
- Initial research input form

**Replacement**: `MultiSequenceLauncher.tsx`
```typescript
interface MultiSequenceLauncherProps {
  onStartComparison: (strategies: SequenceStrategy[], query: string) => void;
  availableStrategies: SequenceStrategy[];
  defaultStrategies: SequenceStrategy[];
}
```

#### 2. Agent Selection Logic in InputForm.tsx
**Location**: `/frontend/src/components/InputForm.tsx` (lines 167-193)
**Reason**: Single agent selection being replaced
**Current Function**:
- Dropdown for agent selection
- Agent-specific configuration (effort levels)
- Model selection per agent

**Replacement**: `GlobalResearchInput.tsx`
```typescript
interface GlobalResearchInputProps {
  onSubmit: (query: string, targetSequences: SequenceStrategy[]) => void;
  activeSequences: Set<SequenceStrategy>;
  onSequenceToggle: (sequence: SequenceStrategy) => void;
  globalSettings: ResearchSettings;
}
```

### Components to Transform

#### 1. App.tsx - Main Application Logic
**Current Structure**:
```typescript
function App() {
  const [selectedAgentId, setSelectedAgentId] = useState(DEFAULT_AGENT);
  const thread = useStream({ assistantId: selectedAgentId, ... });
  
  // Single agent state management
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<ProcessedEvent[]>([]);
  const [delegationEvents, setDelegationEvents] = useState<ProcessedEvent[]>([]);
  
  // Single chat interface rendering
  return isDelegationAgent ? <DelegationDashboard /> : <ChatMessagesView />;
}
```

**Target Structure**:
```typescript
function App() {
  const sequences = useMultiSequenceStreams();
  const [activeSequences, setActiveSequences] = useState<Set<SequenceStrategy>>(new Set());
  
  // Multi-sequence state management
  const sequenceState = useSequenceState();
  const metrics = useSequenceMetrics(sequenceState);
  
  // Multi-chat interface rendering
  return <MultiSequenceLayout sequences={sequences} metrics={metrics} />;
}
```

#### 2. ChatMessagesView.tsx - Core Chat Component
**Transformation**: From single chat to sequence-specific chat
**Current**: Global chat view with agent context
**Target**: Instance-based with sequence identification

**New Component Structure**:
```typescript
// New: SequenceChatView.tsx
export function SequenceChatView({ 
  sequenceStrategy, 
  messages, 
  events, 
  streamInstance,
  isActive 
}: SequenceChatViewProps) {
  // Sequence-specific chat rendering
}

// Modified: ChatMessagesView.tsx becomes container
export function ChatMessagesView({ 
  sequences 
}: MultiSequenceChatProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {sequences.map(sequence => (
        <SequenceChatView key={sequence.strategy} {...sequence} />
      ))}
    </div>
  );
}
```

#### 3. DelegationDashboard.tsx - Multi-Sequence UI
**Current**: Tab-based sequence comparison
**Target**: Parallel chat columns with live comparison

**Transformation Plan**:
- **Keep**: Metrics display logic, sequence event processing
- **Modify**: Layout from tabs to columns
- **Enhance**: Real-time comparison visualization

### Components to Preserve and Enhance

#### 1. ActivityTimeline.tsx
**Preservation Reason**: Core functionality needed for each sequence
**Enhancement**: Add sequence identification and multi-timeline support

**Current Interface**:
```typescript
interface ActivityTimelineProps {
  processedEvents: ProcessedEvent[];
  isLoading: boolean;
}
```

**Enhanced Interface**:
```typescript
interface ActivityTimelineProps {
  processedEvents: ProcessedEvent[];
  isLoading: boolean;
  sequenceId?: SequenceStrategy;
  showSequenceLabel?: boolean;
  compact?: boolean; // For multi-column layout
}
```

#### 2. ToolMessageDisplay.tsx
**Preservation Reason**: Universal tool display component
**No Changes Required**: Already modular and reusable

#### 3. All UI Components (/components/ui/)
**Preservation Reason**: Reusable design system components
**Usage**: Used across all new multi-sequence components

### New Components to Create

#### 1. Multi-Sequence Management
```
/frontend/src/components/multi-sequence/
├── MultiSequenceLauncher.tsx      # Replaces WelcomeScreen
├── MultiSequenceLayout.tsx        # Main layout container
├── SequenceChatView.tsx           # Individual sequence chat
├── GlobalResearchInput.tsx        # Replaces InputForm agent selection
└── SequenceController.tsx         # Start/stop/reset sequences
```

#### 2. Enhanced Metrics and Comparison
```
/frontend/src/components/metrics/
├── LiveSequenceMetrics.tsx        # Real-time metrics per sequence
├── CrossSequenceComparison.tsx    # Side-by-side comparison
└── SequenceProgressIndicator.tsx  # Visual progress tracking
```

#### 3. Custom Hooks
```
/frontend/src/hooks/
├── useMultiSequenceStreams.ts     # WebSocket management
├── useSequenceState.ts            # State management
├── useSequenceMetrics.ts          # Metrics calculation
└── useSequenceEvents.ts           # Event processing
```

## Migration Implementation Strategy

### Phase 1: Parallel Development
1. **Keep Existing**: Maintain current components during development
2. **Feature Flag**: `ENABLE_MULTI_SEQUENCE` environment variable
3. **Gradual Testing**: Test new components alongside existing ones

### Phase 2: Component Replacement
1. **Replace WelcomeScreen**: With MultiSequenceLauncher
2. **Transform InputForm**: Extract reusable parts, replace agent selection
3. **Modify App.tsx**: Switch between old and new architecture based on feature flag

### Phase 3: Clean-up
1. **Remove Deprecated**: Delete old components after verification
2. **Consolidate**: Merge reusable logic into shared utilities
3. **Documentation**: Update component documentation

## WebSocket Architecture Benefits

### Current Architecture Strengths
- **Proven Stability**: Existing useStream hook handles reconnection
- **Event Processing**: Sophisticated event routing already implemented
- **Error Handling**: Built-in error recovery and loading states

### Multi-Sequence Advantages
- **Parallel Processing**: Multiple research strategies simultaneously
- **Independent State**: Each sequence maintains isolated state
- **Scalable**: Can add more sequence strategies without architectural changes
- **Comparative Analysis**: Real-time comparison of research approaches

This mapping provides the complete picture for transforming the single-agent interface into a sophisticated multi-sequence parallel research platform.