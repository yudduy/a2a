# Multi-Sequence Research Interface - Implementation Roadmap

## Project Overview
Transform the current single-agent dropdown interface into a dynamic multi-sequence parallel chat interface that enables real-time comparison of multiple research strategies running simultaneously.

## Implementation Phases

### Phase 1: Core Architecture Transformation
**Duration**: 2-3 days
**Objective**: Establish multi-stream WebSocket architecture and basic parallel chat structure

#### 1.1 Multi-Stream WebSocket Architecture
**Files to Modify**: 
- `/frontend/src/App.tsx`

**Changes Required**:
```typescript
// Current: Single useStream instance
const thread = useStream({ assistantId: selectedAgentId, ... });

// Target: Multiple useStream instances managed by sequence ID
const sequences = {
  'theory-first': useStream({ assistantId: 'theory_first_sequence', ... }),
  'market-first': useStream({ assistantId: 'market_first_sequence', ... }),
  'future-back': useStream({ assistantId: 'future_back_sequence', ... })
};
```

**Implementation Steps**:
1. Create `useMultiSequenceStreams` custom hook
2. Implement sequence lifecycle management (start/stop/reset)
3. Add sequence-specific event routing
4. Maintain backward compatibility with single-agent mode

#### 1.2 State Management Restructuring
**New State Structure**:
```typescript
interface MultiSequenceState {
  activeSequences: Set<SequenceStrategy>;
  sequenceStreams: Record<SequenceStrategy, StreamInstance>;
  sequenceEvents: Record<SequenceStrategy, ProcessedEvent[]>;
  sequenceMessages: Record<SequenceStrategy, Message[]>;
  globalInput: string;
  isComparison: boolean;
}
```

#### 1.3 Event Processing Pipeline
**Create New Files**:
- `/frontend/src/hooks/useMultiSequenceStreams.ts`
- `/frontend/src/utils/sequenceEventRouter.ts`

**Event Flow**:
```
Raw Event → Sequence Identification → Event Processing → UI Update
```

### Phase 2: Component Modularization
**Duration**: 2-3 days  
**Objective**: Transform existing components for multi-instance usage

#### 2.1 ChatMessagesView Modularization
**Current**: Single global chat view
**Target**: Instance-based chat view per sequence

**New Component Structure**:
```typescript
// New: SequenceChatView component
interface SequenceChatViewProps {
  sequenceId: SequenceStrategy;
  messages: Message[];
  events: ProcessedEvent[];
  isLoading: boolean;
  streamInstance: StreamInstance;
}

// Modified: ChatMessagesView becomes container
interface ChatMessagesViewProps {
  sequences: SequenceChatViewProps[];
  layout: 'parallel' | 'tabbed';
}
```

**Files to Create**:
- `/frontend/src/components/SequenceChatView.tsx`
- `/frontend/src/components/MultiSequenceContainer.tsx`

#### 2.2 Input Form Transformation
**Current**: Agent selection dropdown
**Target**: Global input with sequence targeting

**New Component**: `GlobalResearchInput.tsx`
```typescript
interface GlobalResearchInputProps {
  onSubmit: (query: string, targetSequences: SequenceStrategy[]) => void;
  activeSequences: Set<SequenceStrategy>;
  onSequenceToggle: (sequence: SequenceStrategy) => void;
}
```

#### 2.3 Activity Timeline Enhancement
**Enhancement**: Support multiple timelines with sequence identification

**Modified Component**: `ActivityTimeline.tsx`
```typescript
interface ActivityTimelineProps {
  processedEvents: ProcessedEvent[];
  sequenceId?: SequenceStrategy;
  isLoading: boolean;
  showSequenceLabel?: boolean;
}
```

### Phase 3: Layout and UI Implementation
**Duration**: 1-2 days
**Objective**: Create the multi-panel parallel chat interface

#### 3.1 Layout Architecture
**Design**: Three-column responsive layout
```
+------------------+------------------+------------------+
|   Theory First   |   Market First   |   Future Back    |
|   Sequence       |   Sequence       |   Sequence       |
|                  |                  |                  |
|  [Chat Area]     |  [Chat Area]     |  [Chat Area]     |
|  [Timeline]      |  [Timeline]      |  [Timeline]      |
|  [Metrics]       |  [Metrics]       |  [Metrics]       |
+------------------+------------------+------------------+
|           Global Research Input                        |
+-----------------------------------------------------+
```

#### 3.2 Responsive Behavior
- **Desktop**: 3-column side-by-side
- **Tablet**: 2-column with sequence selector
- **Mobile**: Single column with sequence tabs

#### 3.3 New Layout Components
**Files to Create**:
- `/frontend/src/components/layout/MultiSequenceLayout.tsx`
- `/frontend/src/components/layout/SequenceColumn.tsx`
- `/frontend/src/components/layout/ResponsiveSequenceContainer.tsx`

### Phase 4: Real-time Synchronization
**Duration**: 1-2 days
**Objective**: Implement cross-sequence coordination and metrics

#### 4.1 Cross-Sequence Metrics
**Integration**: Connect to existing `MetricsCalculator` from backend
```typescript
interface SequenceMetrics {
  productivity: number;
  insightGeneration: number;
  researchDepth: number;
  executionTime: number;
}

interface CrossSequenceMetrics {
  sequences: Record<SequenceStrategy, SequenceMetrics>;
  comparison: SequenceComparison;
  variance: number;
}
```

#### 4.2 Real-time Updates
**Features**:
- Live metrics updates during execution
- Progress indicators per sequence
- Completion status tracking
- Performance comparison visualization

#### 4.3 Synchronization Components
**Files to Create**:
- `/frontend/src/components/metrics/LiveMetricsPanel.tsx`
- `/frontend/src/components/metrics/SequenceProgressIndicator.tsx`
- `/frontend/src/hooks/useCrossSequenceMetrics.ts`

### Phase 5: Enhanced Features and Polish
**Duration**: 1-2 days
**Objective**: Add advanced features and polish the interface

#### 5.1 Advanced Features
- **Sequence Pausing/Resuming**: Independent control per sequence
- **Result Export**: Export comparison results
- **Configuration Presets**: Saved sequence configurations
- **Research Templates**: Pre-defined research scenarios

#### 5.2 User Experience Enhancements
- **Loading States**: Sophisticated loading indicators
- **Error Handling**: Graceful error recovery per sequence
- **Keyboard Shortcuts**: Quick sequence navigation
- **Accessibility**: Full keyboard navigation and screen reader support

## Technical Implementation Details

### Backend Integration Points

#### Existing Backend API Usage
The current backend already supports the required functionality:

1. **SequenceOptimizationEngine.compare_sequences()**: For parallel execution
2. **Multiple AgentIds**: `THEORY_FIRST_SEQUENCE`, `MARKET_FIRST_SEQUENCE`, `FUTURE_BACK_SEQUENCE`
3. **WebSocket Endpoints**: LangGraph SDK handles multiple concurrent connections

#### No Backend Changes Required
The existing backend architecture fully supports the new interface:
- Sequence patterns already defined
- Agent orchestration implemented
- Metrics calculation available
- WebSocket infrastructure ready

### Custom Hooks Architecture

#### 1. useMultiSequenceStreams
```typescript
interface UseMultiSequenceStreamsReturn {
  sequences: Record<SequenceStrategy, StreamInstance>;
  startSequence: (strategy: SequenceStrategy, query: string) => void;
  stopSequence: (strategy: SequenceStrategy) => void;
  resetSequence: (strategy: SequenceStrategy) => void;
  activeSequences: Set<SequenceStrategy>;
}
```

#### 2. useSequenceMetrics
```typescript
interface UseSequenceMetricsReturn {
  metrics: CrossSequenceMetrics;
  isCalculating: boolean;
  lastUpdate: Date;
}
```

#### 3. useSequenceEvents
```typescript
interface UseSequenceEventsReturn {
  events: Record<SequenceStrategy, ProcessedEvent[]>;
  addEvent: (sequence: SequenceStrategy, event: ProcessedEvent) => void;
  clearEvents: (sequence: SequenceStrategy) => void;
}
```

### State Management Strategy

#### Context-Based Architecture
```typescript
// SequenceContext.tsx
interface SequenceContextValue {
  state: MultiSequenceState;
  actions: {
    startSequence: (strategy: SequenceStrategy, query: string) => void;
    stopSequence: (strategy: SequenceStrategy) => void;
    updateSequenceSettings: (strategy: SequenceStrategy, settings: any) => void;
    resetAll: () => void;
  };
}
```

#### Local State Distribution
- **Global State**: Active sequences, global input, comparison mode
- **Sequence State**: Messages, events, loading status per sequence
- **UI State**: Layout preferences, expanded sections, scroll positions

### Migration Strategy

#### Backward Compatibility
Maintain existing single-agent functionality during transition:

1. **Feature Flag**: `ENABLE_MULTI_SEQUENCE` environment variable
2. **Gradual Rollout**: Single-agent mode remains default initially
3. **Fallback Behavior**: Graceful degradation if multi-sequence fails

#### Migration Steps
1. **Week 1**: Core architecture (Phase 1)
2. **Week 2**: Component transformation (Phase 2)
3. **Week 3**: UI implementation (Phase 3)
4. **Week 4**: Synchronization and metrics (Phase 4)
5. **Week 5**: Polish and testing (Phase 5)

## Success Metrics

### Technical Metrics
- **Performance**: <100ms sequence switching time
- **Reliability**: 99%+ WebSocket connection stability
- **Scalability**: Support 3+ concurrent sequences without degradation

### User Experience Metrics  
- **Usability**: Users can start parallel research within 30 seconds
- **Efficiency**: 40%+ faster research completion vs sequential approach
- **Satisfaction**: Intuitive multi-sequence navigation and comparison

### Quality Metrics
- **Test Coverage**: 90%+ for new components
- **Accessibility**: WCAG 2.1 AA compliance
- **Browser Support**: Chrome, Firefox, Safari, Edge latest versions

This roadmap provides a comprehensive path to transform the Open Deep Research interface into a sophisticated multi-sequence parallel research platform while leveraging the existing robust backend infrastructure.