/**
 * Research Components - Production-ready parallel chat grid UI
 * 
 * This module exports all components for the parallel research interface,
 * providing a complete 3-column chat grid for real-time sequence comparison.
 * 
 * Components:
 * - ParallelChatGrid: Main 3-column layout with responsive design
 * - SequenceChat: Individual sequence chat column with real-time streaming
 * - QueryAnalyzer: Shows sequence selection reasoning and analysis
 * - LiveMetricsBar: Real-time metrics and performance tracking
 * - ComparisonSummary: Final winner analysis and comparison results
 * 
 * Features:
 * - Real-time WebSocket message streaming
 * - Responsive design (desktop 3-column, mobile tabbed)
 * - Professional ShadCN UI components
 * - Live metrics and progress tracking
 * - Agent progression indicators
 * - Comprehensive comparison analysis
 * - Export functionality for results
 * - Accessibility compliance
 * - TypeScript support with full type safety
 */

// Main components
export { ParallelResearchInterface } from './ParallelResearchInterface';
export { ParallelChatGrid } from './ParallelChatGrid';
export { SequenceChat } from './SequenceChat';
export { QueryAnalyzer } from './QueryAnalyzer';
export { LiveMetricsBar } from './LiveMetricsBar';
export { ComparisonSummary } from './ComparisonSummary';

// Default export for convenience
export { ParallelChatGrid as default } from './ParallelChatGrid';

// Re-export types for convenience when using these components
export type {
  SequenceState,
  SequenceStrategy,
  AgentType,
  ParallelSequencesState,
  RealTimeMetrics,
  UseParallelSequencesReturn,
  ConnectionState,
  RoutedMessage,
} from '@/types/parallel';