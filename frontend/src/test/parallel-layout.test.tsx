/**
 * Parallel Layout Tests
 * 
 * Tests the horizontal side-by-side layout implementation
 * to ensure true parallel display without stacking.
 */

import React from 'react';
import { render, screen, within } from '@testing-library/react';
import { vi, describe, test, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import ParallelResearchInterface from '@/components/ParallelResearchInterface';
import { LLMGeneratedSequence, RoutedMessage } from '@/types/parallel';

// Mock ResizeObserver for testing
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock window.innerWidth for responsive testing
const mockWindowWidth = (width: number) => {
  Object.defineProperty(window, 'innerWidth', {
    writable: true,
    configurable: true,
    value: width,
  });
  window.dispatchEvent(new Event('resize'));
};

describe('ParallelResearchInterface Layout', () => {
  const mockSequences: LLMGeneratedSequence[] = [
    {
      sequence_id: 'seq_layout_1',
      sequence_name: 'Academic Research Sequence',
      agent_names: ['research_agent', 'analysis_agent'],
      rationale: 'Academic focus with scholarly sources and peer review analysis',
      research_focus: 'Academic literature and scholarly research',
      confidence_score: 0.88,
      approach_description: 'Comprehensive academic research methodology',
      expected_outcomes: ['Literature review', 'Academic analysis'],
      created_at: new Date().toISOString(),
    },
    {
      sequence_id: 'seq_layout_2', 
      sequence_name: 'Industry Analysis Sequence',
      agent_names: ['market_agent', 'technical_agent'],
      rationale: 'Industry-focused approach with market intelligence',
      research_focus: 'Market trends and industry analysis',
      confidence_score: 0.75,
      approach_description: 'Market research with competitive analysis',
      expected_outcomes: ['Market report', 'Industry insights'],
      created_at: new Date().toISOString(),
    },
    {
      sequence_id: 'seq_layout_3',
      sequence_name: 'Technical Deep Dive Sequence',
      agent_names: ['technical_agent', 'research_agent', 'synthesis_agent'],
      rationale: 'Technical implementation with architectural guidance',
      research_focus: 'Technical architecture and implementation',
      confidence_score: 0.92,
      approach_description: 'Deep technical analysis and design',
      expected_outcomes: ['Technical spec', 'Architecture design', 'Implementation guide'],
      created_at: new Date().toISOString(),
    }
  ];

  const mockParallelMessages: Record<string, RoutedMessage[]> = {
    'seq_layout_1': [
      {
        message_id: 'msg_layout_1_1',
        sequence_id: 'seq_layout_1',
        sequence_name: 'Academic Research Sequence',
        message_type: 'progress',
        timestamp: Date.now(),
        content: 'Starting academic research with literature review...',
        sequence_index: 0,
        routing_timestamp: Date.now(),
        current_agent: 'research_agent',
        agent_type: 'research'
      }
    ],
    'seq_layout_2': [
      {
        message_id: 'msg_layout_2_1',
        sequence_id: 'seq_layout_2',
        sequence_name: 'Industry Analysis Sequence',
        message_type: 'progress',
        timestamp: Date.now(),
        content: 'Initiating market intelligence gathering...',
        sequence_index: 1,
        routing_timestamp: Date.now(),
        current_agent: 'market_agent',
        agent_type: 'market'
      }
    ],
    'seq_layout_3': [
      {
        message_id: 'msg_layout_3_1',
        sequence_id: 'seq_layout_3',
        sequence_name: 'Technical Deep Dive Sequence',
        message_type: 'progress',
        timestamp: Date.now(),
        content: 'Beginning technical architecture analysis...',
        sequence_index: 2,
        routing_timestamp: Date.now(),
        current_agent: 'technical_agent',
        agent_type: 'technical'
      }
    ]
  };

  beforeEach(() => {
    // Default to desktop width
    mockWindowWidth(1280);
  });

  describe('Desktop Horizontal Layout', () => {
    test('renders true side-by-side 3-column layout', () => {
      const { container } = render(
        <div style={{ height: '600px', width: '100%' }}>
          <ParallelResearchInterface
            sequences={mockSequences}
            parallelMessages={mockParallelMessages}
            activeTabId={mockSequences[0].sequence_id}
            isLoading={false}
          />
        </div>
      );

      // Check main container has full height and width
      const mainContainer = container.querySelector('.h-full.w-full');
      expect(mainContainer).toBeInTheDocument();

      // Check flex container for horizontal layout
      const flexContainer = container.querySelector('.flex.h-full.divide-x');
      expect(flexContainer).toBeInTheDocument();

      // Verify 3 columns are present
      const columns = container.querySelectorAll('.flex-1.min-w-0.h-full.overflow-hidden');
      expect(columns).toHaveLength(3);

      // Check each column has proper width styling
      columns.forEach((column, index) => {
        expect(column).toHaveStyle({ width: '33.333%' });
      });
    });

    test('columns display correct sequence names', () => {
      render(
        <div style={{ height: '600px' }}>
          <ParallelResearchInterface
            sequences={mockSequences}
            parallelMessages={mockParallelMessages}
            activeTabId={mockSequences[0].sequence_id}
            isLoading={false}
          />
        </div>
      );

      // Check each sequence name appears in its column
      expect(screen.getByText('Academic Research Sequence')).toBeInTheDocument();
      expect(screen.getByText('Industry Analysis Sequence')).toBeInTheDocument();
      expect(screen.getByText('Technical Deep Dive Sequence')).toBeInTheDocument();
    });

    test('columns have independent scrolling', () => {
      const { container } = render(
        <div style={{ height: '600px' }}>
          <ParallelResearchInterface
            sequences={mockSequences}
            parallelMessages={mockParallelMessages}
            activeTabId={mockSequences[0].sequence_id}
            isLoading={false}
          />
        </div>
      );

      // Each column should have its own ScrollArea
      const scrollAreas = container.querySelectorAll('[data-radix-scroll-area-root]');
      expect(scrollAreas.length).toBeGreaterThanOrEqual(3);
    });

    test('displays confidence scores dynamically', () => {
      render(
        <ParallelResearchInterface
          sequences={mockSequences}
          parallelMessages={mockParallelMessages}
          activeTabId={mockSequences[0].sequence_id}
          isLoading={false}
        />
      );

      // Check dynamic confidence scores (not hard-coded 60%, 50%, 42%)
      expect(screen.getByText('88%')).toBeInTheDocument(); // 0.88 * 100
      expect(screen.getByText('75%')).toBeInTheDocument(); // 0.75 * 100  
      expect(screen.getByText('92%')).toBeInTheDocument(); // 0.92 * 100

      // Ensure hard-coded values are NOT present
      expect(screen.queryByText('60%')).not.toBeInTheDocument();
      expect(screen.queryByText('50%')).not.toBeInTheDocument();
      expect(screen.queryByText('42%')).not.toBeInTheDocument();
    });

    test('displays real conversations in each column', () => {
      render(
        <ParallelResearchInterface
          sequences={mockSequences}
          parallelMessages={mockParallelMessages}
          activeTabId={mockSequences[0].sequence_id}
          isLoading={false}
        />
      );

      // Check actual message content appears (not summaries)
      expect(screen.getByText(/Starting academic research with literature review/)).toBeInTheDocument();
      expect(screen.getByText(/Initiating market intelligence gathering/)).toBeInTheDocument();
      expect(screen.getByText(/Beginning technical architecture analysis/)).toBeInTheDocument();
    });

    test('handles dynamic agent names', () => {
      render(
        <ParallelResearchInterface
          sequences={mockSequences}
          parallelMessages={mockParallelMessages}
          activeTabId={mockSequences[0].sequence_id}
          isLoading={false}
        />
      );

      // Check dynamic agent names appear
      expect(screen.getByText('research agent')).toBeInTheDocument();
      expect(screen.getByText('market agent')).toBeInTheDocument();
      expect(screen.getByText('technical agent')).toBeInTheDocument();
    });

    test('columns maintain equal width with different content lengths', () => {
      // Create sequences with varying content lengths
      const varyingSequences = mockSequences.map((seq, index) => ({
        ...seq,
        approach_description: index === 0 
          ? 'Short description' 
          : index === 1 
            ? 'Medium length description with some additional details about the approach and methodology'
            : 'Very long description that goes into extensive detail about the comprehensive methodology, technical approaches, implementation strategies, expected outcomes, risk mitigation, quality assurance processes, and detailed analysis frameworks that will be employed throughout the research process'
      }));

      const { container } = render(
        <div style={{ height: '600px' }}>
          <ParallelResearchInterface
            sequences={varyingSequences}
            parallelMessages={mockParallelMessages}
            activeTabId={varyingSequences[0].sequence_id}
            isLoading={false}
          />
        </div>
      );

      // All columns should still have equal width
      const columns = container.querySelectorAll('.flex-1.min-w-0.h-full.overflow-hidden');
      columns.forEach(column => {
        expect(column).toHaveStyle({ width: '33.333%' });
      });
    });
  });

  describe('Mobile Responsive Layout', () => {
    test('switches to stacked tabs on mobile', () => {
      mockWindowWidth(800); // Mobile width

      const { container } = render(
        <ParallelResearchInterface
          sequences={mockSequences}
          parallelMessages={mockParallelMessages}
          activeTabId={mockSequences[0].sequence_id}
          isLoading={false}
        />
      );

      // Should show mobile tab selector
      const tabSelector = container.querySelector('.border-b.border-neutral-700.bg-neutral-900');
      expect(tabSelector).toBeInTheDocument();

      // Should show horizontal scrolling tabs
      const tabContainer = container.querySelector('.flex.overflow-x-auto');
      expect(tabContainer).toBeInTheDocument();
    });

    test('mobile tabs show sequence names', () => {
      mockWindowWidth(600);

      render(
        <ParallelResearchInterface
          sequences={mockSequences}
          parallelMessages={mockParallelMessages}
          activeTabId={mockSequences[0].sequence_id}
          isLoading={false}
        />
      );

      // Check tab names are visible in mobile view
      const tabButtons = screen.getAllByRole('button');
      const sequenceNames = mockSequences.map(s => s.sequence_name);
      
      sequenceNames.forEach(name => {
        expect(screen.getByText(name)).toBeInTheDocument();
      });
    });
  });

  describe('Layout Edge Cases', () => {
    test('handles single sequence', () => {
      const { container } = render(
        <ParallelResearchInterface
          sequences={[mockSequences[0]]}
          parallelMessages={{ [mockSequences[0].sequence_id]: mockParallelMessages[mockSequences[0].sequence_id] }}
          activeTabId={mockSequences[0].sequence_id}
          isLoading={false}
        />
      );

      // Should still use flex layout but with single column taking full width
      const columns = container.querySelectorAll('.flex-1.min-w-0.h-full.overflow-hidden');
      expect(columns).toHaveLength(1);
      expect(columns[0]).toHaveStyle({ width: '100%' });
    });

    test('handles two sequences', () => {
      const { container } = render(
        <ParallelResearchInterface
          sequences={mockSequences.slice(0, 2)}
          parallelMessages={{
            [mockSequences[0].sequence_id]: mockParallelMessages[mockSequences[0].sequence_id],
            [mockSequences[1].sequence_id]: mockParallelMessages[mockSequences[1].sequence_id]
          }}
          activeTabId={mockSequences[0].sequence_id}
          isLoading={false}
        />
      );

      const columns = container.querySelectorAll('.flex-1.min-w-0.h-full.overflow-hidden');
      expect(columns).toHaveLength(2);
      columns.forEach(column => {
        expect(column).toHaveStyle({ width: '50%' });
      });
    });

    test('limits to maximum 3 columns', () => {
      const fourSequences = [...mockSequences, {
        ...mockSequences[0],
        sequence_id: 'seq_layout_4',
        sequence_name: 'Fourth Sequence'
      }];

      const { container } = render(
        <ParallelResearchInterface
          sequences={fourSequences}
          parallelMessages={mockParallelMessages}
          activeTabId={fourSequences[0].sequence_id}
          isLoading={false}
        />
      );

      // Should only show first 3 sequences
      const columns = container.querySelectorAll('.flex-1.min-w-0.h-full.overflow-hidden');
      expect(columns).toHaveLength(3);
      
      // Fourth sequence should not be visible
      expect(screen.queryByText('Fourth Sequence')).not.toBeInTheDocument();
    });

    test('handles empty sequences', () => {
      const { container } = render(
        <ParallelResearchInterface
          sequences={[]}
          parallelMessages={{}}
          activeTabId=""
          isLoading={false}
        />
      );

      // Should show empty state
      expect(screen.getByText('No Sequences Available')).toBeInTheDocument();
      expect(screen.getByText('Sequences will appear here when parallel research begins')).toBeInTheDocument();
    });
  });

  describe('Loading States', () => {
    test('shows loading indicators in active columns', () => {
      render(
        <ParallelResearchInterface
          sequences={mockSequences}
          parallelMessages={mockParallelMessages}
          activeTabId={mockSequences[0].sequence_id}
          isLoading={true}
        />
      );

      // Should show loading animation in active column
      expect(screen.getByText('Generating response...')).toBeInTheDocument();
      
      // Check for animated dots
      const loadingDots = document.querySelectorAll('.animate-bounce');
      expect(loadingDots.length).toBeGreaterThan(0);
    });

    test('maintains layout structure during loading', () => {
      const { container } = render(
        <div style={{ height: '600px' }}>
          <ParallelResearchInterface
            sequences={mockSequences}
            parallelMessages={mockParallelMessages}
            activeTabId={mockSequences[0].sequence_id}
            isLoading={true}
          />
        </div>
      );

      // Layout structure should remain intact
      const flexContainer = container.querySelector('.flex.h-full.divide-x');
      expect(flexContainer).toBeInTheDocument();

      const columns = container.querySelectorAll('.flex-1.min-w-0.h-full.overflow-hidden');
      expect(columns).toHaveLength(3);
    });
  });
});