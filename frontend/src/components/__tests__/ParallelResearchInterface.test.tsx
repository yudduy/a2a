/**
 * Test suite for ParallelResearchInterface
 * Basic tests to ensure the component renders correctly
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ParallelResearchInterface from '../ParallelResearchInterface';
import { LLMGeneratedSequence } from '@/types/parallel';

const mockSequences: LLMGeneratedSequence[] = [
  {
    sequence_id: 'test-1',
    sequence_name: 'Test Sequence 1',
    agent_names: ['research_agent'],
    rationale: 'Test rationale',
    research_focus: 'Test focus',
    confidence_score: 0.9,
    approach_description: 'Test approach',
    expected_outcomes: ['Test outcome'],
    created_at: new Date().toISOString(),
  },
  {
    sequence_id: 'test-2',
    sequence_name: 'Test Sequence 2',
    agent_names: ['analysis_agent'],
    rationale: 'Test rationale 2',
    research_focus: 'Test focus 2',
    confidence_score: 0.8,
    approach_description: 'Test approach 2',
    expected_outcomes: ['Test outcome 2'],
    created_at: new Date().toISOString(),
  },
];

describe('ParallelResearchInterface', () => {
  it('renders without crashing', () => {
    render(
      <ParallelResearchInterface
        sequences={mockSequences}
        parallelMessages={{}}
        isLoading={false}
      />
    );
    
    // Should render the sequence names
    expect(screen.getByText('Test Sequence 1')).toBeInTheDocument();
    expect(screen.getByText('Test Sequence 2')).toBeInTheDocument();
  });

  it('shows empty state when no sequences provided', () => {
    render(
      <ParallelResearchInterface
        sequences={[]}
        parallelMessages={{}}
        isLoading={false}
      />
    );
    
    expect(screen.getByText('No Sequences Available')).toBeInTheDocument();
  });

  it('displays loading state correctly', () => {
    render(
      <ParallelResearchInterface
        sequences={mockSequences}
        parallelMessages={{}}
        isLoading={true}
      />
    );
    
    // Should still show sequence names even when loading
    expect(screen.getByText('Test Sequence 1')).toBeInTheDocument();
  });
});