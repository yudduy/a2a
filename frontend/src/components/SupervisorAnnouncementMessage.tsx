import React, { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  ChevronDown, 
  ChevronRight, 
  CheckCircle2,
  ArrowRight,
  Zap,
  Activity
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { LLMGeneratedSequence } from '@/types/parallel';

// ========================================
// Component Interfaces
// ========================================

interface SupervisorAnnouncementMessageProps {
  sequences: LLMGeneratedSequence[];
  onTabsInitialized?: () => void;
  className?: string;
  isLoading?: boolean;
  researchQuery?: string;
}

interface SequencePreviewProps {
  sequence: LLMGeneratedSequence;
  index: number;
  isExpanded: boolean;
  onToggleExpanded: () => void;
}

// ========================================
// Utility Functions
// ========================================

const formatConfidenceScore = (score: number): string => {
  return `${Math.round(score * 100)}%`;
};


// ========================================
// SequencePreview Component
// ========================================

const SequencePreview: React.FC<SequencePreviewProps> = ({
  sequence,
  index,
  isExpanded,
  onToggleExpanded,
}) => {
  return (
    <div className="border border-neutral-700/50 rounded-lg p-3 bg-neutral-800/30 hover:bg-neutral-800/50 transition-colors">
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={onToggleExpanded}
      >
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <Badge variant="outline" className="text-xs flex-shrink-0">
            {index + 1}
          </Badge>
          <div className="min-w-0">
            <span className="font-medium text-neutral-200 truncate">
              {sequence.sequence_name}
            </span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <Badge variant="secondary" className="text-xs">
              {formatConfidenceScore(sequence.confidence_score)}
            </Badge>
            <Badge variant="outline" className="text-xs">
              {sequence.agent_names.length} agents
            </Badge>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-neutral-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-neutral-400" />
          )}
        </div>
      </div>
      
      {/* Brief rationale - always visible */}
      <p className="text-sm text-neutral-400 mt-2">
        {sequence.rationale}
      </p>
      
      {/* Expanded details */}
      {isExpanded && (
        <div className="mt-3 pt-3 border-t border-neutral-700/50 space-y-3">
          {/* Research Focus */}
          <div>
            <h5 className="text-xs font-medium text-neutral-300 mb-1">Research Focus:</h5>
            <p className="text-sm text-neutral-200">{sequence.research_focus}</p>
          </div>
          
          {/* Approach Description */}
          {sequence.approach_description && (
            <div>
              <h5 className="text-xs font-medium text-neutral-300 mb-1">Approach:</h5>
              <p className="text-sm text-neutral-200">{sequence.approach_description}</p>
            </div>
          )}
          
          {/* Agents */}
          <div>
            <h5 className="text-xs font-medium text-neutral-300 mb-1">Agent Pipeline:</h5>
            <div className="flex flex-wrap gap-1">
              {sequence.agent_names.map((agent, idx) => (
                <React.Fragment key={agent}>
                  <Badge variant="outline" className="text-xs">
                    {agent.replace('_', ' ')}
                  </Badge>
                  {idx < sequence.agent_names.length - 1 && (
                    <ArrowRight className="w-3 h-3 text-neutral-500 mt-1" />
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
          
          {/* Expected Outcomes */}
          {sequence.expected_outcomes && sequence.expected_outcomes.length > 0 && (
            <div>
              <h5 className="text-xs font-medium text-neutral-300 mb-1">Expected Outcomes:</h5>
              <ul className="text-sm text-neutral-200 list-disc list-inside space-y-1">
                {sequence.expected_outcomes.map((outcome, idx) => (
                  <li key={idx}>{outcome}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ========================================
// Main SupervisorAnnouncementMessage Component
// ========================================

const SupervisorAnnouncementMessage: React.FC<SupervisorAnnouncementMessageProps> = ({
  sequences,
  onTabsInitialized,
  className,
  isLoading = false,
}) => {
  const [expandedSequences, setExpandedSequences] = useState<Set<string>>(new Set());
  const [hasInitialized, setHasInitialized] = useState(false);

  const toggleSequenceExpanded = (sequenceId: string) => {
    setExpandedSequences(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sequenceId)) {
        newSet.delete(sequenceId);
      } else {
        newSet.add(sequenceId);
      }
      return newSet;
    });
  };

  const handleInitializeTabs = () => {
    setHasInitialized(true);
    onTabsInitialized?.();
  };

  if (sequences.length === 0) {
    return null;
  }

  return (
    <div className={cn(
      "flex items-start gap-3 w-full max-w-none",
      className
    )}>
      <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0 mt-1">
        <Activity className="w-4 h-4 text-green-400" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="bg-neutral-800 rounded-2xl p-4 shadow-sm max-w-none">
          {/* Clean minimal header */}
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle2 className="w-4 h-4 text-green-400" />
              <span className="text-sm font-medium text-neutral-200">Research sequences generated</span>
            </div>
            <p className="text-sm text-neutral-300 leading-relaxed">
              I've generated {sequences.length} specialized research sequences for parallel execution. Each sequence will run independently with different agent teams and approaches.
            </p>
          </div>

          {/* Clean sequences list */}
          <div className="space-y-3 mb-4">
            {sequences.map((sequence, index) => (
              <SequencePreview
                key={sequence.sequence_id}
                sequence={sequence}
                index={index}
                isExpanded={expandedSequences.has(sequence.sequence_id)}
                onToggleExpanded={() => toggleSequenceExpanded(sequence.sequence_id)}
              />
            ))}
          </div>

          {/* Clean action section */}
          <div className="flex items-center justify-between pt-3 border-t border-neutral-700">
            <div className="flex-1">
              <p className="text-sm text-neutral-300 mb-1">Ready to launch parallel research</p>
              <p className="text-xs text-neutral-500">
                Click below to start all sequences simultaneously in side-by-side view
              </p>
            </div>
            <Button 
              onClick={handleInitializeTabs}
              disabled={hasInitialized || isLoading}
              className="bg-blue-600 hover:bg-blue-700 text-white text-sm px-4 py-2 h-auto font-medium transition-colors"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin mr-2" />
                  Launching...
                </>
              ) : hasInitialized ? (
                <>
                  <CheckCircle2 className="w-4 h-4 mr-2" />
                  Active
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Launch Research
                </>
              )}
            </Button>
          </div>

          {/* Clean progress indicator when initialized */}
          {hasInitialized && (
            <div className="mt-4 p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-400" />
                <div>
                  <p className="text-sm font-medium text-green-400">
                    Parallel research active
                  </p>
                  <p className="text-xs text-green-300 mt-1">
                    {sequences.length} sequences running in side-by-side view below
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SupervisorAnnouncementMessage;