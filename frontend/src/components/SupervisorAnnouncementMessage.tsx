import React, { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  GitCompare, 
  PlayCircle, 
  ChevronDown, 
  ChevronRight, 
  Target,
  Clock,
  TrendingUp 
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { LLMGeneratedSequence } from '@/types/parallel';
import { TypedMarkdown } from '@/components/ui/typed-markdown';

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

const getSequenceIcon = (index: number) => {
  const icons = [Target, TrendingUp, Clock];
  const Icon = icons[index] || Target;
  return Icon;
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
  const Icon = getSequenceIcon(index);
  
  return (
    <div className="border border-neutral-600 rounded-lg p-3 bg-neutral-800/50 hover:bg-neutral-800 transition-colors">
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={onToggleExpanded}
      >
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <Badge variant="outline" className="text-xs flex-shrink-0">
            {index + 1}
          </Badge>
          <div className="flex items-center gap-2 min-w-0">
            <Icon className="w-4 h-4 text-blue-400 flex-shrink-0" />
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
      <p className="text-sm text-neutral-400 mt-2 line-clamp-2">
        {sequence.rationale}
      </p>
      
      {/* Expanded details */}
      {isExpanded && (
        <div className="mt-3 pt-3 border-t border-neutral-700 space-y-3 animate-in slide-in-from-top-1 duration-200">
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
          
          {/* Agent Pipeline */}
          <div>
            <h5 className="text-xs font-medium text-neutral-300 mb-2">Agent Pipeline:</h5>
            <div className="flex items-center gap-2 flex-wrap">
              {sequence.agent_names.map((agent, idx) => (
                <React.Fragment key={agent}>
                  <Badge variant="outline" className="text-xs">
                    {agent.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Badge>
                  {idx < sequence.agent_names.length - 1 && (
                    <span className="text-neutral-500 text-xs">→</span>
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
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
  researchQuery = "research request",
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
      "bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-blue-500/10 border border-blue-500/20 rounded-lg p-4 my-4",
      className
    )}>
      {/* Header */}
      <div className="flex items-start gap-3 mb-4">
        <div className="flex-shrink-0 p-2 bg-blue-500/20 rounded-lg">
          <GitCompare className="w-5 h-5 text-blue-400" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <h4 className="font-semibold text-blue-400 text-lg">
              Research Sequences Generated
            </h4>
            <Badge variant="outline" className="text-xs">
              {sequences.length} sequences
            </Badge>
          </div>
          
          {/* Announcement text with typing animation */}
          <div className="text-sm text-neutral-300 mb-3">
            <TypedMarkdown speed={25} hideCursor={true}>
              {`Based on your research request "${researchQuery}", I've generated ${sequences.length} optimized research sequences that will run in parallel to provide comprehensive coverage:`}
            </TypedMarkdown>
          </div>
          
          {/* Sequences overview */}
          <div className="grid gap-2 text-xs text-neutral-400 mb-3">
            <div className="flex items-center gap-4">
              <span>• <strong>Parallel Processing:</strong> All sequences run simultaneously</span>
              <span>• <strong>Specialized Agents:</strong> Domain-specific research focus</span>
            </div>
            <div className="flex items-center gap-4">
              <span>• <strong>Live Updates:</strong> Real-time progress in tabs below</span>
              <span>• <strong>Comprehensive Coverage:</strong> Multi-angle analysis</span>
            </div>
          </div>
        </div>
      </div>

      {/* Sequences List */}
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

      {/* Action Button */}
      <div className="flex items-center justify-between pt-3 border-t border-neutral-700">
        <div className="text-xs text-neutral-500">
          Click below to start parallel research execution and view live progress in tabs
        </div>
        <Button 
          onClick={handleInitializeTabs}
          disabled={hasInitialized || isLoading}
          className="bg-blue-600 hover:bg-blue-700 text-white text-sm px-4 py-2 h-auto"
        >
          {isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin mr-2" />
              Launching...
            </>
          ) : hasInitialized ? (
            <>
              <GitCompare className="w-4 h-4 mr-2" />
              Sequences Active
            </>
          ) : (
            <>
              <PlayCircle className="w-4 h-4 mr-2" />
              Launch Deep Research
            </>
          )}
        </Button>
      </div>

      {/* Progress indicator when initialized */}
      {hasInitialized && (
        <div className="mt-3 p-2 bg-green-500/10 border border-green-500/20 rounded text-center">
          <p className="text-sm text-green-400 font-medium">
            ✅ Parallel research sequences launched successfully! 
            <br />
            <span className="text-xs text-green-300">
              Switch between tabs below to monitor each sequence's progress
            </span>
          </p>
        </div>
      )}
    </div>
  );
};

export default SupervisorAnnouncementMessage;