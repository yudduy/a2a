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
  TrendingUp,
  Brain,
  Users,
  Lightbulb,
  CheckCircle2,
  ArrowRight,
  Zap
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

interface AgentRationaleProps {
  agentName: string;
  expertise: string[];
  selectionReason: string;
  expectedContribution: string;
  confidenceScore: number;
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

const getAgentExpertiseColor = (agentName: string): string => {
  const agentColorMap: Record<string, string> = {
    'research_agent': 'text-green-400 bg-green-400/10',
    'analysis_agent': 'text-amber-400 bg-amber-400/10',
    'market_agent': 'text-purple-400 bg-purple-400/10',
    'technical_agent': 'text-teal-400 bg-teal-400/10',
    'synthesis_agent': 'text-rose-400 bg-rose-400/10',
    'academic_agent': 'text-green-400 bg-green-400/10',
    'industry_agent': 'text-blue-400 bg-blue-400/10'
  };
  return agentColorMap[agentName] || 'text-gray-400 bg-gray-400/10';
};

const getAgentIcon = (agentName: string) => {
  const agentIconMap: Record<string, any> = {
    'research_agent': Brain,
    'analysis_agent': Target,
    'market_agent': TrendingUp,
    'technical_agent': Zap,
    'synthesis_agent': Lightbulb,
    'academic_agent': Brain,
    'industry_agent': Users
  };
  return agentIconMap[agentName] || Brain;
};

const generateAgentRationale = (agentName: string, sequenceContext: string): AgentRationaleProps => {
  // Enhanced agent descriptions based on context
  const agentProfiles: Record<string, Omit<AgentRationaleProps, 'agentName'>> = {
    'research_agent': {
      expertise: ['Academic Research', 'Literature Review', 'Data Collection'],
      selectionReason: 'Selected for comprehensive literature analysis and foundational research depth',
      expectedContribution: 'Provide scholarly foundation and evidence-based insights for the research sequence',
      confidenceScore: 0.92
    },
    'analysis_agent': {
      expertise: ['Data Analysis', 'Pattern Recognition', 'Statistical Modeling'],
      selectionReason: 'Chosen for analytical depth and data-driven insight generation',
      expectedContribution: 'Transform raw research into actionable insights through systematic analysis',
      confidenceScore: 0.88
    },
    'market_agent': {
      expertise: ['Market Intelligence', 'Competitive Analysis', 'Business Strategy'],
      selectionReason: 'Essential for commercial viability assessment and market context',
      expectedContribution: 'Provide market perspective and strategic business implications',
      confidenceScore: 0.85
    },
    'technical_agent': {
      expertise: ['Technical Implementation', 'System Architecture', 'Technology Assessment'],
      selectionReason: 'Required for technical feasibility and implementation considerations',
      expectedContribution: 'Evaluate technical aspects and provide implementation roadmap',
      confidenceScore: 0.90
    },
    'synthesis_agent': {
      expertise: ['Strategic Synthesis', 'Cross-Domain Integration', 'Executive Summary'],
      selectionReason: 'Selected to integrate insights across multiple research streams',
      expectedContribution: 'Synthesize findings into coherent strategic recommendations',
      confidenceScore: 0.94
    }
  };
  
  const profile = agentProfiles[agentName] || agentProfiles['research_agent'];
  return { agentName, ...profile };
};

// ========================================
// AgentRationale Component
// ========================================

const AgentRationale: React.FC<AgentRationaleProps> = ({
  agentName,
  expertise,
  selectionReason,
  expectedContribution,
  confidenceScore,
}) => {
  const colorClass = getAgentExpertiseColor(agentName);
  const AgentIcon = getAgentIcon(agentName);
  
  return (
    <div className="bg-neutral-900/50 border border-neutral-700 rounded-lg p-3 space-y-2">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${colorClass.split(' ')[1]}`}>
          <AgentIcon className={`w-4 h-4 ${colorClass.split(' ')[0]}`} />
        </div>
        <div className="flex-1">
          <h6 className="font-medium text-neutral-200 text-sm">
            {agentName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </h6>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant="outline" className="text-xs">
              {formatConfidenceScore(confidenceScore)} match
            </Badge>
            <div className="flex gap-1">
              {expertise.slice(0, 2).map((skill, idx) => (
                <Badge key={idx} variant="secondary" className="text-xs px-2 py-0">
                  {skill}
                </Badge>
              ))}
              {expertise.length > 2 && (
                <Badge variant="secondary" className="text-xs px-2 py-0">
                  +{expertise.length - 2} more
                </Badge>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <div className="space-y-2 text-xs">
        <div>
          <span className="font-medium text-neutral-300">Selection Reason:</span>
          <p className="text-neutral-400 mt-1">{selectionReason}</p>
        </div>
        <div>
          <span className="font-medium text-neutral-300">Expected Contribution:</span>
          <p className="text-neutral-400 mt-1">{expectedContribution}</p>
        </div>
      </div>
    </div>
  );
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
  const agentRationales = sequence.agent_names.map(agentName => 
    generateAgentRationale(agentName, sequence.research_focus)
  );
  
  return (
    <div className="border border-neutral-600 rounded-lg p-4 bg-gradient-to-r from-neutral-800/50 to-neutral-800/30 hover:from-neutral-800/70 hover:to-neutral-800/50 transition-all duration-200 shadow-lg">
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
          
          {/* Agent Pipeline with Visual Enhancement */}
          <div>
            <h5 className="text-xs font-medium text-neutral-300 mb-2 flex items-center gap-2">
              <Users className="w-3 h-3" />
              Agent Pipeline Strategy:
            </h5>
            <div className="flex items-center gap-2 flex-wrap mb-3">
              {sequence.agent_names.map((agent, idx) => {
                const colorClass = getAgentExpertiseColor(agent);
                const AgentIcon = getAgentIcon(agent);
                return (
                  <React.Fragment key={agent}>
                    <div className={`flex items-center gap-1 px-2 py-1 rounded-md ${colorClass.split(' ')[1]} border border-neutral-600`}>
                      <AgentIcon className={`w-3 h-3 ${colorClass.split(' ')[0]}`} />
                      <span className={`text-xs font-medium ${colorClass.split(' ')[0]}`}>
                        {agent.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                    </div>
                    {idx < sequence.agent_names.length - 1 && (
                      <ArrowRight className="w-3 h-3 text-neutral-500" />
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
          
          {/* Agent Selection Rationale */}
          <div>
            <h5 className="text-xs font-medium text-neutral-300 mb-2 flex items-center gap-2">
              <Brain className="w-3 h-3" />
              Agent Selection Rationale:
            </h5>
            <div className="space-y-2">
              {agentRationales.map((rationale, idx) => (
                <AgentRationale key={idx} {...rationale} />
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
      "bg-gradient-to-br from-slate-900/95 via-blue-950/90 to-slate-900/95 border-2 border-blue-500/30 rounded-xl p-6 my-6 shadow-2xl backdrop-blur-sm",
      "relative overflow-hidden",
      className
    )}>
      {/* Strategic Authority Background Pattern */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-600/5 via-transparent to-purple-600/5 pointer-events-none" />
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-blue-500" />
      {/* Enhanced Strategic Header */}
      <div className="relative z-10">
        <div className="flex items-start gap-4 mb-6">
          <div className="flex-shrink-0 p-3 bg-gradient-to-br from-blue-500/30 to-purple-500/30 rounded-xl border border-blue-400/30 shadow-lg">
            <Brain className="w-6 h-6 text-blue-300" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 mb-3">
              <h3 className="font-bold text-blue-300 text-xl tracking-tight">
                Strategic Research Consultation
              </h3>
              <Badge variant="outline" className="bg-blue-500/20 text-blue-300 border-blue-400/30 font-medium px-3 py-1">
                {sequences.length} Specialized Sequences
              </Badge>
            </div>
            
            <div className="mb-4">
              <div className="flex items-center gap-2 text-sm font-medium text-blue-200 mb-2">
                <CheckCircle2 className="w-4 h-4 text-green-400" />
                AI Supervisor Strategy Analysis Complete
              </div>
              <p className="text-neutral-300 text-sm leading-relaxed">
                I've analyzed your research requirements and generated {sequences.length} optimized research sequences 
                with specialized agent teams for comprehensive parallel execution.
              </p>
            </div>
          
            {/* Strategic Consultation Announcement */}
            <div className="bg-gradient-to-r from-slate-800/50 to-slate-700/30 rounded-lg p-4 border border-slate-600/30 mb-4">
              <div className="flex items-start gap-3 mb-3">
                <Lightbulb className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <h4 className="font-semibold text-yellow-400 text-sm mb-2">Strategic Research Planning</h4>
                  <div className="text-sm text-neutral-300">
                    <TypedMarkdown speed={25} hideCursor={true}>
                      {`Research Query: "${researchQuery}"

I've designed ${sequences.length} complementary research sequences that will execute in parallel, each with specialized agent teams optimized for different analytical approaches. This ensures comprehensive coverage while maximizing efficiency through strategic task distribution.`}
                    </TypedMarkdown>
                  </div>
                </div>
              </div>
              
              {/* Strategic Advantages Grid */}
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="flex items-center gap-2 text-green-300">
                  <Zap className="w-3 h-3" />
                  <span><strong>Parallel Processing:</strong> Simultaneous execution</span>
                </div>
                <div className="flex items-center gap-2 text-blue-300">
                  <Brain className="w-3 h-3" />
                  <span><strong>AI-Optimized:</strong> Strategic agent selection</span>
                </div>
                <div className="flex items-center gap-2 text-purple-300">
                  <Users className="w-3 h-3" />
                  <span><strong>Specialized Teams:</strong> Domain expertise matching</span>
                </div>
                <div className="flex items-center gap-2 text-amber-300">
                  <Target className="w-3 h-3" />
                  <span><strong>Comprehensive:</strong> Multi-perspective analysis</span>
                </div>
              </div>
            </div>
          </div>
        </div>

      {/* Enhanced Sequences List */}
      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-4">
          <GitCompare className="w-4 h-4 text-blue-400" />
          <h4 className="font-semibold text-blue-300 text-lg">Research Sequence Overview</h4>
        </div>
        <div className="space-y-4 mb-6">
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

        </div>
      </div>

      {/* Enhanced Action Section */}
      <div className="relative z-10 bg-gradient-to-r from-slate-800/30 to-slate-700/20 rounded-lg p-4 border border-slate-600/30">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <h5 className="font-medium text-blue-300 text-sm mb-1">Ready for Parallel Execution</h5>
            <p className="text-xs text-neutral-400">
              Launch all research sequences simultaneously and monitor live progress in dedicated tabs below
            </p>
          </div>
          <Button 
            onClick={handleInitializeTabs}
            disabled={hasInitialized || isLoading}
            className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white text-sm px-6 py-3 h-auto font-medium shadow-lg transition-all duration-200 hover:shadow-xl border border-blue-500/30"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin mr-2" />
                Launching Sequences...
              </>
            ) : hasInitialized ? (
              <>
                <CheckCircle2 className="w-4 h-4 mr-2" />
                Sequences Active
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Launch Strategic Research
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Enhanced Progress indicator when initialized */}
      {hasInitialized && (
        <div className="relative z-10 mt-4 p-4 bg-gradient-to-r from-green-900/30 to-emerald-900/20 border border-green-500/30 rounded-lg">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-green-500/20 rounded-lg">
              <CheckCircle2 className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <h5 className="text-sm font-semibold text-green-400">
                Strategic Research Sequences Launched Successfully!
              </h5>
              <p className="text-xs text-green-300 mt-1">
                All {sequences.length} specialized sequences are now executing in parallel with real-time progress tracking
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-green-300">
            <ArrowRight className="w-3 h-3" />
            <span>Switch between tabs below to monitor each sequence's specialized research progress</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default SupervisorAnnouncementMessage;