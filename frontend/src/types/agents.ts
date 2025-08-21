// Specialized Research Agent Types
export enum SpecializedAgentType {
  ACADEMIC = 'academic',
  INDUSTRY = 'industry', 
  TECHNICAL_TRENDS = 'technical_trends',
}

// Research Sequence Types
export enum SequenceType {
  THEORY_FIRST = 'theory_first',
  MARKET_FIRST = 'market_first',
  FUTURE_BACK = 'future_back',
  PARALLEL_ALL = 'parallel_all',
}

// Specialized Research Agent
export interface SpecializedAgent {
  type: SpecializedAgentType;
  name: string;
  description: string;
  icon: string;
  capabilities: string[];
  color: string;
}

// Research Sequence Configuration
export interface ResearchSequence {
  type: SequenceType;
  name: string;
  description: string;
  agents: SpecializedAgentType[];
  parallel: boolean;
}

// Available Specialized Agents
export const SPECIALIZED_AGENTS: SpecializedAgent[] = [
  {
    type: SpecializedAgentType.ACADEMIC,
    name: 'Academic Researcher',
    description: 'Scholarly research with peer-reviewed sources',
    icon: 'graduation-cap',
    capabilities: ['Academic Papers', 'Theoretical Analysis', 'Research Methods'],
    color: 'blue',
  },
  {
    type: SpecializedAgentType.INDUSTRY,
    name: 'Industry Analyst',
    description: 'Market trends and business intelligence',
    icon: 'trending-up',
    capabilities: ['Market Analysis', 'Business Intelligence', 'Industry Reports'],
    color: 'green',
  },
  {
    type: SpecializedAgentType.TECHNICAL_TRENDS,
    name: 'Technical Trends Expert',
    description: 'Emerging technologies and innovation analysis',
    icon: 'zap',
    capabilities: ['Tech Innovation', 'Future Trends', 'Technical Analysis'],
    color: 'purple',
  },
];

// Available Research Sequences
export const RESEARCH_SEQUENCES: ResearchSequence[] = [
  {
    type: SequenceType.THEORY_FIRST,
    name: 'Theory First',
    description: 'Academic → Industry → Technical sequence',
    agents: [SpecializedAgentType.ACADEMIC, SpecializedAgentType.INDUSTRY, SpecializedAgentType.TECHNICAL_TRENDS],
    parallel: false,
  },
  {
    type: SequenceType.MARKET_FIRST,
    name: 'Market First',
    description: 'Industry → Academic → Technical sequence',
    agents: [SpecializedAgentType.INDUSTRY, SpecializedAgentType.ACADEMIC, SpecializedAgentType.TECHNICAL_TRENDS],
    parallel: false,
  },
  {
    type: SequenceType.FUTURE_BACK,
    name: 'Future Back',
    description: 'Technical → Academic → Industry sequence',
    agents: [SpecializedAgentType.TECHNICAL_TRENDS, SpecializedAgentType.ACADEMIC, SpecializedAgentType.INDUSTRY],
    parallel: false,
  },
  {
    type: SequenceType.PARALLEL_ALL,
    name: 'Parallel Analysis',
    description: 'All agents analyze simultaneously',
    agents: [SpecializedAgentType.ACADEMIC, SpecializedAgentType.INDUSTRY, SpecializedAgentType.TECHNICAL_TRENDS],
    parallel: true,
  },
];

// Parallel Chat Interface State
export interface ParallelChatState {
  query: string;
  selectedSequence: SequenceType;
  agentResults: Map<SpecializedAgentType, any>;
  isProcessing: boolean;
}
