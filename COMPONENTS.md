# Open Deep Research - Component Reference

## Overview

This document provides detailed reference information for all major components in the Open Deep Research system. Each component is designed for modularity, extensibility, and production-ready reliability.

## Core Components

### 1. LLM Sequence Generator (`orchestration/llm_sequence_generator.py`)

The LLM Sequence Generator is the intelligent heart of the system, automatically creating optimal agent sequences based on research topic analysis.

#### Key Features
- **Intelligent Topic Analysis**: Automatic domain classification and complexity assessment
- **Strategic Sequence Generation**: Multiple sequence strategies (theory-first, market-first, technical-first, balanced)
- **Agent Capability Matching**: Intelligent matching of research needs to specialized agents
- **Comprehensive Scoring**: Multi-dimensional evaluation of sequence quality

#### Usage Example
```python
from open_deep_research.orchestration.llm_sequence_generator import LLMSequenceGenerator
from open_deep_research.agents.registry import AgentRegistry

registry = AgentRegistry()
generator = LLMSequenceGenerator(registry)

sequences = await generator.generate_sequences(
    research_topic="AI safety research implementation",
    num_sequences=3
)

best_sequence = sequences[0]
print(f"Strategy: {best_sequence.strategy.value}")
print(f"Agents: {' → '.join(best_sequence.agents)}")
print(f"Confidence: {best_sequence.confidence:.2f}")
```

#### Configuration Options
```python
# Initialize with custom configuration
generator = LLMSequenceGenerator(
    agent_registry=registry,
    config=config,  # RunnableConfig for model selection
    debug_mode=True  # Enable detailed logging
)

# Generate with specific parameters
sequences = await generator.generate_sequences(
    research_topic="your topic",
    available_agents=["research_agent", "analysis_agent"],  # Restrict agent pool
    num_sequences=5,  # Generate up to 5 sequences
    strategies=["theory_first", "market_first"]  # Specific strategies only
)
```

#### Topic Analysis Process
1. **Keyword Extraction**: Identifies domain-specific terms
2. **Domain Classification**: Categorizes as academic, market, technical, or mixed
3. **Complexity Assessment**: Evaluates research scope and depth requirements
4. **Agent Requirement Estimation**: Predicts optimal sequence length and specializations

### 2. Sequential Supervisor (`supervisor/sequential_supervisor.py`)

The Sequential Supervisor manages the execution of agent sequences with comprehensive state management and handoff logic.

#### Key Features
- **LangGraph Integration**: Full StateGraph compatibility for workflow orchestration
- **Parallel Execution**: Simultaneous execution of multiple sequences
- **Agent Handoff Management**: Intelligent transitions between agents within sequences
- **State Coordination**: Maintains separate state tracking for each sequence
- **Error Recovery**: Robust error handling and graceful degradation

#### Usage Example
```python
from open_deep_research.supervisor.sequential_supervisor import SequentialSupervisor
from open_deep_research.state import SequentialSupervisorState

supervisor = SequentialSupervisor(agent_registry)

# Create workflow
workflow = await supervisor.create_workflow_graph()
compiled_workflow = workflow.compile()

# Execute sequence
state = SequentialSupervisorState()
state.research_topic = "quantum computing applications"
state.planned_sequence = ["research_agent", "analysis_agent", "synthesis_agent"]

final_state = await compiled_workflow.ainvoke(state)
```

#### Workflow Configuration
```python
# Validate sequence before execution
validation = supervisor.validate_sequence(["research_agent", "analysis_agent"])
if validation["valid"]:
    # Proceed with execution
    pass
else:
    print(f"Validation errors: {validation['errors']}")

# Configure execution parameters
state.max_iterations_per_agent = 3
state.timeout_per_agent = 300  # 5 minutes
state.enable_parallel_research = True
```

### 3. LLM Judge Evaluation System (`evaluation/llm_judge.py`)

The LLM Judge provides comprehensive automated evaluation of research reports with multi-criteria scoring and comparative analysis.

#### Key Features
- **Multi-Criteria Evaluation**: Scores across completeness, depth, coherence, innovation, and actionability
- **Winner Selection**: Automated determination of best-performing sequences
- **Comparative Analysis**: Detailed comparison between different approaches
- **Performance Insights**: Actionable recommendations for sequence optimization
- **Parallel Processing**: Concurrent evaluation of multiple reports

#### Usage Example
```python
from open_deep_research.evaluation.llm_judge import LLMJudge

judge = LLMJudge(config=config)

# Evaluate multiple reports
reports = {
    "theory_first": running_report_1,
    "market_first": running_report_2,
    "technical_first": running_report_3
}

result = await judge.evaluate_reports(
    reports=reports,
    research_topic="AI safety analysis"
)

print(f"Winner: {result.winning_sequence}")
print(f"Score: {result.winning_sequence_score:.1f}/100")
```

#### Evaluation Criteria
- **Completeness (0-10)**: Coverage breadth and thoroughness
- **Depth (0-10)**: Analytical rigor and insight quality
- **Coherence (0-10)**: Logical flow and sequential insight building
- **Innovation (0-10)**: Novel perspectives and creative approaches
- **Actionability (0-10)**: Practical value and implementation guidance

#### Results Analysis
```python
# Access detailed results
for evaluation in result.individual_evaluations:
    print(f"Sequence: {evaluation.sequence_name}")
    print(f"Overall: {evaluation.overall_score:.1f}/100")
    print(f"Strengths: {evaluation.key_strengths}")
    print(f"Weaknesses: {evaluation.key_weaknesses}")

# Get performance comparisons
for comparison in result.comparative_analysis.pairwise_comparisons:
    print(f"{comparison.sequence_a} vs {comparison.sequence_b}")
    print(f"Winner: {comparison.winner} (margin: {comparison.margin:.1f})")
```

### 4. Running Report Builder (`orchestration/report_builder.py`)

The Running Report Builder incrementally constructs research reports as agents complete their work, ensuring consistent formatting and insight connectivity.

#### Key Features
- **Incremental Generation**: Builds reports progressively during research
- **Insight Connection**: Links findings across agents within sequences
- **Format Consistency**: Ensures uniform report structure
- **Real-time Updates**: Supports streaming updates to frontend interfaces
- **Section Management**: Organizes content into logical sections

#### Usage Example
```python
from open_deep_research.orchestration.report_builder import RunningReportBuilder

builder = RunningReportBuilder()

# Initialize report
report = builder.initialize_report(
    research_topic="AI in healthcare",
    sequence_name="theory_first"
)

# Add agent contributions
builder.add_agent_contribution(
    report=report,
    agent_name="research_agent",
    findings=research_findings,
    insights=research_insights
)

# Finalize report
final_report = builder.finalize_report(report)
```

#### Report Structure
```python
class RunningReport:
    title: str
    research_topic: str
    sequence_name: str
    sections: List[ReportSection]
    insights: List[Insight]
    agent_contributions: Dict[str, AgentContribution]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

### 5. Agent Registry (`agents/registry.py`)

The Agent Registry provides centralized management of all available research agents with dynamic discovery and capability assessment.

#### Key Features
- **Dynamic Discovery**: Runtime agent discovery from multiple sources
- **Capability Assessment**: Expertise area matching and validation
- **Agent Validation**: Ensures agents meet interface requirements
- **Precedence Management**: Handles agent priority and selection rules

#### Usage Example
```python
from open_deep_research.agents.registry import AgentRegistry

registry = AgentRegistry()

# List available agents
agents = registry.list_agents()
print(f"Available agents: {agents}")

# Get agent by name
agent = registry.get_agent("research_agent")
if agent:
    print(f"Expertise: {agent.expertise_areas}")

# Find agents by expertise
specialists = registry.find_agents_by_expertise("machine_learning")
```

#### Agent Configuration
```python
# Agent definition example (in agent YAML frontmatter)
"""
---
name: research_agent
description: Specialized academic research agent
expertise_areas:
  - academic_research
  - literature_review
  - peer_review_analysis
capabilities:
  - web_search
  - document_analysis
  - citation_management
---
"""
```

### 6. Specialized Agents (`agents/`)

The system includes several specialized research agents, each optimized for specific research domains.

#### Available Agent Types

##### Research Agent (`agents/research_agent.py`)
- **Expertise**: Academic research, literature reviews, peer-reviewed sources
- **Capabilities**: Web search, document analysis, citation management
- **Best For**: Theory-first approaches, academic foundations

##### Analysis Agent (`agents/analysis_agent.py`)
- **Expertise**: Data analysis, statistical evaluation, trend identification
- **Capabilities**: Quantitative analysis, pattern recognition, comparative studies
- **Best For**: Data-intensive research, analytical deep-dives

##### Market Agent (`agents/market_agent.py`)
- **Expertise**: Market research, competitive analysis, business intelligence
- **Capabilities**: Industry reports, financial analysis, market trend identification
- **Best For**: Business-focused research, market opportunity analysis

##### Technical Agent (`agents/technical_agent.py`)
- **Expertise**: Technical implementation, architecture design, technology evaluation
- **Capabilities**: Code analysis, system design, technology comparison
- **Best For**: Implementation-focused research, technical feasibility studies

##### Synthesis Agent (`agents/synthesis_agent.py`)
- **Expertise**: Information synthesis, report writing, insight integration
- **Capabilities**: Content organization, narrative development, recommendation formulation
- **Best For**: Final report generation, cross-domain integration

#### Agent Interface
```python
from open_deep_research.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="custom_agent",
            description="Custom research specialist",
            expertise_areas=["custom_domain"],
            capabilities=["custom_capability"]
        )
    
    async def research(self, query: str, context: dict) -> dict:
        """Implement agent-specific research logic"""
        return {
            "findings": ["finding1", "finding2"],
            "insights": ["insight1", "insight2"],
            "sources": ["source1", "source2"]
        }
```

## Search and Data Integration

### Search Client (`core/search.py`)

Unified interface for multiple search providers with intelligent result aggregation.

#### Supported Providers
- **Tavily**: Primary web search API (default)
- **Native Search**: OpenAI/Anthropic built-in search
- **DuckDuckGo**: Alternative search provider
- **Exa**: Semantic search capabilities
- **MCP Servers**: Extended data source integration

#### Usage Example
```python
from open_deep_research.core.search import get_search_client

client = get_search_client()
results = await client.search("quantum computing breakthroughs", max_results=10)

for result in results:
    print(f"Title: {result.title}")
    print(f"URL: {result.url}")
    print(f"Snippet: {result.snippet}")
```

### MCP Integration (`core/mcp.py`)

Model Context Protocol integration for extended data source access.

#### Configuration
```python
# MCP server configuration in .env
MCP_SERVERS='[
    {"name": "filesystem", "command": "mcp-filesystem"},
    {"name": "github", "command": "mcp-github", "args": ["--token", "TOKEN"]}
]'
```

## State Management

### Graph State (`state.py`)

Comprehensive state management for LangGraph workflow execution.

#### Core State Classes

##### SequentialSupervisorState
```python
class SequentialSupervisorState:
    research_topic: str
    planned_sequence: List[str]
    current_agent_index: int
    agent_outputs: Dict[str, Any]
    running_report: RunningReport
    handoff_ready: bool
    sequence_complete: bool
    error_state: Optional[str]
```

##### ParallelExecutionState
```python
class ParallelExecutionState:
    sequences: Dict[str, SequentialSupervisorState]
    sequence_results: Dict[str, RunningReport]
    judge_evaluation: Optional[EvaluationResult]
    overall_winner: Optional[str]
    execution_metadata: Dict[str, Any]
```

## Configuration System

### Configuration Management (`configuration.py`)

Centralized configuration system supporting environment variables, runtime configuration, and model provider selection.

#### Key Configuration Options
```python
class ODRConfiguration:
    # Model Configuration
    reasoning_model: str = "claude-3-5-sonnet-20241022"
    reflection_model: str = "claude-3-5-sonnet-20241022"
    evaluation_model: str = "claude-3-5-sonnet-20241022"
    
    # Search Configuration
    search_api: str = "tavily"
    max_search_queries: int = 5
    
    # Sequence Configuration
    enable_dynamic_sequence_generation: bool = True
    max_parallel_sequences: int = 3
    
    # Performance Configuration
    max_research_loops: int = 3
    request_timeout: int = 300
```

#### Usage Example
```python
from open_deep_research.configuration import get_configuration

config = get_configuration()
print(f"Using model: {config.reasoning_model}")
print(f"Search API: {config.search_api}")
print(f"Max sequences: {config.max_parallel_sequences}")
```

## Frontend Components

### React Components (`frontend/src/components/`)

Modern React components for the research interface with TypeScript support.

#### Core Components

##### ParallelResearchInterface
```typescript
interface ParallelResearchInterfaceProps {
  sequences: LLMGeneratedSequence[];
  onSequenceSelect: (sequenceId: string) => void;
  executionResults: Record<string, SequenceResult>;
  judgeEvaluation?: JudgeEvaluation;
}
```

##### ChatInterface
```typescript
interface ChatInterfaceProps {
  messages: Message[];
  isLoading: boolean;
  onSendMessage: (message: string) => void;
  streamInstance?: StreamInstance;
}
```

##### ActivityTimeline
```typescript
interface ActivityTimelineProps {
  processedEvents: ProcessedEvent[];
  sequenceId?: string;
  isLoading: boolean;
  showSequenceLabel?: boolean;
}
```

### Custom Hooks (`frontend/src/hooks/`)

Reusable React hooks for state management and WebSocket communication.

#### useParallelSequences
```typescript
interface UseParallelSequencesReturn {
  sequences: LLMGeneratedSequence[];
  activeSequences: Set<string>;
  executionResults: Record<string, SequenceResult>;
  judgeEvaluation?: JudgeEvaluation;
  startSequence: (sequenceId: string, query: string) => void;
  stopSequence: (sequenceId: string) => void;
}
```

#### useStream
```typescript
interface UseStreamReturn<T> {
  messages: T['messages'];
  input: string;
  handleInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  handleSubmit: (e: React.FormEvent) => void;
  isLoading: boolean;
  error?: Error;
}
```

## Utility Components

### Utilities (`utils.py`)

Common utility functions used throughout the system.

#### Key Functions
```python
# State conversion utilities
def convert_agent_state_to_sequential(state: Any) -> SequentialSupervisorState

# Content processing utilities  
def truncate_content(content: str, max_tokens: int) -> str
def extract_insights(content: str) -> List[str]

# Validation utilities
def validate_sequence(agents: List[str]) -> Dict[str, Any]
def validate_research_topic(topic: str) -> bool
```

### Prompts (`prompts.py`)

System prompts and templates for LLM interactions.

#### Prompt Categories
- **Sequence Generation Prompts**: For intelligent agent sequence creation
- **Research Prompts**: For agent-specific research tasks
- **Evaluation Prompts**: For LLM judge evaluation
- **Synthesis Prompts**: For report generation and insight synthesis

## Error Handling and Logging

### Error Management
The system includes comprehensive error handling patterns:

```python
try:
    result = await component.execute(parameters)
except ValidationError as e:
    logger.error(f"Input validation failed: {e}")
    return create_error_response("invalid_input", str(e))
except TimeoutError as e:
    logger.warning(f"Operation timed out: {e}")
    return create_timeout_response()
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    return create_error_response("system_error", "Please try again")
```

### Logging Configuration
```python
import logging
from open_deep_research.utils import setup_logging

# Configure logging
setup_logging(level=logging.INFO, format="structured")

# Use throughout components
logger = logging.getLogger(__name__)
logger.info("Component initialized", extra={"component": "sequence_generator"})
```

## Performance Considerations

### Optimization Guidelines

#### Model Selection
- **Development**: Use faster models (gpt-4o-mini, claude-3-haiku)
- **Production**: Use higher-quality models (claude-3-5-sonnet, gpt-4o)
- **Cost Optimization**: Configure different models for different tasks

#### Concurrency Management
```python
# Configure parallel execution limits
config.max_parallel_sequences = 3  # Adjust based on resources
config.max_concurrent_agents = 5   # Limit concurrent operations
config.request_timeout = 300       # Set appropriate timeouts
```

#### Memory Management
```python
# Handle large reports and data
def process_large_report(report: str) -> str:
    if len(report) > MAX_CONTENT_LENGTH:
        return truncate_intelligently(report, MAX_CONTENT_LENGTH)
    return report
```

## Extension Points

### Adding New Agents
1. Create agent class extending `BaseAgent`
2. Register in `AgentRegistry`
3. Define expertise areas and capabilities
4. Implement research logic

### Custom Sequence Strategies
1. Extend `SequenceStrategy` enum
2. Implement generation logic in `LLMSequenceGenerator`
3. Add strategy-specific scoring
4. Update documentation

### New Search Providers
1. Implement `SearchProvider` interface
2. Add to search client factory
3. Configure in environment variables
4. Test integration

This component reference provides comprehensive information for understanding, using, and extending the Open Deep Research system. Each component is designed for reliability, extensibility, and ease of integration.