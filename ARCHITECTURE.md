# Open Deep Research - System Architecture

## Overview

Open Deep Research is a configurable, fully open-source deep research agent that works across multiple model providers, search tools, and MCP (Model Context Protocol) servers. The system enables automated research with parallel processing, intelligent agent sequencing, and comprehensive report generation.

## Core Architecture

### System Workflow

```
User Query 
    ↓
Query Processing & Clarification
    ↓
LLM-Based Sequence Generation (1-3 sequences)
    ↓
Parallel Sequence Execution (SequentialSupervisor)
    ↓
Individual Report Generation (RunningReportBuilder)
    ↓
LLM Judge Evaluation & Winner Selection
    ↓
Return Results to Frontend
```

### Key Components

#### 1. LangGraph Integration (`deep_researcher.py`)
- **Entry Point**: `deep_researcher` function for LangGraph deployment
- **State Management**: Comprehensive state tracking through `SequentialSupervisorState`
- **Workflow Orchestration**: Manages complete research workflow from query to final evaluation
- **Configuration Integration**: Respects all configuration settings for model providers and tools

#### 2. Intelligent Sequence Generation (`orchestration/llm_sequence_generator.py`)
- **Strategic Analysis**: LLM-based topic analysis and agent selection
- **Multiple Strategies**: Theory-first, market-first, technical-first, and balanced approaches
- **Agent Capability Matching**: Intelligent matching of research needs to specialized agents
- **Sequence Optimization**: Generates 1-3 optimal sequences based on research complexity

#### 3. Sequential Supervisor (`supervisor/sequential_supervisor.py`)
- **Parallel Execution**: Executes multiple sequences simultaneously
- **Agent Handoff Logic**: Manages transitions between specialized agents within sequences
- **State Coordination**: Maintains separate state for each parallel sequence
- **LangGraph StateGraph Integration**: Full LangGraph workflow compatibility

#### 4. Specialized Agent System (`agents/`)
- **Agent Registry**: Central registry for all available agents
- **Specialized Agents**: Research, analysis, market, technical, and synthesis agents
- **Dynamic Loading**: Runtime agent discovery and capability assessment
- **Expertise Matching**: Agents matched to research domains based on expertise areas

#### 5. LLM Judge Evaluation (`evaluation/llm_judge.py`)
- **Report Comparison**: Comprehensive evaluation of research reports
- **Winner Selection**: Determines best sequence based on quality, completeness, and relevance
- **Scoring Metrics**: Multi-dimensional evaluation including accuracy, depth, and actionability
- **Evaluation Reasoning**: Provides detailed rationale for winner selection

#### 6. Running Report Builder (`orchestration/report_builder.py`)
- **Incremental Generation**: Builds reports progressively as agents complete work
- **Insight Connection**: Links findings across agents within sequences
- **Format Consistency**: Ensures consistent report structure across all sequences
- **Real-time Updates**: Supports streaming updates to frontend interfaces

## Frontend Architecture

### WebSocket Communication
- **Real-time Streaming**: Live updates during sequence execution
- **Event-based Updates**: Structured events for different workflow stages
- **Multiple Stream Management**: Handles parallel sequence streams simultaneously
- **State Synchronization**: Coordinates multiple sequence states in UI

### Component Structure
- **Parallel Chat Interface**: Multiple chat tabs for simultaneous sequence viewing
- **Activity Timeline**: Real-time progress tracking across all sequences
- **Judge Evaluation Display**: Visual presentation of evaluation results and winner
- **Sequence Comparison**: Side-by-side comparison of different approaches

## Technology Stack

### Backend Technologies
- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and tool calling framework
- **Python 3.11+**: Core implementation language
- **FastAPI**: API server for frontend communication
- **WebSockets**: Real-time communication with frontend

### Model Provider Support
- **OpenAI**: GPT-4, GPT-3.5-turbo support
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku support
- **Google**: Gemini Pro support
- **Groq**: Fast inference support
- **DeepSeek**: Cost-effective model support
- **Local Models**: Via MCP server integration

### Search and Data APIs
- **Tavily**: Primary web search API
- **Native Search**: OpenAI/Anthropic built-in search capabilities
- **DuckDuckGo**: Alternative search provider
- **Exa**: Semantic search capabilities
- **MCP Servers**: Extended data source integration

### Frontend Technologies
- **React + TypeScript**: Modern frontend framework
- **Vite**: Fast development and build tooling
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: Component library for consistent UI
- **WebSocket Client**: Real-time communication

## Data Flow Architecture

### Request Flow
1. **Query Input**: User submits research query through frontend
2. **Query Processing**: Backend analyzes query and generates clarifying questions if needed
3. **Sequence Generation**: LLM analyzes topic and generates 1-3 optimal agent sequences
4. **Parallel Execution**: SequentialSupervisor executes all sequences simultaneously
5. **Report Building**: RunningReportBuilder incrementally generates reports per sequence
6. **Evaluation**: LLM Judge compares all reports and selects winner
7. **Response**: Structured results returned to frontend with winner highlighted

### Event Stream Flow
```typescript
interface WorkflowEvent {
  sequence_id: string;
  event_type: 'agent_start' | 'tool_call' | 'agent_complete' | 'sequence_complete';
  timestamp: number;
  data: {
    agent_name?: string;
    tool_name?: string;
    progress?: number;
    report_section?: string;
  };
}
```

### State Management
- **Global State**: Research topic, configuration, overall progress
- **Sequence State**: Individual sequence progress, agent states, intermediate results
- **Report State**: Incremental report building, section completion tracking
- **Evaluation State**: Judge evaluation progress, comparative analysis, winner selection

## Configuration Architecture

### Configuration System (`configuration.py`)
- **Environment Variables**: Primary configuration through `.env` file
- **Runtime Configuration**: Web UI configuration in LangGraph Studio
- **Model Selection**: Configurable model providers and specific models
- **Search API Selection**: Choice between different search providers
- **Concurrency Settings**: Parallel execution limits and timeout configurations

### Key Configuration Options
```python
class ODRConfiguration:
    # Model Configuration
    reasoning_model: str = "claude-3-5-sonnet-20241022"
    reflection_model: str = "claude-3-5-sonnet-20241022"
    
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

## Security Architecture

### Authentication (`security/auth.py`)
- **LangGraph Deployment**: Authentication handler for production deployment
- **API Key Management**: Secure handling of external API keys
- **Environment Isolation**: Separation of development and production configurations

### Data Privacy
- **No Data Persistence**: Research data not stored permanently
- **API Key Protection**: All API keys handled securely through environment variables
- **Session Isolation**: Each research session completely isolated

## Performance Architecture

### Scalability Considerations
- **Parallel Execution**: Multiple sequences run simultaneously
- **Async Processing**: Full async/await support throughout stack
- **Resource Management**: Configurable limits on concurrent operations
- **Timeout Handling**: Comprehensive timeout management for all external calls

### Performance Targets
- **Sequence Generation**: < 10 seconds for topic analysis and agent selection
- **Parallel Execution**: All sequences complete within 5 minutes
- **Judge Evaluation**: < 30 seconds for comparative analysis
- **Total Workflow**: < 6 minutes for complete research cycle

### Monitoring and Observability
- **LangGraph Studio**: Built-in workflow visualization and debugging
- **Structured Logging**: Comprehensive logging throughout execution
- **Performance Metrics**: Execution time tracking for optimization
- **Error Handling**: Graceful degradation and error recovery

## Extension Architecture

### Agent Extension
- **Plugin Architecture**: New agents can be added via agent registry
- **Capability Declaration**: Agents declare expertise areas and capabilities
- **Dynamic Discovery**: Runtime discovery of new agents
- **Custom Logic**: Agents can implement domain-specific research logic

### Model Provider Extension
- **Provider Abstraction**: New model providers can be added through configuration
- **MCP Integration**: Model Context Protocol for extended model access
- **Cost Optimization**: Configurable model selection for cost management

### Search Provider Extension
- **Search Abstraction**: New search APIs can be integrated
- **Result Normalization**: Consistent search result processing
- **Fallback Logic**: Multiple search providers with fallback support

## Deployment Architecture

### Development Deployment
- **LangGraph Studio**: `uvx langgraph dev` for development server
- **Hot Reload**: Automatic reloading during development
- **Debug Mode**: Enhanced logging and debugging capabilities

### Production Deployment
- **LangGraph Cloud**: Cloud deployment with authentication
- **Container Support**: Docker containerization for flexible deployment
- **Environment Management**: Production-specific configuration management

This architecture provides a robust, scalable foundation for automated research with intelligent agent sequencing, parallel processing, and comprehensive evaluation capabilities.