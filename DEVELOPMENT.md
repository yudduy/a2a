# Open Deep Research - Development Guide

## Development Environment Setup

### Prerequisites
- Python 3.11 or higher
- UV package manager (recommended) or pip
- Node.js 18+ (for frontend development)
- Git

### Quick Setup
```bash
# Clone repository
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Copy environment template
cp .env.example .env

# Start development server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

### Environment Configuration
Create a `.env` file with your API keys and configuration:

```bash
# Model Configuration
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key

# Search Configuration  
TAVILY_API_KEY=your_tavily_key

# Development Configuration
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true

# Frontend Configuration
VITE_API_URL=http://localhost:2024
```

## Development Commands

### Core Development
```bash
# Start LangGraph development server with Studio UI
uvx langgraph dev

# Run with custom configuration
uvx langgraph dev --config-file custom_config.yaml

# Production build
uvx langgraph build

# Deploy to LangGraph Platform
uvx langgraph deploy
```

### Code Quality
```bash
# Run linting
ruff check

# Fix linting issues
ruff check --fix

# Type checking
mypy src/

# Format code
ruff format
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src/open_deep_research --cov-report=html

# Run performance tests
pytest tests/test_performance.py -v
```

### Frontend Development
```bash
# Navigate to frontend directory
cd frontend/

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run frontend tests
npm test

# Lint frontend code
npm run lint
```

## Project Structure

### Backend Structure
```
src/open_deep_research/
├── agents/                    # Specialized research agents
│   ├── registry.py           # Agent discovery and management
│   ├── academic_agent.py     # Academic research specialist
│   ├── market_agent.py       # Market analysis specialist
│   └── technical_agent.py    # Technical implementation specialist
├── core/                     # Core research functionality
│   ├── search.py            # Search API integration
│   ├── synthesis.py         # Content synthesis
│   └── validation.py        # Content validation
├── orchestration/           # Workflow orchestration
│   ├── llm_sequence_generator.py  # Intelligent sequence generation
│   ├── sequence_generator.py     # Legacy sequence generation
│   ├── report_builder.py         # Incremental report building
│   └── workflow_manager.py       # Workflow coordination
├── supervisor/              # Multi-agent supervision
│   ├── sequential_supervisor.py  # Sequential workflow management
│   └── parallel_coordinator.py   # Parallel execution coordination
├── evaluation/              # Report evaluation
│   ├── llm_judge.py        # LLM-based report evaluation
│   ├── metrics.py          # Evaluation metrics
│   └── comparative_analysis.py  # Multi-report comparison
├── configuration.py         # System configuration
├── state.py                # Graph state definitions
├── deep_researcher.py      # Main LangGraph entry point
├── prompts.py              # System prompts
└── utils.py                # Utility functions
```

### Frontend Structure
```
frontend/src/
├── components/              # React components
│   ├── research/           # Research-specific components
│   │   ├── ParallelResearchInterface.tsx
│   │   └── SequenceObserver.tsx
│   ├── ui/                 # Reusable UI components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   └── input.tsx
│   ├── ActivityTimeline.tsx
│   ├── ChatInterface.tsx
│   └── InputForm.tsx
├── hooks/                   # React hooks
│   ├── useParallelSequences.ts
│   └── useStream.ts
├── types/                   # TypeScript definitions
│   ├── agents.ts
│   ├── messages.ts
│   └── parallel.ts
└── App.tsx                 # Main application component
```

## Development Patterns

### Agent Development
When creating new specialized agents:

1. **Extend Base Agent**:
```python
from open_deep_research.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="custom_agent",
            description="Specialized agent for custom research",
            expertise_areas=["custom_domain", "specialized_topic"]
        )
    
    async def research(self, query: str, context: dict) -> dict:
        # Implement agent-specific research logic
        pass
```

2. **Register Agent**:
```python
# In agents/registry.py
from .custom_agent import CustomAgent

class AgentRegistry:
    def __init__(self):
        self.agents = {
            "custom_agent": CustomAgent(),
            # ... other agents
        }
```

### Configuration Management
Use the configuration system for all settings:

```python
from open_deep_research.configuration import get_configuration

config = get_configuration()
model = config.reasoning_model
search_api = config.search_api
max_sequences = config.max_parallel_sequences
```

### State Management
Follow the state pattern for workflow management:

```python
from open_deep_research.state import SequentialSupervisorState

# Create state
state = SequentialSupervisorState()
state.research_topic = "AI in healthcare"
state.planned_sequence = ["research_agent", "analysis_agent"]

# Update state
state.add_insight("Key finding from research")
state.update_progress("research_agent", "completed")
```

## Testing Framework

### Unit Testing
Create comprehensive unit tests for all components:

```python
import pytest
from open_deep_research.agents.research_agent import ResearchAgent

class TestResearchAgent:
    def test_agent_initialization(self):
        agent = ResearchAgent()
        assert agent.name == "research_agent"
        assert "academic" in agent.expertise_areas
    
    @pytest.mark.asyncio
    async def test_research_execution(self):
        agent = ResearchAgent()
        result = await agent.research("quantum computing", {})
        assert "findings" in result
        assert len(result["findings"]) > 0
```

### Integration Testing
Test component interactions:

```python
import pytest
from open_deep_research.orchestration.llm_sequence_generator import LLMSequenceGenerator
from open_deep_research.agents.registry import AgentRegistry

class TestSequenceGeneration:
    @pytest.mark.asyncio
    async def test_end_to_end_sequence_generation(self):
        registry = AgentRegistry()
        generator = LLMSequenceGenerator(registry)
        
        sequences = await generator.generate_sequences(
            "AI in healthcare research",
            num_sequences=3
        )
        
        assert len(sequences) == 3
        assert all(seq.confidence > 0.5 for seq in sequences)
        assert all(len(seq.agents) >= 2 for seq in sequences)
```

### Performance Testing
Validate performance requirements:

```python
import time
import pytest
from open_deep_research.orchestration.llm_sequence_generator import LLMSequenceGenerator

class TestPerformance:
    @pytest.mark.asyncio
    async def test_sequence_generation_speed(self):
        start_time = time.time()
        
        generator = LLMSequenceGenerator(registry)
        sequences = await generator.generate_sequences(
            "Complex research topic",
            num_sequences=3
        )
        
        execution_time = time.time() - start_time
        assert execution_time < 10.0  # Should complete in < 10 seconds
        assert len(sequences) == 3
```

## Debugging and Troubleshooting

### LangGraph Studio Debugging
1. **Access Studio UI**: http://127.0.0.1:2024 when running `uvx langgraph dev`
2. **View Execution Graph**: Visual workflow representation
3. **State Inspection**: Examine state at each workflow step
4. **Error Tracking**: Detailed error information and stack traces

### Common Issues and Solutions

#### Model Configuration Issues
```python
# Issue: "Model not found" errors
# Solution: Check model provider configuration
from open_deep_research.configuration import get_configuration
config = get_configuration()
print(f"Current model: {config.reasoning_model}")

# Verify API keys are set
import os
print(f"OpenAI key set: {'OPENAI_API_KEY' in os.environ}")
```

#### Search API Failures
```python
# Issue: Search API timeouts or failures
# Solution: Test search configuration
from open_deep_research.core.search import get_search_client
client = get_search_client()
results = await client.search("test query", max_results=5)
```

#### Agent Registry Loading
```python
# Issue: Agents not loading properly
# Solution: Verify agent registry
from open_deep_research.agents.registry import AgentRegistry
registry = AgentRegistry()
print(f"Available agents: {registry.list_agents()}")
```

### Performance Optimization

#### Model Selection
- **Development**: Use faster, cheaper models (gpt-4o-mini, claude-3-haiku)
- **Production**: Use higher-quality models (gpt-4o, claude-3-5-sonnet)
- **Cost Optimization**: Configure different models for different tasks

#### Parallel Execution
- **Sequence Limits**: Configure `max_parallel_sequences` based on resources
- **Timeout Management**: Set appropriate timeouts for external API calls
- **Resource Monitoring**: Monitor memory and CPU usage during execution

#### Frontend Performance
- **Component Optimization**: Use React.memo for expensive components
- **State Management**: Optimize re-renders with proper state structure
- **WebSocket Management**: Implement connection pooling for multiple sequences

## Evaluation and Quality Assurance

### Deep Research Bench Evaluation
Run comprehensive evaluation against the benchmark:

```bash
# Full evaluation (expensive - $20-$100)
python tests/run_evaluate.py

# Quick evaluation (subset)
python tests/run_evaluate.py --limit 10

# Extract results
python tests/extract_langsmith_data.py --project-name "YOUR_EXPERIMENT_NAME"
```

### Quality Metrics
Track key performance indicators:

- **RACE Score**: LLM-as-a-judge evaluation score
- **Execution Time**: Time from query to final report
- **Cost Per Query**: Token usage and API costs
- **Success Rate**: Percentage of successful completions

### Manual Testing Scenarios
Test these scenarios manually:

1. **Simple Research**: "Latest developments in quantum computing"
2. **Complex Analysis**: "Compare AI frameworks for healthcare vs finance"
3. **Technical Deep-dive**: "Kubernetes security implementation guide"
4. **Market Research**: "Renewable energy investment trends 2024"
5. **Academic Research**: "Machine learning interpretability methods"

## Contributing Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use TypeScript for frontend code
- Write descriptive commit messages
- Include tests for new features
- Update documentation for significant changes

### Pull Request Process
1. **Branch Naming**: `feature/description` or `fix/description`
2. **Code Review**: All changes require review
3. **Testing**: Ensure all tests pass
4. **Documentation**: Update relevant documentation
5. **Performance**: Verify no performance regressions

### Release Process
1. **Version Bumping**: Update version in `pyproject.toml`
2. **Changelog**: Update `CHANGELOG.md`
3. **Testing**: Full test suite and evaluation run
4. **Deployment**: Deploy to staging, then production
5. **Monitoring**: Monitor metrics post-deployment

This development guide provides comprehensive information for contributing to and extending the Open Deep Research project. Follow these patterns and practices to maintain code quality and system reliability.