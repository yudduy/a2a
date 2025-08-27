# Open Deep Research - Claude Instructions

## Project Overview
Open Deep Research is a configurable, fully open-source deep research agent that works across multiple model providers, search tools, and MCP (Model Context Protocol) servers. It enables automated research with parallel processing, intelligent agent sequencing, and comprehensive report generation.

**Repository**: https://github.com/langchain-ai/open_deep_research  
**License**: MIT  
**Status**: Production-ready, actively maintained  
**Community**: Welcome contributors - see CONTRIBUTING.md for guidelines

## System Architecture

### Core Architecture Pattern
The system follows an "always-parallel" architecture where:
1. **Query Processing**: User submits research query through frontend
2. **LLM-Based Sequence Generation**: Generates 1-3 optimal agent sequences based on topic analysis
3. **Parallel Execution**: All sequences execute simultaneously via SequentialSupervisor
4. **Report Building**: Incremental report generation per sequence
5. **LLM Judge Evaluation**: Compares all reports and selects winner

### Key Components
- **LangGraph Integration** (`deep_researcher.py`): Main entry point with state management
- **Intelligent Sequence Generation** (`orchestration/llm_sequence_generator.py`): Strategic agent selection
- **Sequential Supervisor** (`supervisor/sequential_supervisor.py`): Parallel execution management
- **Specialized Agents** (`agents/`): Research, analysis, market, technical, synthesis specialists
- **LLM Judge** (`evaluation/llm_judge.py`): Automated report evaluation and winner selection
- **Running Report Builder** (`orchestration/report_builder.py`): Incremental report construction

## Repository Structure

### Core Implementation (`src/open_deep_research/`)
- `deep_researcher.py` - Main LangGraph implementation (entry point: `deep_researcher`)
- `configuration.py` - System configuration and model provider settings
- `state.py` - Graph state definitions and data structures  
- `prompts.py` - System prompts and templates for LLM interactions
- `utils.py` - Utility functions and helpers

#### Agents System (`agents/`)
- `registry.py` - Central agent discovery and management
- `loader.py` - Dynamic agent loading from markdown files
- `completion_detector.py` - Agent completion detection logic
- `base_agent.py` - Base agent interface and common functionality

#### Orchestration (`orchestration/`)
- `llm_sequence_generator.py` - LLM-based intelligent sequence generation
- `report_builder.py` - Incremental report building
- `sequence_generator.py` - Legacy sequence generation (fallback)

#### Supervisor System (`supervisor/`)
- `sequential_supervisor.py` - Main workflow orchestration
- `llm_sequence_generator.py` - LLM-based strategic sequence planning
- `sequence_models.py` - Data models for sequences and agent capabilities
- `context_manager.py` - State and context management

#### Sequencing (`sequencing/`)
- `simple_sequential_executor.py` - Lightweight sequential execution
- `parallel_executor.py` - Parallel sequence execution
- `sequence_engine.py` - Core sequencing logic
- `metrics.py` - Performance and execution metrics

#### Evaluation (`evaluation/`)
- `llm_judge.py` - LLM-based report evaluation and comparison
- `models.py` - Evaluation data models
- `prompts.py` - Evaluation criteria and prompts

### Legacy Implementations (`src/legacy/`)
Earlier research implementations for reference:
- `graph.py` - Plan-and-execute workflow with human-in-the-loop
- `multi_agent.py` - Supervisor-researcher multi-agent architecture
- `legacy.md` - Legacy implementation documentation

### Frontend (`frontend/`)
Modern React + TypeScript interface:
- `src/components/` - React components for research interface
- `src/hooks/` - React hooks for state management and WebSocket communication
- `src/types/` - TypeScript definitions for agents, messages, and parallel execution

## Development Environment

### Prerequisites
- Python 3.11+ with UV package manager
- Node.js 18+ for frontend development
- API keys for model providers (OpenAI, Anthropic, Google, etc.)
- Search API keys (Tavily, etc.)

### Quick Setup
```bash
# Clone and setup
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
uv venv && source .venv/bin/activate
uv sync

# Configure environment
cp .env.example .env
# Add your API keys to .env

# Start development server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

### Development Commands
- `uvx langgraph dev` - Start LangGraph Studio development server
- `ruff check` - Code linting and formatting
- `mypy src/` - Type checking
- `pytest` - Run test suite
- `python tests/run_evaluate.py` - Run comprehensive evaluations

## Configuration System

The system supports flexible configuration via:
- Environment variables (`.env` file)
- LangGraph Studio web UI
- Runtime configuration objects

### Key Configuration Areas

#### Model Configuration
```python
# Model selection for different tasks
reasoning_model: str = "claude-3-5-sonnet-20241022"
reflection_model: str = "claude-3-5-sonnet-20241022"  
evaluation_model: str = "claude-3-5-sonnet-20241022"
summarization_model: str = "openai:gpt-4.1-mini"
```

#### Search Configuration
```python
# Search provider selection
search_api: str = "tavily"  # tavily, native, duckduckgo, exa
max_search_queries: int = 5
```

#### Sequence Configuration
```python
# Parallel execution settings
enable_dynamic_sequence_generation: bool = True
max_parallel_sequences: int = 3
max_research_loops: int = 3
request_timeout: int = 300
```

## Agent Development

### Agent Structure
Agents are defined in markdown files with YAML frontmatter:

```yaml
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
examples:
  - "Literature review on machine learning interpretability"
  - "Academic paper analysis and synthesis"
completion_indicators:
  - "comprehensive literature review completed"
  - "key findings and gaps identified"
---

## System Prompt
[Agent's detailed system prompt and instructions]
```

### Available Agent Types
- **Research Agent**: Academic research, literature reviews, peer-reviewed sources
- **Analysis Agent**: Data analysis, statistical evaluation, trend identification
- **Market Agent**: Market research, competitive analysis, business intelligence
- **Technical Agent**: Technical implementation, architecture design, technology evaluation
- **Synthesis Agent**: Information synthesis, report writing, insight integration

## Performance and Quality

### Evaluation System
The system includes comprehensive evaluation via:
- **Deep Research Bench**: 100 PhD-level research tasks across 22 domains
- **LLM Judge Evaluation**: Multi-criteria scoring (completeness, depth, coherence, innovation, actionability)
- **Performance Metrics**: Execution time, token usage, cost tracking
- **Success Rate Monitoring**: Completion rates and error handling

### Performance Targets
- **Sequence Generation**: < 10 seconds for topic analysis
- **Parallel Execution**: All sequences complete within 5 minutes
- **Judge Evaluation**: < 30 seconds for comparative analysis
- **Total Workflow**: < 6 minutes for complete research cycle

### Current Performance Results
| Model Configuration | RACE Score | Cost | Tokens | Performance Notes |
|-------------------|------------|------|---------|------------------|
| GPT-5 | 0.4943 | - | 204M | Highest quality |
| Default (GPT-4.1) | 0.4309 | $45.98 | 58M | Balanced performance |
| Claude Sonnet 4 | 0.4401 | $187.09 | 139M | High quality, higher cost |

## Security and Deployment

### Security Features
- **Authentication**: LangGraph deployment authentication handler with JWT token validation
- **API Key Management**: Secure environment variable handling via .env files (never commit secrets)
- **Session Isolation**: Complete isolation between research sessions with sandboxed execution
- **Input Validation**: Comprehensive sanitization of user inputs and AI agent outputs
- **Rate Limiting**: Built-in protection against abuse and DoS attacks
- **Audit Logging**: Security event tracking for monitoring and compliance
- **No Data Persistence**: Research data not stored permanently for privacy protection

### Security Best Practices
- Use strong API keys and rotate them regularly
- Enable branch protection rules in production repositories
- Monitor for security vulnerabilities in dependencies
- Follow OWASP guidelines for AI/ML security
- Implement proper error handling to prevent information leakage

### Deployment Options
- **Development**: Local LangGraph Studio (`uvx langgraph dev --allow-blocking`)
- **Production**: LangGraph Cloud deployment with proper security configuration
- **Open Agent Platform**: UI for non-technical users to configure agents
- **Container Deployment**: Docker support for scalable deployments

### Production Checklist
- [ ] Environment variables properly configured
- [ ] API rate limits configured
- [ ] Monitoring and alerting set up
- [ ] Security scanning enabled
- [ ] Backup and recovery procedures tested

## Contributing Guidelines

### Code Quality Standards
- Follow PEP 8 for Python code style and TypeScript/React best practices for frontend
- Use type hints for better code documentation and maintainability
- Write comprehensive docstrings for public functions and components
- Maintain consistent error handling patterns across the codebase
- Add unit tests for new functionality with appropriate coverage

### Security Guidelines
- **CRITICAL**: Never commit API keys, secrets, or credentials to the repository
- Use environment variables for all configuration values
- Validate all user inputs and AI-generated outputs to prevent injection attacks
- Follow OWASP security guidelines for AI systems and web applications
- Implement proper authentication and authorization for production deployments

### Performance Considerations
- Optimize for token usage and API costs in AI interactions
- Implement proper caching strategies for search results and model outputs
- Use parallel execution patterns where appropriate for better performance
- Monitor memory usage for large contexts and implement limits
- Set reasonable timeouts for all network operations and model calls

### Development Workflow
- Follow the established repository structure and coding conventions
- Test thoroughly before submitting pull requests
- Use descriptive commit messages following conventional commit format
- Update documentation when adding new features or changing APIs
- Ensure all CI/CD checks pass before requesting review

### Open Source Best Practices
- Maintain clear and up-to-date documentation
- Respond promptly to issues and pull requests from community
- Follow semantic versioning for releases
- Keep dependencies updated and secure
- Provide clear examples and tutorials for new users