# Open Deep Research - Claude Instructions

## Project Overview
Open Deep Research is a configurable, fully open-source deep research agent that works across multiple model providers, search tools, and MCP (Model Context Protocol) servers. It enables automated research with parallel processing, intelligent agent sequencing, and comprehensive report generation.

**Repository**: https://github.com/langchain-ai/open_deep_research  
**License**: MIT  
**Status**: Stable / Production-ready as of 2025-08, actively maintained  
**Security**: See `SECURITY.md` for our Security Policy and responsible disclosure process; report issues via GitHub Security Advisories or email `security@your-org.com`.
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
- **Authentication (JWT)**: Only `RS256`/`ES256` accepted; require `iss` and `aud` claims; short access-token TTL (≤15m) with refresh-token rotation and server-side revocation; strict signature verification with clock-skew tolerance ≤120s.
- **API Key Management**: Secrets sourced from environment/secret manager (never committed); least-privilege keys with periodic rotation and scoping.
- **Session Isolation**: Complete isolation between research sessions with sandboxed execution.
- **Input/Output Validation**: Schema/whitelist validation per endpoint; enforce content-type, max field lengths and payload sizes; reject unexpected fields; canonicalize and escape outputs; sanitize/guardrail AI outputs.
- **Rate Limiting**: Per-user and per-IP limits with burst allowance using sliding-window or token-bucket; respond `429` and document exponential backoff policy; thresholds configurable and covered by tests.
- **Audit Logging (Qualified)**: Security/audit logs are retained securely for a limited retention period, protected by access controls and encryption, and automatically purged per retention policy (retention can be configured where appropriate).
- **No Data Persistence (Clarified)**: Research data is not persisted permanently. Only short-term, access-controlled audit logs required for security and compliance may be retained per policy.
- **Verification & CI**: Unit/integration tests and CI checks assert JWT constraints, validation rules, rate limits, and log retention behavior; configuration defaults documented; monitoring/logging enable enforcement and incident investigation.

### Security Best Practices
- Use a managed secrets store (e.g., Vault, AWS Secrets Manager); never commit secrets to the repo.
- Enforce least-privilege API keys with regular rotation; provision environment-specific secrets.
- Implement robust input validation, output redaction, and prompt-injection defenses aligned with OWASP LLM Top 10.
- Apply rate limiting and access controls; log and audit access without leaking sensitive outputs.
- Enable dependency vulnerability scanning (SCA) and schedule regular threat modeling.

### Deployment Options
- **Development**: Local LangGraph Studio (`uvx langgraph dev --allow-blocking`).
- **Containerized Production**: Recommended to deploy containers with restricted permissions and secret files mounted read-only. Example `docker-compose.yml`:

```yaml
version: "3.9"
services:
  deep-researcher:
    image: ghcr.io/langchain-ai/open-deep-research:latest
    restart: unless-stopped
    read_only: true
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=64m
    environment:
      - LOG_LEVEL=info
      - RATE_LIMIT_USER_PER_MINUTE=60
      - RATE_LIMIT_IP_PER_MINUTE=120
      - JWT_AUDIENCE=odr
      - JWT_ISSUER=https://issuer.example.com
      - VITE_API_URL=https://api.example.com
    secrets:
      - openai_api_key
      - anthropic_api_key
    ports:
      - "2024:2024"
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G

secrets:
  openai_api_key:
    file: ./secrets/openai_api_key
  anthropic_api_key:
    file: ./secrets/anthropic_api_key
```

Note: For production at scale, use an orchestrator (e.g., Kubernetes) with network policies, sealed/encrypted secrets, and per-namespace RBAC rather than `env_file`.

### Production Checklist
**Security**
- [ ] Secrets in a managed store; rotation policy documented and automated
- [ ] JWT configuration validated (alg, claims, TTL); refresh token revocation in place
- [ ] Role/permission reviews performed; access audit logging enabled and retained per policy

**Observability**
- [ ] Centralized logs with retention and log aggregation (e.g., ELK, CloudWatch)
- [ ] Metrics exported (latency, error rate, cost) with SLO/SLI defined and alerts configured

**Operations**
- [ ] CI/CD with policy checks, canary/rollback plan tested
- [ ] Incident response runbooks and on-call rotation defined
- [ ] Capacity planning done; autoscaling policies tested under load

**Compliance & Data Protection**
- [ ] Encryption in transit and at rest verified; DLP policies documented
- [ ] Dependency vulnerability scanning and patch cadence established

**Backups & DR**
- [ ] Backups verified; RTO/RPO targets defined and tested

## Contributing Guidelines

### Code Quality Standards
- Pre-commit hooks required (Python: `ruff`, `black`, `isort`, `mypy`; Frontend: `eslint`, `prettier`). Add `.pre-commit-config.yaml` and enforce hooks on commit.
- Enable strict type checks (`mypy --strict`; TypeScript strict mode) in repo configs.
- CI jobs must run linters, type checks, and tests with coverage (`pytest-cov`/`jest`) and enforce ≥80% line/branch coverage.
- Maintain consistent error handling patterns and comprehensive docstrings for public APIs.

### Security Guidelines
- **CRITICAL**: Never commit API keys, secrets, or credentials to the repository
- Use environment variables for all configuration values
- Validate all user inputs and AI-generated outputs to prevent injection attacks
- Follow OWASP security guidelines for AI systems and web applications
- Implement proper authentication and authorization for production deployments

### Performance Considerations
- Apply request timeouts and retry policies with exponential backoff and jitter.
- Implement circuit breakers and bulkhead isolation around external AI/model calls.
- Provide graceful degradation/fallbacks and clear cache invalidation strategies.
- Enforce rate limiting and cost-aware throttling; surface dead-letter handling for failed jobs.
- Expose metrics/alerts with configurable thresholds to tune limits and observe failures.

### Development Workflow
- Include governance artifacts: `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/`, commit conventions in `CONTRIBUTING.md`, and a `CODEOWNERS` file.
- Follow repository structure and coding conventions; ensure all CI checks pass before review.
- Test thoroughly; update documentation when adding features or changing APIs.

### Open Source Best Practices
- Provide and maintain `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and a `MAINTAINERS.md` or `COMMUNITY.md`.
- Enable automated dependency updates (e.g., Dependabot); document schedules and CI badges for builds and code scanning.
- Follow semantic versioning; keep examples and docs current; respond promptly to community contributions.