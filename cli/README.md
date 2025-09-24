# 🔬 Research CLI System

A command-line interface for advanced AI research orchestration, implementing the blueprint architecture with **A2A protocol**, **LangGraph orchestration**, **Langfuse tracing**, and **GRPO learning**.

## 🏗️ Architecture Overview

The CLI system implements a sophisticated research orchestration framework:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Research CLI System                     │
├─────────────────────────────────────────────────────────────────┤
│  🔗 A2A Protocol Layer                                          │
│  • Agent-to-Agent communication                                │
│  • Task delegation and result aggregation                      │
│  • Agent capability discovery                                  │
├─────────────────────────────────────────────────────────────────┤
│  🧠 LangGraph Orchestration                                    │
│  • Graph-based state management                                │
│  • Parallel sequence execution                                 │
│  • Conditional workflow routing                                │
├─────────────────────────────────────────────────────────────────┤
│  📊 Langfuse Observability                                     │
│  • Execution tracing and monitoring                            │
│  • GRPO reinforcement learning                                 │
│  • Performance metrics and analytics                          │
├─────────────────────────────────────────────────────────────────┤
│  🌳 Hierarchical Context Management                            │
│  • Intelligent context compression                             │
│  • Selective context inheritance                               │
│  • Token-efficient context trees                               │
├─────────────────────────────────────────────────────────────────┤
│  🤖 Specialized Research Agents                                │
│  • Academic research agent                                     │
│  • Technical analysis agent                                    │
│  • Market intelligence agent                                   │
│  • Research synthesis agent                                    │
├─────────────────────────────────────────────────────────────────┤
│  💻 Rich CLI Interface                                         │
│  • Beautiful terminal output                                   │
│  • Real-time streaming progress                                │
│  • Interactive research mode                                   │
│  • Command completion and help                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/research-cli.git
cd research-cli

# Install dependencies
pip install -e .

# Run CLI
python -m cli
```

### Basic Usage

```bash
# Interactive mode (recommended)
python -m cli

# Single research query
python -m cli research "quantum computing applications in healthcare"

# Streaming research output
python -m cli research --stream "AI ethics and governance frameworks"

# Training mode (collect episodes for GRPO)
python -m cli train 100

# View system statistics
python -m cli stats

# Show configuration
python -m cli config
```

## 📚 Key Features

### 🔗 A2A Protocol Integration
- **Standardized agent communication** using Google's A2A protocol
- **Agent Cards** for capability advertisement and discovery
- **Task delegation** with structured message passing
- **Error handling** and retry mechanisms

### 🧠 LangGraph Orchestration
- **Graph-based workflows** for complex research execution
- **Parallel sequence generation** with LLM-powered optimization
- **Conditional branching** based on research progress
- **Stateful execution** with checkpoint and recovery

### 📊 Langfuse Observability
- **Comprehensive tracing** of all agent executions
- **GRPO reinforcement learning** from human feedback
- **Performance monitoring** and bottleneck identification
- **Episode collection** for continuous improvement

### 🌳 Hierarchical Context Management
- **Intelligent compression** for different content types
- **Selective inheritance** of relevant context
- **Token-efficient trees** with root (20k) and branch (80k) limits
- **Context optimization** with automatic cleanup

### 🤖 Specialized Agents
- **Academic Agent**: Literature reviews and theoretical analysis
- **Technical Agent**: Implementation and architecture research
- **Market Agent**: Business intelligence and competitive analysis
- **Synthesis Agent**: Research integration and report generation

### 💻 Rich CLI Interface
- **Beautiful terminal output** with Rich formatting
- **Real-time streaming** progress visualization
- **Interactive commands** with auto-completion
- **Progress tracking** with live updates

## 🎯 Usage Examples

### Research Execution

```bash
# Basic research
python -m cli research "renewable energy trends 2024"

# Complex multi-domain research
python -m cli research "blockchain applications in supply chain management"

# Technical deep-dive
python -m cli research "machine learning optimization techniques for large language models"
```

### Training and Learning

```bash
# Collect training episodes
python -m cli train 200

# View learning progress
python -m cli stats

# Analyze orchestration insights
python -m cli insights
```

### Interactive Mode

```bash
python -m cli interactive
```

This launches an interactive session where you can:
- Execute multiple research queries
- View real-time progress
- Access help and configuration
- Monitor system performance

## 🔧 Configuration

### Environment Variables

```bash
# Langfuse Configuration (optional)
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=https://api.langfuse.com

# Model Configuration
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# A2A Server Configuration
A2A_HOST=0.0.0.0
A2A_PORT=8000
```

### Configuration File

Create a `cli_config.json` file:

```json
{
  "langfuse": {
    "public_key": "your-public-key",
    "secret_key": "your-secret-key",
    "host": "https://api.langfuse.com"
  },
  "models": {
    "default_provider": "openai",
    "max_tokens": 16000
  },
  "a2a": {
    "host": "0.0.0.0",
    "port": 8000,
    "timeout": 60
  }
}
```

## 📊 Architecture Details

### A2A Protocol Layer
```
┌─────────────────────────────────────────────────────────────┐
│                     A2A Protocol Layer                      │
├─────────────────────────────────────────────────────────────┤
│  AgentCard Discovery                                        │
│  • Capability advertisement                                 │
│  • Schema validation                                        │
│  • Endpoint registration                                    │
├─────────────────────────────────────────────────────────────┤
│  Task Delegation                                            │
│  • Structured message passing                               │
│  • Context compression                                      │
│  • Result aggregation                                       │
├─────────────────────────────────────────────────────────────┤
│  Error Handling                                             │
│  • Retry mechanisms                                         │
│  • Graceful degradation                                     │
│  • Failure recovery                                         │
└─────────────────────────────────────────────────────────────┘
```

### LangGraph Orchestration
```
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph Orchestration                    │
├─────────────────────────────────────────────────────────────┤
│  Workflow Definition                                        │
│  • StateGraph construction                                  │
│  • Node and edge definition                                 │
│  • Conditional routing                                      │
├─────────────────────────────────────────────────────────────┤
│  Parallel Execution                                         │
│  • Concurrent sequence execution                            │
│  • Resource management                                      │
│  • Load balancing                                           │
├─────────────────────────────────────────────────────────────┤
│  State Management                                           │
│  • Shared state across agents                               │
│  • Checkpoint and recovery                                  │
│  • Streaming updates                                        │
└─────────────────────────────────────────────────────────────┘
```

### Context Management
```
┌─────────────────────────────────────────────────────────────┐
│                 Hierarchical Context Tree                   │
├─────────────────────────────────────────────────────────────┤
│  Root Context (20k tokens)                                  │
│  • Orchestrator coordination                                │
│  • High-level task management                              │
│  • Result aggregation                                       │
├─────────────────────────────────────────────────────────────┤
│  Branch Contexts (80k tokens each)                          │
│  • Agent-specific context                                   │
│  • Selective inheritance                                    │
│  • Intelligent compression                                  │
├─────────────────────────────────────────────────────────────┤
│  Compression Strategy                                       │
│  • Adaptive content compression                             │
│  • Type-specific optimization                               │
│  • Context relevance scoring                                │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python -m pytest cli/tests/

# Run specific test module
python -m pytest cli/tests/test_cli_system.py

# Run with coverage
python -m pytest cli/tests/ --cov=cli/
```

### Integration Tests

```bash
# Test CLI functionality
python -c "
import asyncio
from cli import ResearchCLIApp

async def test():
    app = ResearchCLIApp()
    await app.initialize()
    result = await app.run_research('test query')
    print(f'Result: {result.synthesis[:100]}...')
    await app.close()

asyncio.run(test())
"
```

## 🔍 Monitoring and Debugging

### Logging
```bash
# View logs
tail -f research_cli.log

# Configure log level
export LOG_LEVEL=DEBUG
python -m cli research "debug query"
```

### Performance Monitoring
```bash
# View system statistics
python -m cli stats

# Monitor real-time performance
python -m cli config
```

### Trace Analysis
```bash
# Analyze execution traces
python -c "
from cli.orchestration.trace_collector import TraceCollector
collector = TraceCollector()
stats = collector.get_session_summary()
print(f'Session: {stats}')
"
```

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd research-cli
pip install -e ".[dev]"
```

### Code Structure
```
cli/
├── core/                    # Core system components
│   ├── a2a_client.py       # A2A protocol implementation
│   ├── context_tree.py     # Hierarchical context management
│   └── cli_interface.py    # Rich CLI interface
├── orchestration/          # Orchestration layer
│   ├── langgraph_orchestrator.py  # Graph-based orchestration
│   └── trace_collector.py         # Langfuse tracing
├── agents/                 # Research agents
│   ├── research_agent.py   # Base agent implementation
│   └── [specialized agents]
├── utils/                  # Utilities
│   └── research_types.py   # Type definitions
└── tests/                  # Comprehensive test suite
```

### Adding New Agents
```python
from cli.agents.research_agent import ResearchAgent

class CustomAgent(ResearchAgent):
    def __init__(self):
        super().__init__(
            name="custom_agent",
            description="Custom research agent",
            capabilities=["custom_analysis"]
        )

    async def _execute(self, task):
        # Custom execution logic
        return {"summary": "Custom result", "insights": []}
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙋‍♀️ Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for Q&A
- **Documentation**: This README and inline documentation

---

**Built with ❤️ using cutting-edge AI orchestration technologies**
