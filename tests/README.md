# Testing Documentation

## Test Structure

The test suite validates the Open Deep Research system across multiple dimensions:

- **Unit Tests**: Individual component validation (`test_*.py` files)
- **Integration Tests**: Component interaction validation  
- **Performance Tests**: Timing and resource usage validation
- **Evaluation Tests**: LLM judge and research quality validation

## Running Tests

```bash
# Run full test suite
python -m pytest tests/

# Run evaluation suite with research benchmarks
python tests/run_evaluate.py

# Run specific test categories
python -m pytest tests/test_agent_registry_loading.py
python -m pytest tests/test_llm_judge_integration.py
python -m pytest tests/test_parallel_execution_integration.py
```

## Test Categories

### Core Functionality Tests
- `test_agent_registry_loading.py` - Agent discovery and loading
- `test_completion_detection.py` - Agent completion logic
- `test_configuration_system.py` - Configuration validation
- `test_integration_flow.py` - End-to-end workflow validation

### LLM Integration Tests  
- `test_llm_judge_integration.py` - Report evaluation system
- `test_llm_sequence_generation.py` - Dynamic sequence generation
- `test_sequence_generation_llm_judge.py` - Integrated sequence and judge testing

### Parallel Processing Tests
- `test_parallel_execution_integration.py` - Multi-sequence execution
- `test_sequential_supervisor_integration.py` - Supervisor orchestration
- `test_performance_benchmarks.py` - Performance validation

### Evaluation Framework
- `run_evaluate.py` - Comprehensive research quality evaluation
- `supervisor_parallel_evaluation.py` - Parallel execution benchmarking
- `expt_results/` - Evaluation results from different model configurations

## Performance Targets

- **Sequence Generation**: < 10 seconds for topic analysis
- **Parallel Execution**: All sequences complete within 5 minutes  
- **Judge Evaluation**: < 30 seconds for comparative analysis
- **Total Workflow**: < 6 minutes for complete research cycle

## Evaluation Results

Current performance results from Deep Research Bench (100 PhD-level tasks):

| Model Configuration | RACE Score | Cost | Tokens | Performance Notes |
|-------------------|------------|------|---------|------------------|
| GPT-5 | 0.4943 | - | 204M | Highest quality |
| Default (GPT-4.1) | 0.4309 | $45.98 | 58M | Balanced performance |
| Claude Sonnet 4 | 0.4401 | $187.09 | 139M | High quality, higher cost |

## Dependencies

The test suite requires:
```bash
pip install pytest pytest-asyncio psutil
```

For comprehensive testing and coverage:
```bash
pip install pytest-cov pytest-xdist
```