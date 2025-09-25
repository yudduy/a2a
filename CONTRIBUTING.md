# Contributing to Open Deep Research

Thank you for your interest in contributing to Open Deep Research! We welcome contributions from the community.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker to report bugs
- Include a clear description and steps to reproduce
- Provide system information and error messages
- Check if the issue already exists before creating a new one

### Suggesting Features
- Check existing issues for similar suggestions
- Open a new issue with a clear description
- Explain the use case and expected behavior
- Consider implementation complexity and project scope

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/open_deep_research.git
   cd open_deep_research
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Set up development environment**
   ```bash
   uv venv && source .venv/bin/activate
   uv sync
   cp .env.example .env  # Add your API keys
   ```

4. **Make your changes**
   - Follow the existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

5. **Test your changes**
   ```bash
   # Run linting
   ruff check --fix src/
   ruff format src/
   
   # Run type checking
   mypy src/
   
   # Run tests
   pytest
   
   # Run evaluation (requires API keys)
   python tests/run_evaluate.py
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature
   
   - Add detailed description of changes
   - Reference any related issues
   
   🤖 Generated with Claude Code
   
   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

7. **Push and create PR**
   ```bash
   git push origin feature/amazing-feature
   ```
   Then open a pull request on GitHub.

## Development Guidelines

### Code Style
- **Python**: Follow PEP 8, use type hints, add docstrings
- **TypeScript**: Follow standard TypeScript conventions
- **React**: Use functional components with hooks
- **Tests**: Write comprehensive tests for new features

### Security
- **Never commit API keys or secrets**
- Use environment variables for configuration
- Validate all inputs and outputs
- Follow OWASP guidelines for AI systems

### Performance
- Optimize for token usage and API costs
- Implement proper caching strategies
- Use parallel execution where appropriate
- Monitor memory usage for large contexts

### Documentation
- Update README.md for user-facing changes
- Add docstrings for new functions and classes
- Include examples for complex features
- Update CLAUDE.md for development guidance

## Project Structure

```
src/open_deep_research/
├── deep_researcher.py          # Main LangGraph implementation
├── configuration.py            # System configuration
├── state.py                   # Graph state definitions
├── agents/                    # Agent system
├── orchestration/            # Sequence generation & report building
├── supervisor/               # Workflow orchestration
├── sequencing/              # Execution engines
└── evaluation/              # LLM judge & evaluation
```

## Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Tests
```bash
python tests/run_evaluate.py --limit 5
```


## Release Process

1. Update version in `pyproject.toml`
2. Create a pull request with version bump
3. After merge, create a GitHub release with tag

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/langchain-ai/open_deep_research/issues)
- **Discussions**: [GitHub Discussions](https://github.com/langchain-ai/open_deep_research/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Open Deep Research! 🚀