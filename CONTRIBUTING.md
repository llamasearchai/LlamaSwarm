# Contributing to LlamaSwarm

Thank you for your interest in contributing to LlamaSwarm! We're excited to have you join our community. This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand the expectations we have for all contributors.

## How to Contribute

There are many ways to contribute to LlamaSwarm:

- Reporting bugs and issues
- Suggesting new features or improvements
- Improving documentation
- Adding or improving tests
- Submitting code changes and fixes
- Reviewing pull requests

### Reporting Bugs

If you find a bug, please open an issue on our [GitHub Issues](https://github.com/llamaswarm/llamaswarm/issues) page. When reporting a bug, please include:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior and actual behavior
- Environment information (OS, Python version, LlamaSwarm version)
- Any additional context that might be helpful

### Feature Requests

We welcome suggestions for new features or improvements. To suggest a feature, please open an issue on our [GitHub Issues](https://github.com/llamaswarm/llamaswarm/issues) page with the label "enhancement". Please include:

- A clear, descriptive title
- A detailed description of the proposed feature
- Any relevant background information or use cases
- If possible, an outline of how you envision the feature being implemented

### Contributing Code

1. **Fork the repository**: Create a fork of the LlamaSwarm repository on GitHub.

2. **Clone your fork**: Clone your fork to your local machine.
   ```bash
   git clone https://github.com/YOUR-USERNAME/llamaswarm.git
   cd llamaswarm
   ```

3. **Set up development environment**:
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

4. **Create a branch**: Create a branch for your changes.
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes**: Implement your changes, following our coding standards.

6. **Add tests**: Add tests for your changes to ensure they work as expected.

7. **Run tests**: Make sure all tests pass.
   ```bash
   pytest
   ```

8. **Format your code**: Ensure your code follows our coding standards.
   ```bash
   black llamaswarm
   isort llamaswarm
   flake8 llamaswarm
   ```

9. **Commit your changes**: Commit your changes with a clear, descriptive message.
   ```bash
   git commit -m "Add feature: your feature description"
   ```

10. **Push your changes**: Push your changes to your fork.
    ```bash
    git push origin feature/your-feature-name
    ```

11. **Submit a pull request**: Open a pull request from your fork to the main LlamaSwarm repository.

## Coding Standards

We follow these coding standards for LlamaSwarm:

- **PEP 8**: Follow the [PEP 8](https://pep8.org/) style guide for Python code.
- **Docstrings**: Use NumPy-style docstrings for all functions, classes, and modules.
- **Type Hints**: Use type hints where appropriate to improve code readability and tooling support.
- **Testing**: Write tests for all new features and bug fixes. We use pytest for testing.
- **Code Formatting**: Use Black for code formatting and isort for import sorting.

## Pull Request Guidelines

When submitting a pull request, please:

- Include a clear, descriptive title
- Reference any related issues
- Include a summary of the changes and why they are necessary
- Ensure all tests pass
- Update documentation if necessary
- Follow our coding standards

## Review Process

Pull requests will be reviewed by project maintainers. We may request changes or additional information before merging. Please be responsive to feedback and be willing to make changes if necessary.

## License

By contributing to LlamaSwarm, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Questions and Support

If you have questions or need help, please open an issue on GitHub or reach out to the maintainers directly.

Thank you for contributing to LlamaSwarm! 