# Contributing to Crypto Prediction Platform

Thank you for your interest in contributing to the Crypto Prediction Platform! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Node.js 18+
- Git
- Supabase account (for database)

### Setup Development Environment
```bash
# Clone the repository
git clone <your-fork-url>
cd capstone

# Backend setup
cd backend
pip install -r requirements.txt
cp env_template.txt .env
# Edit .env with your Supabase credentials

# Frontend setup
cd ../frontend
npm install
```

## 📋 Contribution Guidelines

### Code Style

**Python (Backend)**
- Follow PEP 8 style guide
- Use type hints
- Add docstrings to functions
- Keep functions under 50 lines

**TypeScript/React (Frontend)**
- Use TypeScript strict mode
- Follow ESLint rules
- Use functional components with hooks
- Add JSDoc comments for complex functions

### Commit Messages
Use conventional commit format:
```
feat: add new prediction model
fix: resolve timezone issue in data ingestion
docs: update API documentation
test: add unit tests for prediction pipeline
```

### Pull Request Process
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit with conventional commit message
7. Push to your fork
8. Create a Pull Request

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Manual Testing
1. Test API endpoints with Postman/curl
2. Verify frontend functionality
3. Check data flow end-to-end

## 📚 Documentation

### Code Documentation
- Add docstrings to all functions
- Include type hints
- Document complex algorithms
- Add inline comments for tricky logic

### API Documentation
- Update `API_DOCS.md` for new endpoints
- Include request/response examples
- Document error codes

### User Documentation
- Update `README.md` for new features
- Add setup instructions for new dependencies
- Document configuration options

## 🔧 Development Workflow

### Feature Development
1. **Plan**: Create issue describing the feature
2. **Design**: Discuss approach in issue comments
3. **Implement**: Write code with tests
4. **Review**: Self-review before PR
5. **Test**: Ensure all tests pass
6. **Document**: Update relevant docs

### Bug Fixes
1. **Reproduce**: Create minimal test case
2. **Fix**: Implement the fix
3. **Test**: Add regression test
4. **Document**: Update changelog

## 🎯 Areas for Contribution

### High Priority
- **ML Model Improvements**: Better feature engineering, ensemble methods
- **Performance Optimization**: Caching, database queries, API response times
- **Testing**: Unit tests, integration tests, end-to-end tests
- **Documentation**: API docs, user guides, code comments

### Medium Priority
- **UI/UX Improvements**: Better charts, responsive design, accessibility
- **New Features**: Additional cryptocurrencies, advanced analytics
- **Monitoring**: Health checks, error tracking, performance metrics
- **Security**: Input validation, rate limiting, authentication

### Low Priority
- **Infrastructure**: Docker improvements, CI/CD enhancements
- **Tools**: Development scripts, debugging utilities
- **Examples**: Sample data, demo applications

## 🐛 Bug Reports

### Before Submitting
1. Check existing issues
2. Try to reproduce the bug
3. Check if it's a configuration issue

### Bug Report Template
```markdown
**Description**
Brief description of the bug

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10, macOS 12]
- Python: [e.g., 3.9.7]
- Node.js: [e.g., 18.0.0]

**Additional Context**
Screenshots, logs, etc.
```

## 💡 Feature Requests

### Before Submitting
1. Check if feature already exists
2. Consider if it fits the project scope
3. Think about implementation complexity

### Feature Request Template
```markdown
**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
How should this feature work?

**Alternative Solutions**
Other ways to solve this problem

**Additional Context**
Use cases, examples, etc.
```

## 🤝 Code Review

### Review Process
1. **Automated Checks**: CI/CD must pass
2. **Code Review**: At least one approval required
3. **Testing**: Manual testing encouraged
4. **Documentation**: Docs must be updated

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Backward compatibility maintained

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🆘 Getting Help

### Questions & Discussion
- Create a GitHub issue for questions
- Use issue labels appropriately
- Be respectful and constructive

### Resources
- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [React Documentation](https://reactjs.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🙏 Recognition

Contributors will be recognized in:
- Project README
- Release notes
- GitHub contributors page

Thank you for contributing to the Crypto Prediction Platform! 🚀 