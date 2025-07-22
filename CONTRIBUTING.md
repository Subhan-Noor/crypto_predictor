# 🤝 Contributing to Crypto Price Prediction

Thank you for your interest in contributing to the Crypto Price Prediction project! We welcome contributions from developers of all skill levels.

## 📋 Table of Contents

- [🚀 Getting Started](#-getting-started)
- [🛠️ Development Setup](#️-development-setup)
- [📝 Contributing Guidelines](#-contributing-guidelines)
- [🔧 Development Workflow](#-development-workflow)
- [🧪 Testing](#-testing)
- [📖 Documentation](#-documentation)
- [🐛 Bug Reports](#-bug-reports)
- [💡 Feature Requests](#-feature-requests)
- [❓ Questions & Support](#-questions--support)

---

## 🚀 Getting Started

### Prerequisites

Before contributing, make sure you have:

- **Python 3.10+** installed
- **Node.js 18+** and npm installed
- **Git** for version control
- A **Supabase** account (free tier)
- Basic knowledge of React/Next.js and FastAPI

### Areas for Contribution

We welcome contributions in these areas:

- 🧠 **Machine Learning**: Model improvements, new algorithms, feature engineering
- 🎨 **Frontend**: UI/UX improvements, new components, performance optimization
- ⚡ **Backend**: API enhancements, data pipeline improvements, optimization
- 📚 **Documentation**: Guides, tutorials, API documentation
- 🧪 **Testing**: Unit tests, integration tests, end-to-end tests
- 🐛 **Bug Fixes**: Fixing issues and improving stability
- 🔧 **DevOps**: CI/CD improvements, monitoring, infrastructure

---

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/capstone.git
cd capstone

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/capstone.git
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your API keys and configuration

# Run database migrations (if any)
python scripts/setup_database.py

# Start development server
python run.py
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.local.example .env.local
# Edit .env.local with your configuration

# Start development server
npm run dev
```

### 4. Verify Setup

- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs

---

## 📝 Contributing Guidelines

### Code Style

**Python (Backend)**
- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Maximum line length: 88 characters

```bash
# Format code
black .

# Lint code
flake8 .
```

**TypeScript/React (Frontend)**
- Follow [Airbnb Style Guide](https://github.com/airbnb/javascript/tree/master/react)
- Use [ESLint](https://eslint.org/) for linting
- Use [Prettier](https://prettier.io/) for formatting
- Use meaningful component and variable names

```bash
# Lint code
npm run lint

# Format code
npm run format
```

### Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(ml): add LSTM model for price prediction
fix(api): resolve authentication token validation
docs: update setup instructions for Windows
test: add unit tests for price data processing
```

### Branch Naming

Use descriptive branch names:

```
feature/add-lstm-model
fix/authentication-bug
docs/api-documentation
refactor/database-queries
```

---

## 🔧 Development Workflow

### 1. Create a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Follow coding standards

### 3. Test Your Changes

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test

# Run linting
npm run lint  # Frontend
flake8 .      # Backend
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

1. Go to GitHub and create a Pull Request
2. Use the PR template (if available)
3. Provide clear description of changes
4. Link related issues
5. Request review from maintainers

---

## 🧪 Testing

### Backend Testing

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_ml_models.py

# Run specific test
pytest tests/test_api.py::test_prediction_endpoint
```

### Frontend Testing

```bash
cd frontend

# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run with coverage
npm run test:coverage
```

### Test Guidelines

- Write tests for all new features
- Maintain test coverage above 80%
- Use descriptive test names
- Test both success and error cases
- Mock external dependencies

---

## 📖 Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings, inline comments
2. **API Documentation**: FastAPI auto-generated docs
3. **User Documentation**: README, setup guides
4. **Developer Documentation**: Architecture, contributing

### Documentation Standards

**Python Docstrings** (Google Style):
```python
def predict_price(currency: str, days: int) -> dict:
    """Predict cryptocurrency price direction.
    
    Args:
        currency: Currency symbol (BTC, ETH)
        days: Number of days for prediction
        
    Returns:
        Dictionary containing prediction and confidence
        
    Raises:
        ValueError: If currency is not supported
    """
```

**TypeScript/JSDoc**:
```typescript
/**
 * Fetches cryptocurrency price data from API
 * @param currency - Currency symbol (BTC, ETH)
 * @param days - Number of days of historical data
 * @returns Promise resolving to price data array
 */
async function fetchPriceData(currency: string, days: number): Promise<PriceData[]>
```

---

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues to avoid duplicates
2. Try to reproduce the bug
3. Test with latest version
4. Gather system information

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g. Windows 10, macOS Big Sur]
- Browser: [e.g. Chrome 95, Firefox 94]
- Node.js version: [e.g. 18.0.0]
- Python version: [e.g. 3.10.0]

**Additional Context**
- Error logs
- Screenshots
- Any other relevant information
```

---

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Problem/Use Case**
What problem would this feature solve?

**Proposed Solution**
How would you like this feature to work?

**Alternatives Considered**
Any alternative solutions you've considered.

**Additional Context**
- Mockups or examples
- Related issues or discussions
```

---

## 📊 Project Areas

### Machine Learning
- **Skills Needed**: Python, scikit-learn, PyTorch, pandas
- **Tasks**: Model improvements, feature engineering, evaluation metrics
- **Files**: `backend/ml/`, `backend/models/`

### Frontend Development
- **Skills Needed**: React, TypeScript, Next.js, TailwindCSS
- **Tasks**: UI components, dashboard improvements, responsive design
- **Files**: `frontend/app/`, `frontend/components/`

### Backend Development
- **Skills Needed**: Python, FastAPI, PostgreSQL, Redis
- **Tasks**: API endpoints, data processing, performance optimization
- **Files**: `backend/app/`, `backend/services/`

### Data Engineering
- **Skills Needed**: Python, pandas, APIs, data pipelines
- **Tasks**: Data ingestion, processing, sentiment analysis
- **Files**: `backend/scripts/`, `backend/services/`

### DevOps
- **Skills Needed**: Docker, CI/CD, monitoring, deployment
- **Tasks**: Infrastructure improvements, automation, monitoring
- **Files**: `.github/`, `Dockerfile`, deployment configs

---

## ❓ Questions & Support

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Discord**: Real-time chat and support
- **Email**: maintainers@cryptoprediction.com

### Getting Help

1. **Check Documentation**: README, setup guides
2. **Search Issues**: Existing questions and solutions
3. **Ask Questions**: Use GitHub Discussions
4. **Join Discord**: Real-time community support

---

## 🏆 Recognition

Contributors will be recognized in:

- **README**: Contributors section
- **Releases**: Changelog acknowledgments
- **GitHub**: Contributor graphs and stats
- **Discord**: Special contributor roles

---

## 📜 Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

### Our Standards

- **Be Respectful**: Treat everyone with respect and kindness
- **Be Inclusive**: Welcome newcomers and diverse perspectives
- **Be Collaborative**: Work together and help each other
- **Be Professional**: Maintain professional communication

---

## 🔄 Release Process

### Version Numbers

We follow [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes, backward compatible

### Release Cycle

- **Major Releases**: Quarterly
- **Minor Releases**: Monthly
- **Patch Releases**: As needed for critical fixes

---

Thank you for contributing to Crypto Price Prediction! 🚀

Together, we're building the future of cryptocurrency price prediction with AI and machine learning. 