# Crypto Predictor

A real-time cryptocurrency price prediction application using machine learning and websockets. This application provides predictions for various cryptocurrencies including BTC, ETH, XRP, and DOT.

## Features

- Real-time price updates via WebSockets
- Machine learning-based price predictions
- Interactive web interface
- Support for multiple cryptocurrencies

## Project Structure

- `/api`: FastAPI backend with WebSocket endpoints
- `/web`: Frontend application

## Setup Instructions

### Prerequisites

- Python 3.8+
- Virtual environment
- Node.js (for frontend)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Subhan-Noor/crypto_predictor.git
   cd crypto_predictor
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # OR
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the API server:
   ```
   cd api
   uvicorn app:app --reload
   ```

5. Access the application at `http://localhost:8000`

## Technologies Used

- FastAPI
- WebSockets
- Python ML libraries (scikit-learn, pandas, etc.)
- JavaScript/HTML/CSS for frontend

## Project Overview
This project aims to develop a comprehensive cryptocurrency price prediction application that combines historical price data with sentiment analysis to forecast future price movements. The system supports multiple cryptocurrencies and provides an interactive interface for users to explore predictions and historical trends.

## Key Features
- Multi-cryptocurrency support with historical price data collection
- Sentiment analysis integration (news, social media, Fear & Greed Index)
- Multiple prediction models (time series, ML, deep learning)
- Interactive visualization dashboard
- Customizable prediction timeframes
- Backtesting capabilities for model evaluation

## Project Structure
```
crypto-predictor/
├── data_fetchers/             # Data collection modules
│   ├── crypto_data_fetcher.py # Fetch price data for any cryptocurrency
│   ├── sentiment_fetcher.py   # Fetch sentiment data (news, social, F&G index)
│   └── market_data_fetcher.py # Fetch broader market indicators
├── processors/                # Data processing and feature engineering
│   ├── data_processor.py      # Clean and prepare price data
│   └── sentiment_analyzer.py  # Process and score sentiment data
├── models/                    # ML and prediction models
│   ├── model_factory.py       # Model initialization and configuration
│   ├── price_predictor.py     # Price prediction models
│   └── sentiment_model.py     # Sentiment impact models
├── evaluation/                # Testing and evaluation tools
│   ├── backtester.py          # Historical performance testing
│   └── visualizer.py          # Performance visualization
├── api/                       # API for serving predictions
│   └── app.py                 # FastAPI/Flask application
├── frontend/                  # Web interface
│   ├── src/                   # React/Vue components
│   └── public/                # Static assets
├── config/                    # Configuration files
│   ├── settings.py            # Application settings
│   └── api_keys.py            # API key management (gitignored)
└── utils/                     # Utility functions
    ├── logger.py              # Logging configuration
    └── helpers.py             # Common helper functions
```

## Development Roadmap

### Phase 1: Data Infrastructure
1. Extend existing XRP data fetcher to support multiple cryptocurrencies
2. Implement sentiment data collection (Fear & Greed Index, news, social)
3. Create caching mechanisms to manage API limits

### Phase 2: Data Processing
4. Develop data cleaning and normalization routines
5. Implement feature engineering (technical indicators)
6. Build sentiment scoring and normalization

### Phase 3: Model Development
7. Create time series models (ARIMA, Prophet)
8. Implement ML models (Random Forest, XGBoost)
9. Develop deep learning models (LSTM networks)
10. Build ensemble approaches

### Phase 4: Evaluation & Refinement
11. Create comprehensive backtesting framework
12. Implement performance metrics and visualization
13. Optimize models based on backtesting results

### Phase 5: Application Development
14. Develop API endpoints
15. Create interactive web interface
16. Implement real-time updates

## Existing Components
- `fetch_xrp_data.py`: Current implementation for XRP data collection with multi-API support and fallback mechanisms

## Technical Considerations for LLM Assistance

### Data Collection
- The system should support fetching data from multiple APIs with fallback mechanisms
- Implementation should follow the pattern in `fetch_xrp_data.py` but be generalized for any cryptocurrency
- Proper error handling and retry logic is essential

### Sentiment Analysis
- Sentiment data should be normalized to be comparable across sources
- Fear & Greed Index historical data can be used as a baseline
- News sentiment should be scored using NLP techniques
- Social media sentiment requires filtering for relevant content

### Model Development
- Models should be modular and interchangeable
- Feature importance should be tracked to understand prediction factors
- Time series data requires special handling (train/test splits, validation)
- Consider both point predictions and probability distributions

### Frontend Considerations
- Visualizations should clearly communicate prediction confidence
- Interface should allow comparison between different models
- Historical performance should be transparent to users

## API Integration Notes
- CryptoCompare offers comprehensive historical data (2000 days max per request)
- Alpha Vantage provides good daily OHLCV data but has stricter rate limits
- Alternative Market Fear & Greed Index: https://alternative.me/crypto/fear-and-greed-index/
- News APIs: NewsAPI, CryptoPanic, Coinpaprika
- Social sentiment: Twitter/X API (limited access), Reddit API

## Implementation Notes
- Prioritize modular design to allow easy addition of new data sources and models
- Use appropriate caching strategies to minimize API calls
- Consider async programming for better performance in data collection
- Balance model complexity with practical usability/performance

feature_columns = ['open', 'high', 'low', 'volume', 'ma7', 'ma21', 'rsi']
target_column = 'close'