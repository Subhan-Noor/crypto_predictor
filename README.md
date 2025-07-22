# Crypto Price Prediction Web App

This web application uses Machine Learning (ML) to predict if the price of Bitcoin (BTC) and Ethereum (ETH) will increase or decrease over the next 7 days. It leverages historical price data and sentiment analysis (Fear & Greed Index, Twitter, Reddit) to make predictions.

## 🚀 Current Status: Stage 2 (Data Acquisition & Storage)

### ✅ Completed (Stage 1 & 2):
- **Project Setup**: FastAPI backend + Next.js frontend with TypeScript
- **Database**: Supabase PostgreSQL with proper schema
- **Data Fetching**: CoinGecko API integration for historical price data (OHLCV)
- **Sentiment Analysis**: Fear & Greed Index, Twitter, and Reddit sentiment analysis
- **Automated Data Pipeline**: Daily data ingestion via GitHub Actions
- **API Endpoints**: Basic endpoints for price data, sentiment data, and health checks

### 🔄 In Progress:
- ML Model Development (Stage 3)
- Enhanced Frontend Dashboard (Stage 5)

## 🛠 Tech Stack

- **Frontend**: Next.js 15 with TypeScript, TailwindCSS v4
- **Backend**: Python FastAPI with async support
- **Database**: Supabase (PostgreSQL)
- **ML Libraries**: Scikit-learn, Pandas, NumPy, Transformers (HuggingFace)
- **APIs**: CoinGecko, Twitter API v2, Reddit API, Fear & Greed Index
- **Deployment**: GitHub Actions CI/CD

## 📊 Features

### Data Collection
- **Historical Price Data**: OHLCV data for BTC and ETH
- **Sentiment Analysis**: Real-time sentiment from social media and market indicators
- **Automated Updates**: Daily data ingestion via scheduled workflows

### API Endpoints
- `/prices/{currency}` - Historical price data
- `/sentiment/{currency}` - Historical sentiment data
- `/current_prices` - Real-time price data
- `/latest_sentiment` - Latest sentiment indicators
- `/data_status` - Database status and data availability

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Supabase account

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repo-url>
   cd capstone
   ```

2. **Backend setup:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Database setup:**
   - Create a Supabase project
   - Run the SQL schema from `database/schema.sql`
   - Configure environment variables (see `SETUP.md`)

4. **Frontend setup:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

5. **Start backend:**
   ```bash
   cd backend
   python run.py
   ```

6. **Initial data ingestion:**
   ```bash
   python scripts/data_ingestion.py initial 30
   ```

**Detailed setup instructions:** See [SETUP.md](SETUP.md)

## 📖 API Documentation

Once the backend is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Data Status**: http://localhost:8000/data_status

## 🔮 Roadmap

### Stage 3: ML Model Development (Next)
- [ ] Data preprocessing and feature engineering
- [ ] Train baseline models (Logistic Regression, Random Forest)
- [ ] Implement LSTM/neural network models
- [ ] Model evaluation and selection

### Stage 4: Enhanced API
- [ ] Prediction endpoints with ML models
- [ ] Historical prediction tracking
- [ ] Model performance monitoring

### Stage 5: Frontend Dashboard
- [ ] Real-time price and sentiment display
- [ ] Interactive charts and visualizations
- [ ] Prediction accuracy tracking
- [ ] Responsive design with modern UI

### Stage 6: Production Deployment
- [ ] Vercel frontend deployment
- [ ] Backend deployment (Railway/Render)
- [ ] Production monitoring and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes as part of a capstone project.

## 📞 Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Setup Help**: Check [SETUP.md](SETUP.md) for detailed instructions
- **API Questions**: Review the interactive docs at `/docs`
