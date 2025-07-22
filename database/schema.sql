-- Crypto Price Prediction App Database Schema
-- Updated for new data acquisition stack (Twint, Pushshift API, Binance)

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Crypto prices table
CREATE TABLE IF NOT EXISTS crypto_prices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    currency TEXT NOT NULL CHECK (currency IN ('BTC', 'ETH')),
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique combination of currency and date
    UNIQUE(currency, date)
);

-- Crypto sentiment table (updated for new stack)
CREATE TABLE IF NOT EXISTS crypto_sentiment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    currency TEXT NOT NULL CHECK (currency IN ('BTC', 'ETH')),
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    twitter_sentiment DECIMAL(5, 4), -- Sentiment score from Twitter (Twint)
    reddit_sentiment DECIMAL(5, 4),  -- Sentiment score from Reddit (Pushshift)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique combination of currency and date
    UNIQUE(currency, date)
);

-- Predictions table (for Stage 3+)
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    currency TEXT NOT NULL CHECK (currency IN ('BTC', 'ETH')),
    prediction_date TIMESTAMP WITH TIME ZONE NOT NULL,
    prediction_horizon INTEGER NOT NULL, -- Days ahead (e.g., 7)
    predicted_direction TEXT NOT NULL CHECK (predicted_direction IN ('UP', 'DOWN')),
    confidence_score DECIMAL(5, 4) NOT NULL,
    model_version TEXT,
    features_used JSONB, -- Store features used for prediction
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique combination of currency, prediction date, and horizon
    UNIQUE(currency, prediction_date, prediction_horizon)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_crypto_prices_currency_date ON crypto_prices(currency, date);
CREATE INDEX IF NOT EXISTS idx_crypto_sentiment_currency_date ON crypto_sentiment(currency, date);
CREATE INDEX IF NOT EXISTS idx_predictions_currency_date ON predictions(currency, prediction_date);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to update updated_at columns
CREATE TRIGGER update_crypto_prices_updated_at 
    BEFORE UPDATE ON crypto_prices 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_crypto_sentiment_updated_at 
    BEFORE UPDATE ON crypto_sentiment 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column(); 