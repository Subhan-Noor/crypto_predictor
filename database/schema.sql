-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Crypto Prices Table
CREATE TABLE crypto_prices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    currency VARCHAR(10) NOT NULL CHECK (currency IN ('BTC', 'ETH')),
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(currency, date)
);

-- Crypto Sentiment Table
CREATE TABLE crypto_sentiment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    currency VARCHAR(10) NOT NULL CHECK (currency IN ('BTC', 'ETH')),
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    fear_greed_index INTEGER CHECK (fear_greed_index >= 0 AND fear_greed_index <= 100),
    twitter_sentiment DECIMAL(5, 4) CHECK (twitter_sentiment >= -1 AND twitter_sentiment <= 1),
    reddit_sentiment DECIMAL(5, 4) CHECK (reddit_sentiment >= -1 AND reddit_sentiment <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(currency, date)
);

-- Predictions Table
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    currency VARCHAR(10) NOT NULL CHECK (currency IN ('BTC', 'ETH')),
    prediction_date TIMESTAMP WITH TIME ZONE NOT NULL,
    target_date TIMESTAMP WITH TIME ZONE NOT NULL,
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    features_used JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(currency, prediction_date, target_date)
);

-- Create indexes for better query performance
CREATE INDEX idx_crypto_prices_currency_date ON crypto_prices(currency, date DESC);
CREATE INDEX idx_crypto_sentiment_currency_date ON crypto_sentiment(currency, date DESC);
CREATE INDEX idx_predictions_currency_date ON predictions(currency, prediction_date DESC);
CREATE INDEX idx_predictions_target_date ON predictions(target_date DESC);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER set_timestamp_crypto_prices
    BEFORE UPDATE ON crypto_prices
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TRIGGER set_timestamp_crypto_sentiment
    BEFORE UPDATE ON crypto_sentiment
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_set_timestamp(); 