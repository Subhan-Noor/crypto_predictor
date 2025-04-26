// Sentiment thresholds and utilities
export const SENTIMENT_THRESHOLDS = {
    BULLISH: 0.6,
    NEUTRAL: 0.4,
    BEARISH: 0
};

export const SENTIMENT_COLORS = {
    BULLISH: '#4caf50',  // Green
    NEUTRAL: '#ff9800',  // Orange
    BEARISH: '#f44336'   // Red
};

export const getSentimentLabel = (sentiment: number): string => {
    if (sentiment >= SENTIMENT_THRESHOLDS.BULLISH) return 'Bullish';
    if (sentiment >= SENTIMENT_THRESHOLDS.NEUTRAL) return 'Neutral';
    return 'Bearish';
};

export const getSentimentColor = (sentiment: number): string => {
    if (sentiment >= SENTIMENT_THRESHOLDS.BULLISH) return SENTIMENT_COLORS.BULLISH;
    if (sentiment >= SENTIMENT_THRESHOLDS.NEUTRAL) return SENTIMENT_COLORS.NEUTRAL;
    return SENTIMENT_COLORS.BEARISH;
}; 