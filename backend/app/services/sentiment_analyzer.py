"""
Sentiment Analysis Service for Crypto Text Analysis

This module provides:
- Multiple sentiment analysis engines (TextBlob, VADER)
- Crypto-specific text preprocessing
- Batch sentiment analysis
- Confidence scoring
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

logger = logging.getLogger(__name__)


class CryptoSentimentAnalyzer:
    """Advanced sentiment analyzer optimized for cryptocurrency content"""
    
    def __init__(self):
        """Initialize sentiment analyzer with multiple engines"""
        self.vader = SentimentIntensityAnalyzer()
        
        # Crypto-specific keywords for enhanced analysis
        self.bullish_keywords = {
            'moon', 'lambo', 'hodl', 'diamond hands', 'bullish', 'pump', 'rally', 
            'surge', 'breakout', 'bull run', 'to the moon', 'ath', 'all time high',
            'gains', 'profit', 'green', 'buying', 'accumulate', 'dip buying'
        }
        
        self.bearish_keywords = {
            'dump', 'crash', 'bearish', 'panic', 'sell', 'drop', 'fall', 'dip',
            'correction', 'bear market', 'red', 'loss', 'down', 'decline',
            'paper hands', 'fud', 'fear', 'uncertainty', 'doubt'
        }
        
        # Common crypto abbreviations and slang
        self.crypto_slang = {
            'hodl': 'hold',
            'fud': 'fear uncertainty doubt',
            'fomo': 'fear of missing out',
            'btfd': 'buy the fucking dip',
            'diamond hands': 'strong hold',
            'paper hands': 'weak sell',
            'mooning': 'going up',
            'pumping': 'increasing',
            'dumping': 'decreasing',
            'rekt': 'destroyed',
            'wagmi': 'we are going to make it',
            'ngmi': 'not going to make it'
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better sentiment analysis
        
        Args:
            text: Raw text content
            
        Returns:
            Processed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace crypto slang with more standard terms
        for slang, replacement in self.crypto_slang.items():
            text = text.replace(slang, replacement)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags for cleaner analysis (but keep the text)
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove emojis (optional - they might contain sentiment)
        text = re.sub(r'[^\w\s\.\!\?]', '', text)
        
        return text
    
    def analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
                'subjectivity': blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with VADER sentiment scores
        """
        try:
            scores = self.vader.polarity_scores(text)
            return scores  # {'neg': x, 'neu': x, 'pos': x, 'compound': x}
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    def crypto_keyword_boost(self, text: str, base_sentiment: float) -> float:
        """
        Apply crypto-specific keyword boost to sentiment
        
        Args:
            text: Preprocessed text
            base_sentiment: Base sentiment score
            
        Returns:
            Adjusted sentiment score
        """
        text_lower = text.lower()
        
        # Count bullish and bearish keywords
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
        
        # Calculate boost factor
        keyword_diff = bullish_count - bearish_count
        boost_factor = keyword_diff * 0.1  # Adjust strength as needed
        
        # Apply boost but keep within bounds
        adjusted_sentiment = base_sentiment + boost_factor
        return max(-1.0, min(1.0, adjusted_sentiment))
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text (alias for analyze_sentiment for compatibility)
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis results with compound score
        """
        return self.analyze_sentiment(text, method='hybrid')

    def analyze_sentiment(self, text: str, method: str = 'hybrid') -> Dict[str, float]:
        """
        Comprehensive sentiment analysis with multiple methods
        
        Args:
            text: Text to analyze
            method: 'textblob', 'vader', or 'hybrid'
            
        Returns:
            Dictionary with sentiment scores and metadata
        """
        if not text or not isinstance(text, str):
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'method': method,
                'text_length': 0
            }
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'method': method,
                'text_length': 0
            }
        
        if method == 'textblob':
            textblob_result = self.analyze_with_textblob(processed_text)
            sentiment = textblob_result['polarity']
            confidence = abs(sentiment) * textblob_result['subjectivity']
        
        elif method == 'vader':
            vader_result = self.analyze_with_vader(processed_text)
            sentiment = vader_result['compound']
            confidence = abs(sentiment)
        
        else:  # hybrid approach
            textblob_result = self.analyze_with_textblob(processed_text)
            vader_result = self.analyze_with_vader(processed_text)
            
            # Combine scores with weights
            textblob_weight = 0.4
            vader_weight = 0.6
            
            sentiment = (textblob_result['polarity'] * textblob_weight + 
                        vader_result['compound'] * vader_weight)
            
            # Confidence based on agreement between methods
            agreement = 1 - abs(textblob_result['polarity'] - vader_result['compound']) / 2
            confidence = agreement * abs(sentiment)
        
        # Apply crypto-specific keyword boost
        sentiment = self.crypto_keyword_boost(processed_text, sentiment)
        
        return {
            'sentiment': float(sentiment),
            'confidence': float(confidence),
            'method': method,
            'text_length': len(processed_text),
            'original_length': len(text)
        }
    
    def analyze_batch(self, texts: List[str], method: str = 'hybrid') -> Dict[str, float]:
        """
        Analyze sentiment for a batch of texts and return aggregated results
        
        Args:
            texts: List of texts to analyze
            method: Sentiment analysis method
            
        Returns:
            Dictionary with aggregated sentiment statistics
        """
        if not texts:
            return {
                'avg_sentiment': 0.0,
                'weighted_sentiment': 0.0,
                'confidence': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'total_texts': 0
            }
        
        results = []
        total_length = 0
        
        for text in texts:
            result = self.analyze_sentiment(text, method)
            results.append(result)
            total_length += result['text_length']
        
        if not results:
            return {
                'avg_sentiment': 0.0,
                'weighted_sentiment': 0.0,
                'confidence': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'total_texts': 0
            }
        
        sentiments = [r['sentiment'] for r in results]
        confidences = [r['confidence'] for r in results]
        lengths = [r['text_length'] for r in results]
        
        # Calculate average sentiment
        avg_sentiment = np.mean(sentiments)
        
        # Calculate weighted sentiment (by text length and confidence)
        weights = [length * confidence for length, confidence in zip(lengths, confidences)]
        if sum(weights) > 0:
            weighted_sentiment = np.average(sentiments, weights=weights)
        else:
            weighted_sentiment = avg_sentiment
        
        # Calculate sentiment distribution
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        total_count = len(sentiments)
        positive_ratio = positive_count / total_count if total_count > 0 else 0
        negative_ratio = negative_count / total_count if total_count > 0 else 0
        neutral_ratio = neutral_count / total_count if total_count > 0 else 0
        
        # Overall confidence
        avg_confidence = np.mean(confidences)
        
        return {
            'avg_sentiment': float(avg_sentiment),
            'weighted_sentiment': float(weighted_sentiment),
            'confidence': float(avg_confidence),
            'positive_ratio': float(positive_ratio),
            'negative_ratio': float(negative_ratio),
            'neutral_ratio': float(neutral_ratio),
            'total_texts': total_count,
            'sentiment_std': float(np.std(sentiments)),
            'details': {
                'individual_sentiments': sentiments,
                'individual_confidences': confidences
            }
        }


# Global sentiment analyzer instance
sentiment_analyzer = CryptoSentimentAnalyzer() 