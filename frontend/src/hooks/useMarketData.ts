'use client';

import { useState, useEffect } from 'react';
import api, { MarketData, PredictionData, HistoricalData } from '../services/api';

interface UseMarketDataProps {
  symbol?: string;
}

interface MarketDataState {
  marketData: MarketData | null;
  predictionData: PredictionData | null;
  historicalData: HistoricalData[];
  loading: boolean;
  error: string | null;
  source: 'api' | 'mock'; // Track data source
}

// Fallback mock data
const MOCK_DATA = {
  BTC: {
    marketData: {
      symbol: 'BTC',
      price: 94500,
      high_24h: 96000,
      low_24h: 93000,
      volume_24h: 25000000000,
      price_change_24h: 1500
    },
    predictionData: {
      symbol: 'BTC',
      predicted_price: 96000,
      confidence_interval: [93000, 99000] as [number, number],
      sentiment_score: 0.65,
      timestamp: new Date().toISOString()
    },
    historicalData: Array(30).fill(0).map((_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (30 - i));
      const basePrice = 90000 + (i * 150);
      return {
        timestamp: date.toISOString(),
        open: basePrice - 200,
        high: basePrice + 300,
        low: basePrice - 300,
        close: basePrice,
        volume: 20000000000 + (Math.random() * 5000000000)
      };
    })
  }
};

export function useMarketData({ symbol = 'BTC' }: UseMarketDataProps = {}) {
  const [state, setState] = useState<MarketDataState>({
    marketData: null,
    predictionData: null,
    historicalData: [],
    loading: true,
    error: null,
    source: 'api'
  });

  useEffect(() => {
    console.log('useMarketData hook mounted for symbol:', symbol);
    let mounted = true;
    let dataRefreshInterval: NodeJS.Timeout | null = null;

    const fetchData = async () => {
      console.log('Fetching data for symbol:', symbol);
      let isError = false;

      // Step 1: Try to get current price
      let currentPrice = null;
      try {
        console.log('Fetching current price...');
        currentPrice = await api.getCurrentPrice(symbol);
        console.log('REAL API CURRENT PRICE:', currentPrice);
      } catch (error) {
        console.error('Failed to fetch current price:', error);
        isError = true;
      }

      // Step 2: Try to get prediction
      let prediction = null;
      try {
        console.log('Fetching prediction...');
        prediction = await api.getPrediction(symbol);
        console.log('REAL API PREDICTION:', prediction);
      } catch (error) {
        console.error('Failed to fetch prediction:', error);
        isError = true;
      }

      // Step 3: Try to get historical data
      let historical: HistoricalData[] = [];
      try {
        console.log('Fetching historical data...');
        historical = await api.getHistoricalData(symbol, 30, true);
        console.log('REAL API HISTORICAL DATA COUNT:', historical.length);
      } catch (error) {
        console.error('Failed to fetch historical data:', error);
        isError = true;
      }

      // Update state based on what we got
      if (mounted) {
        if (currentPrice && prediction && historical.length > 0) {
          console.log('Using REAL API DATA');
          setState({
            marketData: currentPrice,
            predictionData: prediction,
            historicalData: historical,
            loading: false,
            error: null,
            source: 'api'
          });
        } else {
          console.log('Using MOCK DATA due to missing API data');
          setState({
            marketData: MOCK_DATA.BTC.marketData,
            predictionData: MOCK_DATA.BTC.predictionData,
            historicalData: MOCK_DATA.BTC.historicalData,
            loading: false,
            error: isError ? 'Error fetching data from API' : null,
            source: 'mock'
          });
        }
      }
    };

    // Initial data fetch
    fetchData();
    
    // Set up periodic data refresh
    dataRefreshInterval = setInterval(() => {
      console.log('Refreshing data...');
      fetchData();
    }, 20000); // Every 20 seconds

    return () => {
      console.log('useMarketData hook unmounting');
      mounted = false;
      if (dataRefreshInterval) {
        clearInterval(dataRefreshInterval);
      }
    };
  }, [symbol]);

  return state;
} 