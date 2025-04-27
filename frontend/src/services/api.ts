export interface MarketData {
  symbol: string;
  price: number;
  high_24h: number;
  low_24h: number;
  volume_24h: number;
  price_change_24h: number;
}

export interface PredictionData {
  symbol: string;
  predicted_price: number;
  confidence_interval: [number, number];
  sentiment_score: number;
  timestamp: string;
}

export interface HistoricalData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  indicators?: {
    sma_20?: number;
    ema_20?: number;
    rsi_14?: number;
    macd?: {
      macd: number;
      signal: number;
      histogram: number;
    };
  };
}

// Static fallback mock data if API fails
const MOCK_DATA = {
  BTC: {
    price: 94500,
    high_24h: 96000,
    low_24h: 93000,
    volume_24h: 25000000000,
    price_change_24h: 1500,
    predicted_price: 96000,
    confidence_interval: [93000, 99000] as [number, number],
    sentiment_score: 0.65,
    timestamp: new Date().toISOString()
  }
};

// Use window.location to determine API base URL or fallback to localhost
const API_BASE_URL = typeof window !== 'undefined' && window.location.hostname !== 'localhost' 
  ? `${window.location.protocol}//${window.location.hostname}:8000`
  : import.meta.env.VITE_API_URL || 'http://localhost:8000';

console.log('API_BASE_URL:', API_BASE_URL);

const fetchConfig: RequestInit = {
  mode: 'cors',
  credentials: 'same-origin',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
};

class ApiService {
  async getCurrentPrice(symbol: string): Promise<MarketData> {
    const url = `${API_BASE_URL}/current-price/${symbol}`;
    console.log('Fetching current price from:', url);
    
    try {
      const response = await fetch(url, fetchConfig);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch current price: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Current price response:', data);
      
      if (!data || !data.price) {
        throw new Error('Invalid response for current price');
      }
      
      return {
        symbol: symbol.toUpperCase(),
        price: data.price,
        high_24h: data.price * 1.02, // Estimate high/low
        low_24h: data.price * 0.98,
        volume_24h: 25000000000, // Mock volume
        price_change_24h: data.price * 0.01 // Estimate 1% change
      };
    } catch (error) {
      console.error(`Error fetching current price for ${symbol}:`, error);
      // Fallback to mock data
      if (symbol.toUpperCase() in MOCK_DATA) {
        const mockCrypto = MOCK_DATA[symbol.toUpperCase() as keyof typeof MOCK_DATA];
        return {
          symbol: symbol.toUpperCase(),
          price: mockCrypto.price,
          high_24h: mockCrypto.high_24h,
          low_24h: mockCrypto.low_24h,
          volume_24h: mockCrypto.volume_24h,
          price_change_24h: mockCrypto.price_change_24h
        };
      }
      throw error;
    }
  }

  async getPrediction(symbol: string): Promise<PredictionData> {
    const url = `${API_BASE_URL}/predict`;
    console.log('Fetching prediction from:', url);
    
    try {
      const response = await fetch(url, {
        ...fetchConfig,
        method: 'POST',
        body: JSON.stringify({
          symbol: symbol,
          timeframe: '24h',
          include_sentiment: true
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch prediction: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Prediction response:', data);
      
      if (!data || !data.predicted_price) {
        throw new Error('Invalid response for prediction');
      }
      
      return {
        symbol: data.symbol,
        predicted_price: data.predicted_price,
        confidence_interval: data.confidence_interval as [number, number],
        sentiment_score: data.sentiment_score || 0.5,
        timestamp: data.prediction_time || new Date().toISOString()
      };
    } catch (error) {
      console.error(`Error fetching prediction for ${symbol}:`, error);
      // Fallback to mock data
      if (symbol.toUpperCase() in MOCK_DATA) {
        const mockCrypto = MOCK_DATA[symbol.toUpperCase() as keyof typeof MOCK_DATA];
        return {
          symbol: symbol.toUpperCase(),
          predicted_price: mockCrypto.predicted_price,
          confidence_interval: mockCrypto.confidence_interval,
          sentiment_score: mockCrypto.sentiment_score,
          timestamp: mockCrypto.timestamp
        };
      }
      throw error;
    }
  }

  async getHistoricalData(
    symbol: string,
    days: number = 30,
    includeIndicators: boolean = true
  ): Promise<HistoricalData[]> {
    const url = `${API_BASE_URL}/historical/${symbol}?days=${days}&include_indicators=${includeIndicators}`;
    console.log('Fetching historical data from:', url);
    
    try {
      const response = await fetch(url, fetchConfig);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch historical data: ${response.status} ${response.statusText}`);
      }
      
      const responseJson = await response.json();
      console.log('Historical data response structure:', Object.keys(responseJson));
      
      if (!responseJson || !responseJson.data || !Array.isArray(responseJson.data)) {
        throw new Error('Invalid response for historical data');
      }
      
      console.log('Got', responseJson.data.length, 'historical data points');
      
      return responseJson.data;
    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error);
      // For historical data, we can't easily provide mock data, so we'll create a simple
      // array with simulated price moves based on the current mock price
      if (symbol.toUpperCase() in MOCK_DATA) {
        const mockCrypto = MOCK_DATA[symbol.toUpperCase() as keyof typeof MOCK_DATA];
        const mockHistorical: HistoricalData[] = [];
        
        const currentDate = new Date();
        const basePrice = mockCrypto.price;
        
        // Generate n days of mock data
        for (let i = days; i >= 0; i--) {
          const date = new Date(currentDate);
          date.setDate(date.getDate() - i);
          
          // Random price fluctuation (±5%)
          const randomFactor = 0.9 + Math.random() * 0.2;
          const dayPrice = basePrice * randomFactor;
          
          mockHistorical.push({
            timestamp: date.toISOString(),
            open: dayPrice * 0.99,
            high: dayPrice * 1.02,
            low: dayPrice * 0.98,
            close: dayPrice,
            volume: 20000000000 + Math.random() * 10000000000
          });
        }
        
        return mockHistorical;
      }
      throw error;
    }
  }
}

const api = new ApiService();
export default api; 