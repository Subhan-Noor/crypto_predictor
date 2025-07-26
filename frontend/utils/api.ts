import axios, { AxiosResponse } from 'axios'
import {
  PriceData,
  SentimentData,
  PredictionData,
  CurrentPrice,
  PaginatedResponse,
  APIHealthStatus,
  APIError,
  Currency
} from '../types'

// API Base Configuration with HTTPS enforcement
const getApiBaseUrl = () => {
  let baseUrl = process.env.NEXT_PUBLIC_API_URL || 'https://cryptopredictor-production.up.railway.app'
  
  // Ensure HTTPS in production
  if (process.env.NODE_ENV === 'production' && baseUrl.startsWith('http://')) {
    baseUrl = baseUrl.replace('http://', 'https://')
    console.warn('⚠️ API URL converted to HTTPS for security:', baseUrl)
  }
  
  return baseUrl
}

const API_BASE_URL = getApiBaseUrl()

console.log('🔗 API Base URL:', API_BASE_URL)

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000, // Increased timeout for better reliability
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Requested-With': 'XMLHttpRequest', // CSRF protection
  },
  // Security configurations
  withCredentials: false, // Don't send cookies for security
  xsrfCookieName: 'XSRF-TOKEN',
  xsrfHeaderName: 'X-XSRF-TOKEN',
})

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('API Response Error:', {
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      message: error.message,
      url: error.config?.url,
      method: error.config?.method
    })
    
    // Provide more specific error messages
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      console.error('❌ Backend connection failed. Please check if the backend is running.')
    } else if (error.response?.status === 502) {
      console.error('❌ Backend server error (502). The backend may be down or deploying.')
    } else if (error.response?.status === 404) {
      console.error('❌ API endpoint not found (404). Check the endpoint URL.')
    }
    
    return Promise.reject(error)
  }
)

// API Functions
export const apiService = {
  // Health Check
  async checkHealth(): Promise<APIHealthStatus> {
    const response = await apiClient.get<APIHealthStatus>('/health')
    return response.data
  },

  // Current Prices
  async getCurrentPrices(): Promise<Record<Currency, CurrentPrice>> {
    const response = await apiClient.get<Record<Currency, CurrentPrice>>('/current_prices')
    return response.data
  },

  // Price Data
  async getPriceData(
    currency: Currency,
    page: number = 1,
    perPage: number = 100,
    days?: number
  ): Promise<PaginatedResponse<PriceData>> {
    const params = new URLSearchParams({
      page: page.toString(),
    })
    
    // Set appropriate per_page/limit based on time range
    let actualPerPage = perPage
    if (days === undefined || days > 365) {
      // For All Time or 1+ year, request maximum records
      actualPerPage = 1000
    } else if (days > 90) {
      // For 90+ days, request more records
      actualPerPage = 500
    } else {
      // For shorter periods, use reasonable limit
      actualPerPage = Math.max(perPage, days || 100)
    }
    
    params.append('per_page', actualPerPage.toString())
    
    if (days !== undefined) {
      // Send days parameter directly for enhanced endpoint compatibility
      params.append('days', days.toString())
    } else {
      // For All Time (days undefined), request a very large number of days
      params.append('days', '3650') // ~10 years of data
    }

    // Add limit parameter for backend to fetch more records from database
    if (days === undefined || days > 365) {
      // For All Time or 1+ year, request more records from database
      params.append('limit', '10000')
    } else if (days > 90) {
      // For 90+ days, request more records from database
      params.append('limit', '1000')
    } else {
      // For shorter periods, use default
      params.append('limit', '500')
    }

    const response = await apiClient.get<PaginatedResponse<PriceData>>(
      `/prices/${currency}?${params.toString()}`
    )
    return response.data
  },

  // Sentiment Data
  async getSentimentData(
    currency: Currency,
    page: number = 1,
    perPage: number = 100,
    days?: number
  ): Promise<PaginatedResponse<SentimentData>> {
    const params = new URLSearchParams({
      page: page.toString(),
      per_page: perPage.toString(),
    })
    
    if (days) {
      // Send days parameter directly for enhanced endpoint compatibility
      params.append('days', days.toString())
    }

    const response = await apiClient.get<PaginatedResponse<SentimentData>>(
      `/sentiment/${currency}?${params.toString()}`
    )
    return response.data
  },

  // Predictions
  async getPrediction(currency: Currency): Promise<PredictionData> {
    const response = await apiClient.post<any>(`/predict/${currency}`, {})
    
    // Transform EnhancedPredictionResponse to PredictionData format
    const enhancedResponse = response.data
    return {
      currency: enhancedResponse.currency as Currency,
      prediction: enhancedResponse.predicted_direction as 'UP' | 'DOWN',
      confidence: enhancedResponse.confidence_score || 0.5,
      target_date: enhancedResponse.prediction_date || new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
      created_at: enhancedResponse.prediction_date || new Date().toISOString(),
      features: enhancedResponse.features_importance || {}
    }
  },

  // Basic endpoints (fallback)
  async getBasicPriceData(currency: Currency): Promise<PriceData[]> {
    const response = await apiClient.get<PriceData[]>(`/prices/${currency}/basic`)
    return response.data
  },

  async getBasicSentimentData(currency: Currency): Promise<SentimentData[]> {
    const response = await apiClient.get<SentimentData[]>(`/sentiment/${currency}/basic`)
    return response.data
  },

  async getBasicPrediction(currency: Currency): Promise<PredictionData> {
    const response = await apiClient.post<PredictionData>(`/predict/${currency}/basic`, {})
    return response.data
  },

  // Historical Predictions
  async getPredictionHistory(
    currency: Currency,
    days: number = 30,
    limit: number = 100
  ): Promise<{
    currency: string;
    predictions: any[];
    count: number;
    days: number;
  }> {
    const params = new URLSearchParams({
      days: days.toString(),
      limit: limit.toString(),
    })

    const response = await apiClient.get(
      `/predictions/${currency}/history?${params.toString()}`
    )
    return response.data
  },

  // Predictions Analysis
  async getPredictionAccuracy(currency: Currency, days: number = 30): Promise<{
    currency: string;
    accuracy: number;
    total_predictions: number;
    correct_predictions: number;
    validated_predictions: number;
  }> {
    const response = await apiClient.get(`/predictions/accuracy/${currency}?days=${days}`)
    return response.data
  },

  // Market Analytics
  async getMarketAnalytics(days: number = 30): Promise<{
    timeframe_days: number;
    data_quality: any;
    correlation_metrics: any;
    volatility_metrics: any;
    sentiment_metrics: any;
    performance_metrics: any;
    portfolio_insights: any;
    timestamp: string;
  }> {
    const response = await apiClient.get(`/analytics/market?days=${days}`)
    return response.data
  },

  // Manual Data Validation

  // Get all predictions for the predictions page
  async getAllPredictionHistory(days: number = 30): Promise<{
    BTC: any[];
    ETH: any[];
    combined: any[];
  }> {
    try {
      const [btcHistory, ethHistory] = await Promise.all([
        this.getPredictionHistory('BTC', days, 100),
        this.getPredictionHistory('ETH', days, 100)
      ])

      const combined = [
        ...btcHistory.predictions.map(p => ({ ...p, currency: 'BTC' })),
        ...ethHistory.predictions.map(p => ({ ...p, currency: 'ETH' }))
      ].sort((a, b) => new Date(b.prediction_date).getTime() - new Date(a.prediction_date).getTime())

      return {
        BTC: btcHistory.predictions,
        ETH: ethHistory.predictions,
        combined
      }
    } catch (error) {
      console.error('Error fetching all prediction history:', error)
      throw error
    }
  },
}

// Error handler helper
export const handleAPIError = (error: any): string => {
  if (error.response?.data?.detail) {
    return error.response.data.detail
  }
  if (error.code === 'ERR_NETWORK') {
    return 'Network connection failed. Please check your internet connection or try again later.'
  }
  if (error.code === 'ECONNABORTED') {
    return 'Request timed out. The server may be temporarily unavailable.'
  }
  if (error.response?.status === 404) {
    return 'The requested data was not found.'
  }
  if (error.response?.status === 500) {
    return 'Server error occurred. Please try again later.'
  }
  if (error.response?.status === 503) {
    return 'Service temporarily unavailable. Please try again in a few minutes.'
  }
  if (error.message) {
    return `Connection error: ${error.message}`
  }
  return 'An unexpected error occurred. Please try refreshing the page.'
}

// Data transformation helpers
export const transformPriceDataForChart = (data: PriceData[]) => {
  // Sort data chronologically (oldest to newest) for proper chart display
  const sortedData = [...data].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
  
  return sortedData.map(item => ({
    date: new Date(item.date).toLocaleDateString(),
    price: item.close,
    volume: item.volume,
    high: item.high,
    low: item.low,
    open: item.open,
  }))
}

export const transformSentimentDataForChart = (data: SentimentData[]) => {
  // Sort data chronologically (oldest to newest) for proper chart display
  const sortedData = [...data].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
  
  return sortedData.map(item => ({
    date: new Date(item.date).toLocaleDateString(),
    twitter: item.twitter_sentiment,
    reddit: item.reddit_sentiment,
    overall: item.overall_sentiment || (item.twitter_sentiment + item.reddit_sentiment) / 2,
  }))
} 