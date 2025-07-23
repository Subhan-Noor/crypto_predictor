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

// API Base Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://cryptopredictor-production.up.railway.app'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
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
    console.error('API Response Error:', error.response?.data || error.message)
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
      per_page: perPage.toString(),
    })
    
    if (days) {
      const endDate = new Date()
      const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000)
      params.append('start_date', startDate.toISOString().split('T')[0])
      params.append('end_date', endDate.toISOString().split('T')[0])
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
      const endDate = new Date()
      const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000)
      params.append('start_date', startDate.toISOString().split('T')[0])
      params.append('end_date', endDate.toISOString().split('T')[0])
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
  return data.map(item => ({
    date: new Date(item.date).toLocaleDateString(),
    price: item.close,
    volume: item.volume,
    high: item.high,
    low: item.low,
    open: item.open,
  }))
}

export const transformSentimentDataForChart = (data: SentimentData[]) => {
  return data.map(item => ({
    date: new Date(item.date).toLocaleDateString(),
    twitter: item.twitter_sentiment,
    reddit: item.reddit_sentiment,
    overall: item.overall_sentiment || (item.twitter_sentiment + item.reddit_sentiment) / 2,
  }))
} 