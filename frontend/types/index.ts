// API Response Types
export interface PriceData {
  id: string
  currency: 'BTC' | 'ETH'
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface SentimentData {
  id: string
  currency: 'BTC' | 'ETH'
  date: string
  twitter_sentiment: number
  reddit_sentiment: number
  overall_sentiment: number
}

export interface PredictionData {
  currency: 'BTC' | 'ETH'
  prediction: 'UP' | 'DOWN'
  confidence: number
  target_date: string
  created_at: string
  features?: Record<string, number>
}

export interface CurrentPrice {
  currency: 'BTC' | 'ETH'
  price: number
  change_24h: number
  change_percentage_24h: number
  volume_24h: number
  market_cap?: number
  last_updated: string
}

// API Response Wrappers
export interface PaginatedResponse<T> {
  data: T[]
  total: number
  page: number
  per_page: number
  has_next: boolean
  has_prev: boolean
}

export interface APIHealthStatus {
  status: string
  timestamp: string
  version: string
  uptime: number
  services: {
    database: string
    cache: string
    ml_models: string
  }
}

// Component Props Types
export interface ChartProps {
  data: PriceData[]
  height?: number
  showVolume?: boolean
}

export interface SentimentChartProps {
  data: SentimentData[]
  height?: number
}

export interface PredictionCardProps {
  prediction: PredictionData
  currentPrice?: CurrentPrice
}

// Utility Types
export type Currency = 'BTC' | 'ETH'
export type TimeRange = '1D' | '7D' | '30D' | '90D' | '1Y'
export type ChartType = 'line' | 'candlestick'

// API Error Type
export interface APIError {
  detail: string
  status_code: number
  timestamp: string
} 