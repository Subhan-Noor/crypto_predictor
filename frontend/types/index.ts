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
  actual_direction?: 'UP' | 'DOWN'
  is_correct?: boolean
  price_change_pct?: number
  validated_at?: string
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

// Updated APIHealthStatus to match production backend response
export interface APIHealthStatus {
  status: string
  timestamp: string
  version: string
  environment: string
  services: {
    database: {
      status: string
    }
    cache: {
      status: string
    }
    rate_limiter: {
      status: string
    }
    websocket: {
      service_status: string
      connections: {
        total_connections: number
        subscription_stats: {
          prices: number
          predictions: number
          sentiment: number
          all: number
        }
        currency_stats: {
          BTC: {
            prices: number
            predictions: number
            sentiment: number
          }
          ETH: {
            prices: number
            predictions: number
            sentiment: number
          }
        }
        uptime: number
      }
    }
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

export interface PredictionHistoryItem {
  id: string
  currency: string
  prediction_date: string
  predicted_direction: 'UP' | 'DOWN'
  confidence_score: number
  model_type: string
  actual_direction?: 'UP' | 'DOWN'
  is_correct?: boolean
  price_change_pct?: number
  validated_at?: string
} 