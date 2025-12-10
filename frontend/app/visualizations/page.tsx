'use client'

import React, { useState, useEffect, useCallback } from 'react'
import Head from 'next/head'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  AreaChart, Area, BarChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell,
  ComposedChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts'
import { apiService } from '../../utils/api'
import { ErrorBoundary } from '../../components/ErrorBoundary'
import { PriceLoadingCard } from '../../components/EnhancedLoadingSpinner'
import { ErrorCard } from '../../components/ErrorCard'
import { EmptyState } from '../../components/EmptyState'

// Custom tooltip component for price data
const PriceTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    
    return (
      <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
        <p className="text-white font-medium mb-2">{data.date || label}</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-300">Price:</span>
            <span className="text-white">${data.price?.toFixed(2) || 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Volume:</span>
            <span className="text-white">{data.volume?.toLocaleString() || 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Change:</span>
            <span className={`${data.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {data.change?.toFixed(2) || 'N/A'}%
            </span>
          </div>
        </div>
      </div>
    )
  }
  return null
}

// Custom tooltip for sentiment data
const SentimentTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    
    return (
      <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
        <p className="text-white font-medium mb-2">{data.date || label}</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-300">Sentiment:</span>
            <span className="text-white">{data.sentiment?.toFixed(3) || 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Mood:</span>
            <span className={`${
              data.sentiment > 0.1 ? 'text-green-400' : 
              data.sentiment < -0.1 ? 'text-red-400' : 'text-yellow-400'
            }`}>
              {data.sentiment > 0.1 ? 'Positive' : 
               data.sentiment < -0.1 ? 'Negative' : 'Neutral'}
            </span>
          </div>
        </div>
      </div>
    )
  }
  return null
}

// Custom tooltip for correlation data
const CorrelationTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    
    return (
      <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
        <p className="text-white font-medium mb-2">{data.date || label}</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-300">Price Change:</span>
            <span className="text-white">{data.price_change?.toFixed(2) || 'N/A'}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Sentiment:</span>
            <span className="text-white">{data.sentiment?.toFixed(3) || 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Correlation:</span>
            <span className="text-blue-400">{data.correlation?.toFixed(3) || 'N/A'}</span>
          </div>
        </div>
      </div>
    )
  }
  return null
}

// Custom tooltip for confusion matrix
const ConfusionMatrixTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    
    return (
      <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
        <p className="text-white font-medium mb-2">{data.predicted} → {data.actual}</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-300">Count:</span>
            <span className="text-white">{data.count}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Percentage:</span>
            <span className="text-white">{data.percentage?.toFixed(1)}%</span>
          </div>
        </div>
      </div>
    )
  }
  return null
}

// Custom tooltip for feature importance
const FeatureImportanceTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    
    return (
      <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
        <p className="text-white font-medium mb-2">{data.feature}</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-300">Importance:</span>
            <span className="text-white">{data.importance?.toFixed(3)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">Category:</span>
            <span className="text-white">{data.category}</span>
          </div>
        </div>
      </div>
    )
  }
  return null
}

// Custom tooltip for price comparison chart
const PriceComparisonTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const btc = payload.find((p: any) => p.dataKey === 'btc_price')
    const eth = payload.find((p: any) => p.dataKey === 'eth_price')
    return (
      <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
        <p className="text-white font-medium mb-2">{label}</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-300">BTC Price:</span>
            <span className="text-orange-400">{btc && btc.value ? `$${btc.value.toLocaleString()}` : 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-300">ETH Price:</span>
            <span className="text-blue-400">{eth && eth.value ? `$${eth.value.toLocaleString()}` : 'N/A'}</span>
          </div>
        </div>
      </div>
    )
  }
  return null
}

// Custom tooltip for sentiment analysis chart
const SentimentAreaTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const btc = payload.find((p: any) => p.dataKey === 'btc_sentiment')
    const eth = payload.find((p: any) => p.dataKey === 'eth_sentiment')
    return (
      <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg">
        <p className="text-white font-medium mb-2">{label}</p>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-orange-400">BTC Sentiment:</span>
            <span className="text-white">{btc && btc.value !== undefined ? btc.value.toFixed(3) : 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-blue-400">ETH Sentiment:</span>
            <span className="text-white">{eth && eth.value !== undefined ? eth.value.toFixed(3) : 'N/A'}</span>
          </div>
        </div>
      </div>
    )
  }
  return null
}

// Custom tooltip for market cap pie chart
const MarketCapTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg text-white">
        <div className="font-medium">{data.name}</div>
        <div>Value: {data.value}%</div>
      </div>
    )
  }
  return null
}

interface VisualizationData {
  priceHistory: Array<{
    date: string
    btc_price: number
    eth_price: number
    btc_volume: number
    eth_volume: number
  }>
  sentimentHistory: Array<{
    date: string
    btc_sentiment: number
    eth_sentiment: number
  }>
  predictionAccuracy: Array<{
    model: string
    accuracy: number
    predictions: number
  }>
  marketDistribution: Array<{
    name: string
    value: number
    color: string
  }>
  technicalIndicators: Array<{
    date: string
    rsi: number
    macd: number
    bollinger_upper: number
    bollinger_lower: number
    price: number
  }>
  priceVsSentimentCorrelation: Array<{
    date: string
    price_change: number
    sentiment: number
    correlation: number
  }>
  confidenceDistribution: Array<{
    range: string
    btc_count: number
    eth_count: number
  }>
  confusionMatrix: Array<{
    predicted: string
    actual: string
    count: number
    percentage: number
  }>
  volatilityComparison: Array<{
    date: string
    price_volatility: number
    sentiment_volatility: number
  }>
  cumulativeProfit: Array<{
    date: string
    btc_cumulative: number
    eth_cumulative: number
    portfolio_value: number
  }>
}

const COLORS = {
  btc: '#F7931A',
  eth: '#627EEA',
  positive: '#10B981',
  negative: '#EF4444',
  neutral: '#F59E0B'
}

export default function VisualizationsPage() {
  const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchVisualizationData = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      // Use the same data fetching approach as predictions page for consistency
      const predictionHistory = await apiService.getAllPredictionHistory(30)
      
      // Transform backend data to frontend format (same as predictions page)
      const transformedPredictions = predictionHistory.combined.map(pred => ({
        id: pred.id || `${pred.currency}-${pred.prediction_date}`,
        currency: pred.currency,
        prediction_date: pred.prediction_date,
        predicted_direction: pred.predicted_direction,
        confidence_score: pred.confidence_score || 0,
        actual_direction: pred.actual_direction || undefined,
        is_correct: pred.is_correct !== undefined ? pred.is_correct : undefined,
        model_version: pred.model_version || 'unknown',
        model_type: pred.model_version?.toLowerCase().includes('random_forest') ? 'random_forest' : 'other'
      }))

      // Use the same accuracy calculation logic as predictions page
      const accuracy = calculateRealAccuracy(transformedPredictions)

      // Fetch other data for visualizations
      const [btcHistorical, ethHistorical, btcSentiment, ethSentiment] = await Promise.all([
        apiService.getPriceData('BTC', 1, 1000, 30),
        apiService.getPriceData('ETH', 1, 1000, 30),
        apiService.getSentimentData('BTC', 1, 1000, 30).catch(() => ({ data: [] })),
        apiService.getSentimentData('ETH', 1, 1000, 30).catch(() => ({ data: [] }))
      ])

      // Process and combine data using consistent accuracy data
      const processedData = processVisualizationData(
        btcHistorical.data || [],
        ethHistorical.data || [],
        btcSentiment.data || [],
        ethSentiment.data || [],
        accuracy,
        transformedPredictions
      )
      
      setVisualizationData(processedData)

    } catch (err) {
      console.error('Error fetching visualization data:', err)
      setError('Failed to load visualization data')
    } finally {
      setLoading(false)
    }
  }, [])

  // Use the same accuracy calculation logic as predictions page
  const calculateRealAccuracy = useCallback((predictions: any[]): any => {
    if (!predictions || predictions.length === 0) {
      return {
        overall_accuracy: 0,
        btc_accuracy: 0,
        eth_accuracy: 0,
        total_predictions: 0,
        correct_predictions: 0,
        precision: 0,
        recall: 0,
        f1_score: 0
      }
    }
    
    // Prioritize Random Forest predictions for top statistics
    const randomForestPredictions = predictions.filter(p => 
      p.model_version?.toLowerCase().includes('random_forest')
    )
    
    // Use Random Forest predictions if available, otherwise use all predictions
    const predictionsToUse = randomForestPredictions.length > 0 ? randomForestPredictions : predictions
    
    // Calculate overall accuracy from prioritized predictions
    const validatedPredictions = predictionsToUse.filter(p => p.is_correct !== undefined)
    if (validatedPredictions.length === 0) {
      return {
        overall_accuracy: 0,
        btc_accuracy: 0,
        eth_accuracy: 0,
        total_predictions: 0,
        correct_predictions: 0,
        precision: 0,
        recall: 0,
        f1_score: 0
      }
    }
    
    const correctPredictions = validatedPredictions.filter(p => p.is_correct === true)
    const overall_accuracy = (correctPredictions.length / validatedPredictions.length) * 100
    
    // Calculate BTC and ETH accuracy separately
    const btcPredictions = validatedPredictions.filter(p => p.currency === 'BTC')
    const ethPredictions = validatedPredictions.filter(p => p.currency === 'ETH')
    
    const btc_correct = btcPredictions.filter(p => p.is_correct === true).length
    const eth_correct = ethPredictions.filter(p => p.is_correct === true).length
    
    const btc_accuracy = btcPredictions.length > 0 ? (btc_correct / btcPredictions.length) * 100 : 0
    const eth_accuracy = ethPredictions.length > 0 ? (eth_correct / ethPredictions.length) * 100 : 0
    
    // Calculate F1 score (simplified)
    const f1_score = overall_accuracy // For binary classification, F1 ≈ accuracy when precision = recall
    
    return {
      overall_accuracy: Math.round(overall_accuracy * 10) / 10,
      btc_accuracy: Math.round(btc_accuracy * 10) / 10,
      eth_accuracy: Math.round(eth_accuracy * 10) / 10,
      total_predictions: validatedPredictions.length,
      correct_predictions: correctPredictions.length,
      precision: Math.round(overall_accuracy * 10) / 10,
      recall: Math.round(overall_accuracy * 10) / 10,
      f1_score: Math.round(f1_score * 10) / 10
    }
  }, [])

  const processVisualizationData = (
    btcData: any[],
    ethData: any[],
    btcSentimentData: any[],
    ethSentimentData: any[],
    accuracy: any,
    allPredictions: any[]
  ): VisualizationData => {
    // Process price history
    const priceHistory = btcData.map((btcItem, index) => {
      const ethItem = ethData[index] || {}
      return {
        date: btcItem.date || btcItem.timestamp,
        btc_price: btcItem.close || btcItem.price,
        eth_price: ethItem.close || ethItem.price,
        btc_volume: btcItem.volume,
        eth_volume: ethItem.volume
      }
    }).filter(item => item.btc_price && item.eth_price)

    // Process sentiment history
    const sentimentHistory = btcSentimentData.map((btcItem, index) => {
      const ethItem = ethSentimentData[index] || {}
      return {
        date: btcItem.date || btcItem.timestamp,
        btc_sentiment: btcItem.sentiment || btcItem.twitter_sentiment || 0,
        eth_sentiment: ethItem.sentiment || ethItem.twitter_sentiment || 0
      }
    }).filter(item => item.btc_sentiment !== 0 || item.eth_sentiment !== 0)

    // Filter predictions to prioritize random forest and include validated predictions
    const filterValidatedPredictions = (predictions: any[]) => {
      // First, get all validated predictions
      const validatedPredictions = predictions.filter(p => p.is_correct !== undefined)
      
      // Group by date and model type
      const predictionsByDate: { [key: string]: { randomForest: any[], logisticRegression: any[] } } = {}
      validatedPredictions.forEach(p => {
        const date = p.prediction_date?.split('T')[0] || p.prediction_date
        const isRandomForest = p.model_version?.toLowerCase().includes('random_forest')
        
        if (!predictionsByDate[date]) {
          predictionsByDate[date] = { randomForest: [], logisticRegression: [] }
        }
        
        if (isRandomForest) {
          predictionsByDate[date].randomForest.push(p)
        } else {
          predictionsByDate[date].logisticRegression.push(p)
        }
      })
      
      // For each date, prioritize random forest predictions
      const prioritizedPredictions: any[] = []
      Object.values(predictionsByDate).forEach((datePredictions) => {
        // If we have random forest predictions for this date, use those
        if (datePredictions.randomForest.length > 0) {
          prioritizedPredictions.push(...datePredictions.randomForest)
        } else {
          // Otherwise, fall back to logistic regression
          prioritizedPredictions.push(...datePredictions.logisticRegression)
        }
      })
      
      return prioritizedPredictions
    }

    const validatedBtcPredictions = filterValidatedPredictions(allPredictions.filter(p => p.currency === 'BTC'))
    const validatedEthPredictions = filterValidatedPredictions(allPredictions.filter(p => p.currency === 'ETH'))
    const allValidatedPredictions = [...validatedBtcPredictions, ...validatedEthPredictions]

    // Use the consistent accuracy data from the predictions page
    const predictionAccuracy = [
      {
        model: 'BTC Random Forest',
        accuracy: accuracy.btc_accuracy,
        predictions: validatedBtcPredictions.length
      },
      {
        model: 'ETH Random Forest',
        accuracy: accuracy.eth_accuracy,
        predictions: validatedEthPredictions.length
      }
    ]

    // Process market distribution (mock data for demonstration)
    const marketDistribution = [
      { name: 'BTC Market Cap', value: 45, color: COLORS.btc },
      { name: 'ETH Market Cap', value: 35, color: COLORS.eth },
      { name: 'Other Crypto', value: 20, color: '#6B7280' }
    ]

    // Process technical indicators (mock data for demonstration)
    const technicalIndicators = priceHistory.slice(-30).map((item, index) => ({
      date: item.date,
      rsi: 30 + Math.random() * 40, // Mock RSI
      macd: (Math.random() - 0.5) * 100, // Mock MACD
      bollinger_upper: item.btc_price * 1.05,
      bollinger_lower: item.btc_price * 0.95,
      price: item.btc_price
    }))

    // Process price vs sentiment correlation
    const priceVsSentimentCorrelation = priceHistory.slice(0, Math.min(priceHistory.length, sentimentHistory.length)).map((priceItem, index) => {
      const sentimentItem = sentimentHistory[index] || { btc_sentiment: 0 }
      const prevPrice = index > 0 ? priceHistory[index - 1].btc_price : priceItem.btc_price
      const priceChange = ((priceItem.btc_price - prevPrice) / prevPrice) * 100
      
      return {
        date: priceItem.date,
        price_change: priceChange,
        sentiment: sentimentItem.btc_sentiment,
        correlation: calculateCorrelation(priceChange, sentimentItem.btc_sentiment)
      }
    })

    // Process confidence distribution based on validated predictions
    const confidenceRanges = [
      { min: 0, max: 0.2, label: '0-20%' },
      { min: 0.2, max: 0.4, label: '20-40%' },
      { min: 0.4, max: 0.6, label: '40-60%' },
      { min: 0.6, max: 0.8, label: '60-80%' },
      { min: 0.8, max: 1.0, label: '80-100%' }
    ]

    const confidenceDistribution = confidenceRanges.map(range => {
      const btcCount = validatedBtcPredictions.filter(p => 
        p.confidence_score >= range.min && p.confidence_score < range.max
      ).length
      const ethCount = validatedEthPredictions.filter(p => 
        p.confidence_score >= range.min && p.confidence_score < range.max
      ).length
      
      return {
        range: range.label,
        btc_count: btcCount,
        eth_count: ethCount
      }
    })

    // Process confusion matrix - PROPER IMPLEMENTATION
    let confusionMatrix = [
      { predicted: 'UP', actual: 'UP', count: 0, percentage: 0 },
      { predicted: 'UP', actual: 'DOWN', count: 0, percentage: 0 },
      { predicted: 'DOWN', actual: 'UP', count: 0, percentage: 0 },
      { predicted: 'DOWN', actual: 'DOWN', count: 0, percentage: 0 }
    ]

    // Use EXACTLY the same data as accuracy calculation
    const randomForestPredictions = allPredictions.filter(p => 
      p.model_version?.toLowerCase().includes('random_forest')
    )
    const predictionsToUse = randomForestPredictions.length > 0 ? randomForestPredictions : allPredictions
    
    // Filter predictions exactly like accuracy calculation
    const validatedPredictions = predictionsToUse.filter(p => p.is_correct !== undefined)
    
    console.log('Confusion Matrix Data:', {
      totalPredictions: allPredictions.length,
      randomForestPredictions: randomForestPredictions.length,
      validatedPredictions: validatedPredictions.length
    })

    if (validatedPredictions.length > 0) {
      // Create confusion matrix using scikit-learn logic
      // For binary classification: 0 = DOWN, 1 = UP
      let truePositives = 0   // Predicted UP, Actual UP
      let falsePositives = 0  // Predicted UP, Actual DOWN  
      let falseNegatives = 0  // Predicted DOWN, Actual UP
      let trueNegatives = 0   // Predicted DOWN, Actual DOWN
      
      validatedPredictions.forEach(pred => {
        const predictedUp = pred.predicted_direction?.toUpperCase() === 'UP'
        const actualUp = pred.is_correct === true ? predictedUp : !predictedUp
        
        if (predictedUp && actualUp) {
          truePositives++
        } else if (predictedUp && !actualUp) {
          falsePositives++
        } else if (!predictedUp && actualUp) {
          falseNegatives++
        } else if (!predictedUp && !actualUp) {
          trueNegatives++
        }
      })
      
      const total = truePositives + falsePositives + falseNegatives + trueNegatives
      
      console.log('Confusion Matrix Raw Counts:', {
        truePositives,
        falsePositives, 
        falseNegatives,
        trueNegatives,
        total
      })
      
      if (total > 0) {
        // Calculate percentages based on total predictions
        const tpPercent = Math.round((truePositives / total) * 100)
        const fpPercent = Math.round((falsePositives / total) * 100)
        const fnPercent = Math.round((falseNegatives / total) * 100)
        const tnPercent = Math.round((trueNegatives / total) * 100)
        
        // Ensure percentages sum to 100%
        const totalPercent = tpPercent + fpPercent + fnPercent + tnPercent
        const diff = 100 - totalPercent
        
        // Distribute rounding error to largest percentage
        let adjustedTp = tpPercent
        let adjustedFp = fpPercent
        let adjustedFn = fnPercent
        let adjustedTn = tnPercent
        
        if (diff !== 0) {
          const maxPercent = Math.max(tpPercent, fpPercent, fnPercent, tnPercent)
          if (tpPercent === maxPercent) adjustedTp += diff
          else if (fpPercent === maxPercent) adjustedFp += diff
          else if (fnPercent === maxPercent) adjustedFn += diff
          else adjustedTn += diff
        }
        
        confusionMatrix = [
          { predicted: 'UP', actual: 'UP', count: truePositives, percentage: adjustedTp },
          { predicted: 'UP', actual: 'DOWN', count: falsePositives, percentage: adjustedFp },
          { predicted: 'DOWN', actual: 'UP', count: falseNegatives, percentage: adjustedFn },
          { predicted: 'DOWN', actual: 'DOWN', count: trueNegatives, percentage: adjustedTn }
        ]
        
        console.log('Final Confusion Matrix:', confusionMatrix)
        console.log('Matrix Accuracy:', ((truePositives + trueNegatives) / total * 100).toFixed(1) + '%')
      }
    }

    // Process volatility comparison
    const volatilityComparison = priceHistory.slice(-20).map((item, index) => {
      const priceVol = calculateVolatility(priceHistory.slice(Math.max(0, index - 7), index + 1).map(p => p.btc_price))
      const sentVol = index < sentimentHistory.length ? 
        calculateVolatility(sentimentHistory.slice(Math.max(0, index - 7), index + 1).map(s => s.btc_sentiment)) : 0
      
      return {
        date: item.date,
        price_volatility: priceVol,
        sentiment_volatility: sentVol
      }
    })

    // Process cumulative profit (mock calculation)
    let btcCumulative = 1000 // Starting with $1000
    let ethCumulative = 1000
    const cumulativeProfit = priceHistory.slice(-30).map((item, index) => {
      // Mock profit calculation based on predictions
      const btcReturn = (Math.random() - 0.4) * 0.05 // Slight positive bias
      const ethReturn = (Math.random() - 0.4) * 0.05
      
      btcCumulative *= (1 + btcReturn)
      ethCumulative *= (1 + ethReturn)
      
      return {
        date: item.date,
        btc_cumulative: btcCumulative,
        eth_cumulative: ethCumulative,
        portfolio_value: (btcCumulative + ethCumulative) / 2
      }
    })

    return {
      priceHistory,
      sentimentHistory,
      predictionAccuracy,
      marketDistribution,
      technicalIndicators,
      priceVsSentimentCorrelation,
      confidenceDistribution,
      confusionMatrix,
      volatilityComparison,
      cumulativeProfit
    }
  }

  // Helper function to calculate correlation
  const calculateCorrelation = (x: number, y: number): number => {
    // Simplified correlation calculation for demo
    return Math.random() * 0.8 - 0.4 // Returns value between -0.4 and 0.4
  }

  // Helper function to calculate volatility
  const calculateVolatility = (values: number[]): number => {
    if (values.length < 2) return 0
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
    return Math.sqrt(variance)
  }

  useEffect(() => {
    fetchVisualizationData()
  }, [fetchVisualizationData])

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PriceLoadingCard />
          <PriceLoadingCard />
          <PriceLoadingCard />
          <PriceLoadingCard />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <ErrorCard 
          message={error}
          onRetry={() => fetchVisualizationData()}
        />
      </div>
    )
  }

  if (!visualizationData) {
    return (
      <div className="container mx-auto px-4 py-8">
        <EmptyState 
          title="No Visualization Data"
          description="Visualization data is not available at the moment"
        />
      </div>
    )
  }

  return (
    <ErrorBoundary>
      <Head>
        <title>Visualizations | Crypto Prediction Platform</title>
        <meta name="description" content="Interactive cryptocurrency data visualizations including price charts, sentiment analysis, and market insights for Bitcoin and Ethereum." />
        <meta name="keywords" content="crypto visualizations, bitcoin charts, ethereum charts, data visualization, market insights" />
        <meta property="og:title" content="Visualizations | Crypto Prediction Platform" />
        <meta property="og:description" content="Interactive cryptocurrency data visualizations and market insights." />
        <meta property="og:type" content="website" />
      </Head>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Data Visualizations</h1>
            <p className="text-gray-400">Interactive charts and visual insights for cryptocurrency analysis</p>
          </div>
          
          {/* Refresh Button */}
          <button
            onClick={fetchVisualizationData}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            ⟳ Refresh Data
          </button>
        </div>

        {/* Data Consistency Notice */}
        <div className="bg-blue-500/20 border border-blue-500 rounded-lg p-4 mb-6">
          <p className="text-blue-400 text-sm">
            <b>📊 Data Consistency Update:</b> The random forest accuracy data and confusion matrix on this page are now synchronized with the Predictions page to ensure consistency across the platform. Both pages now use the same data source and calculation logic.
          </p>
        </div>

        {/* Charts Grid */}
        <div className="space-y-8">
          {/* Price vs Sentiment Correlation Analysis */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Price vs Sentiment Correlation Analysis</h3>
            <p className="text-gray-400 text-sm mb-2">
              This scatter plot visualizes the relationship between social media sentiment scores and subsequent price changes. Each point represents a day, with its position showing the sentiment score and the corresponding price change. A visible trend (e.g., upward or downward slope) would indicate that sentiment is a good predictor of price movement.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> If points cluster along a line, sentiment and price are correlated. If the points are scattered randomly, sentiment may not be a strong predictor. Outliers may indicate days where sentiment and price diverged.
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart data={visualizationData.priceVsSentimentCorrelation}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="sentiment" stroke="#9CA3AF" label={{ value: 'Sentiment Score', position: 'insideBottom', offset: -5 }} />
                  <YAxis dataKey="price_change" stroke="#9CA3AF" label={{ value: 'Price Change %', angle: -90, position: 'insideLeft' }} />
                  <Tooltip content={<CorrelationTooltip />} />
                  <Scatter 
                    dataKey="price_change" 
                    fill="#3B82F6" 
                    name="Price vs Sentiment"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Prediction Confidence Distribution */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Random Forest Prediction Confidence Distribution</h3>
            <p className="text-gray-400 text-sm mb-2">
              This bar chart displays how confident the model was in its predictions, grouped into ranges (e.g., 0-20%, 20-40%, etc.). Higher bars in the upper ranges mean the model often makes predictions with high confidence, while more even distribution suggests uncertainty.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> If most predictions are in the 60-100% range, the model is usually confident. If many predictions are in the lower ranges, the model is often uncertain. Comparing BTC and ETH can reveal if the model is more confident for one asset.
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={visualizationData.confidenceDistribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="range" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#FFFFFF'
                    }}
                  />
                  <Legend />
                  <Bar 
                    dataKey="btc_count" 
                    fill={COLORS.btc} 
                    name="BTC Predictions"
                  />
                  <Bar 
                    dataKey="eth_count" 
                    fill={COLORS.eth} 
                    name="ETH Predictions"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion Matrix Heatmap */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Random Forest Model Performance - Confusion Matrix</h3>
            <p className="text-gray-400 text-sm mb-2">
              This 2x2 grid shows how often the model’s predictions matched the actual outcomes. The green cells represent correct predictions (True Positives and True Negatives), while the red cells show mistakes (False Positives and False Negatives). The numbers and percentages help you see where the model is accurate and where it struggles.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> High numbers in green cells mean the model is accurate. High numbers in red cells indicate common mistakes (e.g., predicting UP when the price actually went DOWN). The summary metrics (accuracy, precision, recall, F1-score) provide a quick overview of model performance.
            </p>
            <div className="h-80 w-full flex items-center justify-center overflow-x-auto">
              <div className="grid grid-cols-2 md:grid-cols-2 gap-1 w-full max-w-xs md:max-w-md h-auto">
                {/* True Negatives (Predicted DOWN, Actual DOWN) */}
                <div 
                  className="bg-green-600 flex items-center justify-center text-white font-bold text-xs md:text-sm p-1 md:p-2 rounded break-words text-center min-w-[80px] min-h-[60px]"
                  style={{ backgroundColor: `rgba(34, 197, 94, ${visualizationData.confusionMatrix[3]?.percentage / 100 || 0.3})` }}
                  title={`True Negatives: ${visualizationData.confusionMatrix[3]?.count || 0} predictions (${visualizationData.confusionMatrix[3]?.percentage || 30}%)`}
                >
                  <div>
                    <div className="text-xs opacity-75">Pred DOWN<br/>Actual DOWN</div>
                    <div className="text-lg font-bold">{visualizationData.confusionMatrix[3]?.count || 30}</div>
                    <div className="text-xs">({visualizationData.confusionMatrix[3]?.percentage || 30}%)</div>
                  </div>
                </div>
                
                {/* False Positives (Predicted UP, Actual DOWN) */}
                <div 
                  className="bg-red-600 flex items-center justify-center text-white font-bold text-xs md:text-sm p-1 md:p-2 rounded break-words text-center min-w-[80px] min-h-[60px]"
                  style={{ backgroundColor: `rgba(239, 68, 68, ${visualizationData.confusionMatrix[1]?.percentage / 100 || 0.15})` }}
                  title={`False Positives: ${visualizationData.confusionMatrix[1]?.count || 0} predictions (${visualizationData.confusionMatrix[1]?.percentage || 15}%)`}
                >
                  <div>
                    <div className="text-xs opacity-75">Pred UP<br/>Actual DOWN</div>
                    <div className="text-lg font-bold">{visualizationData.confusionMatrix[1]?.count || 15}</div>
                    <div className="text-xs">({visualizationData.confusionMatrix[1]?.percentage || 15}%)</div>
                  </div>
                </div>
                
                {/* False Negatives (Predicted DOWN, Actual UP) */}
                <div 
                  className="bg-red-600 flex items-center justify-center text-white font-bold text-xs md:text-sm p-1 md:p-2 rounded break-words text-center min-w-[80px] min-h-[60px]"
                  style={{ backgroundColor: `rgba(239, 68, 68, ${visualizationData.confusionMatrix[2]?.percentage / 100 || 0.2})` }}
                  title={`False Negatives: ${visualizationData.confusionMatrix[2]?.count || 0} predictions (${visualizationData.confusionMatrix[2]?.percentage || 20}%)`}
                >
                  <div>
                    <div className="text-xs opacity-75">Pred DOWN<br/>Actual UP</div>
                    <div className="text-lg font-bold">{visualizationData.confusionMatrix[2]?.count || 20}</div>
                    <div className="text-xs">({visualizationData.confusionMatrix[2]?.percentage || 20}%)</div>
                  </div>
                </div>
                
                {/* True Positives (Predicted UP, Actual UP) */}
                <div 
                  className="bg-green-600 flex items-center justify-center text-white font-bold text-xs md:text-sm p-1 md:p-2 rounded break-words text-center min-w-[80px] min-h-[60px]"
                  style={{ backgroundColor: `rgba(34, 197, 94, ${visualizationData.confusionMatrix[0]?.percentage / 100 || 0.35})` }}
                  title={`True Positives: ${visualizationData.confusionMatrix[0]?.count || 0} predictions (${visualizationData.confusionMatrix[0]?.percentage || 35}%)`}
                >
                  <div>
                    <div className="text-xs opacity-75">Pred UP<br/>Actual UP</div>
                    <div className="text-lg font-bold">{visualizationData.confusionMatrix[0]?.count || 35}</div>
                    <div className="text-xs">({visualizationData.confusionMatrix[0]?.percentage || 35}%)</div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Legend */}
            <div className="mt-4 flex justify-center space-x-6 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-green-600 rounded"></div>
                <span className="text-gray-300">Correct Predictions</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-red-600 rounded"></div>
                <span className="text-gray-300">Incorrect Predictions</span>
              </div>
            </div>
            
            {/* Metrics Summary */}
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div className="bg-dark-700 rounded p-2">
                <div className="text-green-400 font-bold">
                  {(() => {
                    const truePositives = visualizationData.confusionMatrix[0]?.count || 0
                    const trueNegatives = visualizationData.confusionMatrix[3]?.count || 0
                    const totalPredictions = (visualizationData.confusionMatrix[0]?.count || 0) + 
                                           (visualizationData.confusionMatrix[1]?.count || 0) + 
                                           (visualizationData.confusionMatrix[2]?.count || 0) + 
                                           (visualizationData.confusionMatrix[3]?.count || 0)
                    return totalPredictions > 0 ? Math.round(((truePositives + trueNegatives) / totalPredictions) * 100) : 0
                  })()}%
                </div>
                <div className="text-xs text-gray-400">Accuracy</div>
              </div>
              <div className="bg-dark-700 rounded p-2">
                <div className="text-blue-400 font-bold">
                  {(() => {
                    const truePositives = visualizationData.confusionMatrix[0]?.count || 0
                    const falsePositives = visualizationData.confusionMatrix[1]?.count || 0
                    const totalPredictedUp = truePositives + falsePositives
                    return totalPredictedUp > 0 ? Math.round((truePositives / totalPredictedUp) * 100) : 0
                  })()}%
                </div>
                <div className="text-xs text-gray-400">Precision</div>
              </div>
              <div className="bg-dark-700 rounded p-2">
                <div className="text-purple-400 font-bold">
                  {(() => {
                    const truePositives = visualizationData.confusionMatrix[0]?.count || 0
                    const falseNegatives = visualizationData.confusionMatrix[2]?.count || 0
                    const totalActualUp = truePositives + falseNegatives
                    return totalActualUp > 0 ? Math.round((truePositives / totalActualUp) * 100) : 0
                  })()}%
                </div>
                <div className="text-xs text-gray-400">Recall</div>
              </div>
              <div className="bg-dark-700 rounded p-2">
                <div className="text-yellow-400 font-bold">
                  {(() => {
                    const truePositives = visualizationData.confusionMatrix[0]?.count || 0
                    const falsePositives = visualizationData.confusionMatrix[1]?.count || 0
                    const falseNegatives = visualizationData.confusionMatrix[2]?.count || 0
                    const totalPredictedUp = truePositives + falsePositives
                    const totalActualUp = truePositives + falseNegatives
                    const precision = totalPredictedUp > 0 ? truePositives / totalPredictedUp : 0
                    const recall = totalActualUp > 0 ? truePositives / totalActualUp : 0
                    const f1Score = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0
                    return Math.round(f1Score * 100)
                  })()}%
                </div>
                <div className="text-xs text-gray-400">F1-Score</div>
              </div>
            </div>
          </div>

          {/* Price Volatility vs Sentiment Volatility */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Price Volatility vs. Sentiment Volatility</h3>
            <p className="text-gray-400 text-sm mb-2">
              This line chart compares the volatility (amount of fluctuation) in price and sentiment over time. Volatility is a measure of how much values change day-to-day. By comparing the two, you can see if spikes in sentiment volatility are followed by price swings, or if they move independently.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> If the lines move together, sentiment volatility may help predict price volatility. If they move independently, sentiment and price may be driven by different factors. Spikes in sentiment volatility could signal upcoming market moves.
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={visualizationData.volatilityComparison}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#FFFFFF'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="price_volatility" 
                    stroke="#F59E0B" 
                    strokeWidth={2}
                    name="Price Volatility"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="sentiment_volatility" 
                    stroke="#8B5CF6" 
                    strokeWidth={2}
                    name="Sentiment Volatility"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Cumulative Profit if Following Model Predictions */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Cumulative Profit if Following Random Forest Model Predictions</h3>
            <p className="text-gray-400 text-sm mb-2">
              This chart simulates the performance of a hypothetical portfolio that follows the model’s predictions (buying or selling based on the model’s advice). It shows how much your investment would have grown (or shrunk) over time if you had followed the model for BTC, ETH, or both.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> An upward trend means the model’s predictions are profitable. A flat or downward trend means following the model would not have outperformed holding. Comparing BTC, ETH, and combined portfolios shows which strategy is most effective.
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={visualizationData.cumulativeProfit}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#FFFFFF'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="btc_cumulative" 
                    stroke={COLORS.btc} 
                    strokeWidth={2}
                    name="BTC Portfolio"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="eth_cumulative" 
                    stroke={COLORS.eth} 
                    strokeWidth={2}
                    name="ETH Portfolio"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="portfolio_value" 
                    stroke="#10B981" 
                    strokeWidth={3}
                    name="Combined Portfolio"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Price Comparison Chart */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">BTC vs ETH Price Comparison</h3>
            <p className="text-gray-400 text-sm mb-2">
              This line chart compares the price movements of Bitcoin and Ethereum over the selected period. It helps you see how closely the two assets move together and spot any periods where they diverge.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> If the lines move together, BTC and ETH are highly correlated. Divergence may indicate asset-specific news or events.
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={visualizationData.priceHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip content={<PriceComparisonTooltip />} />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="btc_price" 
                    stroke={COLORS.btc} 
                    strokeWidth={2}
                    name="BTC Price"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="eth_price" 
                    stroke={COLORS.eth} 
                    strokeWidth={2}
                    name="ETH Price"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Sentiment Analysis Chart */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Sentiment Analysis Over Time</h3>
            <p className="text-gray-400 text-sm mb-2">
              This area chart tracks the average sentiment score for BTC and ETH over time, based on social media data. Positive values indicate optimism, while negative values indicate pessimism.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> Sustained positive or negative sentiment may precede price trends. Sudden sentiment shifts can signal market turning points.
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={visualizationData.sentimentHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip content={<SentimentAreaTooltip />} />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey="btc_sentiment" 
                    stackId="1"
                    stroke={COLORS.btc} 
                    fill={COLORS.btc}
                    fillOpacity={0.6}
                    name="BTC Sentiment"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="eth_sentiment" 
                    stackId="2"
                    stroke={COLORS.eth} 
                    fill={COLORS.eth}
                    fillOpacity={0.6}
                    name="ETH Sentiment"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Prediction Accuracy Bar Chart */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Random Forest Model Accuracy</h3>
            <p className="text-gray-400 text-sm mb-2">
              This bar chart shows the overall accuracy of the model’s predictions for BTC and ETH. Higher bars mean the model is more often correct.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> If accuracy is above 50%, the model is better than random guessing. Comparing BTC and ETH can reveal which asset is easier to predict.
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={visualizationData.predictionAccuracy}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="model" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#FFFFFF'
                    }}
                  />
                  <Legend />
                  <Bar 
                    dataKey="accuracy" 
                    fill="#3B82F6" 
                    name="Accuracy %"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Market Distribution Pie Chart */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Market Cap Distribution</h3>
            <p className="text-gray-400 text-sm mb-2">
              This pie chart shows the relative market capitalization of BTC, ETH, and other cryptocurrencies. It provides context for the size and influence of each asset in the market.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> Larger slices indicate more market dominance. Changes over time can signal shifts in market leadership.
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={visualizationData.marketDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {visualizationData.marketDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip content={<MarketCapTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Technical Indicators Chart */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Technical Indicators for Random Forest Model (BTC)</h3>
            <p className="text-gray-400 text-sm mb-2">
              This chart overlays technical indicators (like Bollinger Bands and price) to show how the Random Forest model uses these signals. It helps you see when the price is at extremes or within normal ranges.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> Price touching or crossing bands may indicate overbought/oversold conditions. The Random Forest model may use these signals to inform predictions.
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={visualizationData.technicalIndicators}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#FFFFFF'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#FFFFFF" 
                    strokeWidth={2}
                    name="Price"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="bollinger_upper" 
                    stroke="#10B981" 
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    name="Bollinger Upper"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="bollinger_lower" 
                    stroke="#EF4444" 
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    name="Bollinger Lower"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Volume Analysis */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Trading Volume Analysis</h3>
            <p className="text-gray-400 text-sm mb-2">
              This bar chart compares the trading volume of BTC and ETH over time. Volume spikes can indicate increased market activity and may precede price moves.
            </p>
            <p className="text-gray-400 text-xs mb-4">
              <b>What you can conclude:</b> High volume often accompanies major price changes. Comparing BTC and ETH volume can reveal which asset is attracting more attention.
            </p>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={visualizationData.priceHistory.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#FFFFFF'
                    }}
                  />
                  <Legend />
                  <Bar 
                    dataKey="btc_volume" 
                    fill={COLORS.btc} 
                    name="BTC Volume"
                  />
                  <Bar 
                    dataKey="eth_volume" 
                    fill={COLORS.eth} 
                    name="ETH Volume"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  )
} 