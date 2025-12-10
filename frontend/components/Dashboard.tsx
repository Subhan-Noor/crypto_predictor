'use client'

import React, { useState, useEffect } from 'react'
import { PriceCard } from './PriceCard'
import { PredictionCard } from './PredictionCard'
import { PriceChart } from './PriceChart'
import { LoadingSpinner } from './LoadingSpinner'
import { apiService, handleAPIError } from '../utils/api'
import { CurrentPrice, PredictionData, PriceData, Currency } from '../types'

export const Dashboard: React.FC = () => {
  const [currentPrices, setCurrentPrices] = useState<Record<Currency, CurrentPrice> | null>(null)
  const [predictions, setPredictions] = useState<Record<Currency, PredictionData> | null>(null)
  const [priceData, setPriceData] = useState<Record<Currency, PriceData[]>>({ BTC: [], ETH: [] })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadDashboardData = async () => {
      setLoading(true)
      setError(null)

      try {
        // Load current prices
        try {
          const prices = await apiService.getCurrentPrices()
          setCurrentPrices(prices)
        } catch (err) {
          console.error('Failed to load current prices:', err)
        }

        // Load predictions for both currencies
        const predictionPromises = ['BTC' as Currency, 'ETH' as Currency].map(async (currency) => {
          try {
            const prediction = await apiService.getPrediction(currency)
            return { currency, prediction }
          } catch (err) {
            console.error(`Failed to load ${currency} prediction:`, err)
            return null
          }
        })

        const predictionResults = await Promise.all(predictionPromises)
        const predictionsMap: Record<Currency, PredictionData> = {} as any
        predictionResults.forEach(result => {
          if (result) {
            predictionsMap[result.currency] = result.prediction
          }
        })
        setPredictions(predictionsMap)

        // Load recent price data for charts
        const pricePromises = ['BTC' as Currency, 'ETH' as Currency].map(async (currency) => {
          try {
            const response = await apiService.getPriceData(currency, 1, 30, 30) // Last 30 days
            return { currency, data: response.data }
          } catch (err) {
            console.error(`Failed to load ${currency} price data:`, err)
            // Try basic endpoint as fallback
            try {
              const basicData = await apiService.getBasicPriceData(currency)
              return { currency, data: basicData.slice(-30) } // Last 30 entries
            } catch (fallbackErr) {
              console.error(`Fallback failed for ${currency}:`, fallbackErr)
              return { currency, data: [] }
            }
          }
        })

        const priceResults = await Promise.all(pricePromises)
        const priceDataMap: Record<Currency, PriceData[]> = { BTC: [], ETH: [] }
        priceResults.forEach(result => {
          priceDataMap[result.currency] = result.data
        })
        setPriceData(priceDataMap)

      } catch (err) {
        setError(handleAPIError(err))
      } finally {
        setLoading(false)
      }
    }

    loadDashboardData()

    // Set up auto-refresh every 30 seconds
    const interval = setInterval(loadDashboardData, 30000)
    return () => clearInterval(interval)
  }, [])

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="bg-red-500/20 border border-red-500 rounded-lg p-6 text-center">
          <h2 className="text-xl font-semibold text-red-400 mb-2">Error Loading Dashboard</h2>
          <p className="text-red-300">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">
          Crypto Price Prediction Dashboard
        </h1>
        <p className="text-gray-400">
          AI-powered Bitcoin and Ethereum price forecasting with sentiment analysis
        </p>
      </div>

      {/* Current Prices */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {loading ? (
          <>
            <PriceCard price={{} as CurrentPrice} isLoading={true} />
            <PriceCard price={{} as CurrentPrice} isLoading={true} />
          </>
        ) : (
          <>
            {currentPrices?.BTC && <PriceCard price={currentPrices.BTC} />}
            {currentPrices?.ETH && <PriceCard price={currentPrices.ETH} />}
          </>
        )}
      </div>

      {/* Predictions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div>
          <h2 className="text-2xl font-semibold text-white mb-4">AI Predictions</h2>
          {loading ? (
            <PredictionCard prediction={{} as PredictionData} isLoading={true} />
          ) : predictions?.BTC ? (
            <PredictionCard prediction={predictions.BTC} />
          ) : (
            <div className="bg-dark-800 rounded-lg p-6 text-center">
              <p className="text-gray-400">BTC prediction unavailable</p>
            </div>
          )}
        </div>
        
        <div className="mt-8 md:mt-0">
          <h2 className="text-2xl font-semibold text-white mb-4 invisible md:visible">&nbsp;</h2>
          {loading ? (
            <PredictionCard prediction={{} as PredictionData} isLoading={true} />
          ) : predictions?.ETH ? (
            <PredictionCard prediction={predictions.ETH} />
          ) : (
            <div className="bg-dark-800 rounded-lg p-6 text-center">
              <p className="text-gray-400">ETH prediction unavailable</p>
            </div>
          )}
        </div>
      </div>

      {/* Price Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          {loading ? (
            <LoadingSpinner message="Loading BTC chart..." />
          ) : (
            <PriceChart 
              data={priceData.BTC} 
              currency="BTC" 
              height={300} 
            />
          )}
        </div>
        
        <div>
          {loading ? (
            <LoadingSpinner message="Loading ETH chart..." />
          ) : (
            <PriceChart 
              data={priceData.ETH} 
              currency="ETH" 
              height={300} 
            />
          )}
        </div>
      </div>

      {/* Status Bar */}
      <div className="mt-8 text-center">
        <div className="inline-flex items-center space-x-2 text-sm text-gray-400">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span>Live data • Auto-refresh every 30s</span>
        </div>
      </div>
    </div>
  )
} 