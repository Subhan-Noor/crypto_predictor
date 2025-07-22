'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { PriceCard } from './PriceCard'
import { PredictionCard } from './PredictionCard'
import { PriceChart } from './PriceChart'
import { ErrorBoundary } from './ErrorBoundary'
import { 
  NoDataState, 
  NoPredictionsState, 
  NoChartDataState 
} from './EmptyState'
import { 
  APIErrorCard, 
  DataErrorCard, 
  PredictionErrorCard 
} from './ErrorCard'
import { 
  PriceLoadingCard, 
  PredictionLoadingCard, 
  ChartLoadingCard 
} from './EnhancedLoadingSpinner'
import { apiService, handleAPIError } from '../utils/api'
import { CurrentPrice, PredictionData, PriceData, Currency } from '../types'

interface DashboardState {
  currentPrices: Record<Currency, CurrentPrice> | null
  predictions: Record<Currency, PredictionData> | null
  priceData: Record<Currency, PriceData[]>
  loading: {
    prices: boolean
    predictions: boolean
    charts: boolean
  }
  errors: {
    prices: string | null
    predictions: string | null
    charts: string | null
  }
  lastUpdate: Date | null
}

export const EnhancedDashboard: React.FC = () => {
  const [state, setState] = useState<DashboardState>({
    currentPrices: null,
    predictions: null,
    priceData: { BTC: [], ETH: [] },
    loading: {
      prices: true,
      predictions: true,
      charts: true
    },
    errors: {
      prices: null,
      predictions: null,
      charts: null
    },
    lastUpdate: null
  })

  const updateLoadingState = useCallback((key: keyof DashboardState['loading'], value: boolean) => {
    setState(prev => ({
      ...prev,
      loading: { ...prev.loading, [key]: value }
    }))
  }, [])

  const updateErrorState = useCallback((key: keyof DashboardState['errors'], value: string | null) => {
    setState(prev => ({
      ...prev,
      errors: { ...prev.errors, [key]: value }
    }))
  }, [])

  const loadCurrentPrices = useCallback(async () => {
    updateLoadingState('prices', true)
    updateErrorState('prices', null)
    
    try {
      const prices = await apiService.getCurrentPrices()
      setState(prev => ({ ...prev, currentPrices: prices }))
    } catch (err) {
      console.error('Failed to load current prices:', err)
      updateErrorState('prices', handleAPIError(err))
    } finally {
      updateLoadingState('prices', false)
    }
  }, [updateLoadingState, updateErrorState])

  const loadPredictions = useCallback(async () => {
    updateLoadingState('predictions', true)
    updateErrorState('predictions', null)
    
    try {
      const currencies: Currency[] = ['BTC', 'ETH']
      const predictionPromises = currencies.map(async (currency) => {
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
      
      setState(prev => ({ ...prev, predictions: predictionsMap }))
    } catch (err) {
      console.error('Failed to load predictions:', err)
      updateErrorState('predictions', handleAPIError(err))
    } finally {
      updateLoadingState('predictions', false)
    }
  }, [updateLoadingState, updateErrorState])

  const loadPriceData = useCallback(async () => {
    updateLoadingState('charts', true)
    updateErrorState('charts', null)
    
    try {
      const currencies: Currency[] = ['BTC', 'ETH']
      const pricePromises = currencies.map(async (currency) => {
        try {
          const response = await apiService.getPriceData(currency, 1, 30, 30)
          return { currency, data: response.data }
        } catch (err) {
          console.error(`Failed to load ${currency} price data:`, err)
          try {
            const basicData = await apiService.getBasicPriceData(currency)
            return { currency, data: basicData.slice(-30) }
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
      
      setState(prev => ({ ...prev, priceData: priceDataMap }))
    } catch (err) {
      console.error('Failed to load price data:', err)
      updateErrorState('charts', handleAPIError(err))
    } finally {
      updateLoadingState('charts', false)
    }
  }, [updateLoadingState, updateErrorState])

  const loadAllData = useCallback(async () => {
    setState(prev => ({ ...prev, lastUpdate: new Date() }))
    await Promise.all([
      loadCurrentPrices(),
      loadPredictions(),
      loadPriceData()
    ])
  }, [loadCurrentPrices, loadPredictions, loadPriceData])

  // Initial load
  useEffect(() => {
    loadAllData()
  }, [loadAllData])

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(loadAllData, 30000)
    return () => clearInterval(interval)
  }, [loadAllData])

  const renderPriceSection = () => {
    if (state.errors.prices) {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <APIErrorCard onRetry={loadCurrentPrices} />
          <APIErrorCard onRetry={loadCurrentPrices} />
        </div>
      )
    }

    if (state.loading.prices) {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <PriceLoadingCard />
          <PriceLoadingCard />
        </div>
      )
    }

    if (!state.currentPrices) {
      return (
        <div className="bg-dark-800 rounded-lg p-6 mb-8">
          <NoDataState onRetry={loadCurrentPrices} />
        </div>
      )
    }

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {state.currentPrices.BTC && <PriceCard price={state.currentPrices.BTC} />}
        {state.currentPrices.ETH && <PriceCard price={state.currentPrices.ETH} />}
      </div>
    )
  }

  const renderPredictionSection = () => {
    if (state.errors.predictions) {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <PredictionErrorCard currency="BTC" onRetry={loadPredictions} />
          <PredictionErrorCard currency="ETH" onRetry={loadPredictions} />
        </div>
      )
    }

    if (state.loading.predictions) {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <PredictionLoadingCard />
          <PredictionLoadingCard />
        </div>
      )
    }

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div>
          <h2 className="text-2xl font-semibold text-white mb-4">AI Predictions</h2>
          {state.predictions?.BTC ? (
            <PredictionCard prediction={state.predictions.BTC} />
          ) : (
            <div className="bg-dark-800 rounded-lg">
              <NoPredictionsState currency="BTC" onRetry={loadPredictions} />
            </div>
          )}
        </div>
        
        <div>
          <h2 className="text-2xl font-semibold text-white mb-4 invisible md:visible">&nbsp;</h2>
          {state.predictions?.ETH ? (
            <PredictionCard prediction={state.predictions.ETH} />
          ) : (
            <div className="bg-dark-800 rounded-lg">
              <NoPredictionsState currency="ETH" onRetry={loadPredictions} />
            </div>
          )}
        </div>
      </div>
    )
  }

  const renderChartsSection = () => {
    if (state.errors.charts) {
      return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <DataErrorCard dataType="BTC chart data" onRetry={loadPriceData} />
          <DataErrorCard dataType="ETH chart data" onRetry={loadPriceData} />
        </div>
      )
    }

    if (state.loading.charts) {
      return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ChartLoadingCard />
          <ChartLoadingCard />
        </div>
      )
    }

    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          {state.priceData.BTC.length > 0 ? (
            <PriceChart 
              data={state.priceData.BTC} 
              currency="BTC" 
              height={300} 
            />
          ) : (
            <div className="bg-dark-800 rounded-lg">
              <NoChartDataState currency="BTC" onRetry={loadPriceData} />
            </div>
          )}
        </div>
        
        <div>
          {state.priceData.ETH.length > 0 ? (
            <PriceChart 
              data={state.priceData.ETH} 
              currency="ETH" 
              height={300} 
            />
          ) : (
            <div className="bg-dark-800 rounded-lg">
              <NoChartDataState currency="ETH" onRetry={loadPriceData} />
            </div>
          )}
        </div>
      </div>
    )
  }

  const isAnyLoading = Object.values(state.loading).some(loading => loading)
  const hasAnyErrors = Object.values(state.errors).some(error => error !== null)

  return (
    <ErrorBoundary>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                Crypto Price Prediction Dashboard
              </h1>
              <p className="text-gray-400">
                AI-powered Bitcoin and Ethereum price forecasting with sentiment analysis
              </p>
            </div>
            
            {/* Refresh Button */}
            <button
              onClick={loadAllData}
              disabled={isAnyLoading}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                isAnyLoading
                  ? 'bg-dark-700 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700 hover:scale-105'
              }`}
            >
              {isAnyLoading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                  <span>Refreshing...</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  <span>Refresh</span>
                </div>
              )}
            </button>
          </div>
          
          {/* Status Indicators */}
          {(hasAnyErrors || state.lastUpdate) && (
            <div className="mt-4 flex items-center space-x-4 text-sm">
              {hasAnyErrors && (
                <div className="flex items-center space-x-1 text-red-400">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                  </svg>
                  <span>Some data failed to load</span>
                </div>
              )}
              
              {state.lastUpdate && (
                <div className="flex items-center space-x-1 text-gray-400">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span>
                    Last updated: {state.lastUpdate.toLocaleTimeString()}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Current Prices */}
        {renderPriceSection()}

        {/* Predictions */}
        {renderPredictionSection()}

        {/* Price Charts */}
        {renderChartsSection()}

        {/* Status Bar */}
        <div className="mt-8 text-center">
          <div className="inline-flex items-center space-x-2 text-sm text-gray-400">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span>Live data • Auto-refresh every 30s</span>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  )
} 