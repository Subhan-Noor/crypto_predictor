'use client'

import React, { useState, useEffect, useCallback } from 'react'
import Head from 'next/head'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { apiService } from '../../utils/api'
import { ErrorBoundary } from '../../components/ErrorBoundary'
import { PredictionLoadingCard } from '../../components/EnhancedLoadingSpinner'
import { ErrorCard } from '../../components/ErrorCard'
import { EmptyState } from '../../components/EmptyState'

interface PredictionHistoryItem {
  id: string
  currency: string
  prediction_date: string
  predicted_direction: 'UP' | 'DOWN'
  confidence_score: number
  actual_direction?: 'UP' | 'DOWN'
  is_correct?: boolean
  model_version: string
  model_type: string
}

interface AccuracyMetrics {
  overall_accuracy: number
  btc_accuracy: number
  eth_accuracy: number
  total_predictions: number
  correct_predictions: number
  precision: number
  recall: number
  f1_score: number
}

interface ModelPerformance {
  model_type: string
  accuracy: number
  predictions_count: number
  avg_confidence: number
  last_prediction: string
}

const COLORS = ['#10B981', '#EF4444', '#F59E0B', '#8B5CF6', '#3B82F6']

export default function PredictionsPage() {
  const [predictionHistory, setPredictionHistory] = useState<PredictionHistoryItem[]>([])
  const [accuracyMetrics, setAccuracyMetrics] = useState<AccuracyMetrics | null>(null)
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance[]>([])
  const [selectedCurrency, setSelectedCurrency] = useState<'ALL' | 'BTC' | 'ETH'>('ALL')
  const [selectedTimeRange, setSelectedTimeRange] = useState(30)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchPredictionData = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      // Fetch real prediction data from the backend
      const predictionHistory = await apiService.getAllPredictionHistory(selectedTimeRange)
      
      // Transform backend data to frontend format
      const transformedPredictions: PredictionHistoryItem[] = predictionHistory.combined.map(pred => ({
        id: pred.id || `${pred.currency}-${pred.prediction_date}`,
        currency: pred.currency,
        prediction_date: pred.prediction_date,
        predicted_direction: pred.predicted_direction,
        confidence_score: pred.confidence_score || 0,
        actual_direction: pred.actual_direction || undefined,
        is_correct: pred.is_correct !== undefined ? pred.is_correct : undefined,
        model_version: pred.model_version || 'unknown',
        model_type: pred.model_version ? pred.model_version.split('_')[1] || 'unknown' : 'unknown'
      }))

      // Get accuracy data for both currencies
      const [btcAccuracy, ethAccuracy] = await Promise.all([
        apiService.getPredictionAccuracy('BTC', selectedTimeRange),
        apiService.getPredictionAccuracy('ETH', selectedTimeRange)
      ])

      // Calculate accuracy metrics
      const accuracy = calculateRealAccuracy(transformedPredictions, btcAccuracy, ethAccuracy)
      const modelPerformance = calculateRealModelPerformance(transformedPredictions)

      setPredictionHistory(transformedPredictions)
      setAccuracyMetrics(accuracy)
      setModelPerformance(modelPerformance)

    } catch (err) {
      console.error('Error fetching prediction data:', err)
      setError('Failed to load prediction data. Please try again.')
      
      // Fallback to empty state instead of mock data
      setPredictionHistory([])
      setAccuracyMetrics({
        overall_accuracy: 0,
        btc_accuracy: 0,
        eth_accuracy: 0,
        total_predictions: 0,
        correct_predictions: 0,
        precision: 0,
        recall: 0,
        f1_score: 0
      })
      setModelPerformance([])
    } finally {
      setLoading(false)
    }
  }, [selectedTimeRange])

  const calculateRealAccuracy = (
    predictions: PredictionHistoryItem[],
    btcAccuracy: any,
    ethAccuracy: any
  ): AccuracyMetrics => {
    // Use backend accuracy data for each currency separately
    const btc_accuracy = btcAccuracy.accuracy
    const eth_accuracy = ethAccuracy.accuracy
    
    // Calculate overall accuracy as weighted average
    const btc_total = btcAccuracy.total_predictions
    const eth_total = ethAccuracy.total_predictions
    const total = btc_total + eth_total
    
    const overall_accuracy = total > 0 ? 
      ((btc_accuracy * btc_total + eth_accuracy * eth_total) / total) : 0
    
    const total_correct = btcAccuracy.correct_predictions + ethAccuracy.correct_predictions
    
    // Calculate proper precision and recall
    const precision = total > 0 ? (total_correct / total) : 0
    const recall = total > 0 ? (total_correct / total) : 0
    
    // Calculate proper F1 score: F1 = 2 * (precision * recall) / (precision + recall)
    const f1_score = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0
    
    return {
      overall_accuracy: overall_accuracy,
      btc_accuracy: btc_accuracy,
      eth_accuracy: eth_accuracy,
      total_predictions: total,
      correct_predictions: total_correct,
      precision: precision * 100, // Convert to percentage
      recall: recall * 100, // Convert to percentage
      f1_score: f1_score * 100 // Convert to percentage
    }
  }

  const calculateRealModelPerformance = (predictions: PredictionHistoryItem[]): ModelPerformance[] => {
    const modelTypes = ['logistic_regression', 'random_forest', 'lstm', 'regression', 'unknown']
    
    return modelTypes.map(modelType => {
      const modelPredictions = predictions.filter(p => 
        p.model_type === modelType || 
        (modelType === 'unknown' && !['logistic_regression', 'random_forest', 'lstm', 'regression'].includes(p.model_type))
      )
      
      // Only count validated predictions
      const validatedPredictions = modelPredictions.filter(p => p.is_correct !== undefined)
      const correct = validatedPredictions.filter(p => p.is_correct === true).length
      const avgConfidence = modelPredictions.reduce((sum, p) => sum + p.confidence_score, 0) / (modelPredictions.length || 1)
      
      return {
        model_type: modelType,
        accuracy: validatedPredictions.length > 0 ? (correct / validatedPredictions.length) * 100 : 0,
        predictions_count: modelPredictions.length,
        validated_count: validatedPredictions.length,
        avg_confidence: avgConfidence * 100, // Convert to percentage
        last_prediction: modelPredictions[0]?.prediction_date || 'N/A'
      }
    }).filter(model => model.predictions_count > 0) // Only show models with predictions
  }

  const getFilteredPredictions = () => {
    return predictionHistory.filter(prediction => {
      if (selectedCurrency !== 'ALL' && prediction.currency !== selectedCurrency) {
        return false
      }
      return true
    })
  }

  const getDirectionDistribution = () => {
    const filtered = getFilteredPredictions()
    const upPredictions = filtered.filter(p => p.predicted_direction === 'UP').length
    const downPredictions = filtered.filter(p => p.predicted_direction === 'DOWN').length
    
    return [
      { name: 'UP Predictions', value: upPredictions, color: '#10B981' },
      { name: 'DOWN Predictions', value: downPredictions, color: '#EF4444' }
    ]
  }

  useEffect(() => {
    fetchPredictionData()
  }, [fetchPredictionData])

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PredictionLoadingCard />
          <PredictionLoadingCard />
          <PredictionLoadingCard />
          <PredictionLoadingCard />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <ErrorCard 
          message={error}
          onRetry={fetchPredictionData}
        />
      </div>
    )
  }

  if (!accuracyMetrics || predictionHistory.length === 0) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center py-12">
          <div className="mb-6">
            <h1 className="text-4xl font-bold text-white mb-4">Predictions Dashboard</h1>
            <p className="text-gray-400 mb-8">Real ML predictions and historical accuracy analysis</p>
          </div>
          
          {!accuracyMetrics ? (
            <EmptyState 
              title="No Prediction Data"
              description="Prediction data is not available at the moment"
            />
          ) : (
            <div className="bg-dark-800 rounded-lg p-8 max-w-2xl mx-auto">
              <div className="text-6xl mb-4">🤖</div>
              <h2 className="text-2xl font-semibold text-white mb-4">
                Welcome to Real ML Predictions!
              </h2>
              <p className="text-gray-400 mb-6">
                We&apos;ve generated your first real machine learning predictions using historical data and sentiment analysis. 
                Check back daily to see new predictions and track accuracy over time.
              </p>
              <div className="bg-green-500/20 border border-green-500 rounded-lg p-4 mb-6">
                <p className="text-green-400 font-medium">
                  ✅ Real predictions are now active using trained ML models!
                </p>
              </div>
              <button 
                onClick={fetchPredictionData}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors font-medium"
              >
                🔄 Check for New Predictions
              </button>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <ErrorBoundary>
      <Head>
        <title>Real ML Predictions - Historical Accuracy and Model Performance</title>
        <meta name="description" content="Track the accuracy of your real machine learning predictions across Bitcoin and Ethereum. Analyze model performance and prediction trends." />
        <meta name="keywords" content="real ml predictions, cryptocurrency predictions, bitcoin predictions, ethereum predictions, machine learning, accuracy, model performance" />
        <meta name="author" content="Real ML" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Head>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Predictions Dashboard</h1>
            <p className="text-gray-400">Historical predictions and model performance analysis</p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Currency Filter */}
            <select
              value={selectedCurrency}
              onChange={(e) => setSelectedCurrency(e.target.value as 'ALL' | 'BTC' | 'ETH')}
              className="px-3 py-2 bg-dark-800 text-white rounded-lg border border-dark-700 focus:border-blue-500 text-sm font-medium"
              style={{ 
                fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                fontSize: '14px',
                fontWeight: '500'
              }}
            >
              <option value="ALL">All Currencies</option>
              <option value="BTC">Bitcoin</option>
              <option value="ETH">Ethereum</option>
            </select>

            {/* Time Range Filter */}
            <select
              value={selectedTimeRange}
              onChange={(e) => setSelectedTimeRange(Number(e.target.value))}
              className="px-3 py-2 bg-dark-800 text-white rounded-lg border border-dark-700 focus:border-blue-500 text-sm font-medium"
              style={{ 
                fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                fontSize: '14px',
                fontWeight: '500'
              }}
            >
              <option value={7}>Last 7 Days</option>
              <option value={30}>Last 30 Days</option>
              <option value={90}>Last 90 Days</option>
            </select>

            {/* Refresh Button */}
            <button
              onClick={fetchPredictionData}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
            >
              ⟳ Refresh
            </button>
          </div>
        </div>

        {/* Accuracy Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-2">Overall Accuracy</h3>
            <div className="text-3xl font-bold text-green-400">
              {accuracyMetrics.overall_accuracy.toFixed(1)}%
            </div>
            <p className="text-sm text-gray-400 mt-2">
              {accuracyMetrics.correct_predictions} / {accuracyMetrics.total_predictions} predictions
            </p>
          </div>

          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-2">Bitcoin Accuracy</h3>
            <div className="text-3xl font-bold text-orange-400">
              {accuracyMetrics.btc_accuracy.toFixed(1)}%
            </div>
            <p className="text-sm text-gray-400 mt-2">BTC predictions only</p>
          </div>

          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-2">Ethereum Accuracy</h3>
            <div className="text-3xl font-bold text-purple-400">
              {accuracyMetrics.eth_accuracy.toFixed(1)}%
            </div>
            <p className="text-sm text-gray-400 mt-2">ETH predictions only</p>
          </div>

          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-2">F1 Score</h3>
            <div className="text-3xl font-bold text-blue-400">
              {accuracyMetrics.f1_score.toFixed(1)}%
            </div>
            <p className="text-sm text-gray-400 mt-2">Model performance metric</p>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-1 gap-8 mb-8">
          {/* Prediction Direction Distribution */}
          <div className="bg-dark-800 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Prediction Distribution</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={getDirectionDistribution()}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {getDirectionDistribution().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Model Performance Comparison */}
        <div className="bg-dark-800 rounded-lg p-6 mb-8">
          <h3 className="text-xl font-semibold text-white mb-4">Model Performance Comparison</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelPerformance}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="model_type" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                <Bar dataKey="accuracy" fill="#3B82F6" name="Accuracy %" />
                <Bar dataKey="avg_confidence" fill="#10B981" name="Avg Confidence" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Predictions Table */}
        <div className="bg-dark-800 rounded-lg p-6">
          <h3 className="text-xl font-semibold text-white mb-4">Recent Predictions</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-dark-700">
                  <th className="text-left py-3 px-4 text-gray-400">Date</th>
                  <th className="text-left py-3 px-4 text-gray-400">Currency</th>
                  <th className="text-left py-3 px-4 text-gray-400">Prediction</th>
                  <th className="text-left py-3 px-4 text-gray-400">Confidence</th>
                  <th className="text-left py-3 px-4 text-gray-400">Actual</th>
                  <th className="text-left py-3 px-4 text-gray-400">Result</th>
                  <th className="text-left py-3 px-4 text-gray-400">Model</th>
                </tr>
              </thead>
              <tbody>
                {getFilteredPredictions().slice(0, 10).map((prediction) => (
                  <tr key={prediction.id} className="border-b border-dark-700/50 hover:bg-dark-700/30">
                    <td className="py-3 px-4 text-white">
                      {new Date(prediction.prediction_date).toLocaleDateString()}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        prediction.currency === 'BTC' ? 'bg-orange-500/20 text-orange-400' : 'bg-purple-500/20 text-purple-400'
                      }`}>
                        {prediction.currency}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        prediction.predicted_direction === 'UP' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                      }`}>
                        {prediction.predicted_direction}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-white">
                      {(prediction.confidence_score * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {prediction.actual_direction ? (
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          prediction.actual_direction === 'UP' 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {prediction.actual_direction}
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                          Pending
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {prediction.is_correct === true ? (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          ✅ Correct
                        </span>
                      ) : prediction.is_correct === false ? (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                          ❌ Wrong
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                          Pending
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-gray-400">
                      {prediction.model_type}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {getFilteredPredictions().length === 0 && (
            <div className="text-center py-8 text-gray-400">
              No predictions found for the selected criteria
            </div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  )
} 