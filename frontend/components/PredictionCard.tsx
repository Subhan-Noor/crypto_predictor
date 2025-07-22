import React from 'react'
import { PredictionData, Currency } from '../types'

interface PredictionCardProps {
  prediction: PredictionData
  isLoading?: boolean
}

export const PredictionCard: React.FC<PredictionCardProps> = ({ 
  prediction, 
  isLoading = false 
}) => {
  if (isLoading) {
    return (
      <div className="bg-dark-800 rounded-lg p-6 animate-pulse">
        <div className="flex items-center justify-between mb-4">
          <div className="w-24 h-6 bg-dark-700 rounded"></div>
          <div className="w-16 h-6 bg-dark-700 rounded"></div>
        </div>
        <div className="w-16 h-8 bg-dark-700 rounded mb-2"></div>
        <div className="w-20 h-4 bg-dark-700 rounded"></div>
      </div>
    )
  }

  const isUp = prediction.prediction === 'UP'
  const predictionColor = isUp ? 'crypto-green' : 'crypto-red'
  const bgGradient = isUp 
    ? 'from-green-500/20 to-transparent' 
    : 'from-red-500/20 to-transparent'

  const confidenceLevel = prediction.confidence >= 0.7 ? 'High' : 
                         prediction.confidence >= 0.5 ? 'Medium' : 'Low'

  return (
    <div className={`bg-gradient-to-br ${bgGradient} bg-dark-800 rounded-lg p-6 border border-dark-700`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">
          {prediction.currency} Prediction
        </h3>
        <span className="text-xs text-gray-400 bg-dark-700 px-2 py-1 rounded">
          7 Days
        </span>
      </div>
      
      <div className="mb-3">
        <div className="flex items-center space-x-2">
          <span className={`text-2xl font-bold text-${predictionColor}`}>
            {prediction.prediction}
          </span>
          <div className={`w-6 h-6 rounded-full bg-${predictionColor} flex items-center justify-center`}>
            {isUp ? (
              <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M3.293 9.707a1 1 0 010-1.414l6-6a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V17a1 1 0 11-2 0V5.414L4.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 10.293a1 1 0 010 1.414l-6 6a1 1 0 01-1.414 0l-6-6a1 1 0 111.414-1.414L9 14.586V3a1 1 0 012 0v11.586l4.293-4.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            )}
          </div>
        </div>
      </div>
      
      <div className="flex items-center justify-between text-sm">
        <div>
          <span className="text-gray-400">Confidence: </span>
          <span className="text-white font-medium">
            {(prediction.confidence * 100).toFixed(1)}%
          </span>
        </div>
        <span className={`px-2 py-1 rounded text-xs font-medium ${
          confidenceLevel === 'High' ? 'bg-green-500/20 text-green-400' :
          confidenceLevel === 'Medium' ? 'bg-yellow-500/20 text-yellow-400' :
          'bg-red-500/20 text-red-400'
        }`}>
          {confidenceLevel}
        </span>
      </div>

      <div className="mt-3 pt-3 border-t border-dark-700">
        <div className="text-xs text-gray-400">
          Target: {new Date(prediction.target_date).toLocaleDateString()}
        </div>
      </div>
    </div>
  )
} 