import React from 'react'
import { CurrentPrice, Currency } from '../types'

interface PriceCardProps {
  price: CurrentPrice
  isLoading?: boolean
}

export const PriceCard: React.FC<PriceCardProps> = ({ price, isLoading = false }) => {
  if (isLoading) {
    return (
      <div className="bg-dark-800 rounded-lg p-6 animate-pulse">
        <div className="flex items-center justify-between mb-4">
          <div className="w-16 h-6 bg-dark-700 rounded"></div>
          <div className="w-8 h-8 bg-dark-700 rounded-full"></div>
        </div>
        <div className="w-24 h-8 bg-dark-700 rounded mb-2"></div>
        <div className="w-16 h-4 bg-dark-700 rounded"></div>
      </div>
    )
  }

  const isPositive = price.change_percentage_24h >= 0
  const changeColor = isPositive ? 'text-crypto-green' : 'text-crypto-red'
  const bgColor = price.currency === 'BTC' ? 'crypto-bitcoin' : 'crypto-ethereum'

  return (
    <div className="bg-dark-800 rounded-lg p-6 border border-dark-700 hover:border-dark-600 transition-colors">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">
          {price.currency}
        </h3>
        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm bg-${bgColor}`}>
          {price.currency === 'BTC' ? '₿' : 'Ξ'}
        </div>
      </div>
      
      <div className="mb-2">
        <span className="text-2xl font-bold text-white">
          ${price.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
      </div>
      
      <div className="flex items-center space-x-4 text-sm">
        <span className={`${changeColor} font-medium`}>
          {isPositive ? '+' : ''}
          {price.change_percentage_24h.toFixed(2)}%
        </span>
        <span className="text-gray-400">
          ${Math.abs(price.change_24h).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
      </div>

      {price.volume_24h && (
        <div className="mt-3 pt-3 border-t border-dark-700">
          <span className="text-xs text-gray-400">24h Volume</span>
          <div className="text-sm text-white">
            ${price.volume_24h.toLocaleString('en-US', { maximumFractionDigits: 0 })}
          </div>
        </div>
      )}
    </div>
  )
} 