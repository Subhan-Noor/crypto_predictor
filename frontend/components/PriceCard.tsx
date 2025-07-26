import React from 'react'
import { CurrentPrice, Currency } from '../types'
import { SkeletonCard } from './LoadingSpinner'

interface PriceCardProps {
  price: CurrentPrice
  isLoading?: boolean
}

export const PriceCard: React.FC<PriceCardProps> = ({ price, isLoading = false }) => {
  if (isLoading) {
    return <SkeletonCard />
  }

  if (!price || price.price === undefined) {
    return (
      <div 
        className="bg-dark-800 rounded-lg p-6 border border-dark-700 text-center"
        role="alert"
        aria-label="Price data unavailable"
      >
        <div className="text-gray-400">Price data unavailable</div>
      </div>
    )
  }

  const isPositive = typeof price.change_percentage_24h === 'number' ? price.change_percentage_24h >= 0 : true;
  const changeColor = isPositive ? 'text-crypto-green' : 'text-crypto-red';
  const bgColor = price.currency === 'BTC' ? 'crypto-bitcoin' : 'crypto-ethereum';
  
  const changeAriaLabel = `24 hour change: ${isPositive ? 'increased' : 'decreased'} by ${Math.abs(price.change_percentage_24h || 0).toFixed(2)} percent`;

  return (
    <article 
      className="bg-dark-800 rounded-lg p-6 border border-dark-700 hover:border-dark-600 transition-colors focus-within:ring-2 focus-within:ring-blue-500"
      role="region"
      aria-label={`${price.currency} price information`}
    >
      <header className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">
          {price.currency}
        </h3>
        <div 
          className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm bg-${bgColor}`}
          role="img"
          aria-label={`${price.currency} logo`}
        >
          {price.currency === 'BTC' ? '\u20bf' : '\u039e'}
        </div>
      </header>
      
      <div className="mb-2">
        <span 
          className="text-2xl font-bold text-white"
          aria-label={`Current price: $${price.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
        >
          ${price.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
      </div>
      
      <div className="flex items-center space-x-4 text-sm">
        <span 
          className={`${changeColor} font-medium`}
          aria-label={changeAriaLabel}
        >
          {typeof price.change_percentage_24h === 'number' ? (isPositive ? '+' : '') + price.change_percentage_24h.toFixed(2) + '%' : 'N/A'}
        </span>
        <span 
          className="text-gray-400"
          aria-label={`24 hour price change: $${Math.abs(price.change_24h || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
        >
          {typeof price.change_24h === 'number' ? `$${Math.abs(price.change_24h).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : 'N/A'}
        </span>
      </div>
    </article>
  );
} 