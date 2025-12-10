import React from 'react'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  variant?: 'default' | 'button' | 'overlay'
  message?: string
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  size = 'md', 
  variant = 'default',
  message 
}) => {
  const getSizeClasses = () => {
    switch (size) {
      case 'sm': return 'w-4 h-4'
      case 'md': return 'w-6 h-6'
      case 'lg': return 'w-8 h-8'
      default: return 'w-6 h-6'
    }
  }

  const getVariantClasses = () => {
    switch (variant) {
      case 'button': return 'text-white'
      case 'overlay': return 'text-blue-500'
      default: return 'text-blue-500'
    }
  }

  const SpinnerSVG = () => (
    <svg 
      className={`animate-spin ${getSizeClasses()} ${getVariantClasses()}`} 
      fill="none" 
      viewBox="0 0 24 24"
      role="img"
      aria-label="Loading"
    >
      <circle 
        className="opacity-25" 
        cx="12" 
        cy="12" 
        r="10" 
        stroke="currentColor" 
        strokeWidth="4"
      />
      <path 
        className="opacity-75" 
        fill="currentColor" 
        d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  )

  if (variant === 'overlay') {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-dark-800 rounded-lg p-6 flex flex-col items-center space-y-4">
          <SpinnerSVG />
          {message && (
            <p className="text-white text-center">{message}</p>
          )}
        </div>
      </div>
    )
  }

  if (variant === 'button') {
    return (
      <div className="flex items-center justify-center space-x-2">
        <SpinnerSVG />
        {message && <span className="text-white">{message}</span>}
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center justify-center space-y-2 py-8">
      <SpinnerSVG />
      {message && (
        <p className="text-gray-400 text-center">{message}</p>
      )}
    </div>
  )
}

// Skeleton loading component for cards
export const SkeletonCard: React.FC = () => (
  <div className="bg-dark-800 rounded-lg p-6 animate-pulse" role="status" aria-label="Loading content">
    <div className="flex items-center justify-between mb-4">
      <div className="w-16 h-6 bg-dark-700 rounded"></div>
      <div className="w-8 h-8 bg-dark-700 rounded-full"></div>
    </div>
    <div className="w-24 h-8 bg-dark-700 rounded mb-2"></div>
    <div className="w-16 h-4 bg-dark-700 rounded"></div>
  </div>
)

// Skeleton loading component for charts
export const SkeletonChart: React.FC<{ height?: number }> = ({ height = 300 }) => (
  <div 
    className="bg-dark-800 rounded-lg p-6 animate-pulse" 
    style={{ height }}
    role="status" 
    aria-label="Loading chart"
  >
    <div className="flex items-center justify-between mb-4">
      <div className="w-32 h-6 bg-dark-700 rounded"></div>
      <div className="w-16 h-4 bg-dark-700 rounded"></div>
    </div>
    <div className="w-full h-full bg-dark-700 rounded"></div>
  </div>
) 