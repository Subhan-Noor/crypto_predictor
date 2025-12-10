import React from 'react'

interface LoadingSpinnerProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  color?: 'blue' | 'green' | 'purple' | 'red' | 'yellow'
  className?: string
}

export const EnhancedLoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  size = 'md', 
  color = 'blue',
  className = '' 
}) => {
  const sizeClasses = {
    xs: 'w-3 h-3',
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  }

  const colorClasses = {
    blue: 'border-blue-500',
    green: 'border-green-500',
    purple: 'border-purple-500',
    red: 'border-red-500',
    yellow: 'border-yellow-500'
  }

  return (
    <div className={`flex items-center justify-center ${className}`}>
      <div 
        className={`${sizeClasses[size]} animate-spin rounded-full border-2 border-gray-600 ${colorClasses[color]} border-opacity-25`}
        style={{
          borderTopColor: `var(--${color}-500)`,
          animation: 'spin 1s linear infinite'
        }}
      />
    </div>
  )
}

interface PulseLoaderProps {
  count?: number
  size?: 'sm' | 'md' | 'lg'
  color?: 'blue' | 'green' | 'purple'
  className?: string
}

export const PulseLoader: React.FC<PulseLoaderProps> = ({
  count = 3,
  size = 'md',
  color = 'blue',
  className = ''
}) => {
  const sizeClasses = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4'
  }

  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500'
  }

  return (
    <div className={`flex space-x-1 ${className}`}>
      {Array.from({ length: count }).map((_, index) => (
        <div
          key={index}
          className={`${sizeClasses[size]} ${colorClasses[color]} rounded-full animate-pulse`}
          style={{
            animationDelay: `${index * 0.15}s`,
            animationDuration: '1s'
          }}
        />
      ))}
    </div>
  )
}

interface SkeletonProps {
  className?: string
  width?: string | number
  height?: string | number
  rounded?: boolean
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  width = '100%',
  height = '1rem',
  rounded = false
}) => {
  const widthStyle = typeof width === 'number' ? `${width}px` : width
  const heightStyle = typeof height === 'number' ? `${height}px` : height

  return (
    <div
      className={`bg-dark-700 animate-pulse ${rounded ? 'rounded-full' : 'rounded'} ${className}`}
      style={{ width: widthStyle, height: heightStyle }}
    />
  )
}

interface LoadingCardProps {
  title?: string
  message?: string
  showSpinner?: boolean
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export const EnhancedLoadingCard: React.FC<LoadingCardProps> = ({
  title = 'Loading...',
  message,
  showSpinner = true,
  size = 'md',
  className = ''
}) => {
  const sizeClasses = {
    sm: 'min-h-[150px] p-4',
    md: 'min-h-[200px] p-6',
    lg: 'min-h-[300px] p-8'
  }

  return (
    <div className={`bg-dark-800 rounded-lg border border-dark-700 flex flex-col items-center justify-center ${sizeClasses[size]} ${className}`}>
      {showSpinner && (
        <EnhancedLoadingSpinner size="lg" className="mb-4" />
      )}
      <h3 className="text-lg font-semibold text-white mb-2">
        {title}
      </h3>
      {message && (
        <p className="text-gray-400 text-center max-w-md">
          {message}
        </p>
      )}
    </div>
  )
}

// Specific loading components for common scenarios
export const PriceLoadingCard: React.FC = () => (
  <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
    <div className="flex items-center justify-between mb-4">
      <Skeleton width="4rem" height="1.5rem" />
      <Skeleton width="2rem" height="2rem" rounded />
    </div>
    <Skeleton width="6rem" height="2rem" className="mb-2" />
    <div className="flex items-center space-x-4">
      <Skeleton width="3rem" height="1rem" />
      <Skeleton width="4rem" height="1rem" />
    </div>
    <div className="mt-3 pt-3 border-t border-dark-700">
      <Skeleton width="5rem" height="1rem" />
    </div>
  </div>
)

export const PredictionLoadingCard: React.FC = () => (
  <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
    <div className="flex items-center justify-between mb-4">
      <Skeleton width="8rem" height="1.5rem" />
      <Skeleton width="3rem" height="1.5rem" />
    </div>
    <div className="flex items-center space-x-2 mb-3">
      <Skeleton width="3rem" height="2rem" />
      <Skeleton width="1.5rem" height="1.5rem" rounded />
    </div>
    <div className="flex items-center justify-between">
      <Skeleton width="5rem" height="1rem" />
      <Skeleton width="3rem" height="1.5rem" />
    </div>
    <div className="mt-3 pt-3 border-t border-dark-700">
      <Skeleton width="6rem" height="1rem" />
    </div>
  </div>
)

export const ChartLoadingCard: React.FC = () => (
  <div className="bg-dark-800 rounded-lg p-6 border border-dark-700">
    <div className="flex items-center justify-between mb-4">
      <Skeleton width="6rem" height="1.5rem" />
      <Skeleton width="4rem" height="1rem" />
    </div>
    <div className="space-y-2">
      {Array.from({ length: 6 }).map((_, index) => (
        <div key={index} className="flex items-end space-x-1">
          {Array.from({ length: 10 }).map((_, barIndex) => (
            <Skeleton
              key={barIndex}
              width="1rem"
              height={`${Math.random() * 60 + 20}px`}
            />
          ))}
        </div>
      ))}
    </div>
  </div>
) 