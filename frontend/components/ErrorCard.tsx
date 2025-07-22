import React from 'react'

interface ErrorCardProps {
  title?: string
  message: string
  type?: 'error' | 'warning' | 'info'
  onRetry?: () => void
  onDismiss?: () => void
  className?: string
}

export const ErrorCard: React.FC<ErrorCardProps> = ({
  title = 'Error',
  message,
  type = 'error',
  onRetry,
  onDismiss,
  className = ''
}) => {
  const typeStyles = {
    error: {
      bg: 'bg-red-500/10 border-red-500/20',
      icon: 'text-red-400',
      title: 'text-red-400',
      text: 'text-red-300'
    },
    warning: {
      bg: 'bg-yellow-500/10 border-yellow-500/20',
      icon: 'text-yellow-400',
      title: 'text-yellow-400',
      text: 'text-yellow-300'
    },
    info: {
      bg: 'bg-blue-500/10 border-blue-500/20',
      icon: 'text-blue-400',
      title: 'text-blue-400',
      text: 'text-blue-300'
    }
  }

  const styles = typeStyles[type]

  const getIcon = () => {
    switch (type) {
      case 'error':
        return (
          <svg className={`w-5 h-5 ${styles.icon}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        )
      case 'warning':
        return (
          <svg className={`w-5 h-5 ${styles.icon}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        )
      case 'info':
        return (
          <svg className={`w-5 h-5 ${styles.icon}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
    }
  }

  return (
    <div className={`rounded-lg border p-4 ${styles.bg} ${className}`}>
      <div className="flex items-start">
        <div className="flex-shrink-0">
          {getIcon()}
        </div>
        <div className="ml-3 flex-1">
          <h3 className={`text-sm font-medium ${styles.title}`}>
            {title}
          </h3>
          <p className={`mt-1 text-sm ${styles.text}`}>
            {message}
          </p>
          {(onRetry || onDismiss) && (
            <div className="mt-3 flex space-x-2">
              {onRetry && (
                <button
                  onClick={onRetry}
                  className={`text-sm font-medium ${styles.title} hover:opacity-80 transition-opacity`}
                >
                  Try Again
                </button>
              )}
              {onDismiss && (
                <button
                  onClick={onDismiss}
                  className="text-sm font-medium text-gray-400 hover:text-gray-300 transition-colors"
                >
                  Dismiss
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Specific error components for common scenarios
export const APIErrorCard: React.FC<{ onRetry?: () => void }> = ({ onRetry }) => (
  <ErrorCard
    title="Connection Error"
    message="Unable to connect to our servers. Please check your internet connection and try again."
    type="error"
    onRetry={onRetry}
  />
)

export const DataErrorCard: React.FC<{ dataType: string; onRetry?: () => void }> = ({ 
  dataType, 
  onRetry 
}) => (
  <ErrorCard
    title="Data Loading Error"
    message={`Failed to load ${dataType}. This might be a temporary issue with our data sources.`}
    type="warning"
    onRetry={onRetry}
  />
)

export const PredictionErrorCard: React.FC<{ currency: string; onRetry?: () => void }> = ({ 
  currency, 
  onRetry 
}) => (
  <ErrorCard
    title="Prediction Unavailable"
    message={`Unable to generate ${currency} predictions at the moment. Our AI models may be updating.`}
    type="info"
    onRetry={onRetry}
  />
) 