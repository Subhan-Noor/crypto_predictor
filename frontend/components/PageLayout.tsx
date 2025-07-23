import React, { ReactNode } from 'react'

interface PageLayoutProps {
  title: string
  description?: string
  children: ReactNode
  headerActions?: ReactNode
  showBackToTop?: boolean
}

export const PageLayout: React.FC<PageLayoutProps> = ({
  title,
  description,
  children,
  headerActions,
  showBackToTop = true
}) => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900">
      <div className="container mx-auto px-4 py-8">
        {/* Standardized Header */}
        <header className="flex flex-col lg:flex-row lg:justify-between lg:items-center mb-8 space-y-4 lg:space-y-0">
          <div className="space-y-2">
            <h1 className="text-4xl lg:text-5xl font-bold text-white leading-tight">
              {title}
            </h1>
            {description && (
              <p className="text-lg text-gray-400 max-w-2xl">
                {description}
              </p>
            )}
          </div>
          
          {headerActions && (
            <div className="flex flex-wrap items-center gap-3">
              {headerActions}
            </div>
          )}
        </header>

        {/* Main Content */}
        <main className="space-y-8">
          {children}
        </main>

        {/* Back to Top Button */}
        {showBackToTop && (
          <button
            onClick={scrollToTop}
            className="fixed bottom-6 right-6 p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg transition-all duration-300 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-dark-900"
            aria-label="Back to top"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>
          </button>
        )}
      </div>
    </div>
  )
}

// Standardized Button Components
interface ButtonProps {
  children: ReactNode
  onClick?: () => void
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  loading?: boolean
  className?: string
}

export const Button: React.FC<ButtonProps> = ({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  className = ''
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-900'
  
  const variantClasses = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500',
    secondary: 'bg-gray-600 hover:bg-gray-700 text-white focus:ring-gray-500',
    success: 'bg-green-600 hover:bg-green-700 text-white focus:ring-green-500',
    warning: 'bg-yellow-600 hover:bg-yellow-700 text-white focus:ring-yellow-500',
    danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500'
  }
  
  const sizeClasses = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-2.5 text-sm',
    lg: 'px-6 py-3 text-base'
  }
  
  const disabledClasses = 'opacity-50 cursor-not-allowed'
  
  const classes = `
    ${baseClasses}
    ${variantClasses[variant]}
    ${sizeClasses[size]}
    ${disabled || loading ? disabledClasses : ''}
    ${className}
  `.trim()

  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={classes}
    >
      {loading && (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      )}
      {children}
    </button>
  )
}

// Standardized Card Component
interface CardProps {
  children: ReactNode
  className?: string
  padding?: 'sm' | 'md' | 'lg'
  hover?: boolean
}

export const Card: React.FC<CardProps> = ({
  children,
  className = '',
  padding = 'md',
  hover = false
}) => {
  const paddingClasses = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  }
  
  const hoverClasses = hover ? 'hover:bg-dark-700/50 transition-colors duration-200' : ''
  
  const classes = `
    bg-dark-800 rounded-lg border border-dark-700/50
    ${paddingClasses[padding]}
    ${hoverClasses}
    ${className}
  `.trim()

  return (
    <div className={classes}>
      {children}
    </div>
  )
}

// Standardized Section Component
interface SectionProps {
  title?: string
  description?: string
  children: ReactNode
  className?: string
}

export const Section: React.FC<SectionProps> = ({
  title,
  description,
  children,
  className = ''
}) => {
  return (
    <section className={`space-y-6 ${className}`}>
      {(title || description) && (
        <div className="space-y-2">
          {title && (
            <h2 className="text-2xl font-bold text-white">
              {title}
            </h2>
          )}
          {description && (
            <p className="text-gray-400">
              {description}
            </p>
          )}
        </div>
      )}
      {children}
    </section>
  )
}

// Standardized Filter Bar Component
interface FilterOption {
  value: string
  label: string
}

interface FilterBarProps {
  filters: Array<{
    label: string
    value: string
    options: FilterOption[]
    onChange: (value: string) => void
  }>
  actions?: ReactNode
}

export const FilterBar: React.FC<FilterBarProps> = ({ filters, actions }) => {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 p-4 bg-dark-800/50 rounded-lg border border-dark-700/50">
      <div className="flex flex-wrap items-center gap-4">
        {filters.map((filter, index) => (
          <div key={index} className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-300">
              {filter.label}:
            </label>
            <select
              value={filter.value}
              onChange={(e) => filter.onChange(e.target.value)}
              className="px-3 py-2 bg-dark-800 text-white rounded-lg border border-dark-700 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            >
              {filter.options.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
      
      {actions && (
        <div className="flex items-center space-x-3">
          {actions}
        </div>
      )}
    </div>
  )
}

// Status Badge Component
interface StatusBadgeProps {
  status: 'operational' | 'degraded' | 'down' | 'success' | 'warning' | 'error' | 'info'
  children: ReactNode
  size?: 'sm' | 'md'
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  children,
  size = 'md'
}) => {
  const statusClasses = {
    operational: 'bg-green-500/20 text-green-400 border-green-500/30',
    success: 'bg-green-500/20 text-green-400 border-green-500/30',
    degraded: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    warning: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    down: 'bg-red-500/20 text-red-400 border-red-500/30',
    error: 'bg-red-500/20 text-red-400 border-red-500/30',
    info: 'bg-blue-500/20 text-blue-400 border-blue-500/30'
  }
  
  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-1.5 text-sm'
  }
  
  const classes = `
    inline-flex items-center font-medium rounded-full border
    ${statusClasses[status]}
    ${sizeClasses[size]}
  `.trim()

  return (
    <span className={classes}>
      {children}
    </span>
  )
} 