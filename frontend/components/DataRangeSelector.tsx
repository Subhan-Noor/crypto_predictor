import React from 'react'

export interface TimeRange {
  value: string
  label: string
  days: number
}

export const TIME_RANGES: TimeRange[] = [
  { value: '1D', label: '1 Day', days: 1 },
  { value: '7D', label: '7 Days', days: 7 },
  { value: '30D', label: '30 Days', days: 30 },
  { value: '90D', label: '90 Days', days: 90 },
  { value: '1Y', label: '1 Year', days: 365 },
  { value: 'ALL', label: 'All Time', days: 1000 }
]

interface DataRangeSelectorProps {
  selectedRange: TimeRange
  onRangeChange: (range: TimeRange) => void
  className?: string
  variant?: 'buttons' | 'dropdown'
}

export const DataRangeSelector: React.FC<DataRangeSelectorProps> = ({
  selectedRange,
  onRangeChange,
  className = '',
  variant = 'buttons'
}) => {
  if (variant === 'dropdown') {
    return (
      <select
        value={selectedRange.value}
        onChange={(e) => {
          const range = TIME_RANGES.find(r => r.value === e.target.value)
          if (range) onRangeChange(range)
        }}
        className={`px-3 py-2 bg-dark-800 text-white rounded-lg border border-dark-700 focus:border-blue-500 focus:outline-none text-sm font-medium ${className}`}
        style={{ 
          fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          fontSize: '14px',
          fontWeight: '500'
        }}
      >
        {TIME_RANGES.map((range) => (
          <option 
            key={range.value} 
            value={range.value} 
            className="text-sm font-medium"
            style={{ 
              fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            {range.label}
          </option>
        ))}
      </select>
    )
  }

  return (
    <div className={`flex bg-dark-800 rounded-lg p-1 ${className}`}>
      {TIME_RANGES.map((range) => (
        <button
          key={range.value}
          onClick={() => onRangeChange(range)}
          className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
            selectedRange.value === range.value
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-dark-700'
          }`}
        >
          {range.label}
        </button>
      ))}
    </div>
  )
} 