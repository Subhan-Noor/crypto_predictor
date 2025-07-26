import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts'
import { PriceData } from '../types'
import { SkeletonChart } from './LoadingSpinner'

interface PriceChartProps {
  data: PriceData[]
  height?: number
  showVolume?: boolean
  currency: 'BTC' | 'ETH'
  isLoading?: boolean
}

export const PriceChart: React.FC<PriceChartProps> = ({
  data,
  height = 300,
  showVolume = false,
  currency,
  isLoading = false
}) => {
  if (isLoading) {
    return <SkeletonChart height={height} />
  }

  // Sort data chronologically (oldest to newest) for proper chart display
  const sortedData = [...data].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
  
  const chartData = sortedData.map(item => ({
    date: new Date(item.date).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric'  // Add year to fix multi-year chart tooltips
    }),
    price: item.close,
    volume: item.volume,
    fullDate: item.date  // Keep original date for tooltip
  }))

  const currencyColor = currency === 'BTC' ? '#f7931a' : '#627eea'

  if (data.length === 0) {
    return (
      <div 
        className="bg-dark-800 rounded-lg p-6 flex items-center justify-center" 
        style={{ height }}
        role="alert"
        aria-label="No chart data available"
      >
        <p className="text-gray-400">No price data available</p>
      </div>
    )
  }

  // Custom tooltip component to show full date
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload
      const fullDate = new Date(dataPoint.fullDate).toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      })
      
      return (
        <div 
          className="bg-dark-700 border border-dark-600 rounded-lg p-3 shadow-lg"
          role="tooltip"
          aria-label={`Price data for ${fullDate}`}
        >
          <p className="text-white font-medium">{fullDate}</p>
          <p className="text-gray-300">
            Price: <span className="text-white font-semibold">${payload[0].value.toLocaleString()}</span>
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <section 
      className="bg-dark-800 rounded-lg p-6"
      role="region"
      aria-labelledby={`${currency}-chart-title`}
    >
      <header className="flex items-center justify-between mb-4">
        <h3 
          id={`${currency}-chart-title`}
          className="text-lg font-semibold text-white"
        >
          {currency} Price Chart
        </h3>
        <div className="flex items-center space-x-2">
          <div 
            className="w-3 h-3 rounded-full" 
            style={{ backgroundColor: currencyColor }}
            role="img"
            aria-label="Price line indicator"
          />
          <span className="text-sm text-gray-400">Price</span>
        </div>
      </header>
      
      <div 
        role="img" 
        aria-label={`${currency} price chart showing ${chartData.length} data points`}
        tabIndex={0}
      >
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="date" 
              stroke="#9CA3AF" 
              fontSize={12}
              tickLine={false}
              aria-label="Date axis"
            />
            <YAxis 
              stroke="#9CA3AF" 
              fontSize={12}
              tickLine={false}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
              aria-label="Price axis"
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="price"
              stroke={currencyColor}
              strokeWidth={2}
              fill={`${currencyColor}20`}
              fillOpacity={0.1}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      
      {showVolume && (
        <div className="mt-4 pt-4 border-t border-dark-700">
          <h4 className="sr-only">Volume Chart</h4>
          <div 
            role="img" 
            aria-label={`${currency} volume chart`}
            tabIndex={0}
          >
            <ResponsiveContainer width="100%" height={100}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9CA3AF" fontSize={10} tickLine={false} />
                <YAxis stroke="#9CA3AF" fontSize={10} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: '6px',
                    color: '#ffffff'
                  }}
                  formatter={(value: number) => [value.toLocaleString(), 'Volume']}
                />
                <Area
                  type="monotone"
                  dataKey="volume"
                  stroke="#10b981"
                  strokeWidth={1}
                  fill="#10b98120"
                  fillOpacity={0.1}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </section>
  )
} 