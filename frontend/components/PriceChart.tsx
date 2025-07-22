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

interface PriceChartProps {
  data: PriceData[]
  height?: number
  showVolume?: boolean
  currency: 'BTC' | 'ETH'
}

export const PriceChart: React.FC<PriceChartProps> = ({
  data,
  height = 300,
  showVolume = false,
  currency
}) => {
  const chartData = data.map(item => ({
    date: new Date(item.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    price: item.close,
    volume: item.volume,
  }))

  const currencyColor = currency === 'BTC' ? '#f7931a' : '#627eea'

  if (data.length === 0) {
    return (
      <div className="bg-dark-800 rounded-lg p-6 flex items-center justify-center" style={{ height }}>
        <p className="text-gray-400">No price data available</p>
      </div>
    )
  }

  return (
    <div className="bg-dark-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">
          {currency} Price Chart
        </h3>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: currencyColor }}></div>
          <span className="text-sm text-gray-400">Price</span>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            dataKey="date" 
            stroke="#9CA3AF" 
            fontSize={12}
            tickLine={false}
          />
          <YAxis 
            stroke="#9CA3AF" 
            fontSize={12}
            tickLine={false}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #475569',
              borderRadius: '6px',
              color: '#ffffff'
            }}
            formatter={(value: number) => [`$${value.toLocaleString()}`, 'Price']}
          />
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
      
      {showVolume && (
        <div className="mt-4 pt-4 border-t border-dark-700">
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
              />
              <Area
                type="monotone"
                dataKey="volume"
                stroke="#64748b"
                fill="#64748b20"
                fillOpacity={0.1}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
} 