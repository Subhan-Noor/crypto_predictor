'use client'

import React, { useState, useEffect } from 'react'

interface HealthResponse {
  status: string
  timestamp: string
  version: string
  environment: string
}

export const ConnectionTest: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const testConnection = async () => {
      try {
        setLoading(true)
        setError(null)
        
        const url = process.env.NEXT_PUBLIC_API_URL
        if (!url) {
          throw new Error('API URL not configured')
        }

        const response = await fetch(`${url}/health`)
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const healthData: HealthResponse = await response.json()
        setHealth(healthData)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    testConnection()
  }, [])

  if (loading) {
    return <div className="text-gray-400">Testing connection...</div>
  }

  if (error) {
    return <div className="text-red-400">❌ Connection failed: {error}</div>
  }

  if (!health) {
    return <div className="text-yellow-400">⚠️ No health data received</div>
  }

  return (
    <div className="text-green-400">
      ✅ Backend connected successfully
      <div className="text-sm text-gray-400 mt-1">
        Status: {health.status} | Version: {health.version}
      </div>
    </div>
  )
} 