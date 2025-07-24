'use client'

import React, { useState, useEffect } from 'react'
import { apiService } from '../utils/api'

export const ConnectionTest: React.FC = () => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [error, setError] = useState<string>('')
  const [apiUrl, setApiUrl] = useState<string>('')

  useEffect(() => {
    const testConnection = async () => {
      try {
        setStatus('loading')
        setError('')
        
        // Log the API URL being used
        const url = process.env.NEXT_PUBLIC_API_URL || 'https://cryptopredictor-production.up.railway.app'
        setApiUrl(url)
        console.log('🔗 Testing connection to:', url)
        
        // Test the health endpoint
        const health = await apiService.checkHealth()
        console.log('✅ Backend health check successful:', health)
        setStatus('success')
        
      } catch (err: any) {
        console.error('❌ Connection test failed:', err)
        setError(err.message || 'Unknown error')
        setStatus('error')
      }
    }

    testConnection()
  }, [])

  if (status === 'loading') {
    return (
      <div className="bg-blue-500/20 border border-blue-500 rounded-lg p-4 mb-4">
        <p className="text-blue-400">🔄 Testing backend connection...</p>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 mb-4">
        <h3 className="text-red-400 font-semibold mb-2">❌ Backend Connection Failed</h3>
        <p className="text-red-400 text-sm mb-2">API URL: {apiUrl}</p>
        <p className="text-red-400 text-sm">Error: {error}</p>
        <p className="text-red-400 text-xs mt-2">
          This usually means the Railway backend is down or deploying. Check the Railway dashboard.
        </p>
      </div>
    )
  }

  return (
    <div className="bg-green-500/20 border border-green-500 rounded-lg p-4 mb-4">
      <h3 className="text-green-400 font-semibold mb-2">✅ Backend Connected Successfully</h3>
      <p className="text-green-400 text-sm">API URL: {apiUrl}</p>
    </div>
  )
} 