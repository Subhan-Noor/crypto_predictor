'use client'

import React, { useState, useEffect, useCallback } from 'react'
import Head from 'next/head'
import { apiService } from '../../utils/api'
import { APIHealthStatus } from '../../types'

interface SystemStatus {
  api: 'operational' | 'degraded' | 'down'
  database: 'operational' | 'degraded' | 'down'
  cache: 'operational' | 'degraded' | 'down'
  websocket: 'operational' | 'degraded' | 'down'
  lastUpdate: Date | null
}

export default function StatusPage() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    api: 'down',
    database: 'down', 
    cache: 'down',
    websocket: 'down',
    lastUpdate: null
  })
  const [loading, setLoading] = useState(true)
  const [apiHealth, setApiHealth] = useState<APIHealthStatus | null>(null)

  const checkSystemHealth = useCallback(async () => {
    try {
      setLoading(true)
      const health = await apiService.checkHealth()
      setApiHealth(health)
      
      // Parse health response based on production API format
      const newStatus: SystemStatus = {
        api: health?.status === 'healthy' || health?.status === 'ok' ? 'operational' : 'degraded',
        database: health?.services?.database?.status === 'healthy' ? 'operational' : 'degraded',
        cache: health?.services?.cache?.status === 'available' || 
               health?.services?.cache?.status === 'healthy' || 
               health?.services?.cache?.status === 'connected' ? 'operational' : 'degraded',
        websocket: health?.services?.websocket?.service_status === 'running' ? 'operational' : 'degraded',
        lastUpdate: new Date()
      }
      
      setSystemStatus(newStatus)
    } catch (error) {
      console.error('System health check failed:', error)
      setSystemStatus({
        api: 'down',
        database: 'down',
        cache: 'down', 
        websocket: 'down',
        lastUpdate: new Date()
      })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    checkSystemHealth()
    // Auto-refresh every 30 seconds
    const interval = setInterval(checkSystemHealth, 30000)
    return () => clearInterval(interval)
  }, [checkSystemHealth])

  const getStatusColor = (status: 'operational' | 'degraded' | 'down') => {
    switch (status) {
      case 'operational':
        return 'text-green-400'
      case 'degraded':
        return 'text-yellow-400'
      case 'down':
        return 'text-red-400'
      default:
        return 'text-gray-400'
    }
  }

  const getStatusIcon = (status: 'operational' | 'degraded' | 'down') => {
    switch (status) {
      case 'operational':
        return '🟢'
      case 'degraded':
        return '🟡'
      case 'down':
        return '🔴'
      default:
        return '⚪'
    }
  }

  const getOverallStatus = () => {
    const statuses = [systemStatus.api, systemStatus.database, systemStatus.cache, systemStatus.websocket]
    const operationalCount = statuses.filter(s => s === 'operational').length
    const downCount = statuses.filter(s => s === 'down').length
    
    if (downCount > 0) return 'down'
    if (operationalCount === statuses.length) return 'operational'
    return 'degraded'
  }

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString()
    } catch {
      return timestamp
    }
  }

  return (
    <div className="min-h-screen bg-dark-900 text-white">
      <Head>
        <title>System Status - Crypto Prediction API</title>
        <meta name="description" content="Real-time monitoring of the Crypto Prediction API services." />
        <meta name="keywords" content="crypto, prediction, api, status, monitoring" />
        <meta name="author" content="Crypto Prediction API" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Head>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-4">System Status</h1>
            <p className="text-gray-400">
              Real-time monitoring of the Crypto Prediction API services
            </p>
          </div>

          {/* Overall Status */}
          <div className="bg-dark-800 rounded-lg p-6 mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-semibold mb-2">Overall System Status</h2>
                <p className="text-gray-400">
                  {loading ? 'Checking system health...' : 'Last updated: ' + 
                    (systemStatus.lastUpdate?.toLocaleTimeString() || 'Unknown')}
                </p>
              </div>
              <div className={`text-4xl ${getStatusColor(getOverallStatus())}`}>
                {getStatusIcon(getOverallStatus())}
              </div>
            </div>
          </div>

          {/* Service Status Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {/* API Status */}
            <div className="bg-dark-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">API Service</h3>
                <span className={`text-2xl ${getStatusColor(systemStatus.api)}`}>
                  {getStatusIcon(systemStatus.api)}
                </span>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className={getStatusColor(systemStatus.api)}>
                    {systemStatus.api.charAt(0).toUpperCase() + systemStatus.api.slice(1)}
                  </span>
                </div>
                {apiHealth && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Version:</span>
                      <span>{apiHealth.version}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Environment:</span>
                      <span>{apiHealth.environment}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Last Check:</span>
                      <span>{formatTimestamp(apiHealth.timestamp)}</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Database Status */}
            <div className="bg-dark-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">Database</h3>
                <span className={`text-2xl ${getStatusColor(systemStatus.database)}`}>
                  {getStatusIcon(systemStatus.database)}
                </span>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className={getStatusColor(systemStatus.database)}>
                    {systemStatus.database.charAt(0).toUpperCase() + systemStatus.database.slice(1)}
                  </span>
                </div>
                {apiHealth?.services?.database && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Connection:</span>
                    <span className={getStatusColor(systemStatus.database)}>
                      {apiHealth.services.database.status}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Cache Status */}
            <div className="bg-dark-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">Cache Service</h3>
                <span className={`text-2xl ${getStatusColor(systemStatus.cache)}`}>
                  {getStatusIcon(systemStatus.cache)}
                </span>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className={getStatusColor(systemStatus.cache)}>
                    {systemStatus.cache.charAt(0).toUpperCase() + systemStatus.cache.slice(1)}
                  </span>
                </div>
                {apiHealth?.services?.cache && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Service:</span>
                    <span className={getStatusColor(systemStatus.cache)}>
                      {apiHealth.services.cache.status}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* WebSocket Status */}
            <div className="bg-dark-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">WebSocket Service</h3>
                <span className={`text-2xl ${getStatusColor(systemStatus.websocket)}`}>
                  {getStatusIcon(systemStatus.websocket)}
                </span>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className={getStatusColor(systemStatus.websocket)}>
                    {systemStatus.websocket.charAt(0).toUpperCase() + systemStatus.websocket.slice(1)}
                  </span>
                </div>
                {apiHealth?.services?.websocket && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Service:</span>
                      <span className={getStatusColor(systemStatus.websocket)}>
                        {apiHealth.services.websocket.service_status}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Connections:</span>
                      <span>{apiHealth.services.websocket.connections.total_connections}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Uptime:</span>
                      <span>{apiHealth.services.websocket.connections.uptime}s</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Manual Refresh Button */}
          <div className="text-center">
            <button
              onClick={checkSystemHealth}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50 text-white px-6 py-3 rounded-lg transition-colors font-medium"
            >
              {loading ? 'Checking...' : 'Refresh Status'}
            </button>
          </div>

          {/* Raw API Response (for debugging) */}
          {apiHealth && (
            <div className="mt-8 bg-dark-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">API Response</h3>
              <pre className="bg-dark-900 p-4 rounded text-sm overflow-x-auto">
                {JSON.stringify(apiHealth, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 