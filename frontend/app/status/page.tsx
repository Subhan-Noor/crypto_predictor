'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { apiService } from '../../utils/api'

interface SystemStatus {
  api: 'operational' | 'degraded' | 'down'
  database: 'operational' | 'degraded' | 'down'
  cache: 'operational' | 'degraded' | 'down'
  websocket: 'operational' | 'degraded' | 'down'
  lastUpdate: Date | null
}

interface APIHealthResponse {
  status?: string
  services?: {
    database?: {
      status?: string
    }
    cache?: {
      status?: string
    }
    websocket?: {
      service_status?: string
    }
  }
  version?: string
  environment?: string
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
  const [apiHealth, setApiHealth] = useState<APIHealthResponse | null>(null)

  const checkSystemHealth = useCallback(async () => {
    try {
      setLoading(true)
      const health = await apiService.checkHealth() as APIHealthResponse
      setApiHealth(health)
      
      // Parse health response to determine individual service status
      const newStatus: SystemStatus = {
        api: health?.status === 'ok' || health?.status === 'healthy' ? 'operational' : 'degraded',
        database: health?.services?.database?.status === 'operational' || health?.services?.database?.status === 'healthy' ? 'operational' : 'degraded',
        cache: health?.services?.cache?.status === 'operational' || health?.services?.cache?.status === 'healthy' ? 'operational' : 'degraded',
        websocket: health?.services?.websocket?.service_status === 'operational' || health?.services?.websocket?.service_status === 'healthy' ? 'operational' : 'degraded',
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
      case 'operational': return 'text-green-400'
      case 'degraded': return 'text-yellow-400'
      case 'down': return 'text-red-400'
    }
  }

  const getStatusIcon = (status: 'operational' | 'degraded' | 'down') => {
    switch (status) {
      case 'operational': return '✅'
      case 'degraded': return '⚠️'
      case 'down': return '❌'
    }
  }

  const getOverallStatus = () => {
    const statuses = [systemStatus.api, systemStatus.database, systemStatus.cache, systemStatus.websocket]
    if (statuses.every(s => s === 'operational')) return 'operational'
    if (statuses.some(s => s === 'down')) return 'down'
    return 'degraded'
  }

  const overallStatus = getOverallStatus()

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-white mb-4">System Status</h1>
        <div className={`text-2xl font-semibold ${getStatusColor(overallStatus)}`}>
          {getStatusIcon(overallStatus)} {overallStatus.charAt(0).toUpperCase() + overallStatus.slice(1)}
        </div>
        {systemStatus.lastUpdate && (
          <p className="text-gray-400 mt-2">
            Last updated: {systemStatus.lastUpdate.toLocaleString()}
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* API Status */}
        <div className="bg-dark-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">API</h3>
            <span className="text-2xl">{getStatusIcon(systemStatus.api)}</span>
          </div>
          <p className={`text-sm ${getStatusColor(systemStatus.api)}`}>
            {systemStatus.api.charAt(0).toUpperCase() + systemStatus.api.slice(1)}
          </p>
          {apiHealth?.version && (
            <p className="text-xs text-gray-400 mt-2">Version: {apiHealth.version}</p>
          )}
        </div>

        {/* Database Status */}
        <div className="bg-dark-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Database</h3>
            <span className="text-2xl">{getStatusIcon(systemStatus.database)}</span>
          </div>
          <p className={`text-sm ${getStatusColor(systemStatus.database)}`}>
            {systemStatus.database.charAt(0).toUpperCase() + systemStatus.database.slice(1)}
          </p>
          <p className="text-xs text-gray-400 mt-2">Supabase PostgreSQL</p>
        </div>

        {/* Cache Status */}
        <div className="bg-dark-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Cache</h3>
            <span className="text-2xl">{getStatusIcon(systemStatus.cache)}</span>
          </div>
          <p className={`text-sm ${getStatusColor(systemStatus.cache)}`}>
            {systemStatus.cache.charAt(0).toUpperCase() + systemStatus.cache.slice(1)}
          </p>
          <p className="text-xs text-gray-400 mt-2">Redis Cache</p>
        </div>

        {/* WebSocket Status */}
        <div className="bg-dark-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">WebSocket</h3>
            <span className="text-2xl">{getStatusIcon(systemStatus.websocket)}</span>
          </div>
          <p className={`text-sm ${getStatusColor(systemStatus.websocket)}`}>
            {systemStatus.websocket.charAt(0).toUpperCase() + systemStatus.websocket.slice(1)}
          </p>
          <p className="text-xs text-gray-400 mt-2">Real-time Updates</p>
        </div>
      </div>

      {/* Detailed System Information */}
      {!loading && apiHealth && (
        <div className="bg-dark-800 rounded-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-white mb-4">System Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h3 className="text-white font-medium mb-2">API Details</h3>
              <div className="space-y-1 text-gray-400">
                <div className="flex justify-between">
                  <span>Status:</span>
                  <span className={getStatusColor(systemStatus.api)}>{apiHealth.status || 'Unknown'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Version:</span>
                  <span>{apiHealth.version || 'Unknown'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Environment:</span>
                  <span>{apiHealth.environment || 'Production'}</span>
                </div>
              </div>
            </div>
            <div>
              <h3 className="text-white font-medium mb-2">Services</h3>
              <div className="space-y-1 text-gray-400">
                {apiHealth.services && Object.entries(apiHealth.services).map(([service, info]: [string, any]) => (
                  <div key={service} className="flex justify-between">
                    <span>{service}:</span>
                    <span className={getStatusColor(info?.status === 'operational' || info?.status === 'healthy' ? 'operational' : 'degraded')}>
                      {info?.status || 'unknown'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Performance Metrics */}
      <div className="bg-dark-800 rounded-lg p-6">
        <h2 className="text-2xl font-semibold text-white mb-4">Recent Activity</h2>
        <div className="space-y-3">
          <div className="flex items-center space-x-3">
            <span className="w-2 h-2 bg-green-500 rounded-full"></span>
            <span className="text-white">System monitoring active</span>
          </div>
          <div className="flex items-center space-x-3">
            <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
            <span className="text-white">Auto-refresh enabled (30s interval)</span>
          </div>
          <div className="flex items-center space-x-3">
            <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
            <span className="text-white">Crypto prediction services running</span>
          </div>
        </div>
      </div>

      {/* Refresh Button */}
      <div className="mt-8 text-center">
        <button
          onClick={checkSystemHealth}
          disabled={loading}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
        >
          {loading ? 'Checking...' : 'Refresh Status'}
        </button>
      </div>
    </div>
  )
} 