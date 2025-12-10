import React from 'react'
import { EnhancedDashboard } from '../components/EnhancedDashboard'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export default function Home() {
  return (
    <div>
      <EnhancedDashboard />
    </div>
  )
} 