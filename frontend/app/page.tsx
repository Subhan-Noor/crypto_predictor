import React from 'react'
import { Dashboard } from '../components/Dashboard'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export default function Home() {
  return (
    <div>
      <Dashboard />
    </div>
  )
} 