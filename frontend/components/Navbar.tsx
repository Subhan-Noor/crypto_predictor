import React from 'react'
import Link from 'next/link'

export const Navbar: React.FC = () => {
  return (
    <nav className="bg-dark-800 border-b border-dark-700">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo/Brand */}
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-crypto-bitcoin to-crypto-ethereum rounded-full flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M3 18h6v-2H3v2zM3 6v2h6V6H3zm0 7h9v-2H3v2zm10-2v2h8v-2h-8zm0-5v2h8V6h-8zm0 7v2h8v-2h-8z"/>
              </svg>
            </div>
            <span className="text-xl font-bold text-white">
              Crypto Prediction
            </span>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-8">
            <Link 
              href="/" 
              className="text-gray-300 hover:text-white transition-colors duration-200"
            >
              Dashboard
            </Link>
            <Link 
              href="/analytics" 
              className="text-gray-300 hover:text-white transition-colors duration-200"
            >
              Analytics
            </Link>
            <Link 
              href="/predictions" 
              className="text-gray-300 hover:text-white transition-colors duration-200"
            >
              Predictions
            </Link>
            <Link 
              href="/status" 
              className="text-gray-300 hover:text-white transition-colors duration-200"
            >
              Status
            </Link>
            <Link 
              href="/about" 
              className="text-gray-300 hover:text-white transition-colors duration-200"
            >
              About
            </Link>
          </div>

          {/* Status Indicator */}
          <div className="flex items-center space-x-4">
            <div className="hidden sm:flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-400">Live</span>
            </div>
            
            {/* Mobile menu button */}
            <button className="md:hidden text-gray-300 hover:text-white">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  )
} 