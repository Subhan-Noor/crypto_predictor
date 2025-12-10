'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

export const Navbar: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const pathname = usePathname()

  const navItems = [
    { href: '/', label: 'Dashboard' },
    { href: '/analytics', label: 'Analytics' },
    { href: '/visualizations', label: 'Visualizations' },
    { href: '/predictions', label: 'Predictions' },
    { href: '/status', label: 'Status' },
    { href: '/about', label: 'About' }
  ]

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen)
  }

  const isActiveLink = (href: string) => {
    return pathname === href
  }

  return (
    <nav className="bg-dark-800 border-b border-dark-700" role="navigation" aria-label="Main navigation">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo/Brand */}
          <Link 
            href="/" 
            className="flex items-center space-x-2 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-lg p-1"
            aria-label="Crypto Prediction - Home"
          >
            <div 
              className="w-8 h-8 bg-gradient-to-r from-crypto-bitcoin to-crypto-ethereum rounded-full flex items-center justify-center"
              role="img"
              aria-label="Crypto Prediction logo"
            >
              <svg 
                className="w-4 h-4 text-white" 
                fill="currentColor" 
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path d="M3 18h6v-2H3v2zM3 6v2h6V6H3zm0 7h9v-2H3v2zm10-2v2h8v-2h-8zm0-5v2h8V6h-8zm0 7v2h8v-2h-8z"/>
              </svg>
            </div>
            <span className="text-xl font-bold text-white">
              Crypto Prediction
            </span>
          </Link>

          {/* Desktop Navigation Links */}
          <div className="hidden md:flex items-center space-x-8" role="menubar">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                role="menuitem"
                className={`transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded px-2 py-1 ${
                  isActiveLink(item.href)
                    ? 'text-white border-b-2 border-blue-500'
                    : 'text-gray-300 hover:text-white'
                }`}
                aria-current={isActiveLink(item.href) ? 'page' : undefined}
              >
                {item.label}
              </Link>
            ))}
          </div>

          {/* Status Indicator and Mobile Menu */}
          <div className="flex items-center space-x-4">
            {/* Live Status Indicator */}
            <div className="hidden sm:flex items-center space-x-2">
              <div 
                className="w-2 h-2 bg-green-500 rounded-full animate-pulse"
                role="status"
                aria-label="System status: Live"
              />
              <span className="text-sm text-gray-400">Live</span>
            </div>
            
            {/* Mobile menu button */}
            <button 
              className="md:hidden text-gray-300 hover:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 rounded p-1"
              onClick={toggleMobileMenu}
              aria-expanded={isMobileMenuOpen}
              aria-controls="mobile-menu"
              aria-label="Toggle mobile menu"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {isMobileMenuOpen && (
          <div 
            id="mobile-menu"
            className="md:hidden py-4 border-t border-dark-700"
            role="menu"
            aria-labelledby="mobile-menu-button"
          >
            <div className="space-y-2">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  role="menuitem"
                  className={`block px-4 py-2 rounded transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    isActiveLink(item.href)
                      ? 'text-white bg-dark-700'
                      : 'text-gray-300 hover:text-white hover:bg-dark-700'
                  }`}
                  onClick={() => setIsMobileMenuOpen(false)}
                  aria-current={isActiveLink(item.href) ? 'page' : undefined}
                >
                  {item.label}
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </nav>
  )
} 