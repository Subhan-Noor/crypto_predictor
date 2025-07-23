'use client'

import React from 'react'
import { ErrorBoundary } from '../../components/ErrorBoundary'

export default function AboutPage() {
  const techStack = [
    {
      category: 'Frontend',
      technologies: [
        { name: 'Next.js 14', description: 'React framework with SSR and SSG' },
        { name: 'TypeScript', description: 'Type-safe JavaScript development' },
        { name: 'TailwindCSS', description: 'Utility-first CSS framework' },
        { name: 'Recharts', description: 'Interactive charts and data visualization' },
        { name: 'React Hooks', description: 'Modern React state management' }
      ]
    },
    {
      category: 'Backend',
      technologies: [
        { name: 'FastAPI', description: 'High-performance Python web framework' },
        { name: 'Python 3.11', description: 'Modern Python with async support' },
        { name: 'Pydantic', description: 'Data validation and serialization' },
        { name: 'SQLAlchemy', description: 'Database ORM and query builder' },
        { name: 'Redis', description: 'In-memory caching and session storage' }
      ]
    },
    {
      category: 'Machine Learning',
      technologies: [
        { name: 'Scikit-learn', description: 'Traditional ML algorithms and metrics' },
        { name: 'TensorFlow/Keras', description: 'Deep learning and neural networks' },
        { name: 'Pandas', description: 'Data manipulation and analysis' },
        { name: 'NumPy', description: 'Numerical computing and array operations' },
        { name: 'Feature Engineering', description: 'Technical indicators and sentiment analysis' }
      ]
    },
    {
      category: 'Database & Hosting',
      technologies: [
        { name: 'Supabase', description: 'PostgreSQL database with real-time features' },
        { name: 'Vercel', description: 'Frontend deployment and CDN' },
        { name: 'Railway', description: 'Backend hosting and deployment' },
        { name: 'GitHub Actions', description: 'CI/CD and automated workflows' },
        { name: 'Docker', description: 'Containerization and deployment' }
      ]
    }
  ]

  const features = [
    {
      title: 'AI-Powered Predictions',
      description: 'Multiple ML models including Random Forest, Logistic Regression, and LSTM networks to predict cryptocurrency price movements with confidence scores.',
      icon: '🤖'
    },
    {
      title: 'Real-Time Data',
      description: 'Live price feeds from Binance API with WebSocket connections for real-time updates and market data visualization.',
      icon: '📊'
    },
    {
      title: 'Sentiment Analysis',
      description: 'Social media sentiment analysis from Twitter and Reddit to incorporate market mood into prediction algorithms.',
      icon: '💭'
    },
    {
      title: 'Advanced Analytics',
      description: 'Comprehensive market analysis including correlation studies, volatility metrics, and risk assessment tools.',
      icon: '📈'
    },
    {
      title: 'Performance Tracking',
      description: 'Historical prediction accuracy tracking with detailed performance metrics and model comparison analysis.',
      icon: '🎯'
    },
    {
      title: 'Modern UI/UX',
      description: 'Responsive design with dark theme, interactive charts, and professional dashboard interface for optimal user experience.',
      icon: '✨'
    }
  ]

  const developmentStages = [
    { stage: 1, title: 'Project Setup & Initialization', status: '✅ Complete' },
    { stage: 2, title: 'Data Acquisition & Storage', status: '✅ Complete' },
    { stage: 3, title: 'Data Preprocessing & ML Model Development', status: '✅ Complete' },
    { stage: 4, title: 'Backend API Development (FastAPI)', status: '✅ Complete' },
    { stage: 5, title: 'Frontend Web Application Development (Next.js)', status: '✅ Complete' },
    { stage: 6, title: 'Integrations & Automation', status: '✅ Complete' },
    { stage: 7, title: 'Testing, Deployment & Monitoring', status: '✅ Complete' },
    { stage: 8, title: 'Documentation & Improvements', status: '✅ Complete' },
    { stage: 9, title: 'Feature Completion & Enhancement', status: '🟡 In Progress' }
  ]

  return (
    <ErrorBoundary>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">About Crypto Prediction</h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            An AI-powered cryptocurrency price prediction platform leveraging machine learning, 
            sentiment analysis, and real-time market data to forecast Bitcoin and Ethereum price movements.
          </p>
        </div>

        {/* Project Overview */}
        <div className="bg-dark-800 rounded-lg p-8 mb-12">
          <h2 className="text-3xl font-bold text-white mb-6">Project Overview</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold text-blue-400 mb-4">Mission</h3>
              <p className="text-gray-300 mb-6">
                To create a comprehensive, AI-driven platform that provides accurate cryptocurrency 
                price predictions by combining traditional financial analysis with modern machine 
                learning techniques and social sentiment data.
              </p>
              
              <h3 className="text-xl font-semibold text-green-400 mb-4">Key Objectives</h3>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">•</span>
                  Predict BTC and ETH price movements with high accuracy
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">•</span>
                  Integrate multiple data sources and ML models
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">•</span>
                  Provide real-time market insights and analytics
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-2">•</span>
                  Track and improve prediction accuracy over time
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold text-purple-400 mb-4">Technical Approach</h3>
              <p className="text-gray-300 mb-6">
                The platform uses a multi-model ensemble approach combining traditional machine 
                learning algorithms with deep learning networks, enhanced by sentiment analysis 
                from social media platforms.
              </p>
              
              <h3 className="text-xl font-semibold text-orange-400 mb-4">Data Sources</h3>
              <ul className="space-y-2 text-gray-300">
                <li className="flex items-start">
                  <span className="text-orange-400 mr-2">•</span>
                  Binance API for real-time price data
                </li>
                <li className="flex items-start">
                  <span className="text-orange-400 mr-2">•</span>
                  Twitter sentiment analysis
                </li>
                <li className="flex items-start">
                  <span className="text-orange-400 mr-2">•</span>
                  Reddit community sentiment
                </li>
                <li className="flex items-start">
                  <span className="text-orange-400 mr-2">•</span>
                  Technical indicators and market metrics
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">Platform Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div key={index} className="bg-dark-800 rounded-lg p-6 hover:bg-dark-700 transition-colors">
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Technology Stack */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">Technology Stack</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {techStack.map((category, categoryIndex) => (
              <div key={categoryIndex} className="bg-dark-800 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-blue-400 mb-4">{category.category}</h3>
                <div className="space-y-4">
                  {category.technologies.map((tech, techIndex) => (
                    <div key={techIndex} className="border-l-2 border-blue-500 pl-4">
                      <h4 className="font-medium text-white">{tech.name}</h4>
                      <p className="text-sm text-gray-400">{tech.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Development Timeline */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">Development Timeline</h2>
          <div className="bg-dark-800 rounded-lg p-6">
            <div className="space-y-4">
              {developmentStages.map((stage, index) => (
                <div key={index} className="flex items-center space-x-4 p-4 rounded-lg bg-dark-700/50">
                  <div className="flex-shrink-0 w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold">
                    {stage.stage}
                  </div>
                  <div className="flex-grow">
                    <h3 className="text-white font-medium">{stage.title}</h3>
                  </div>
                  <div className="flex-shrink-0">
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                      stage.status.includes('Complete') 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {stage.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Architecture Overview */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">System Architecture</h2>
          <div className="bg-dark-800 rounded-lg p-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-white text-xl">🌐</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">Frontend Layer</h3>
                <p className="text-gray-400 text-sm">
                  Next.js application with TypeScript, deployed on Vercel with global CDN for optimal performance.
                </p>
              </div>
              
              <div className="text-center">
                <div className="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-white text-xl">⚡</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">API Layer</h3>
                <p className="text-gray-400 text-sm">
                  FastAPI backend with async/await, Redis caching, and WebSocket support for real-time updates.
                </p>
              </div>
              
              <div className="text-center">
                <div className="w-16 h-16 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-white text-xl">🗄️</span>
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">Data Layer</h3>
                <p className="text-gray-400 text-sm">
                  PostgreSQL database on Supabase with automated ML pipeline and scheduled data ingestion.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Links and Resources */}
        <div className="bg-dark-800 rounded-lg p-8">
          <h2 className="text-3xl font-bold text-white mb-6 text-center">Resources & Links</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="w-12 h-12 bg-gray-600 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-white">📚</span>
              </div>
              <h3 className="font-semibold text-white mb-2">Documentation</h3>
              <p className="text-gray-400 text-sm">
                Comprehensive guides and API documentation
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-12 h-12 bg-gray-600 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-white">🔧</span>
              </div>
              <h3 className="font-semibold text-white mb-2">Setup Guide</h3>
              <p className="text-gray-400 text-sm">
                Local development and deployment instructions
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-12 h-12 bg-gray-600 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-white">🤝</span>
              </div>
              <h3 className="font-semibold text-white mb-2">Contributing</h3>
              <p className="text-gray-400 text-sm">
                Guidelines for contributing to the project
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-12 h-12 bg-gray-600 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-white">📧</span>
              </div>
              <h3 className="font-semibold text-white mb-2">Support</h3>
              <p className="text-gray-400 text-sm">
                Get help and report issues
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 py-8 border-t border-dark-700">
          <p className="text-gray-400">
            Built with ❤️ using modern web technologies and machine learning
          </p>
          <p className="text-gray-500 text-sm mt-2">
            © 2024 Crypto Prediction Platform. Open source project under MIT License.
          </p>
        </div>
      </div>
    </ErrorBoundary>
  )
} 