# Crypto Prediction Frontend

Next.js frontend application for the Crypto Price Prediction Platform.

## Features

- **Real-time Dashboard**: Live price data and predictions
- **Analytics**: Advanced charts and market insights
- **Predictions**: Historical prediction tracking and accuracy
- **Status Monitoring**: System health and performance metrics
- **Responsive Design**: Mobile-friendly interface

## Tech Stack

- **Framework**: Next.js 14.2.3
- **Language**: TypeScript
- **Styling**: TailwindCSS
- **Charts**: Recharts
- **Icons**: FontAwesome, Lucide React
- **HTTP Client**: Axios

## Quick Start

### Prerequisites
- Node.js 18.17.0+
- npm 8.0.0+

### Installation
```bash
# Install dependencies
npm install

# Set environment variables
cp .env.example .env.local
# Edit .env.local with your API URL

# Run development server
npm run dev
```

### Environment Variables
```bash
# Required
NEXT_PUBLIC_API_URL=https://your-backend-url.com

# Optional
NEXT_PUBLIC_APP_NAME=Crypto Prediction Platform
NEXT_PUBLIC_APP_VERSION=1.0.0
```

## Development

### Available Scripts
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
```

### Project Structure
```
frontend/
├── app/                    # Next.js app directory
│   ├── analytics/         # Analytics page
│   ├── predictions/       # Predictions page
│   ├── status/           # Status page
│   ├── about/            # About page
│   ├── layout.tsx        # Root layout
│   └── page.tsx          # Home page
├── components/            # React components
├── utils/                # Utility functions
├── types/                # TypeScript types
├── styles/               # Global styles
└── public/               # Static assets
```

### Component Guidelines
- Use TypeScript for all components
- Follow functional component pattern with hooks
- Add proper error boundaries
- Include loading states
- Use semantic HTML and ARIA labels

## Deployment

### Vercel (Recommended)
1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### Other Platforms
```bash
# Build the application
npm run build

# Start production server
npm run start
```

## Performance

### Optimizations
- **Code Splitting**: Automatic with Next.js
- **Image Optimization**: Next.js Image component
- **Caching**: Static generation where possible
- **Bundle Analysis**: Use `@next/bundle-analyzer`

### Monitoring
- **Core Web Vitals**: Track performance metrics
- **Error Tracking**: Monitor for runtime errors
- **Analytics**: User behavior tracking

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details.
