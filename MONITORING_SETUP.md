# 📊 Monitoring & Logging Setup Guide

This guide covers setting up comprehensive monitoring and logging for your deployed crypto prediction application.

## 🎯 Overview

The application includes built-in monitoring endpoints and logging. This guide will help you set up additional monitoring and alerting for production.

---

## 🏥 1. Built-in Health Monitoring

### Health Endpoints

The application provides several monitoring endpoints:

```bash
# Overall API health
GET /health

# Database and data status  
GET /data_status

# Automation system status
GET /automation/status

# Automation history and metrics
GET /automation/history
```

### Health Check Response Format

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "database": {
    "status": "connected",
    "response_time_ms": 45
  },
  "services": {
    "binance": "operational",
    "supabase": "operational"
  },
  "system": {
    "environment": "production",
    "version": "1.0.0"
  }
}
```

---

## 🔧 2. Platform-Specific Monitoring

### Railway Monitoring

1. **Built-in Metrics**
   - CPU and Memory usage
   - Response times
   - Error rates
   - Request volume

2. **Custom Metrics**
   - Set up custom metrics dashboard
   - Monitor specific endpoints
   - Track deployment history

3. **Alerts**
   - Configure alerts for high error rates
   - Set up notifications for downtime
   - Monitor resource usage thresholds

### Render Monitoring

1. **Service Metrics**
   - Response times
   - Error rates  
   - Memory and CPU usage
   - Build and deployment status

2. **Log Aggregation**
   - Centralized logging
   - Log search and filtering
   - Export logs for analysis

### Vercel Monitoring

1. **Function Metrics**
   - Function invocations
   - Duration and timeout errors
   - Cold start performance

2. **Web Vitals**
   - Core Web Vitals monitoring
   - Real User Monitoring (RUM)
   - Performance insights

---

## 📈 3. External Monitoring Services

### Option A: UptimeRobot (Free)

1. **Setup**
   - Sign up at [UptimeRobot](https://uptimerobot.com)
   - Add HTTP(S) monitors for:
     - Frontend: `https://your-app.vercel.app`
     - Backend Health: `https://your-backend.railway.app/health`
     - API Status: `https://your-backend.railway.app/data_status`

2. **Configuration**
   ```
   Monitor Type: HTTP(S)
   URL: https://your-backend.railway.app/health
   Monitoring Interval: 5 minutes
   Monitor Timeout: 30 seconds
   ```

3. **Alerts**
   - Email notifications for downtime
   - SMS alerts (premium feature)
   - Slack/Discord webhooks

### Option B: Pingdom (Freemium)

1. **Setup**
   - Sign up at [Pingdom](https://pingdom.com)
   - Create uptime checks for key endpoints

2. **Advanced Monitoring**
   - Transaction monitoring for user flows
   - Page speed monitoring
   - Real user monitoring

### Option C: StatusCake (Free Tier)

1. **Features**
   - Website monitoring
   - API monitoring
   - SSL certificate monitoring
   - Domain monitoring

---

## 📊 4. Custom Monitoring Dashboard

### Option A: Simple Status Page

Create a simple status page that aggregates health checks:

```typescript
// pages/status.tsx (Frontend)
import { useEffect, useState } from 'react'
import { apiService } from '../utils/api'

interface SystemStatus {
  frontend: 'healthy' | 'degraded' | 'down'
  backend: 'healthy' | 'degraded' | 'down'
  database: 'healthy' | 'degraded' | 'down'
  automation: 'healthy' | 'degraded' | 'down'
}

export default function StatusPage() {
  const [status, setStatus] = useState<SystemStatus>({
    frontend: 'healthy',
    backend: 'down',
    database: 'down',
    automation: 'down'
  })

  useEffect(() => {
    async function checkStatus() {
      try {
        const health = await apiService.checkHealth()
        // Update status based on response
        setStatus({
          frontend: 'healthy',
          backend: health.status === 'healthy' ? 'healthy' : 'degraded',
          database: health.database?.status === 'connected' ? 'healthy' : 'down',
          automation: health.automation?.status === 'operational' ? 'healthy' : 'degraded'
        })
      } catch (error) {
        setStatus(prev => ({ ...prev, backend: 'down' }))
      }
    }

    checkStatus()
    const interval = setInterval(checkStatus, 30000) // Check every 30 seconds
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="status-dashboard">
      <h1>System Status</h1>
      <div className="status-grid">
        <StatusCard name="Frontend" status={status.frontend} />
        <StatusCard name="Backend API" status={status.backend} />
        <StatusCard name="Database" status={status.database} />
        <StatusCard name="Automation" status={status.automation} />
      </div>
    </div>
  )
}
```

### Option B: Grafana Dashboard (Advanced)

1. **Setup Grafana**
   - Use Grafana Cloud (free tier)
   - Connect to your application metrics

2. **Dashboard Panels**
   - API response times
   - Error rates
   - Prediction accuracy over time
   - Data ingestion success rates

---

## 🚨 5. Alerting Setup

### GitHub Actions Alerts

The existing GitHub Actions workflow already includes error reporting. You can enhance it:

```yaml
# Add to .github/workflows/ci.yml
- name: Send Slack notification on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    text: "Daily automation failed! Check the logs."
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### Webhook Alerts

Set up webhook endpoints for monitoring services:

```python
# Add to backend/app/main.py
@app.post("/webhooks/alert")
async def handle_alert(alert_data: dict):
    """Handle incoming alert webhooks"""
    # Process alert and send notifications
    # Log critical issues
    # Trigger recovery procedures if needed
    pass
```

---

## 📝 6. Logging Configuration

### Backend Logging

The application uses Python's logging module. Enhance it for production:

```python
# backend/app/logger.py (already exists, but can be enhanced)
import logging
import sys
from datetime import datetime

def setup_production_logging():
    """Configure logging for production environment"""
    
    # JSON formatter for better log parsing
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            return json.dumps(log_data)
    
    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger
```

### Frontend Logging

Set up client-side error tracking:

```typescript
// utils/monitoring.ts
export class FrontendMonitoring {
  static logError(error: Error, context?: any) {
    console.error('Frontend Error:', error, context)
    
    // Send to backend logging endpoint
    fetch('/api/logs/error', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        error: error.message,
        stack: error.stack,
        context,
        timestamp: new Date().toISOString(),
        url: window.location.href
      })
    }).catch(console.error)
  }
  
  static logPerformance(metric: string, value: number) {
    // Log performance metrics
    console.log(`Performance: ${metric} = ${value}ms`)
  }
}
```

---

## 🔍 7. Log Analysis

### Log Aggregation

1. **Simple Log Collection**
   - Railway/Render: Built-in log viewing
   - Export logs for analysis
   - Set up log retention policies

2. **Advanced Log Analysis**
   - Use ELK Stack (Elasticsearch, Logstash, Kibana)
   - Ship logs to external services
   - Set up log-based alerts

### Key Metrics to Track

1. **Application Metrics**
   - API response times
   - Error rates by endpoint
   - Prediction accuracy
   - Data ingestion success rates

2. **Business Metrics**
   - User engagement
   - Feature usage
   - Prediction requests
   - Data freshness

3. **System Metrics**
   - CPU and memory usage
   - Database performance
   - External API response times
   - Cache hit rates

---

## 📊 8. Performance Monitoring

### Frontend Performance

1. **Core Web Vitals**
   - Largest Contentful Paint (LCP)
   - First Input Delay (FID)  
   - Cumulative Layout Shift (CLS)

2. **Custom Metrics**
   - API call durations
   - Chart rendering times
   - Page load times

### Backend Performance

1. **API Performance**
   - Response times by endpoint
   - Database query performance
   - ML model prediction times
   - Data processing speeds

2. **Resource Usage**
   - Memory consumption
   - CPU utilization
   - Database connections
   - Cache performance

---

## ✅ 9. Monitoring Checklist

### Basic Setup
- [ ] Health endpoints accessible
- [ ] Uptime monitoring configured
- [ ] Basic alerting set up
- [ ] Log aggregation working

### Advanced Setup
- [ ] Custom dashboard created
- [ ] Performance monitoring active
- [ ] Error tracking implemented
- [ ] Business metrics tracked

### Alerting
- [ ] Downtime alerts configured
- [ ] Error rate alerts set up
- [ ] Performance threshold alerts
- [ ] Automation failure alerts

### Documentation
- [ ] Monitoring runbook created
- [ ] Alert response procedures documented
- [ ] Dashboard access documented
- [ ] Log analysis procedures defined

---

## 🎯 Success Metrics

Your monitoring setup should track:

- **Uptime**: > 99.9%
- **Response Time**: < 2 seconds for API calls
- **Error Rate**: < 1% for critical endpoints  
- **Prediction Accuracy**: Track over time
- **Data Freshness**: Daily updates successful

This comprehensive monitoring setup will give you visibility into your application's health and performance in production. 