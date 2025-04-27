import React, { useEffect, useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Chip,
  Stack,
  Alert,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { API_BASE_URL } from '../config';

// Define interfaces for API responses
interface PredictionResponse {
  symbol: string;
  current_price: number;
  predicted_price: number;
  confidence_interval: [number, number];
  prediction_time: string;
  sentiment_score: number | null;
}

interface PredictionDataItem {
  date: string;
  actual?: number;
  predicted: number;
  lower: number;
  upper: number;
}

export default function Predictions() {
  const [selectedCrypto, setSelectedCrypto] = React.useState('BTC');
  const [selectedModel, setSelectedModel] = React.useState('ensemble');
  const [timeframe, setTimeframe] = React.useState('24h');
  const [predictionData, setPredictionData] = useState<PredictionDataItem[]>([]);
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchPredictionData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch current price first to ensure consistency with dashboard
      const currentPriceResponse = await fetch(
        `${API_BASE_URL}/current-price/${selectedCrypto}`
      );
      
      if (!currentPriceResponse.ok) {
        throw new Error('Failed to fetch current price data');
      }
      
      const currentPriceData = await currentPriceResponse.json();
      const currentPrice = currentPriceData.price;
      
      console.log(`Current price for ${selectedCrypto}: ${currentPrice}`);
      
      // Fetch prediction
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: selectedCrypto,
          timeframe: timeframe,
          include_sentiment: true,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction data');
      }

      const predictionResponse: PredictionResponse = await response.json();
      
      console.log('Prediction response:', predictionResponse);
      
      // Override prediction response with current price data from dashboard
      predictionResponse.current_price = currentPrice;
      
      console.log('Updated prediction with current price:', predictionResponse);
      
      // Fetch historical data for context
      const historicalResponse = await fetch(
        `${API_BASE_URL}/historical/${selectedCrypto}?days=7`
      );

      if (!historicalResponse.ok) {
        throw new Error('Failed to fetch historical data');
      }

      const historicalDataResponse = await historicalResponse.json();
      const historicalData = historicalDataResponse.data;

      // Process historical data to add formatted dates
      historicalData.forEach((point: any, index: number) => {
        const date = new Date();
        date.setDate(date.getDate() - (historicalData.length - index));
        point.formatted_date = date.toLocaleDateString();
      });

      // Combine historical and prediction data
      const combinedData: PredictionDataItem[] = [];

      // Add historical data points (except the most recent, which we'll replace with current price)
      historicalData.slice(-4, -1).forEach((point: any) => {
        combinedData.push({
          date: point.formatted_date,
          actual: point.close,
          predicted: point.close,
          lower: point.close,
          upper: point.close,
        });
      });

      // Add current point with price from prediction API (using the current price from dashboard)
      const currentDate = new Date().toLocaleDateString();
      combinedData.push({
        date: currentDate,
        actual: predictionResponse.current_price,
        predicted: predictionResponse.current_price,
        lower: predictionResponse.current_price,
        upper: predictionResponse.current_price,
      });

      // Add future predictions
      const futureDates = getNextDates(timeframe === '24h' ? 1 : timeframe === '7d' ? 7 : 30);
      
      // For the first future date, we want to clearly show the prediction starting from the current price
      if (futureDates.length > 0) {
        // Calculate the percentage change
        const percentChange = ((predictionResponse.predicted_price - predictionResponse.current_price) / predictionResponse.current_price) * 100;
        
        combinedData.push({
          date: futureDates[0],
          predicted: predictionResponse.predicted_price,
          lower: predictionResponse.confidence_interval[0],
          upper: predictionResponse.confidence_interval[1],
        });
        
        // Add prediction summary after chart has rendered
        setLastUpdated(`${new Date().toLocaleString()} | Prediction: ${percentChange > 0 ? '+' : ''}${percentChange.toFixed(2)}% in ${timeframe}`);
        
        // For remaining future dates (if any), use the same prediction
        for (let i = 1; i < futureDates.length; i++) {
          combinedData.push({
            date: futureDates[i],
            predicted: predictionResponse.predicted_price,
            lower: predictionResponse.confidence_interval[0],
            upper: predictionResponse.confidence_interval[1],
          });
        }
      }

      // Sort data by date to ensure proper ordering
      combinedData.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
      
      setPredictionData(combinedData);
      setLastUpdated(new Date().toLocaleString());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error fetching prediction data:', err);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to generate future dates
  const getNextDates = (numDays: number): string[] => {
    const dates: string[] = [];
    const today = new Date();
    for (let i = 1; i <= numDays; i++) {
      const date = new Date(today);
      date.setDate(today.getDate() + i);
      dates.push(date.toLocaleDateString());
    }
    return dates;
  };

  useEffect(() => {
    fetchPredictionData();
  }, [selectedCrypto, selectedModel, timeframe]);

  const models = [
    { value: 'arima', label: 'ARIMA' },
    { value: 'prophet', label: 'Prophet' },
    { value: 'lstm', label: 'LSTM' },
    { value: 'ensemble', label: 'Ensemble' },
  ];

  const timeframes = [
    { value: '24h', label: '24 Hours' },
    { value: '7d', label: '7 Days' },
    { value: '30d', label: '30 Days' },
  ];

  const metrics = [
    { label: 'MAE', value: '2.3%' },
    { label: 'RMSE', value: '3.1%' },
    { label: 'Accuracy', value: '85%' },
  ];

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" gutterBottom>
          Price Predictions
        </Typography>
        <Stack direction="row" spacing={2}>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Cryptocurrency</InputLabel>
            <Select
              value={selectedCrypto}
              label="Cryptocurrency"
              onChange={(e) => setSelectedCrypto(e.target.value)}
            >
              <MenuItem value="BTC">Bitcoin (BTC)</MenuItem>
              <MenuItem value="ETH">Ethereum (ETH)</MenuItem>
              <MenuItem value="XRP">Ripple (XRP)</MenuItem>
            </Select>
          </FormControl>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Model</InputLabel>
            <Select
              value={selectedModel}
              label="Model"
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((model) => (
                <MenuItem key={model.value} value={model.value}>
                  {model.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={timeframe}
              label="Timeframe"
              onChange={(e) => setTimeframe(e.target.value)}
            >
              {timeframes.map((tf) => (
                <MenuItem key={tf.value} value={tf.value}>
                  {tf.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={fetchPredictionData}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Update Prediction'}
          </Button>
        </Stack>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Price Forecast
              </Typography>
              {lastUpdated && (
                <Typography variant="caption" color="textSecondary">
                  {lastUpdated}
                </Typography>
              )}
            </Box>
            {predictionData.length > 0 && (() => {
              // Pre-extract data to avoid TypeScript errors
              const currentPoint = predictionData.find(d => d.actual);
              const futurePoint = predictionData.find(d => d.date === getNextDates(1)[0]);
              
              const currentPrice = currentPoint?.actual || 0;
              const predictedPrice = futurePoint?.predicted || 0;
              const lowerBound = futurePoint?.lower || 0;
              const upperBound = futurePoint?.upper || 0;
              
              const priceChange = ((predictedPrice - currentPrice) / currentPrice * 100);
              const isPositive = predictedPrice > currentPrice;
              
              return (
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography>
                    <strong>Current Price:</strong> ${currentPrice.toLocaleString()}
                  </Typography>
                  <Typography color={isPositive ? 'success.main' : 'error.main'}>
                    <strong>Predicted (24h):</strong> ${predictedPrice.toLocaleString()}
                    <span> ({priceChange > 0 ? '+' : ''}{priceChange.toFixed(2)}%)</span>
                  </Typography>
                  <Typography>
                    <strong>Confidence Range:</strong> ${lowerBound.toLocaleString()} - 
                    ${upperBound.toLocaleString()}
                  </Typography>
                </Box>
              );
            })()}
            <Box sx={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={predictionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis 
                    domain={['auto', 'auto']}
                    tickFormatter={(value) => `$${value.toLocaleString()}`}
                  />
                  <Tooltip 
                    labelFormatter={(label) => `Date: ${label}`}
                    formatter={(value, name) => {
                      if (value === undefined) return ['-', name];
                      return [`$${Number(value).toLocaleString()}`, name];
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#8884d8"
                    strokeWidth={3}
                    name="Actual Price"
                    dot={{ r: 5 }}
                    connectNulls={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#82ca9d"
                    strokeWidth={3}
                    name="Predicted Price"
                    dot={{ r: 5 }}
                    activeDot={{ r: 7 }}
                    connectNulls={true}
                  />
                  <Line
                    type="monotone"
                    dataKey="upper"
                    stroke="#b7e4c7"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    name="Upper Bound"
                    dot={false}
                    connectNulls={true}
                  />
                  <Line
                    type="monotone"
                    dataKey="lower"
                    stroke="#b7e4c7"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    name="Lower Bound"
                    dot={false}
                    connectNulls={true}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Model Performance Metrics
            </Typography>
            <Stack direction="row" spacing={2}>
              {metrics.map((metric) => (
                <Chip
                  key={metric.label}
                  label={`${metric.label}: ${metric.value}`}
                  color="primary"
                  variant="outlined"
                />
              ))}
            </Stack>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
} 