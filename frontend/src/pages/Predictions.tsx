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
      // Fetch current prediction
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
      
      // Fetch historical data for context
      const historicalResponse = await fetch(
        `${API_BASE_URL}/historical/${selectedCrypto}?days=7`
      );

      if (!historicalResponse.ok) {
        throw new Error('Failed to fetch historical data');
      }

      const historicalData = await historicalResponse.json();

      // Combine historical and prediction data
      const combinedData: PredictionDataItem[] = [];

      // Add historical data points
      historicalData.slice(-3).forEach((point: any) => {
        combinedData.push({
          date: point.formatted_date,
          actual: point.close,
          predicted: point.close,
          lower: point.close,
          upper: point.close,
        });
      });

      // Add current point
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
      futureDates.forEach((date, index) => {
        combinedData.push({
          date: date,
          predicted: predictionResponse.predicted_price,
          lower: predictionResponse.confidence_interval[0],
          upper: predictionResponse.confidence_interval[1],
        });
      });

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
                  Last updated: {lastUpdated}
                </Typography>
              )}
            </Box>
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
                    formatter={(value) => [`$${Number(value).toLocaleString()}`, null]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#8884d8"
                    strokeWidth={2}
                    name="Actual Price"
                    dot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#82ca9d"
                    strokeWidth={2}
                    name="Predicted Price"
                    strokeDasharray="5 5"
                    dot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="upper"
                    stroke="#b7e4c7"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    name="Upper Bound"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="lower"
                    stroke="#b7e4c7"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    name="Lower Bound"
                    dot={false}
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