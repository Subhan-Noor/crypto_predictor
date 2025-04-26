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

// Define the type for prediction data items
interface PredictionDataItem {
  date: string;
  actual?: number;
  predicted: number;
  lower: number;
  upper: number;
}

// Generate data based on current date
const generatePredictionData = (): PredictionDataItem[] => {
  const data: PredictionDataItem[] = [];
  const today = new Date();
  
  // Generate last 3 days + 2 future days
  for (let i = -3; i <= 2; i++) {
    const date = new Date(today);
    date.setDate(today.getDate() + i);
    const dateStr = date.toISOString().split('T')[0];
    
    const basePrice = 45000 + Math.random() * 1000;
    const item: PredictionDataItem = { 
      date: dateStr,
      predicted: basePrice + Math.random() * 500, 
      lower: basePrice - Math.random() * 1000, 
      upper: basePrice + Math.random() * 1000
    };
    
    // Only include actual price for past dates
    if (i <= 0) {
      item.actual = basePrice;
    }
    
    data.push(item);
  }
  
  return data;
};

export default function Predictions() {
  const [selectedCrypto, setSelectedCrypto] = React.useState('BTC');
  const [selectedModel, setSelectedModel] = React.useState('ensemble');
  const [timeframe, setTimeframe] = React.useState('7d');
  const [predictionData, setPredictionData] = useState<PredictionDataItem[]>([]);

  useEffect(() => {
    setPredictionData(generatePredictionData());
  }, []);

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

  const handleUpdatePrediction = () => {
    // Generate new data when user clicks update
    setPredictionData(generatePredictionData());
  };

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
          <Button variant="contained" color="primary" onClick={handleUpdatePrediction}>
            Update Prediction
          </Button>
        </Stack>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Price Forecast
            </Typography>
            <Box sx={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={predictionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#8884d8"
                    strokeWidth={2}
                    name="Actual Price"
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#82ca9d"
                    strokeWidth={2}
                    name="Predicted Price"
                  />
                  <Line
                    type="monotone"
                    dataKey="upper"
                    stroke="#82ca9d"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    name="Upper Bound"
                  />
                  <Line
                    type="monotone"
                    dataKey="lower"
                    stroke="#82ca9d"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    name="Lower Bound"
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