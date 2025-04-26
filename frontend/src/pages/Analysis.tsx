import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  LinearProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';

const mockSentimentData = [
  { date: '2024-02-01', sentiment: 0.8, volume: 1200 },
  { date: '2024-02-02', sentiment: 0.6, volume: 1500 },
  { date: '2024-02-03', sentiment: 0.3, volume: 1800 },
  { date: '2024-02-04', sentiment: 0.7, volume: 1300 },
  { date: '2024-02-05', sentiment: 0.9, volume: 1600 },
];

const mockIndicators = [
  { name: 'RSI', value: 65, interpretation: 'Slightly Overbought' },
  { name: 'MACD', value: 125, interpretation: 'Bullish Trend' },
  { name: 'Moving Average (50)', value: 45200, interpretation: 'Above MA' },
  { name: 'Volume', value: '1.2B', interpretation: 'Above Average' },
];

const mockNewsSentiment = [
  { source: 'Twitter', sentiment: 0.75 },
  { source: 'Reddit', sentiment: 0.65 },
  { source: 'News Articles', sentiment: 0.55 },
  { source: 'Fear & Greed Index', sentiment: 0.70 },
];

export default function Analysis() {
  const [selectedCrypto, setSelectedCrypto] = React.useState('BTC');
  const [timeframe, setTimeframe] = React.useState('7d');

  const getSentimentColor = (sentiment: number) => {
    if (sentiment >= 0.7) return '#4caf50';
    if (sentiment >= 0.5) return '#ff9800';
    return '#f44336';
  };

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" gutterBottom>
          Market Analysis
        </Typography>
        <Box>
          <FormControl sx={{ minWidth: 120, mr: 2 }}>
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
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={timeframe}
              label="Timeframe"
              onChange={(e) => setTimeframe(e.target.value)}
            >
              <MenuItem value="24h">24 Hours</MenuItem>
              <MenuItem value="7d">7 Days</MenuItem>
              <MenuItem value="30d">30 Days</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Sentiment Analysis Over Time
            </Typography>
            <Box sx={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockSentimentData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="sentiment"
                    stroke="#8884d8"
                    strokeWidth={2}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="volume"
                    stroke="#82ca9d"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Current Sentiment
            </Typography>
            {mockNewsSentiment.map((item) => (
              <Box key={item.source} sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  {item.source}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={item.sentiment * 100}
                  sx={{
                    height: 10,
                    borderRadius: 5,
                    backgroundColor: '#e0e0e0',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: getSentimentColor(item.sentiment),
                    },
                  }}
                />
                <Typography variant="body2" color="textSecondary">
                  {(item.sentiment * 100).toFixed(1)}%
                </Typography>
              </Box>
            ))}
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Typography variant="h6" gutterBottom>
            Technical Indicators
          </Typography>
          <Grid container spacing={2}>
            {mockIndicators.map((indicator) => (
              <Grid item xs={12} sm={6} md={3} key={indicator.name}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      {indicator.name}
                    </Typography>
                    <Typography variant="h5" component="div">
                      {indicator.value}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {indicator.interpretation}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
} 