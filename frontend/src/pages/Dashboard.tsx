import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
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
  Area,
  AreaChart,
} from 'recharts';
import { RealTimePrice } from '../components/RealTimePrice';
import { API_BASE_URL } from '../config';

const SUPPORTED_CRYPTOS = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT'];

interface MarketData {
  price: number;
  high24h: number;
  low24h: number;
  volume24h: number;
  change24h: number;
  changepct24h: number;
}

interface HistoricalDataPoint {
  date: string;
  formatted_date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const Dashboard: React.FC = () => {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMarketData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/current-price/${selectedCrypto}`);
      if (!response.ok) throw new Error('Failed to fetch market data');
      const data = await response.json();
      setMarketData(data);
    } catch (err) {
      console.error('Error fetching market data:', err);
      setError('Failed to fetch market data');
    }
  };

  const fetchHistoricalData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/historical/${selectedCrypto}?days=30`);
      if (!response.ok) throw new Error('Failed to fetch historical data');
      const data = await response.json();
      setHistoricalData(data);
    } catch (err) {
      console.error('Error fetching historical data:', err);
      setError('Failed to fetch historical data');
    }
  };

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([fetchMarketData(), fetchHistoricalData()])
      .finally(() => setLoading(false));
  }, [selectedCrypto]);

  const formatPrice = (price: number) => `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  const formatPercent = (percent: number) => `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
  const formatVolume = (volume: number) => `$${(volume / 1_000_000).toFixed(2)}M`;

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4">
          Cryptocurrency Dashboard
        </Typography>
        <FormControl variant="outlined" sx={{ minWidth: 120 }}>
          <InputLabel>Cryptocurrency</InputLabel>
          <Select
            value={selectedCrypto}
            onChange={(e) => setSelectedCrypto(e.target.value as string)}
            label="Cryptocurrency"
          >
            {SUPPORTED_CRYPTOS.map((crypto) => (
              <MenuItem key={crypto} value={crypto}>
                {crypto}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <RealTimePrice symbol={selectedCrypto} />
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Market Overview
            </Typography>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : marketData ? (
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary" gutterBottom>
                        24h High
                      </Typography>
                      <Typography variant="h6">
                        {formatPrice(marketData.high24h)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary" gutterBottom>
                        24h Low
                      </Typography>
                      <Typography variant="h6">
                        {formatPrice(marketData.low24h)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary" gutterBottom>
                        24h Volume
                      </Typography>
                      <Typography variant="h6">
                        {formatVolume(marketData.volume24h)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary" gutterBottom>
                        24h Change
                      </Typography>
                      <Typography 
                        variant="h6" 
                        color={marketData.changepct24h >= 0 ? 'success.main' : 'error.main'}
                      >
                        {formatPercent(marketData.changepct24h)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            ) : null}
          </Paper>
        </Grid>
        
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Historical Performance
            </Typography>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : historicalData.length > 0 ? (
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={historicalData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="formatted_date" />
                    <YAxis 
                      domain={['auto', 'auto']}
                      tickFormatter={(value) => `$${value.toLocaleString()}`}
                    />
                    <Tooltip
                      formatter={(value: any) => [`$${Number(value).toLocaleString()}`, 'Price']}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="close"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.3}
                      name="Price"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            ) : null}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 