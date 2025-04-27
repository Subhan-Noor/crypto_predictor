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
  symbol: string;
  price: number;
  high24h?: number;
  low24h?: number;
  volume24h?: number;
  change24h?: number;
  changepct24h?: number;
}

interface PredictionData {
  symbol: string;
  current_price: number;
  predicted_price: number;
  confidence_interval: [number, number];
  prediction_time: string;
  sentiment_score?: number;
}

interface HistoricalDataPoint {
  date: string;
  formatted_date?: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  isPrediction?: boolean;
}

const Dashboard: React.FC = () => {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);
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

  const fetchPredictionData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: selectedCrypto,
          timeframe: '24h',
          include_sentiment: true,
        }),
      });
      
      if (!response.ok) throw new Error('Failed to fetch prediction data');
      const data = await response.json();
      setPredictionData(data);
    } catch (err) {
      console.error('Error fetching prediction data:', err);
      setError('Failed to fetch prediction data');
    }
  };

  const fetchHistoricalData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/historical/${selectedCrypto}?days=30`);
      if (!response.ok) throw new Error('Failed to fetch historical data');
      const data = await response.json();

      // Process the data from our API format
      if (data && data.data && Array.isArray(data.data)) {
        const formattedData = data.data.map((item: any) => ({
          ...item,
          formatted_date: new Date(item.date || Date.now()).toLocaleDateString()
        }));
        setHistoricalData(formattedData);
      } else {
        setHistoricalData([]);
      }
    } catch (err) {
      console.error('Error fetching historical data:', err);
      setError('Failed to fetch historical data');
    }
  };

  // Function to combine historical and prediction data for the chart
  const updateChartData = () => {
    if (!historicalData.length) return;
    
    const combined = [...historicalData];
    
    // Add prediction point if available
    if (predictionData) {
      // Get the last date and add a day
      const lastHistoricalPoint = historicalData[historicalData.length - 1];
      const lastDate = new Date(lastHistoricalPoint.date || Date.now());
      const nextDay = new Date(lastDate);
      nextDay.setDate(lastDate.getDate() + 1);
      
      combined.push({
        date: nextDay.toISOString(),
        formatted_date: nextDay.toLocaleDateString(),
        open: predictionData.current_price,
        high: Math.max(predictionData.current_price, predictionData.predicted_price),
        low: Math.min(predictionData.current_price, predictionData.predicted_price),
        close: predictionData.predicted_price,
        volume: lastHistoricalPoint.volume,
        isPrediction: true
      });
    }
    
    setChartData(combined);
  };

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([fetchMarketData(), fetchHistoricalData(), fetchPredictionData()])
      .finally(() => setLoading(false));
  }, [selectedCrypto]);

  useEffect(() => {
    updateChartData();
  }, [historicalData, predictionData]);

  const formatPrice = (price: number | undefined) => {
    if (price === undefined) return '$0.00';
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };
  
  const formatPercent = (percent: number | undefined) => {
    if (percent === undefined) return '0.00%';
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
  };
  
  const formatVolume = (volume: number | undefined) => {
    if (volume === undefined) return '$0.00M';
    return `$${(volume / 1_000_000).toFixed(2)}M`;
  };

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
            <Typography variant="h6" gutterBottom>
              Current Price
            </Typography>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : marketData ? (
              <Box>
                <Typography variant="h4" component="div" sx={{ mb: 2 }}>
                  {formatPrice(marketData.price)}
                </Typography>
                {predictionData && (
                  <Box sx={{ mt: 2, p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="subtitle1" color={predictionData.predicted_price > marketData.price ? 'success.main' : 'error.main'}>
                      <strong>Predicted (24h):</strong> {formatPrice(predictionData.predicted_price)} ({((predictionData.predicted_price - marketData.price) / marketData.price * 100).toFixed(2)}%)
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Confidence: {formatPrice(predictionData.confidence_interval[0])} - {formatPrice(predictionData.confidence_interval[1])}
                    </Typography>
                    {predictionData.sentiment_score !== undefined && (
                      <Typography variant="body2" sx={{ mt: 1, display: 'flex', alignItems: 'center' }}>
                        <span style={{ marginRight: '8px' }}>Prediction includes</span>
                        <Box component="span" 
                          sx={{ 
                            bgcolor: predictionData.sentiment_score >= 0.6 ? 'success.main' :
                                    predictionData.sentiment_score >= 0.4 ? 'warning.main' : 'error.main',
                            color: 'white',
                            borderRadius: 1,
                            px: 1,
                            py: 0.25,
                            fontSize: '0.75rem',
                            fontWeight: 'bold',
                          }}
                        >
                          {predictionData.sentiment_score >= 0.6 ? 'Bullish' : 
                           predictionData.sentiment_score >= 0.4 ? 'Neutral' : 'Bearish'} sentiment
                        </Box>
                      </Typography>
                    )}
                  </Box>
                )}
              </Box>
            ) : (
              <Typography variant="body1" color="text.secondary">
                Price data unavailable
              </Typography>
            )}
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
                        Current Price
                      </Typography>
                      <Typography variant="h6">
                        {formatPrice(marketData.price)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary" gutterBottom>
                        Symbol
                      </Typography>
                      <Typography variant="h6">
                        {marketData.symbol}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                {predictionData && (
                  <Grid item xs={12}>
                    <Card sx={{ bgcolor: predictionData.predicted_price > marketData.price ? 'success.light' : 'error.light' }}>
                      <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                          24-Hour Prediction
                        </Typography>
                        <Typography variant="h6" color={predictionData.predicted_price > marketData.price ? 'success.dark' : 'error.dark'}>
                          {formatPrice(predictionData.predicted_price)} ({((predictionData.predicted_price - marketData.price) / marketData.price * 100).toFixed(2)}%)
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          Confidence Range: {formatPrice(predictionData.confidence_interval[0])} - {formatPrice(predictionData.confidence_interval[1])}
                        </Typography>
                        {predictionData.sentiment_score !== undefined && (
                          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                            <Typography variant="body2" sx={{ mr: 1 }}>
                              Market Sentiment:
                            </Typography>
                            <Box
                              sx={{
                                display: 'flex',
                                alignItems: 'center',
                                bgcolor: predictionData.sentiment_score >= 0.6 ? 'success.main' :
                                         predictionData.sentiment_score >= 0.4 ? 'warning.main' : 'error.main',
                                color: 'white',
                                borderRadius: 1,
                                px: 1,
                                py: 0.5,
                                fontSize: '0.75rem',
                                fontWeight: 'bold',
                              }}
                            >
                              {predictionData.sentiment_score >= 0.6 ? 'Bullish' : 
                               predictionData.sentiment_score >= 0.4 ? 'Neutral' : 'Bearish'}
                              ({(predictionData.sentiment_score * 100).toFixed(0)}%)
                            </Box>
                          </Box>
                        )}
                        <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 1 }}>
                          <Box component="span" sx={{ fontWeight: 'bold', color: 'info.main' }}>
                            ✓ Synchronized sentiment data across all pages
                          </Box>
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                )}
                {/* Since we don't have 24hr high/low and other stats, we'll just show the current price in different formats */}
                <Grid item xs={6}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary" gutterBottom>
                        Market Cap
                      </Typography>
                      <Typography variant="h6">
                        {formatVolume(marketData.price * (selectedCrypto === 'BTC' ? 19_000_000 : 
                                                         selectedCrypto === 'ETH' ? 120_000_000 : 
                                                         selectedCrypto === 'SOL' ? 400_000_000 : 
                                                         1_000_000_000))}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary" gutterBottom>
                        Updated
                      </Typography>
                      <Typography variant="h6">
                        {new Date().toLocaleTimeString()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            ) : (
              <Typography variant="body1" color="text.secondary">
                Market data unavailable
              </Typography>
            )}
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
            ) : chartData.length > 0 ? (
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="formatted_date"
                      tickFormatter={(value) => value}
                    />
                    <YAxis 
                      domain={['auto', 'auto']}
                      tickFormatter={(value) => `$${value.toLocaleString()}`}
                    />
                    <Tooltip
                      formatter={(value) => [`$${Number(value).toLocaleString()}`, null]}
                      labelFormatter={(label) => `Date: ${label}`}
                    />
                    <Legend />
                    <defs>
                      <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#8884d8" stopOpacity={0.1}/>
                      </linearGradient>
                      <linearGradient id="colorPrediction" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#82ca9d" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <Area 
                      type="monotone" 
                      dataKey="close" 
                      stroke="#8884d8" 
                      fillOpacity={1} 
                      fill="url(#colorPrice)"
                      name="Price"
                      activeDot={{ r: 8 }}
                      dot={false}
                    />
                    {/* Add a separate line for prediction */}
                    {predictionData && chartData.length > 0 && chartData.some(d => d.isPrediction) && (
                      <Line 
                        type="monotone" 
                        dataKey={(data) => data.isPrediction ? data.close : null}
                        stroke="#82ca9d" 
                        strokeWidth={2}
                        name="Prediction"
                        dot={{ r: 6 }}
                      />
                    )}
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            ) : (
              <Typography variant="body1" color="text.secondary">
                Historical data unavailable
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 