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
  Card,
  CardContent,
  LinearProgress,
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
} from 'recharts';
import { getSentimentColor } from '../utils/sentiment';
import { API_BASE_URL } from '../config';

// Define types for sentiment data
interface SentimentDataItem {
  date: string;
  formatted_date: string;
  sentiment: number;
  volume: number;
  positive_mentions: number;
  negative_mentions: number;
  neutral_mentions: number;
}

interface NewsSentiment {
  source: string;
  sentiment: number;
}

export default function Analysis() {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [timeframe, setTimeframe] = useState('7d');
  const [sentimentData, setSentimentData] = useState<SentimentDataItem[]>([]);
  const [newsSentiment, setNewsSentiment] = useState<NewsSentiment[]>([]);
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Get number of days from timeframe
  const getDays = (tf: string): number => {
    switch (tf) {
      case '24h': return 1;
      case '7d': return 7;
      case '30d': return 30;
      default: return 7;
    }
  };

  // Fetch sentiment data from API
  const fetchSentimentData = async () => {
    setLoading(true);
    setError(null);
    try {
      const days = getDays(timeframe);
      const response = await fetch(`${API_BASE_URL}/sentiment/${selectedCrypto}?days=${days}`);
      if (!response.ok) {
        throw new Error('Failed to fetch sentiment data');
      }
      const data = await response.json();
      setSentimentData(data);

      // Calculate news sentiment from the most recent data point
      const latestData = data[data.length - 1];
      const totalMentions = latestData.positive_mentions + latestData.negative_mentions + latestData.neutral_mentions;
      
      setNewsSentiment([
        {
          source: 'Social Media',
          sentiment: latestData.positive_mentions / totalMentions
        },
        {
          source: 'News Articles',
          sentiment: (latestData.positive_mentions + latestData.neutral_mentions * 0.5) / totalMentions
        },
        {
          source: 'Market Analysis',
          sentiment: latestData.sentiment
        },
        {
          source: 'Overall Trend',
          sentiment: data.slice(-7).reduce((acc: number, curr: SentimentDataItem): number => acc + curr.sentiment, 0) / Math.min(7, data.length)
        }
      ]);

      setLastUpdated(new Date().toLocaleString());
    } catch (err) {
      console.error('Error fetching sentiment data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSentimentData();
  }, [selectedCrypto, timeframe]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
        <CircularProgress />
      </Box>
    );
  }

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
              <MenuItem value="SOL">Solana (SOL)</MenuItem>
              <MenuItem value="DOT">Polkadot (DOT)</MenuItem>
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

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Sentiment Analysis Over Time
              </Typography>
              {lastUpdated && (
                <Typography variant="caption" color="textSecondary">
                  Last updated: {lastUpdated}
                </Typography>
              )}
            </Box>
            <Box sx={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sentimentData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="formatted_date" />
                  <YAxis 
                    yAxisId="left" 
                    domain={[0, 1]} 
                    tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                  />
                  <YAxis 
                    yAxisId="right" 
                    orientation="right" 
                    tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`}
                  />
                  <Tooltip 
                    labelFormatter={(label) => `Date: ${label}`}
                    formatter={(value: any, name: string) => [
                      name === "Sentiment" ? `${(value * 100).toFixed(1)}%` : `${(value / 1000000).toFixed(1)}M`,
                      name
                    ]}
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="sentiment"
                    stroke="#8884d8"
                    strokeWidth={2}
                    name="Sentiment"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="volume"
                    stroke="#82ca9d"
                    strokeWidth={2}
                    name="Volume"
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
            {newsSentiment.map((item) => (
              <Box key={item.source} sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2">
                    {item.source}
                  </Typography>
                  <Typography variant="subtitle2" color="textSecondary">
                    {(item.sentiment * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={item.sentiment * 100}
                  sx={{
                    height: 10,
                    borderRadius: 5,
                    backgroundColor: '#e0e0e0',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: getSentimentColor(item.sentiment),
                      borderRadius: 5,
                    },
                  }}
                />
              </Box>
            ))}
            <Box sx={{ mt: 3, pt: 2, borderTop: 1, borderColor: 'divider' }}>
              <Typography variant="subtitle2" gutterBottom>
                Overall Sentiment
              </Typography>
              {(() => {
                const avgSentiment = newsSentiment.reduce((acc, curr) => acc + curr.sentiment, 0) / newsSentiment.length;
                return (
                  <>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" fontWeight="medium">
                        {avgSentiment >= 0.6 ? 'Bullish' : avgSentiment >= 0.4 ? 'Neutral' : 'Bearish'}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        {(avgSentiment * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={avgSentiment * 100}
                      sx={{
                        height: 10,
                        borderRadius: 5,
                        backgroundColor: '#e0e0e0',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: getSentimentColor(avgSentiment),
                          borderRadius: 5,
                        },
                      }}
                    />
                  </>
                );
              })()}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
} 