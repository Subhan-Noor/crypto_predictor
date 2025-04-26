import React, { useState } from 'react';
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
import { RealTimePrice } from '../components/RealTimePrice';

const SUPPORTED_CRYPTOS = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT'];

const mockData = [
  { date: '2024-01-01', price: 45000 },
  { date: '2024-01-02', price: 46000 },
  { date: '2024-01-03', price: 44000 },
  { date: '2024-01-04', price: 47000 },
  { date: '2024-01-05', price: 48000 },
];

const Dashboard: React.FC = () => {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');

  const metrics = [
    { title: 'Current Price', value: '$47,123.45' },
    { title: '24h Change', value: '+2.5%' },
    { title: 'Market Cap', value: '$890B' },
    { title: 'Volume (24h)', value: '$28.5B' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Cryptocurrency Dashboard</h1>
        <FormControl variant="outlined" style={{ minWidth: 120 }}>
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
      </div>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper elevation={0} className="h-full">
            <RealTimePrice symbol={selectedCrypto} />
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper elevation={0} className="h-full p-6">
            <h2 className="text-xl font-semibold mb-4">Market Overview</h2>
            {/* Add market overview content */}
          </Paper>
        </Grid>
        
        <Grid item xs={12}>
          <Paper elevation={0} className="p-6">
            <h2 className="text-xl font-semibold mb-4">Historical Performance</h2>
            {/* Add historical chart */}
          </Paper>
        </Grid>
      </Grid>
    </div>
  );
};

export default Dashboard; 