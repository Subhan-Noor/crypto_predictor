import React, { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000';

const DiagnosticPage: React.FC = () => {
  const [currentPrice, setCurrentPrice] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [historical, setHistorical] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Test current price endpoint
        console.log('Testing current price endpoint...');
        const priceResponse = await fetch(`${API_BASE_URL}/current-price/BTC`);
        if (!priceResponse.ok) {
          throw new Error(`Current price request failed: ${priceResponse.status}`);
        }
        const priceData = await priceResponse.json();
        setCurrentPrice(priceData);
        
        // Test prediction endpoint
        console.log('Testing prediction endpoint...');
        const predictionResponse = await fetch(`${API_BASE_URL}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            symbol: 'BTC',
            timeframe: '24h',
            include_sentiment: true
          })
        });
        if (!predictionResponse.ok) {
          throw new Error(`Prediction request failed: ${predictionResponse.status}`);
        }
        const predictionData = await predictionResponse.json();
        setPrediction(predictionData);
        
        // Test historical data endpoint
        console.log('Testing historical data endpoint...');
        const historicalResponse = await fetch(
          `${API_BASE_URL}/historical/BTC?days=30&include_indicators=true`
        );
        if (!historicalResponse.ok) {
          throw new Error(`Historical data request failed: ${historicalResponse.status}`);
        }
        const historicalData = await historicalResponse.json();
        setHistorical(historicalData.data || []);
        
        setLoading(false);
      } catch (err) {
        console.error('Diagnostic test failed:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', padding: '20px' }}>
      <h1>API Diagnostic Page</h1>
      <p>This page tests direct API connectivity</p>
      
      {loading ? (
        <p>Loading data from API...</p>
      ) : error ? (
        <div style={{ color: 'red' }}>
          <h2>Error</h2>
          <pre>{error}</pre>
        </div>
      ) : (
        <div>
          <h2>API Test Results</h2>
          
          <h3>Current Price</h3>
          <pre style={{ background: '#f0f0f0', padding: '10px', overflow: 'auto' }}>
            {JSON.stringify(currentPrice, null, 2)}
          </pre>
          
          <h3>Prediction</h3>
          <pre style={{ background: '#f0f0f0', padding: '10px', overflow: 'auto' }}>
            {JSON.stringify(prediction, null, 2)}
          </pre>
          
          <h3>Historical Data (first 3 items)</h3>
          <pre style={{ background: '#f0f0f0', padding: '10px', overflow: 'auto' }}>
            {JSON.stringify(historical.slice(0, 3), null, 2)}
          </pre>
          
          <p><strong>Total historical data points:</strong> {historical.length}</p>
        </div>
      )}
      
      <div style={{ marginTop: '20px' }}>
        <p>Raw test: <a href={`${API_BASE_URL}/current-price/BTC`} target="_blank" rel="noopener noreferrer">Open current price endpoint in new tab</a></p>
      </div>
    </div>
  );
};

export default DiagnosticPage; 