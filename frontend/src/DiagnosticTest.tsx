import React, { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000';

const DiagnosticTest: React.FC = () => {
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
    <div style={{ fontFamily: 'Arial, sans-serif', padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h1>API Diagnostic Test</h1>
      <p>This page directly tests the API connectivity without any complex state management.</p>
      
      {loading ? (
        <p>Loading data from API...</p>
      ) : error ? (
        <div style={{ color: 'red', backgroundColor: '#ffeeee', padding: '10px', borderRadius: '5px' }}>
          <h2>Error</h2>
          <pre>{error}</pre>
        </div>
      ) : (
        <div>
          <h2>API Test Results</h2>
          
          <div style={{ backgroundColor: '#eeffee', padding: '10px', borderRadius: '5px', marginBottom: '20px' }}>
            <h3>Current Price</h3>
            <pre style={{ background: '#f0f0f0', padding: '10px', overflow: 'auto' }}>
              {JSON.stringify(currentPrice, null, 2)}
            </pre>
          </div>
          
          <div style={{ backgroundColor: '#eeeeff', padding: '10px', borderRadius: '5px', marginBottom: '20px' }}>
            <h3>Prediction</h3>
            <pre style={{ background: '#f0f0f0', padding: '10px', overflow: 'auto' }}>
              {JSON.stringify(prediction, null, 2)}
            </pre>
          </div>
          
          <div style={{ backgroundColor: '#ffffee', padding: '10px', borderRadius: '5px', marginBottom: '20px' }}>
            <h3>Historical Data (first 3 items)</h3>
            <pre style={{ background: '#f0f0f0', padding: '10px', overflow: 'auto' }}>
              {JSON.stringify(historical.slice(0, 3), null, 2)}
            </pre>
            
            <p><strong>Total historical data points:</strong> {historical.length}</p>
          </div>
        </div>
      )}
      
      <div style={{ marginTop: '30px', padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '5px' }}>
        <h3>Direct API Links</h3>
        <ul>
          <li><a href={`${API_BASE_URL}/current-price/BTC`} target="_blank" rel="noopener noreferrer">Current Price Endpoint</a></li>
          <li><a href={`${API_BASE_URL}/historical/BTC?days=30&include_indicators=true`} target="_blank" rel="noopener noreferrer">Historical Data Endpoint</a></li>
          <li>(Prediction endpoint requires POST request)</li>
        </ul>
      </div>
    </div>
  );
};

export default DiagnosticTest; 