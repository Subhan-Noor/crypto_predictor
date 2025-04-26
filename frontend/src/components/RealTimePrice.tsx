import React from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { getSentimentLabel, getSentimentColor } from '../utils/sentiment';

interface PriceData {
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

interface Prediction {
    price: number;
    confidence_interval: [number, number];
}

interface WebSocketData {
    type: string;
    symbol: string;
    timestamp: string;
    price_data: PriceData;
    prediction: Prediction;
    sentiment: number;
}

interface RealTimePriceProps {
    symbol: string;
}

export const RealTimePrice: React.FC<RealTimePriceProps> = ({ symbol }) => {
    const { data, error, isConnected } = useWebSocket(symbol);

    if (error) {
        return (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
                Error connecting to real-time updates: {error}
            </div>
        );
    }

    if (!isConnected) {
        return (
            <div className="animate-pulse bg-gray-100 p-4 rounded-lg">
                Connecting to real-time updates...
            </div>
        );
    }

    if (!data || !data.price_data || !data.prediction) {
        return (
            <div className="animate-pulse bg-gray-100 p-4 rounded-lg">
                Waiting for data...
            </div>
        );
    }

    const wsData = data as WebSocketData;
    const { price_data, prediction, sentiment } = wsData;
    const priceChange = ((price_data.close - price_data.open) / price_data.open) * 100;
    const isPriceUp = price_data.close > price_data.open;

    return (
        <div className="bg-white shadow-lg rounded-lg p-6 space-y-4">
            <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold">{symbol} Real-Time</h2>
                <div className="text-sm text-gray-500">
                    Last updated: {new Date(wsData.timestamp).toLocaleTimeString()}
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                    <div className="text-gray-600">Current Price</div>
                    <div className="text-3xl font-bold">
                        ${price_data.close.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </div>
                    <div className={`text-sm ${isPriceUp ? 'text-green-600' : 'text-red-600'}`}>
                        {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                    </div>
                </div>

                <div className="space-y-2">
                    <div className="text-gray-600">Predicted Price (24h)</div>
                    <div className="text-2xl font-semibold">
                        ${prediction.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </div>
                    <div className="text-sm text-gray-500">
                        Range: ${prediction.confidence_interval[0].toFixed(2)} - ${prediction.confidence_interval[1].toFixed(2)}
                    </div>
                </div>
            </div>

            <div className="border-t pt-4">
                <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                        <div className="text-gray-600">24h High</div>
                        <div className="font-semibold">${price_data.high.toLocaleString()}</div>
                    </div>
                    <div>
                        <div className="text-gray-600">24h Low</div>
                        <div className="font-semibold">${price_data.low.toLocaleString()}</div>
                    </div>
                    <div>
                        <div className="text-gray-600">Volume</div>
                        <div className="font-semibold">${price_data.volume.toLocaleString()}</div>
                    </div>
                </div>
            </div>

            <div className="border-t pt-4">
                <div className="text-gray-600">Market Sentiment</div>
                <div className="mt-2 h-2 bg-gray-200 rounded-full">
                    <div
                        className="h-full rounded-full"
                        style={{ 
                            width: `${sentiment * 100}%`,
                            backgroundColor: getSentimentColor(sentiment)
                        }}
                    />
                </div>
                <div className="text-sm text-gray-500 mt-1">
                    {getSentimentLabel(sentiment)}
                </div>
            </div>
        </div>
    );
}; 