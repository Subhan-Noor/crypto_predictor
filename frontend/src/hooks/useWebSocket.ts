import { useState, useEffect, useCallback } from 'react';
import { wsService } from '../services/websocket';

interface CryptoUpdate {
    type: string;
    symbol: string;
    timestamp: string;
    price_data: {
        open: number;
        high: number;
        low: number;
        close: number;
        volume: number;
    };
    prediction: {
        price: number;
        confidence_interval: [number, number];
    };
    sentiment: number;
}

export function useWebSocket(symbol: string) {
    const [data, setData] = useState<CryptoUpdate | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isConnected, setIsConnected] = useState(false);

    const handleUpdate = useCallback((update: CryptoUpdate) => {
        if (update.type === 'update' && update.price_data && update.prediction) {
            setData(update);
        }
    }, []);

    const handleError = useCallback((err: string) => {
        setError(err);
        setIsConnected(false);
    }, []);

    const handleConnectionChange = useCallback((connected: boolean) => {
        setIsConnected(connected);
        if (!connected) {
            setData(null);
        }
    }, []);

    useEffect(() => {
        setData(null);
        setError(null);
        setIsConnected(false);

        const unsubscribe = wsService.subscribe(
            symbol,
            handleUpdate,
            handleError,
            handleConnectionChange
        );

        return () => {
            unsubscribe();
            setIsConnected(false);
            setData(null);
            setError(null);
        };
    }, [symbol, handleUpdate, handleError, handleConnectionChange]);

    return {
        data,
        error,
        isConnected
    };
} 