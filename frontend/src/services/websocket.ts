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

type UpdateCallback = (update: CryptoUpdate) => void;
type ErrorCallback = (error: string) => void;
type ConnectionCallback = (connected: boolean) => void;

interface CallbackSet {
    update: UpdateCallback;
    error: ErrorCallback;
    connection: ConnectionCallback;
}

class WebSocketService {
    private static instance: WebSocketService;
    private connections: Map<string, { 
        ws: WebSocket | null;
        callbacks: Set<CallbackSet>;
        heartbeatInterval?: number;
        reconnectTimeout?: number;
        isReconnecting: boolean;
        reconnectAttempts: number;
    }>;
    private baseUrl: string;
    private readonly MAX_RECONNECT_ATTEMPTS = 5;
    private readonly INITIAL_RECONNECT_DELAY = 1000;

    private constructor() {
        this.connections = new Map();
        this.baseUrl = 'ws://localhost:8000/ws';
    }

    public static getInstance(): WebSocketService {
        if (!WebSocketService.instance) {
            WebSocketService.instance = new WebSocketService();
        }
        return WebSocketService.instance;
    }

    private startHeartbeat(ws: WebSocket): number {
        return window.setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                try {
                    ws.send(JSON.stringify({ type: 'heartbeat' }));
                } catch (error) {
                    console.warn('Failed to send heartbeat:', error);
                }
            }
        }, 30000) as unknown as number;
    }

    private getReconnectDelay(attempts: number): number {
        // Exponential backoff with max delay of 30 seconds
        return Math.min(this.INITIAL_RECONNECT_DELAY * Math.pow(2, attempts), 30000);
    }

    public subscribe(
        symbol: string,
        onUpdate: UpdateCallback,
        onError: ErrorCallback,
        onConnection: ConnectionCallback
    ): () => void {
        const callbacks: CallbackSet = { update: onUpdate, error: onError, connection: onConnection };
        
        // Get or create connection entry
        let connection = this.connections.get(symbol);
        if (!connection) {
            connection = {
                ws: null,
                callbacks: new Set([callbacks]),
                isReconnecting: false,
                reconnectAttempts: 0
            };
            this.connections.set(symbol, connection);
            this.createAndConnectWebSocket(symbol);
        } else {
            connection.callbacks.add(callbacks);
            // If there's an active connection, notify the new subscriber
            if (connection.ws?.readyState === WebSocket.OPEN) {
                onConnection(true);
            }
        }

        // Return unsubscribe function
        return () => {
            const conn = this.connections.get(symbol);
            if (conn) {
                conn.callbacks.delete(callbacks);
                if (conn.callbacks.size === 0) {
                    this.cleanupConnection(symbol);
                }
            }
        };
    }

    private cleanupConnection(symbol: string) {
        const connection = this.connections.get(symbol);
        if (connection) {
            if (connection.heartbeatInterval) {
                clearInterval(connection.heartbeatInterval);
            }
            if (connection.reconnectTimeout) {
                clearTimeout(connection.reconnectTimeout);
            }
            if (connection.ws && connection.ws.readyState === WebSocket.OPEN) {
                try {
                    connection.ws.close(1000, 'Normal closure');
                } catch (error) {
                    console.warn('Error closing WebSocket:', error);
                }
            }
            this.connections.delete(symbol);
        }
    }

    private createAndConnectWebSocket(symbol: string) {
        const connection = this.connections.get(symbol);
        if (!connection) return;

        try {
            const ws = new WebSocket(`${this.baseUrl}/${symbol}`);
            connection.ws = ws;

            ws.onopen = () => {
                console.log(`WebSocket connection opened for ${symbol}`);
                connection.isReconnecting = false;
                connection.reconnectAttempts = 0;
                
                // Send initial message
                try {
                    ws.send(JSON.stringify({ type: 'subscribe', symbol }));
                    // Start heartbeat
                    connection.heartbeatInterval = this.startHeartbeat(ws);
                    // Notify all subscribers
                    connection.callbacks.forEach(cb => cb.connection(true));
                } catch (error) {
                    console.error('Error in onopen handler:', error);
                }
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'heartbeat') return;
                    
                    // Notify all subscribers
                    connection.callbacks.forEach(cb => cb.update(data));
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                    connection.callbacks.forEach(cb => 
                        cb.error('Failed to parse server message')
                    );
                }
            };

            ws.onclose = (event) => {
                console.log(`WebSocket connection closed for ${symbol}`, event.code, event.reason);
                
                // Clear heartbeat interval
                if (connection.heartbeatInterval) {
                    clearInterval(connection.heartbeatInterval);
                    connection.heartbeatInterval = undefined;
                }
                
                // Notify all subscribers
                connection.callbacks.forEach(cb => cb.connection(false));
                
                // Attempt to reconnect after a delay if it wasn't a normal closure
                if (event.code !== 1000 && !connection.isReconnecting) {
                    this.handleReconnection(symbol, connection);
                }
            };

            ws.onerror = (error) => {
                console.error(`WebSocket error for ${symbol}:`, error);
                // Only notify of errors if we're not already reconnecting
                if (!connection.isReconnecting) {
                    connection.callbacks.forEach(cb => 
                        cb.error('WebSocket connection error')
                    );
                }
            };
        } catch (error) {
            console.error('Error creating WebSocket:', error);
            this.handleReconnection(symbol, connection);
        }
    }

    private handleReconnection(symbol: string, connection: any) {
        if (connection.reconnectAttempts >= this.MAX_RECONNECT_ATTEMPTS) {
            console.log(`Max reconnection attempts reached for ${symbol}`);
            connection.callbacks.forEach((cb: CallbackSet) => 
                cb.error('Failed to establish connection after multiple attempts')
            );
            return;
        }

        connection.isReconnecting = true;
        connection.reconnectAttempts++;
        
        const delay = this.getReconnectDelay(connection.reconnectAttempts);
        console.log(`Attempting to reconnect to ${symbol} in ${delay}ms (attempt ${connection.reconnectAttempts})`);
        
        connection.reconnectTimeout = window.setTimeout(() => {
            if (connection.callbacks.size > 0) {
                this.createAndConnectWebSocket(symbol);
            }
        }, delay);
    }

    public unsubscribeAll(symbol: string) {
        this.cleanupConnection(symbol);
    }
}

export const wsService = WebSocketService.getInstance(); 