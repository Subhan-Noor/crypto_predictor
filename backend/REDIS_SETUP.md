# Redis Setup Guide for Windows

This guide will help you set up Redis for the Enhanced Crypto Prediction API.

## Option 1: Redis Cloud (Recommended for Windows)

### 1. Sign up for Redis Cloud
- Visit: https://redis.com/try-free/
- Create a free account
- Create a new database

### 2. Get your Redis URL
- After creating the database, you'll get a Redis URL that looks like:
  ```
  redis://username:password@host:port
  ```

### 3. Update your .env file
Add this to your `backend/.env` file:
```env
REDIS_URL=redis://username:password@host:port
REDIS_ENABLED=true
```

## Option 2: WSL (Windows Subsystem for Linux)

### 1. Install WSL
```bash
wsl --install
```

### 2. Install Redis in WSL
```bash
# Open WSL terminal
sudo apt update
sudo apt install redis-server
```

### 3. Start Redis
```bash
sudo service redis-server start
```

### 4. Test Redis
```bash
redis-cli ping
# Should return: PONG
```

### 5. Update your .env file
```env
REDIS_URL=redis://localhost:6379
REDIS_ENABLED=true
```

## Option 3: Docker (Advanced)

### 1. Install Docker Desktop
- Download from: https://www.docker.com/products/docker-desktop/

### 2. Run Redis container
```bash
docker run -d --name redis-cache -p 6379:6379 redis:latest
```

### 3. Update your .env file
```env
REDIS_URL=redis://localhost:6379
REDIS_ENABLED=true
```

## Option 4: Run Without Redis (Fallback)

If you can't set up Redis, the API will still work but without caching and rate limiting:

### Update your .env file
```env
REDIS_ENABLED=false
```

## Testing Redis Connection

After setup, test your Redis connection:

```bash
# Test with Python
python -c "import redis; r = redis.from_url('your_redis_url'); print(r.ping())"
```

## Troubleshooting

### Common Issues:

1. **Connection refused**: Redis server not running
2. **Authentication failed**: Wrong password in Redis URL
3. **Timeout**: Network issues or wrong host/port

### Solutions:

1. **Check if Redis is running**:
   ```bash
   # WSL
   sudo service redis-server status
   
   # Docker
   docker ps | grep redis
   ```

2. **Test connection manually**:
   ```bash
   redis-cli -u your_redis_url ping
   ```

3. **Check firewall settings**: Make sure port 6379 is open

## Performance Impact

- **With Redis**: 5x faster response times, rate limiting active
- **Without Redis**: Normal performance, no rate limiting

The API will work in both modes, but Redis provides significant performance benefits. 