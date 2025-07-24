# 🚀 Railway Deployment Setup Guide

## Environment Variables Required

You need to set these environment variables in your Railway project dashboard:

### **Required Variables:**

1. **SUPABASE_URL**
   - Go to your Supabase project dashboard
   - Navigate to Settings → API
   - Copy the "Project URL" (starts with `https://`)

2. **SUPABASE_KEY**
   - In the same Supabase Settings → API page
   - Copy the "anon public" key (starts with `eyJ`)

3. **SUPABASE_SERVICE_ROLE_KEY**
   - In the same Supabase Settings → API page
   - Copy the "service_role" key (starts with `eyJ`)
   - ⚠️ **Keep this secret** - it has admin privileges

### **Optional Variables:**

4. **REDIS_URL** (optional)
   - If you have a Redis instance, add the connection URL
   - Format: `redis://username:password@host:port`
   - If not set, the app will work without caching

5. **ENVIRONMENT**
   - Set to `production` for Railway deployment

## How to Set Environment Variables in Railway:

### **Method 1: Railway Dashboard**
1. Go to your Railway project dashboard
2. Click on your backend service
3. Go to the "Variables" tab
4. Add each variable:
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ENVIRONMENT=production
   ```

### **Method 2: Railway CLI**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Link to your project
railway link

# Set variables
railway variables set SUPABASE_URL=https://your-project.supabase.co
railway variables set SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
railway variables set SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
railway variables set ENVIRONMENT=production
```

## Verification Steps:

### **1. Check Environment Variables**
After setting the variables, redeploy and check the logs:
```bash
railway logs
```

You should see:
```
🔧 Testing environment variables...
  ✅ SUPABASE_URL: https://your-project.supabase.co
  ✅ SUPABASE_KEY: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
  ✅ SUPABASE_SERVICE_ROLE_KEY: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### **2. Test Health Endpoint**
Once deployed, test the health endpoint:
```bash
curl https://your-railway-app.up.railway.app/health
```

Should return:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-23T...",
  "version": "2.0.0",
  "environment": "production"
}
```

### **3. Test Frontend Connection**
Visit your Vercel frontend and check the browser console for:
```
🔗 API Base URL: https://your-railway-app.up.railway.app
✅ Backend Connected Successfully
```

## Troubleshooting:

### **If Environment Variables Are Missing:**
- Double-check the variable names (case-sensitive)
- Make sure you're setting them for the correct service
- Redeploy after setting variables

### **If Redis Connection Fails:**
- The app will work without Redis (caching disabled)
- Check logs for "Redis cache unavailable - using fallback mode"

### **If Database Connection Fails:**
- Verify your Supabase credentials are correct
- Check if your Supabase project is active
- Ensure the database tables exist

## Security Notes:

- ⚠️ **Never commit environment variables to Git**
- 🔒 **Keep SUPABASE_SERVICE_ROLE_KEY secret**
- 🔐 **Use Railway's built-in secrets management**
- 🌐 **Consider using Railway's built-in PostgreSQL instead of Supabase for production**

## Next Steps:

1. Set the environment variables in Railway
2. Redeploy the backend
3. Test the health endpoint
4. Update your frontend to use the new Railway URL
5. Test the full application

---

**Need Help?**
- Check Railway documentation: https://docs.railway.app/
- Check Supabase documentation: https://supabase.com/docs
- Review the deployment logs for specific error messages 