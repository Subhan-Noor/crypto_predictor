# User Input Checklist Before Stage 3

To ensure the project is ready for ML model development, please complete the following steps:

---

## 1. Supabase Setup

1. **Create a Supabase Project:**
   - Go to [https://supabase.com](https://supabase.com) and sign up/log in.
   - Click "New Project" and follow the prompts.
   - Wait for the project to initialize.

2. **Get Supabase Credentials:**
   - In the dashboard, go to **Settings → API**.
   - Copy the following:
     - **Project URL** (e.g., `https://xyzcompany.supabase.co`)
     - **anon public key** (for SUPABASE_KEY)

---

## 2. .env File Setup (backend/.env)

Create a file named `.env` in the `backend/` directory with the following content:

```
# Database Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

# API Keys (Optional for testing, but recommended for full functionality)
# No API keys required for Twint and Pushshift API

# Redis (for caching, optional)
REDIS_URL=redis://localhost:6379

# Application Settings
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO
```

**Replace all `your_...` values with your actual credentials.**

---

## 3. Initial Data Ingestion

After completing the above steps:

1. **Activate your Python virtual environment:**
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run initial data ingestion:**
   ```bash
   # New data ingestion script will be provided in Stage 3
   ```

---

## 4. Verify Setup

- Visit `http://localhost:8000/health` to check API health.
- If you see errors about missing credentials or failed connections, double-check your `.env` file and Supabase setup.

---

## 5. (Optional) Enable Daily Automation

- Add your Supabase credentials as GitHub repository secrets if you want to use the automated daily ingestion workflow.
- The workflow will run daily at 6 AM UTC.

---

**Once all steps are complete and data is flowing, you are ready for Stage 3 (ML model development)!** 