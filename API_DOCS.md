# Crypto Prediction API Documentation

## Endpoints

### 1. Get Prediction History
**GET** `/predictions/{currency}/history?days=30&limit=100`

Returns all predictions for a currency, including validation fields.

**Example Response:**
```json
{
  "currency": "BTC",
  "predictions": [
    {
      "id": "uuid",
      "currency": "BTC",
      "prediction_date": "2025-07-17T00:00:00Z",
      "predicted_direction": "DOWN",
      "confidence_score": 0.45,
      "model_version": "random_forest_20250724_012746",
      "actual_direction": "UP",
      "is_correct": false,
      "price_change_pct": 0.96,
      "validated_at": "2025-07-24T01:32:23Z"
    },
    ...
  ],
  "count": 30,
  "days": 30
}
```

**Fields:**
- `actual_direction`: 'UP' or 'DOWN' (real result, if validated)
- `is_correct`: true/false (if validated)
- `price_change_pct`: % change over prediction horizon
- `validated_at`: ISO timestamp when validated

---

### 2. Get Prediction Accuracy
**GET** `/predictions/accuracy/{currency}?days=30`

Returns accuracy stats and recent predictions.

**Example Response:**
```json
{
  "currency": "BTC",
  "accuracy": 91.7,
  "total_predictions": 30,
  "validated_predictions": 24,
  "correct_predictions": 22,
  "recent_predictions": [ ... ],
  "prediction_distribution": { "up": 12, "down": 18 }
}
```

---

### 3. Generate Prediction (Manual)
**POST** `/predict/{currency}`

Request body:
```json
{
  "prediction_horizon": 7
}
```

Response:
```json
{
  "currency": "BTC",
  "prediction_date": "2025-07-24T00:00:00Z",
  "predicted_direction": "UP",
  "confidence_score": 0.67,
  "model_version": "random_forest_20250724_012746",
  "model_type": "random_forest"
}
```

---

### 4. Trigger Validation (Manual/Admin)
**GET** `/predictions/auto-validate`

Returns summary of auto-validation run.

---

### 5. Get Prices
**GET** `/prices/{currency}?days=30`

---

### 6. Get Sentiment
**GET** `/sentiment/{currency}?days=30`

---

## Notes
- All endpoints return JSON.
- All date/times are ISO8601 UTC.
- For more, see `/docs` (Swagger UI) on your deployed backend. 