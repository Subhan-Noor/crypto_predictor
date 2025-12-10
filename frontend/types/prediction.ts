export type Prediction = {
  id: string;
  currency: string;
  prediction_date: string;
  predicted_direction: 'UP' | 'DOWN';
  confidence_score: number;
  model_version: string;
  // Validation fields:
  actual_direction?: 'UP' | 'DOWN';
  is_correct?: boolean;
  price_change_pct?: number;
  validated_at?: string;
}; 