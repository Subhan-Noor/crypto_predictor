import React from 'react';
import { Prediction } from '../types/prediction';

export const PredictionRow: React.FC<{ prediction: Prediction }> = ({ prediction }) => {
  const {
    prediction_date,
    predicted_direction,
    confidence_score,
    actual_direction,
    is_correct,
    price_change_pct,
    validated_at,
  } = prediction;

  return (
    <tr>
      <td>{new Date(prediction_date).toLocaleDateString()}</td>
      <td>{predicted_direction}</td>
      <td>{actual_direction ?? <span style={{ color: '#aaa' }}>Pending</span>}</td>
      <td>
        {is_correct === true && <span style={{ color: 'green' }}>✅</span>}
        {is_correct === false && <span style={{ color: 'red' }}>❌</span>}
        {is_correct === undefined && <span style={{ color: '#aaa' }}>Pending</span>}
      </td>
      <td>{price_change_pct !== undefined ? `${price_change_pct.toFixed(2)}%` : '-'}</td>
      <td>{(confidence_score * 100).toFixed(1)}%</td>
      <td>{validated_at ? new Date(validated_at).toLocaleDateString() : '-'}</td>
    </tr>
  );
}; 