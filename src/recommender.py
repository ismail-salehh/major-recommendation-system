"""
Inference engine and recommendation service for student academic majors.
"""
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

from src.major_mapper import generalize_major

DEFAULT_MODEL_PATH = "models/major_recommender_ensemble.pkl"
DEFAULT_ENCODERS_PATH = "models/label_encoders.pkl"
DEFAULT_METADATA_PATH = "models/model_metadata.json"


class MajorRecommender:
    """
    Main Recommendation Engine for serving major predictions.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 encoders_path: str = DEFAULT_ENCODERS_PATH,
                 metadata_path: str = DEFAULT_METADATA_PATH):
        self.model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoders_path)

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.feature_columns = self.metadata.get('feature_columns', [])
        self.class_ceiling = self.metadata.get('class_ceiling', {})

    def validate_and_format_input(self, raw_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Validates student feature input dictionary and converts categorical inputs into encoded numbers.
        """
        formatted = {}

        # GPA
        gpa = float(raw_input.get('GPA', 80.0))
        gpa = max(60.0, min(100.0, gpa))
        formatted['GPA'] = gpa

        # Numerical Interests (1-5)
        for key in ['Interest in Technology', 'Interest in Health and Medicine',
                    'Interest in Business and Management', 'Interest in Arts and Creativity']:
            formatted[key] = int(raw_input.get(key, 3))

        # Categorical String Encodings
        cat_mappings = {
            'Thinking Style': raw_input.get('Thinking Style', 'تفكير منطقي'),
            'Personality Type': raw_input.get('Personality Type', 'اجتماعي (بتحب تختلط بالناس بكثرة )'),
            'Preferred Study Method': raw_input.get('Preferred Study Method', 'مزيج بين النظري والعملي'),
            'Preferred Work Style': raw_input.get('Preferred Work Style', 'الاثنين معا')
        }

        for col, str_val in cat_mappings.items():
            le = self.label_encoders[col]
            # Handle unseen labels gracefully by mapping to closest or first class
            if str_val in le.classes_:
                formatted[col] = int(le.transform([str_val])[0])
            else:
                formatted[col] = 0

        df_input = pd.DataFrame([formatted], columns=self.feature_columns)
        return df_input

    def recommend(self, student_input: Dict[str, Any], top_k: int = 5,
                  use_relative_scaling: bool = False) -> List[Dict[str, Any]]:
        """
        Generates top_k recommended majors with confidence scores and domain categories.
        """
        df_input = self.validate_and_format_input(student_input)
        probas = self.model.predict_proba(df_input)[0]

        top_indices = np.argsort(probas)[-top_k:][::-1]
        major_encoder = self.label_encoders['Major']

        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            raw_prob = float(probas[idx])
            major_name = major_encoder.inverse_transform([idx])[0]
            category = generalize_major(major_name)

            if use_relative_scaling:
                ceiling = float(self.class_ceiling.get(str(idx), 1e-6))
                score = min(1.0, raw_prob / ceiling) if ceiling > 0 else raw_prob
            else:
                score = min(1.0, raw_prob)

            recommendations.append({
                'rank': rank,
                'major': major_name,
                'category': category,
                'confidence': round(score, 4),
                'confidence_pct': f"{score * 100:.1f}%",
                'raw_probability': round(raw_prob, 4)
            })

        return recommendations
