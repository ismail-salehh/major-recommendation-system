"""
Script to retrain models, evaluate performance, and export model artifacts.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_pipeline import prepare_train_test_pipeline
from src.models import (
    build_random_forest_baseline,
    build_extra_trees,
    build_xgboost,
    build_lightgbm,
    build_knn,
    build_soft_voting_ensemble
)
from src.evaluator import compute_top_k_accuracy, compute_class_ceiling, extract_feature_importances


def main():
    print("=" * 65)
    print("  MAJOR RECOMMENDATION SYSTEM - MODEL TRAINING & BENCHMARK")
    print("=" * 65)

    os.makedirs('models', exist_ok=True)

    print("\n[1/4] Running data preprocessing and SMOTE pipeline...")
    pipeline = prepare_train_test_pipeline(
        csv_path='data/data.csv',
        occurrence_threshold=4,
        test_size=0.2,
        random_state=42,
        use_smote=True
    )

    X_train = pipeline['X_train']
    X_test = pipeline['X_test']
    y_train = pipeline['y_train']
    y_test = pipeline['y_test']
    label_encoders = pipeline['label_encoders']
    feature_columns = pipeline['feature_columns']

    major_classes = label_encoders['Major'].classes_
    print(f"      - Filtered Dataset Shape: {pipeline['df_clean'].shape}")
    print(f"      - Target Major Classes ({len(major_classes)} majors): {list(major_classes[:5])} ...")

    models_to_evaluate = {
        'Random Forest (Notebook Baseline)': build_random_forest_baseline(42),
        'Extra Trees (Tuned)': build_extra_trees(42),
        'XGBoost Classifier': build_xgboost(42),
        'LightGBM Classifier': build_lightgbm(42),
        'KNN (n=5)': build_knn(5),
        'Soft Voting Ensemble (ET + RF + XGB + LGB)': build_soft_voting_ensemble(42)
    }

    print("\n[2/4] Training and benchmarking candidate models...")
    benchmark_results = {}
    fitted_models = {}

    for name, clf in models_to_evaluate.items():
        print(f"      -> Fitting {name}...")
        clf.fit(X_train, y_train)
        acc_dict = compute_top_k_accuracy(clf, X_test, y_test, k_values=[1, 3, 5, 10])
        benchmark_results[name] = acc_dict
        fitted_models[name] = clf

    print("\n" + "=" * 65)
    print(f"{'Model Architecture':<42} | {'Top-1':<7} | {'Top-3':<7} | {'Top-5':<7} | {'Top-10':<7}")
    print("-" * 75)
    for name, acc in benchmark_results.items():
        print(f"{name:<42} | {acc['Top-1']:<7.4f} | {acc['Top-3']:<7.4f} | {acc['Top-5']:<7.4f} | {acc['Top-10']:<7.4f}")
    print("=" * 75)

    print("\n[3/4] Exporting model artifacts...")

    # Best model selection (Ensemble)
    best_model_name = 'Soft Voting Ensemble (ET + RF + XGB + LGB)'
    best_model = fitted_models[best_model_name]
    baseline_rf = fitted_models['Random Forest (Notebook Baseline)']

    # Save pickles
    joblib.dump(best_model, 'models/major_recommender_ensemble.pkl')
    joblib.dump(baseline_rf, 'models/random_forest_baseline.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')

    # Compute class ceiling for relative probability scaling
    class_ceilings = compute_class_ceiling(best_model, X_test, y_test)

    # Feature importances
    df_imp = extract_feature_importances(best_model, feature_columns)

    metadata = {
        'best_model_name': best_model_name,
        'feature_columns': feature_columns,
        'categorical_columns': pipeline['categorical_columns'],
        'num_classes': len(major_classes),
        'major_classes': list(major_classes),
        'benchmark_metrics': benchmark_results,
        'class_ceiling': {str(k): float(v) for k, v in class_ceilings.items()},
        'feature_importances': df_imp.to_dict(orient='records')
    }

    with open('models/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("      [OK] Saved best model to: models/major_recommender_ensemble.pkl")
    print("      [OK] Saved baseline model to: models/random_forest_baseline.pkl")
    print("      [OK] Saved label encoders to: models/label_encoders.pkl")
    print("      [OK] Saved metadata to: models/model_metadata.json")

    print("\n[4/4] Pipeline training completed successfully!")


if __name__ == '__main__':
    main()
