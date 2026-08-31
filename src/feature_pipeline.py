"""
Data feature preprocessing, encoding, SMOTE oversampling, and dataset splitting pipeline.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.text_cleaner import clean_gpa
from src.major_mapper import map_major

FEATURE_COLUMNS = [
    'GPA',
    'Interest in Technology',
    'Interest in Health and Medicine',
    'Interest in Business and Management',
    'Interest in Arts and Creativity',
    'Thinking Style',
    'Personality Type',
    'Preferred Study Method',
    'Preferred Work Style'
]

CATEGORICAL_COLUMNS = [
    'Major',
    'Thinking Style',
    'Personality Type',
    'Preferred Study Method',
    'Preferred Work Style'
]


def load_raw_dataset(csv_path: str = 'data/data.csv') -> pd.DataFrame:
    """Loads and returns raw student survey dataset with renamed standard column titles."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path, encoding='utf-8')
    if 'طابع زمني' in df.columns:
        df.drop('طابع زمني', axis=1, inplace=True)

    df.columns = [
        'GPA', 'Major', 'Interest in Technology', 'Interest in Health and Medicine',
        'Interest in Business and Management', 'Interest in Arts and Creativity',
        'Thinking Style', 'Personality Type', 'Preferred Study Method', 'Preferred Work Style',
        'Satisfaction with Current Major', 'Would an Intelligent System Change Your Major?'
    ]

    # Drop survey feedback columns not used for major recommendations
    df.drop(columns=['Satisfaction with Current Major', 'Would an Intelligent System Change Your Major?'], inplace=True, errors='ignore')
    return df


def preprocess_dataset(df: pd.DataFrame, occurrence_threshold: int = 4):
    """
    Cleans GPA, maps raw major strings, removes low-frequency majors,
    fits LabelEncoders, and prepares X and y DataFrames.
    """
    df_clean = df.copy()

    # Clean numerical GPA
    df_clean['GPA'] = df_clean['GPA'].apply(clean_gpa)

    # Map raw majors to canonical labels
    df_clean['Major'] = df_clean['Major'].apply(map_major)
    df_clean.dropna(subset=['GPA', 'Major'], inplace=True)

    # Filter out rare major labels below occurrence_threshold
    counts = df_clean['Major'].value_counts()
    to_remove = counts[counts < occurrence_threshold].index
    df_clean['Major'] = df_clean['Major'].apply(lambda x: None if x in to_remove else x)
    df_clean.dropna(subset=['Major'], inplace=True)

    # Label Encoders dictionary
    label_encoders = {}
    df_encoded = df_clean.copy()

    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    X = df_encoded[FEATURE_COLUMNS].copy()
    y = df_encoded['Major'].copy()

    return df_clean, df_encoded, X, y, label_encoders


def prepare_train_test_pipeline(
    csv_path: str = 'data/data.csv',
    occurrence_threshold: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True
):
    """
    Complete end-to-end dataset preprocessing & train-test split pipeline.
    """
    df_raw = load_raw_dataset(csv_path)
    df_clean, df_encoded, X, y, label_encoders = preprocess_dataset(df_raw, occurrence_threshold)

    if use_smote:
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X, y)
    else:
        X_res, y_res = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, random_state=random_state, stratify=y_res
    )

    pipeline_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_raw_features': X,
        'y_raw_targets': y,
        'df_clean': df_clean,
        'df_encoded': df_encoded,
        'label_encoders': label_encoders,
        'feature_columns': FEATURE_COLUMNS,
        'categorical_columns': CATEGORICAL_COLUMNS
    }

    return pipeline_data
