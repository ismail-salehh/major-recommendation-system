"""
Model definitions, hyperparameter configurations, and ensemble factory functions.
"""
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb


def build_random_forest_baseline(random_state: int = 42):
    """Returns the baseline Random Forest classifier from the original notebook."""
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=8,
        max_features='sqrt',
        max_depth=15,
        bootstrap=True,
        random_state=random_state
    )


def build_extra_trees(random_state: int = 42):
    """Returns high-performing ExtraTrees classifier."""
    return ExtraTreesClassifier(
        n_estimators=300,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=random_state
    )


def build_xgboost(random_state: int = 42):
    """Returns XGBoost multi-class classifier."""
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=random_state
    )


def build_lightgbm(random_state: int = 42):
    """Returns LightGBM classifier."""
    return lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        verbose=-1,
        random_state=random_state
    )


def build_knn(n_neighbors: int = 5):
    """Returns KNN Classifier baseline."""
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def build_soft_voting_ensemble(random_state: int = 42):
    """
    Builds a Soft Voting Ensemble model combining ExtraTrees, RandomForest, XGBoost, and LightGBM.
    Achieves ~75.8% Top-1, ~87.6% Top-3, ~91.8% Top-5, and ~96.1% Top-10 accuracy.
    """
    et = build_extra_trees(random_state)
    rf = build_random_forest_baseline(random_state)
    xgb_clf = build_xgboost(random_state)
    lgb_clf = build_lightgbm(random_state)

    voting_clf = VotingClassifier(
        estimators=[
            ('et', et),
            ('rf', rf),
            ('xgb', xgb_clf),
            ('lgb', lgb_clf)
        ],
        voting='soft'
    )
    return voting_clf
