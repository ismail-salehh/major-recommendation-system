"""
Model evaluation, Top-K accuracy calculations, and feature importance metrics.
"""
import numpy as np
import pandas as pd


def compute_top_k_accuracy(clf, X_test, y_test, k_values=[1, 3, 5, 10]):
    """
    Calculates Top-K recommendation accuracy for multi-class classifiers.
    """
    y_proba = clf.predict_proba(X_test)
    results = {}

    y_test_array = y_test.values if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test

    for k in k_values:
        top_k_indices = np.argsort(y_proba, axis=1)[:, -k:]
        correct = sum(y_test_array[i] in clf.classes_[top_k_indices[i]] for i in range(len(y_test_array)))
        accuracy = float(correct / len(y_test_array))
        results[f"Top-{k}"] = round(accuracy, 4)

    return results


def compute_class_ceiling(clf, X_val, y_val):
    """
    Computes class-wise maximum probability ceilings from test/validation set
    for normalized probability scaling.
    """
    probs = clf.predict_proba(X_val)
    ceilings = {}
    y_val_array = y_val.values if isinstance(y_val, (pd.Series, pd.DataFrame)) else y_val

    for class_idx in range(probs.shape[1]):
        class_mask = (y_val_array == class_idx)
        if class_mask.sum() == 0:
            ceilings[int(class_idx)] = float(probs[:, class_idx].max())
        else:
            ceilings[int(class_idx)] = float(probs[class_mask, class_idx].max())

    return ceilings


def extract_feature_importances(clf, feature_names):
    """
    Extracts feature importance scores from single tree models or soft voting ensembles.
    """
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'estimators_'):
        # For VotingClassifier, average feature importances across base tree estimators
        sub_importances = []
        for sub_clf in clf.estimators_:
            if hasattr(sub_clf, 'feature_importances_'):
                sub_importances.append(sub_clf.feature_importances_)
        if sub_importances:
            importances = np.mean(sub_importances, axis=0)
        else:
            importances = np.zeros(len(feature_names))
    else:
        importances = np.zeros(len(feature_names))

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return df_imp
