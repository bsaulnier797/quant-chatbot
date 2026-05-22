import xgboost as xgb
import sklearn.metrics as metrics
import pandas as pd
import numpy as np


def build_features(ticker: str, period: str = '2y'):
    """Pull feature matrix, target, and raw prices for a given ticker."""
    from ml.features import build_feature_matrix
    X, y, prices = build_feature_matrix(ticker, period)
    return X, y, prices


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_size: int = None,
    step_size: int = 21
) -> list[dict]:
    """
    Perform walk-forward validation on the dataset.

    Trains on all data before each window and tests on the next step_size days.
    Never shuffles -- time series order is always preserved.

    Args:
        X:                  feature DataFrame
        y:                  target Series
        initial_train_size: samples in the first training set. Defaults to
                            60% of the dataset so short-history tickers still
                            produce folds rather than returning an empty list.
        step_size:          samples per test window (default: ~1 month of trading days)

    Returns:
        List of dicts with accuracy, precision, recall, and f1 for each fold.
        Returns empty list only if dataset is too small to form a single fold.
    """
    # Adaptive initial training size: 60% of data, floored at 60 rows minimum
    if initial_train_size is None:
        initial_train_size = max(60, int(len(X) * 0.6))

    # Safety: if we can't even form one fold, return empty immediately
    if len(X) < initial_train_size + step_size:
        return []

    # Clean NaNs consistently with train_model
    X_clean = X.ffill().bfill()

    results = []

    for start in range(initial_train_size, len(X_clean), step_size):
        end = start + step_size
        if end > len(X_clean):
            break

        X_train, y_train = X_clean.iloc[:start], y.iloc[:start]
        X_test, y_test = X_clean.iloc[start:end], y.iloc[start:end]

        model = _make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'fold_start': start,
            'fold_end': end,
            'accuracy': metrics.accuracy_score(y_test, y_pred),
            'precision': metrics.precision_score(y_test, y_pred, zero_division=0),
            'recall': metrics.recall_score(y_test, y_pred, zero_division=0),
            'f1_score': metrics.f1_score(y_test, y_pred, zero_division=0),
        })

    return results


def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """
    Train a final XGBoost classifier on the full dataset.

    XGBoost is tree-based so it does not require feature scaling --
    it splits on thresholds, not distances.

    Returns:
        Fitted XGBClassifier
    """
    X_clean = X.ffill().bfill()

    model = _make_model()
    model.fit(X_clean, y)
    return model


def evaluate_model(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate a trained model on a held-out test set.

    Returns:
        Dict with accuracy, precision, recall, and f1
    """
    X_clean = X.ffill().bfill()
    y_pred = model.predict(X_clean)

    return {
        'accuracy': metrics.accuracy_score(y, y_pred),
        'precision': metrics.precision_score(y, y_pred, zero_division=0),
        'recall': metrics.recall_score(y, y_pred, zero_division=0),
        'f1_score': metrics.f1_score(y, y_pred, zero_division=0),
    }


def get_feature_importance(model: xgb.XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame of feature importances sorted descending.
    Used to power the feature importance bar chart in Streamlit.
    """
    return pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)


def predict_next_day(model: xgb.XGBClassifier, X: pd.DataFrame) -> dict:
    """
    Predict direction for the most recent trading day.

    Returns:
        Dict with signal ("Buy" or "Hold"), confidence %, and date
    """
    X_clean = X.ffill().bfill()
    last_row = X_clean.iloc[[-1]]  # keep as DataFrame for predict_proba

    prediction = model.predict(last_row)[0]
    confidence = model.predict_proba(last_row)[0][prediction]

    return {
        'signal': 'Buy' if prediction == 1 else 'Hold',
        'confidence': round(float(confidence) * 100, 1),
        'date': X.index[-1].strftime('%Y-%m-%d') if hasattr(X.index[-1], 'strftime') else str(X.index[-1]),
    }


def _make_model() -> xgb.XGBClassifier:
    """Internal helper to keep model hyperparameters consistent across training and validation."""
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
    )