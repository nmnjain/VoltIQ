"""
Utility functions for renewable energy forecasting prototype.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import os


def create_feature_scaler(X_train):
    """Create and fit a StandardScaler for features."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_model(model, filepath):
    """Save a model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a model from disk."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, and R² metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def print_metrics(metrics, model_name=""):
    """Pretty print metrics."""
    print(f"\n{model_name} Metrics:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  R²:   {metrics['R2']:.4f}")
