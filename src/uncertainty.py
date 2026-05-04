"""
Uncertainty estimation using residual-based quantiles.
"""
import numpy as np
from xgboost import XGBRegressor
from .utils import save_model, compute_metrics, print_metrics


class QuantilePredictor:
    """Estimate uncertainty using residual-based quantiles."""
    
    def __init__(self):
        self.model_p50 = None  # Mean model
        self.quantile_factors = {}  # Store quantile multipliers
        self.residuals_train = None
    
    def train_quantile_models(self, X_train, X_test, y_train, y_test):
        """
        Train a mean model and estimate quantile factors from residuals.
        This is a practical approach when native quantile regression isn't available.
        """
        print("Training uncertainty model (P10, P50, P90)...")
        
        # Train main XGBoost model (P50)
        print(f"\n  Training P50 (mean model)...")
        self.model_p50 = XGBRegressor(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            verbosity=0
        )
        
        self.model_p50.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model_p50.predict(X_train)
        y_pred_test = self.model_p50.predict(X_test)
        
        metrics = compute_metrics(y_test, y_pred_test)
        print_metrics(metrics, "  P50 (Mean Model)")
        
        # Compute residuals on training set
        train_residuals = np.abs(y_train - y_pred_train)
        self.residuals_train = train_residuals
        
        # Estimate quantile factors based on residuals
        p10_factor = np.percentile(train_residuals, 10) / np.median(train_residuals)
        p90_factor = np.percentile(train_residuals, 90) / np.median(train_residuals)
        
        self.quantile_factors = {
            'p10_factor': p10_factor,
            'p90_factor': p90_factor,
            'median_residual': np.median(train_residuals)
        }
        
        print(f"\n  Uncertainty factors:")
        print(f"    P10 factor: {p10_factor:.3f}")
        print(f"    P90 factor: {p90_factor:.3f}")
        print(f"    Median residual: {np.median(train_residuals):.3f}")
        
        return [None, self.model_p50, None]  # Return for compatibility
    
    def predict_quantiles(self, X):
        """Generate P10, P50, P90 for samples."""
        # Handle NaN values (cold-start scenario with new plants)
        X_clean = self._handle_nans(X)
        
        p50 = self.model_p50.predict(X_clean)
        median_residual = self.quantile_factors['median_residual']
        p10_factor = self.quantile_factors['p10_factor']
        p90_factor = self.quantile_factors['p90_factor']
        
        p10 = np.maximum(0, p50 - p10_factor * median_residual)
        p90 = p50 + p90_factor * median_residual
        
        return p10, p50, p90
    
    def _handle_nans(self, X):
        """
        Replace NaN values (from cold-start missing lags) with zeros.
        
        Safe because lag_missing_flag tells model the context:
        - flag=1: these lags are missing (NaN→0), model should discount them
        - flag=0: these lags are real, including actual zero generation values
        """
        X_clean = np.array(X, copy=True, dtype=np.float32)
        nan_mask = np.isnan(X_clean)
        X_clean[nan_mask] = 0  # Replace NaN with 0 (context provided by missing_flag)
        
        return X_clean
    
    def save_all(self, base_path):
        """Save model and factors."""
        save_model(self.model_p50, f"{base_path}/xgboost_p50.pkl")
        save_model(self.quantile_factors, f"{base_path}/quantile_factors.pkl")
        print(f"Uncertainty models saved to {base_path}")
