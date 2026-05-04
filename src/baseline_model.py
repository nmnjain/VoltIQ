"""
Baseline models for comparison.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .utils import compute_metrics, print_metrics


class BaselineModels:
    """Implement baseline models for performance comparison."""
    
    def __init__(self):
        self.persistence_metrics = None
        self.linear_metrics = None
    
    def persistence_baseline(self, y_test, y_test_lag24):
        """Persistence baseline: y(t) = y(t-24h)."""
        print("Computing persistence baseline (y(t) = y(t-24))...")
        
        self.persistence_metrics = compute_metrics(y_test, y_test_lag24)
        print_metrics(self.persistence_metrics, "Persistence Baseline")
        
        return self.persistence_metrics
    
    def linear_baseline(self, X_train, X_test, y_train, y_test):
        """Simple linear regression baseline."""
        print("Training linear regression baseline...")
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        
        self.linear_metrics = compute_metrics(y_test, y_pred)
        print_metrics(self.linear_metrics, "Linear Regression Baseline")
        
        return self.linear_metrics, lr
    
    def comparison_table(self, xgboost_metrics):
        """Create comparison table of all models."""
        comparison = pd.DataFrame({
            'Model': ['Persistence (t-24)', 'Linear Regression', 'XGBoost'],
            'MAE': [
                self.persistence_metrics['MAE'],
                self.linear_metrics['MAE'],
                xgboost_metrics['MAE']
            ],
            'RMSE': [
                self.persistence_metrics['RMSE'],
                self.linear_metrics['RMSE'],
                xgboost_metrics['RMSE']
            ],
            'R²': [
                self.persistence_metrics['R2'],
                self.linear_metrics['R2'],
                xgboost_metrics['R2']
            ]
        })
        
        print("\n" + "="*60)
        print("MODEL COMPARISON TABLE")
        print("="*60)
        print(comparison.to_string(index=False))
        print("="*60)
        
        # Calculate improvement
        persistence_r2 = self.persistence_metrics['R2']
        xgboost_r2 = xgboost_metrics['R2']
        improvement = ((xgboost_r2 - persistence_r2) / persistence_r2) * 100
        
        print(f"\nXGBoost vs Persistence improvement: +{improvement:.1f}% (R²)")
        
        return comparison
