"""
Model training for renewable energy forecasting.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pickle
from .utils import save_model, load_model, compute_metrics, print_metrics


class ModelTrainer:
    """Train XGBoost global model for renewable energy forecasting."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.feature_importance = None
    
    def train_test_split_temporal(self, X, y, test_size=0.2):
        """Split data temporally (train on earlier data, test on later)."""
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data(self, X, y, test_size=0.2):
        """Prepare data for training."""
        print("Preparing data...")
        
        # Temporal split (respect time ordering)
        X_train, X_test, y_train, y_test = self.train_test_split_temporal(X, y, test_size)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"  Train set: {X_train_scaled.shape}")
        print(f"  Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost regressor."""
        print("Training XGBoost model...")
        
        self.model = XGBRegressor(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            early_stopping_rounds=10,
            objective='reg:squarederror',
            random_state=42,
            verbosity=1
        )
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        print("✓ Training complete")
        
        return self.model
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        """Evaluate model performance."""
        print("\nEvaluating model...")
        
        # Train metrics
        y_train_pred = self.model.predict(X_train)
        train_metrics = compute_metrics(y_train, y_train_pred)
        print_metrics(train_metrics, "Train")
        
        # Test metrics
        y_test_pred = self.model.predict(X_test)
        test_metrics = compute_metrics(y_test, y_test_pred)
        print_metrics(test_metrics, "Test")
        
        return train_metrics, test_metrics
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='r2'
        )
        
        print(f"  R² scores: {scores}")
        print(f"  Mean R²: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores
    
    def get_feature_importance(self, feature_names):
        """Extract and return feature importance."""
        importance = self.model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importance[indices]
        })
        
        self.feature_importance = importance_df
        
        print("\nTop 10 Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {i+1}. {row['feature']:<30} {row['importance']:.4f}")
        
        return importance_df
    
    def save_all(self, model_path, scaler_path):
        """Save model and scaler."""
        save_model(self.model, model_path)
        save_model(self.scaler, scaler_path)


if __name__ == "__main__":
    from data_generation import SyntheticDataGenerator
    from feature_engineering import FeatureEngineer
    
    print("Testing model training pipeline...")
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    generator = SyntheticDataGenerator()
    generator.generate_plant_metadata()
    data = generator.generate_dataset(days=365)
    
    # Engineer features
    print("\n2. Engineering features...")
    engineer = FeatureEngineer()
    df_engineered, feature_cols = engineer.prepare_features(data)
    X, y = engineer.get_X_y(df_engineered, feature_cols)
    
    # Train model
    print("\n3. Training model...")
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    trainer.train_xgboost(X_train, X_test, y_train, y_test)
    
    # Evaluate
    print("\n4. Evaluating model...")
    train_metrics, test_metrics = trainer.evaluate(X_train, X_test, y_train, y_test)
    
    # Feature importance
    print("\n5. Computing feature importance...")
    importance_df = trainer.get_feature_importance(feature_cols)
    
    print("\n✅ Model training test complete!")
