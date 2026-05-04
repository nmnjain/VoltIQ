"""
Feature engineering pipeline for renewable energy forecasting.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle


class FeatureEngineer:
    """Engineer features for renewable energy forecasting model."""
    
    def __init__(self):
        self.scaler = None
        self.categorical_encoder = None
        self.feature_names = None
        self.plant_ids = None
    
    def engineer_temporal_features(self, df):
        """Create temporal cyclic features."""
        df = df.copy()
        
        # Extract hour of day and month
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Cyclic encoding (sine/cosine)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def engineer_lag_features(self, df, lags=[1, 3, 24]):
        """Create lagged generation features."""
        df = df.copy()
        
        for lag in lags:
            df[f'lag_{lag}h'] = df.groupby('plant_id')['actual_generation_mw'].shift(lag)
        
        # Forward fill and backward fill missing values (first entries)
        df = df.bfill().ffill()
        
        return df
    
    def engineer_rolling_features(self, df, window=3):
        """Create rolling window features."""
        df = df.copy()
        
        df[f'rolling_mean_{window}h'] = (
            df.groupby('plant_id')['actual_generation_mw']
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        
        return df
    
    def engineer_derived_features(self, df):
        """Create derived weather features."""
        df = df.copy()
        
        # Solar effect: interactions between irradiance and cloud cover
        df['solar_effect'] = df['irradiance'] * (1 - df['cloud_cover'])
        
        # Wind effect: interaction with capacity
        df['wind_effect'] = df['wind_speed'] * df['capacity_mw']
        
        return df
    
    def engineer_plant_features(self, df):
        """Encode plant type as categorical."""
        df = df.copy()
        
        # One-hot encode plant type
        plant_type_dummies = pd.get_dummies(df['plant_type'], prefix='plant_type', drop_first=False)
        df = pd.concat([df, plant_type_dummies], axis=1)
        
        return df
    
    def prepare_features(self, df):
        """Execute full feature engineering pipeline."""
        print("Engineering features...")
        
        # 1. Temporal features
        df = self.engineer_temporal_features(df)
        print("  ✓ Temporal features")
        
        # 2. Lag features
        df = self.engineer_lag_features(df)
        print("  ✓ Lag features")
        
        # 3. Rolling features
        df = self.engineer_rolling_features(df)
        print("  ✓ Rolling features")
        
        # 4. Derived features
        df = self.engineer_derived_features(df)
        print("  ✓ Derived features")
        
        # 5. Plant features
        df = self.engineer_plant_features(df)
        print("  ✓ Plant features")
        
        # Select final feature columns (exclude metadata and target)
        exclude_cols = [
            'timestamp', 'plant_id', 'actual_generation_mw', 
            'hour', 'month', 'plant_type'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        self.plant_ids = df['plant_id'].unique()
        
        print(f"  → {len(feature_cols)} features generated")
        
        return df, feature_cols
    
    def get_X_y(self, df, feature_cols):
        """Extract features (X) and target (y) from dataframe."""
        X = df[feature_cols].values
        y = df['actual_generation_mw'].values
        
        return X, y
    
    def save_metadata(self, filepath):
        """Save feature engineering metadata."""
        metadata = {
            'feature_names': self.feature_names,
            'plant_ids': self.plant_ids
        }
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Feature metadata saved to {filepath}")


if __name__ == "__main__":
    from data_generation import SyntheticDataGenerator
    
    print("Testing feature engineering pipeline...")
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    generator = SyntheticDataGenerator()
    generator.generate_plant_metadata()
    data = generator.generate_dataset(days=7)  # Quick test with 1 week
    
    # Engineer features
    print("\n2. Engineering features...")
    engineer = FeatureEngineer()
    df_engineered, feature_cols = engineer.prepare_features(data)
    
    print(f"\n3. Feature columns: {len(feature_cols)}")
    print(f"   {feature_cols[:10]}...")  # Print first 10
    
    print(f"\n4. Dataset shape: {df_engineered.shape}")
    print(f"   No NaN values: {df_engineered[feature_cols].isna().sum().sum() == 0}")
    
    print("\n✅ Feature engineering test complete!")
