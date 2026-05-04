"""
Synthetic data generation for renewable energy forecasting.
Generates realistic multi-plant renewable generation dataset with weather data.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class SyntheticDataGenerator:
    """Generate synthetic renewable energy data with realistic patterns."""
    
    def __init__(self, seed=42, realism_level='high'):
        """
        realism_level: 'clean' (original), 'medium', 'high' (realistic)
        - clean: R² ~0.98 (simple physics)
        - medium: R² ~0.90 (moderate noise + some events)
        - high: R² ~0.85 (correlated noise, outages, curtailment, extremes)
        """
        self.seed = seed
        self.realism_level = realism_level
        np.random.seed(seed)
        self.plants = None
        self.data = None
        
        # For correlated noise (AR(1) process)
        self.prev_cloud = {}
        self.prev_wind = {}
    
    def generate_plant_metadata(self, n_solar=3, n_wind=2):
        """Create plant metadata with realistic attributes."""
        plants = []
        
        # Solar plants
        for i in range(n_solar):
            plants.append({
                'plant_id': f'SOLAR_{i+1:03d}',
                'plant_type': 'solar',
                'capacity_mw': np.random.uniform(5, 30),
                'latitude': np.random.uniform(20, 40),
                'longitude': np.random.uniform(-100, -80)
            })
        
        # Wind plants
        for i in range(n_wind):
            plants.append({
                'plant_id': f'WIND_{i+1:03d}',
                'plant_type': 'wind',
                'capacity_mw': np.random.uniform(10, 50),
                'latitude': np.random.uniform(30, 45),
                'longitude': np.random.uniform(-110, -90)
            })
        
        self.plants = pd.DataFrame(plants)
        return self.plants
    
    def generate_weather(self, date, hour, plant_id, plant_type, base_seed=None):
        """Generate realistic weather patterns with CORRELATED noise (not IID)."""
        if base_seed is not None:
            np.random.seed(base_seed)
        
        # Hour of year (0-8759)
        day_of_year = date.timetuple().tm_yday
        hour_of_year = (day_of_year - 1) * 24 + hour
        
        # ============================================================
        # CLOUD COVER: AR(1) CORRELATED NOISE (persists over hours)
        # ============================================================
        cloud_base = np.abs(np.sin(2 * np.pi * hour_of_year / 8760) * 0.2) + 0.3
        
        # Initialize or continue AR(1) process
        if plant_id not in self.prev_cloud:
            self.prev_cloud[plant_id] = cloud_base
        
        if self.realism_level in ['medium', 'high']:
            # AR(1): cloud_t = 0.75 * cloud_t-1 + noise (persists!)
            cloud_cover = 0.75 * self.prev_cloud[plant_id] + 0.25 * cloud_base
            cloud_noise_std = 0.20 if self.realism_level == 'high' else 0.15
            cloud_cover += np.random.normal(0, cloud_noise_std)
            self.prev_cloud[plant_id] = cloud_cover
        else:
            # Original: IID noise
            cloud_cover = cloud_base + np.random.normal(0, 0.15)
        
        cloud_cover = np.clip(cloud_cover, 0, 1)
        
        # ============================================================
        # IRRADIANCE with nonlinear distortions
        # ============================================================
        hour_normalized = (hour + 12) % 24
        irradiance_base = max(0, 800 * np.sin(np.pi * hour_normalized / 24))
        irradiance = irradiance_base * (1 - 0.8 * cloud_cover)
        
        # Higher noise in high-realism mode
        if self.realism_level == 'high':
            noise_std = 40  # Was 20 → doubled
            # Add nonlinear distortion (breaks perfect physics)
            irradiance *= (1 + np.random.normal(0, 0.08))  # ±8% multiplicative
        else:
            noise_std = 20
        
        irradiance += np.random.normal(0, noise_std)
        irradiance = np.clip(irradiance, 0, 1000)
        
        # ============================================================
        # TEMPERATURE
        # ============================================================
        seasonal_temp = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 1) / 365)
        daily_temp = 5 * np.sin(np.pi * (hour_normalized - 6) / 12)
        temp_noise_std = 3 if self.realism_level == 'high' else 2
        temperature = seasonal_temp + daily_temp + np.random.normal(0, temp_noise_std)
        temperature = np.clip(temperature, 10, 40)
        
        # ============================================================
        # WIND SPEED: AR(1) CORRELATED + GUST SPIKES
        # ============================================================
        wind_base = np.abs(np.random.lognormal(mean=1.2, sigma=0.8))
        wind_base += 0.3 * np.sin(2 * np.pi * hour_of_year / 168)  # Weekly cycle
        
        if plant_id not in self.prev_wind:
            self.prev_wind[plant_id] = wind_base
        
        if self.realism_level in ['medium', 'high']:
            # AR(1): wind persists
            wind_speed = 0.70 * self.prev_wind[plant_id] + 0.30 * wind_base
            self.prev_wind[plant_id] = wind_speed
        else:
            wind_speed = wind_base
        
        # Extreme gust events (~1-2% of time)
        if self.realism_level == 'high' and np.random.random() < 0.02:
            wind_speed *= np.random.uniform(1.5, 2.5)  # Sudden gust spike
        else:
            wind_speed += np.random.normal(0, 1)  # Normal variation
        
        wind_speed = np.clip(wind_speed, 0, 15)
        
        return {
            'cloud_cover': cloud_cover,
            'irradiance': irradiance,
            'temperature': temperature,
            'wind_speed': wind_speed
        }
    
    def simulate_generation(self, weather, plant_type, capacity_mw):
        """Simulate generation with realistic imperfections."""
        if plant_type == 'solar':
            # Base solar physics
            generation = (
                capacity_mw * 
                (weather['irradiance'] / 1000) * 
                (1 - 0.8 * weather['cloud_cover'])
            )
            
            # Noise magnitude depends on realism level
            if self.realism_level == 'high':
                noise = np.random.normal(0, capacity_mw * 0.10)  # 10% noise (was 2%)
            elif self.realism_level == 'medium':
                noise = np.random.normal(0, capacity_mw * 0.05)  # 5% noise
            else:
                noise = np.random.normal(0, capacity_mw * 0.02)  # 2% noise
            
            generation = np.clip(generation + noise, 0, capacity_mw)
        
        elif plant_type == 'wind':
            # Base wind physics
            wind_factor = min(1.0, (weather['wind_speed'] / 12) ** 3)
            generation = capacity_mw * wind_factor
            
            # Higher noise for wind
            if self.realism_level == 'high':
                noise = np.random.normal(0, capacity_mw * 0.12)  # 12% noise (was 5%)
            elif self.realism_level == 'medium':
                noise = np.random.normal(0, capacity_mw * 0.08)  # 8% noise
            else:
                noise = np.random.normal(0, capacity_mw * 0.05)  # 5% noise
            
            generation = np.clip(generation + noise, 0, capacity_mw)
        
        else:
            raise ValueError(f"Unknown plant type: {plant_type}")
        
        # ============================================================
        # REALISTIC IMPERFECTIONS (high realism mode)
        # ============================================================
        if self.realism_level == 'high':
            # 1. EFFICIENCY DROP (equipment aging, dust, etc.)
            # Random efficiency 85-100% of theoretical
            efficiency = np.random.uniform(0.85, 1.0)
            generation *= efficiency
            
            # 2. OUTAGES (1-2% of time, equipment fails)
            if np.random.random() < 0.01:  # 1% outage rate
                generation = 0
            
            # 3. CURTAILMENT (grid limit, 2-3% of time)
            elif np.random.random() < 0.02:  # 2% curtailment rate
                curtailment_factor = np.random.uniform(0.5, 0.95)  # Cap at 50-95%
                generation *= curtailment_factor
        
        return np.clip(generation, 0, capacity_mw)
    
    def generate_dataset(self, days=365, start_date=None):
        """Generate full synthetic dataset."""
        if start_date is None:
            start_date = datetime(2023, 1, 1)
        
        if self.plants is None:
            self.generate_plant_metadata()
        
        records = []
        
        for day_offset in range(days):
            date = start_date + timedelta(days=day_offset)
            
            for hour in range(24):
                for _, plant in self.plants.iterrows():
                    # Generate weather
                    weather = self.generate_weather(
                        date, hour, plant['plant_id'], plant['plant_type'],
                        base_seed=self.seed + day_offset * 1000 + hour
                    )
                    
                    # Simulate generation
                    generation = self.simulate_generation(
                        weather,
                        plant['plant_type'],
                        plant['capacity_mw']
                    )
                    
                    timestamp = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=hour)
                    
                    records.append({
                        'timestamp': timestamp,
                        'plant_id': plant['plant_id'],
                        'plant_type': plant['plant_type'],
                        'capacity_mw': plant['capacity_mw'],
                        'cloud_cover': weather['cloud_cover'],
                        'irradiance': weather['irradiance'],
                        'temperature': weather['temperature'],
                        'wind_speed': weather['wind_speed'],
                        'actual_generation_mw': generation
                    })
        
        self.data = pd.DataFrame(records)
        return self.data
    
    def save_dataset(self, filepath):
        """Save dataset to CSV."""
        if self.data is None:
            raise ValueError("No dataset generated. Call generate_dataset() first.")
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        self.data.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return self.data


# Main execution
if __name__ == "__main__":
    print("Generating synthetic renewable energy dataset...")
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate plant metadata
    print("\n1. Generating plant metadata...")
    plants = generator.generate_plant_metadata(n_solar=3, n_wind=2)
    print(f"Generated {len(plants)} plants:")
    print(plants)
    
    # Generate full dataset
    print("\n2. Generating 365 days of hourly data...")
    data = generator.generate_dataset(days=365)
    print(f"Generated {len(data)} records")
    
    # Save dataset
    print("\n3. Saving dataset...")
    output_path = "data/synthetic/raw_data_with_weather.csv"
    generator.save_dataset(output_path)
    
    # Print data validation
    print("\n4. Data Validation:")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Plants: {data['plant_id'].unique().tolist()}")
    print(f"\nGeneration statistics (MW):")
    print(data['actual_generation_mw'].describe())
    
    print(f"\nCloud cover statistics:")
    print(data['cloud_cover'].describe())
    
    # Check correlations for solar plants
    solar_data = data[data['plant_type'] == 'solar']
    if len(solar_data) > 0:
        solar_corr = solar_data[['cloud_cover', 'actual_generation_mw']].corr().iloc[0, 1]
        print(f"\nSolar: Cloud Cover vs Generation correlation: {solar_corr:.3f}")
    
    # Check wind data variability
    wind_data = data[data['plant_type'] == 'wind']
    if len(wind_data) > 0:
        wind_std = wind_data['actual_generation_mw'].std()
        wind_mean = wind_data['actual_generation_mw'].mean()
        print(f"Wind: Mean {wind_mean:.2f} MW, Std {wind_std:.2f} MW (CV: {wind_std/wind_mean:.2f})")
    
    print("\n✅ Synthetic data generation complete!")
