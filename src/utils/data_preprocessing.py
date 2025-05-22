"""
Data preprocessing and feature engineering for vehicle sensor data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle


class VehicleDataPreprocessor:
    """Preprocess and engineer features from vehicle sensor data"""
    
    def __init__(self, data_path='data/raw/vehicle_sensor_data.csv'):
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'anomaly'
        
    def load_data(self):
        """Load raw sensor data"""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df)} records for {df['vehicle_id'].nunique()} vehicles")
        return df
    
    def clean_data(self, df):
        """Clean and filter noisy data"""
        print("Cleaning data...")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Remove outliers using IQR method for key sensors
        sensor_cols = [col for col in df.columns if col not in ['vehicle_id', 'timestamp', 'anomaly']]
        
        for col in sensor_cols:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        print(f"✓ Cleaned data: {len(df)} records remaining")
        return df
    
    def apply_moving_average(self, df, window=5):
        """Apply moving average filter to reduce noise"""
        print(f"Applying moving average filter (window={window})...")
        
        sensor_cols = [col for col in df.columns if col not in ['vehicle_id', 'timestamp', 'anomaly']]
        
        # Group by vehicle and apply rolling average
        for col in sensor_cols:
            df[f'{col}_ma'] = df.groupby('vehicle_id')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        print(f"✓ Applied moving average to {len(sensor_cols)} sensors")
        return df
    
    def engineer_features(self, df):
        """Create domain-specific features"""
        print("Engineering features...")
        
        # Rate of change features
        sensor_cols = [col for col in df.columns if col not in ['vehicle_id', 'timestamp', 'anomaly'] and not col.endswith('_ma')]
        
        for col in sensor_cols:
            # Rate of change
            df[f'{col}_rate'] = df.groupby('vehicle_id')[col].diff()
            
            # Rolling statistics
            df[f'{col}_std'] = df.groupby('vehicle_id')[col].transform(
                lambda x: x.rolling(window=10, min_periods=1).std()
            )
        
        # Domain-specific features
        # Temperature differential
        df['temp_differential'] = df['engine_temp'] - df['coolant_temp']
        
        # Tire pressure imbalance
        df['tire_pressure_imbalance'] = df[['tire_pressure_fl', 'tire_pressure_fr', 
                                             'tire_pressure_rl', 'tire_pressure_rr']].std(axis=1)
        
        # Engine stress indicator
        df['engine_stress'] = (df['rpm'] / 1000) * (df['engine_temp'] / 100)
        
        # Battery health indicator
        df['battery_health'] = df['battery_voltage'] / 12.6  # Normalized to ideal voltage
        
        # Fill NaN values created by diff and rolling operations
        df = df.fillna(0)
        
        print(f"✓ Engineered features: {df.shape[1]} total columns")
        return df
    
    def normalize_features(self, df, fit=True):
        """Normalize sensor values"""
        print("Normalizing features...")
        
        # Select feature columns (exclude metadata and target)
        exclude_cols = ['vehicle_id', 'timestamp', 'anomaly']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        if fit:
            df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        else:
            df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        print(f"✓ Normalized {len(self.feature_columns)} features")
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        print("Splitting data...")
        
        # Split by vehicle to avoid data leakage
        vehicle_ids = df['vehicle_id'].unique()
        
        # First split: train+val vs test
        train_val_ids, test_ids = train_test_split(
            vehicle_ids, test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_size/(1-test_size), random_state=42
        )
        
        train_df = df[df['vehicle_id'].isin(train_ids)]
        val_df = df[df['vehicle_id'].isin(val_ids)]
        test_df = df[df['vehicle_id'].isin(test_ids)]
        
        print(f"✓ Train: {len(train_df)} records ({len(train_ids)} vehicles)")
        print(f"✓ Val: {len(val_df)} records ({len(val_ids)} vehicles)")
        print(f"✓ Test: {len(test_df)} records ({len(test_ids)} vehicles)")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df, val_df, test_df, output_dir='data/processed'):
        """Save processed datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving processed data to {output_path}...")
        
        train_df.to_csv(output_path / 'train.csv', index=False)
        val_df.to_csv(output_path / 'val.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)
        
        # Save scaler
        with open(output_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature columns
        with open(output_path / 'feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print("✓ Saved all processed datasets and preprocessing artifacts")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"Train anomaly rate: {train_df['anomaly'].mean():.2%}")
        print(f"Val anomaly rate: {val_df['anomaly'].mean():.2%}")
        print(f"Test anomaly rate: {test_df['anomaly'].mean():.2%}")
    
    def preprocess_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("="*60)
        print("VEHICLE DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Clean data
        df = self.clean_data(df)
        
        # Apply filters
        df = self.apply_moving_average(df, window=5)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Normalize features
        df = self.normalize_features(df, fit=True)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Save processed data
        self.save_processed_data(train_df, val_df, test_df)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)
        
        return train_df, val_df, test_df


if __name__ == '__main__':
    preprocessor = VehicleDataPreprocessor()
    train_df, val_df, test_df = preprocessor.preprocess_pipeline()
