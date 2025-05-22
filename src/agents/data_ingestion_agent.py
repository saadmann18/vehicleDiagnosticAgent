"""
Data Ingestion Agent - Loads and prepares sensor data for analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Optional


class DataIngestionAgent:
    """
    Agent responsible for loading and preparing vehicle sensor data
    """
    
    def __init__(self, data_dir='data/processed'):
        self.data_dir = Path(data_dir)
        self.scaler = None
        self.feature_columns = None
        self._load_preprocessing_artifacts()
    
    def _load_preprocessing_artifacts(self):
        """Load scaler and feature columns"""
        scaler_path = self.data_dir / 'scaler.pkl'
        features_path = self.data_dir / 'feature_columns.pkl'
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        if features_path.exists():
            with open(features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test dataset"""
        test_path = self.data_dir / 'test.csv'
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")
        
        df = pd.read_csv(test_path)
        return df
    
    def get_vehicle_data(self, vehicle_id: int, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get sensor data for a specific vehicle
        
        Args:
            vehicle_id: ID of the vehicle
            df: Optional dataframe to filter from, otherwise loads test data
            
        Returns:
            DataFrame with vehicle sensor data
        """
        if df is None:
            df = self.load_test_data()
        
        vehicle_data = df[df['vehicle_id'] == vehicle_id].copy()
        
        if len(vehicle_data) == 0:
            raise ValueError(f"No data found for vehicle_id {vehicle_id}")
        
        return vehicle_data
    
    def get_latest_readings(self, vehicle_id: int, n_readings: int = 50) -> pd.DataFrame:
        """
        Get the latest N sensor readings for a vehicle
        
        Args:
            vehicle_id: ID of the vehicle
            n_readings: Number of recent readings to retrieve
            
        Returns:
            DataFrame with latest sensor readings
        """
        vehicle_data = self.get_vehicle_data(vehicle_id)
        latest_data = vehicle_data.tail(n_readings)
        return latest_data
    
    def prepare_for_analysis(self, vehicle_data: pd.DataFrame) -> Dict:
        """
        Prepare vehicle data for downstream agents
        
        Args:
            vehicle_data: Raw vehicle sensor data
            
        Returns:
            Dictionary containing prepared data and metadata
        """
        vehicle_id = vehicle_data['vehicle_id'].iloc[0]
        
        # Extract features
        if self.feature_columns:
            features = vehicle_data[self.feature_columns].values
        else:
            # Fallback: use all numeric columns except metadata
            exclude_cols = ['vehicle_id', 'timestamp', 'anomaly']
            feature_cols = [col for col in vehicle_data.columns if col not in exclude_cols]
            features = vehicle_data[feature_cols].values
        
        # Get ground truth if available
        ground_truth = vehicle_data['anomaly'].values if 'anomaly' in vehicle_data.columns else None
        
        prepared_data = {
            'vehicle_id': vehicle_id,
            'features': features,
            'feature_names': self.feature_columns if self.feature_columns else feature_cols,
            'timestamps': vehicle_data['timestamp'].values,
            'raw_data': vehicle_data,
            'ground_truth': ground_truth,
            'num_readings': len(vehicle_data),
            'time_range': (vehicle_data['timestamp'].min(), vehicle_data['timestamp'].max())
        }
        
        return prepared_data
    
    def get_sensor_summary(self, vehicle_data: pd.DataFrame) -> Dict:
        """
        Get summary statistics for sensor readings
        
        Args:
            vehicle_data: Vehicle sensor data
            
        Returns:
            Dictionary with sensor statistics
        """
        sensor_cols = [col for col in vehicle_data.columns 
                      if col not in ['vehicle_id', 'timestamp', 'anomaly']]
        
        summary = {}
        for col in sensor_cols:
            summary[col] = {
                'mean': float(vehicle_data[col].mean()),
                'std': float(vehicle_data[col].std()),
                'min': float(vehicle_data[col].min()),
                'max': float(vehicle_data[col].max()),
                'latest': float(vehicle_data[col].iloc[-1])
            }
        
        return summary
    
    def run(self, vehicle_id: int, n_readings: Optional[int] = None) -> Dict:
        """
        Main execution method for the Data Ingestion Agent
        
        Args:
            vehicle_id: ID of the vehicle to analyze
            n_readings: Optional number of recent readings to analyze
            
        Returns:
            Dictionary containing prepared data for downstream agents
        """
        print(f"\n{'='*60}")
        print(f"DATA INGESTION AGENT - Vehicle {vehicle_id}")
        print(f"{'='*60}")
        
        # Load vehicle data
        if n_readings:
            vehicle_data = self.get_latest_readings(vehicle_id, n_readings)
            print(f"✓ Loaded latest {n_readings} readings for vehicle {vehicle_id}")
        else:
            vehicle_data = self.get_vehicle_data(vehicle_id)
            print(f"✓ Loaded all {len(vehicle_data)} readings for vehicle {vehicle_id}")
        
        # Prepare data for analysis
        prepared_data = self.prepare_for_analysis(vehicle_data)
        print(f"✓ Prepared {prepared_data['num_readings']} readings for analysis")
        print(f"  Time range: {prepared_data['time_range'][0]} to {prepared_data['time_range'][1]}")
        print(f"  Features: {len(prepared_data['feature_names'])}")
        
        # Get sensor summary
        sensor_summary = self.get_sensor_summary(vehicle_data)
        prepared_data['sensor_summary'] = sensor_summary
        
        print(f"✓ Generated sensor summary statistics")
        print(f"{'='*60}\n")
        
        return prepared_data


if __name__ == '__main__':
    # Test the Data Ingestion Agent
    agent = DataIngestionAgent()
    
    # Test with a vehicle from test set
    test_df = agent.load_test_data()
    test_vehicle_id = test_df['vehicle_id'].iloc[0]
    
    result = agent.run(test_vehicle_id, n_readings=100)
    
    print("\nSample sensor summary:")
    for sensor, stats in list(result['sensor_summary'].items())[:3]:
        print(f"  {sensor}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
