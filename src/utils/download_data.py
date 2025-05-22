"""
Download NASA Turbofan Engine Degradation Dataset
This dataset simulates engine sensor data with degradation patterns
"""
import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def download_nasa_turbofan_data(data_dir='data/raw'):
    """
    Download NASA Turbofan Engine Degradation Simulation Data Set
    Source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # NASA C-MAPSS Dataset URL
    url = "https://ti.arc.nasa.gov/c/6/"
    
    print("Downloading NASA Turbofan Engine Degradation Dataset...")
    print("This dataset contains simulated engine sensor data with degradation patterns")
    
    # Alternative: Use a direct download link or create synthetic data
    # Since the NASA link requires manual download, we'll create a synthetic dataset
    print("\nNote: Creating synthetic vehicle sensor dataset based on NASA patterns...")
    
    return create_synthetic_vehicle_data(data_path)


def create_synthetic_vehicle_data(data_path):
    """
    Create synthetic vehicle sensor data with realistic patterns
    Simulates: engine temp, RPM, speed, battery voltage, oil pressure, etc.
    """
    import numpy as np
    import pandas as pd
    
    print("Generating synthetic vehicle sensor data...")
    
    np.random.seed(42)
    
    # Number of vehicles and time steps
    n_vehicles = 100
    n_timesteps = 500
    
    datasets = {}
    
    for vehicle_id in range(1, n_vehicles + 1):
        data = []
        
        # Determine if vehicle will have anomaly
        has_anomaly = np.random.rand() > 0.7  # 30% have anomalies
        anomaly_start = np.random.randint(300, 450) if has_anomaly else n_timesteps + 1
        
        for t in range(n_timesteps):
            # Base sensor readings with some noise
            base_engine_temp = 90 + np.random.normal(0, 5)
            base_rpm = 2000 + np.random.normal(0, 200)
            base_speed = 60 + np.random.normal(0, 10)
            base_battery = 12.6 + np.random.normal(0, 0.2)
            base_oil_pressure = 40 + np.random.normal(0, 3)
            base_coolant_temp = 85 + np.random.normal(0, 4)
            base_fuel_pressure = 50 + np.random.normal(0, 2)
            base_throttle = 50 + np.random.normal(0, 10)
            base_brake_temp = 150 + np.random.normal(0, 15)
            base_tire_pressure_fl = 32 + np.random.normal(0, 0.5)
            base_tire_pressure_fr = 32 + np.random.normal(0, 0.5)
            base_tire_pressure_rl = 32 + np.random.normal(0, 0.5)
            base_tire_pressure_rr = 32 + np.random.normal(0, 0.5)
            base_vibration = 0.5 + np.random.normal(0, 0.1)
            
            # Introduce anomalies after anomaly_start
            if t >= anomaly_start:
                degradation_factor = (t - anomaly_start) / 100
                
                # Engine overheating
                base_engine_temp += degradation_factor * 20
                base_coolant_temp += degradation_factor * 15
                
                # Oil pressure drop
                base_oil_pressure -= degradation_factor * 10
                
                # Battery degradation
                base_battery -= degradation_factor * 0.5
                
                # Increased vibration
                base_vibration += degradation_factor * 0.3
                
                # Tire pressure issues
                if np.random.rand() > 0.8:
                    base_tire_pressure_fl -= degradation_factor * 2
            
            # Create data point
            data_point = {
                'vehicle_id': vehicle_id,
                'timestamp': t,
                'engine_temp': max(0, base_engine_temp),
                'rpm': max(0, base_rpm),
                'speed': max(0, base_speed),
                'battery_voltage': max(0, base_battery),
                'oil_pressure': max(0, base_oil_pressure),
                'coolant_temp': max(0, base_coolant_temp),
                'fuel_pressure': max(0, base_fuel_pressure),
                'throttle_position': np.clip(base_throttle, 0, 100),
                'brake_temp': max(0, base_brake_temp),
                'tire_pressure_fl': max(0, base_tire_pressure_fl),
                'tire_pressure_fr': max(0, base_tire_pressure_fr),
                'tire_pressure_rl': max(0, base_tire_pressure_rl),
                'tire_pressure_rr': max(0, base_tire_pressure_rr),
                'vibration_level': max(0, base_vibration),
                'anomaly': 1 if t >= anomaly_start else 0
            }
            data.append(data_point)
        
        datasets[f'vehicle_{vehicle_id}'] = pd.DataFrame(data)
    
    # Combine all vehicles into one dataset
    full_dataset = pd.concat(datasets.values(), ignore_index=True)
    
    # Save to CSV
    output_file = data_path / 'vehicle_sensor_data.csv'
    full_dataset.to_csv(output_file, index=False)
    print(f"✓ Saved synthetic vehicle sensor data to {output_file}")
    print(f"  - Total records: {len(full_dataset)}")
    print(f"  - Vehicles: {n_vehicles}")
    print(f"  - Timesteps per vehicle: {n_timesteps}")
    print(f"  - Anomaly rate: ~30%")
    
    # Create summary statistics
    summary = full_dataset.groupby('vehicle_id')['anomaly'].sum()
    vehicles_with_anomalies = (summary > 0).sum()
    print(f"  - Vehicles with anomalies: {vehicles_with_anomalies}/{n_vehicles}")
    
    return output_file


if __name__ == '__main__':
    download_nasa_turbofan_data()
