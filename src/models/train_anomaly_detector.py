"""
Train the LSTM Anomaly Detection Model
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import pickle
from anomaly_detector import AnomalyDetectionModel


def load_data(data_dir='data/processed'):
    """Load preprocessed data"""
    data_path = Path(data_dir)
    
    train_df = pd.read_csv(data_path / 'train.csv')
    val_df = pd.read_csv(data_path / 'val.csv')
    
    # Load feature columns
    with open(data_path / 'feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    return train_df, val_df, feature_columns


def prepare_data_by_vehicle(df, feature_columns, sequence_length=50):
    """Prepare sequences grouped by vehicle"""
    all_sequences = []
    all_labels = []
    
    for vehicle_id in df['vehicle_id'].unique():
        vehicle_data = df[df['vehicle_id'] == vehicle_id]
        
        features = vehicle_data[feature_columns].values
        labels = vehicle_data['anomaly'].values
        
        # Create sequences for this vehicle
        for i in range(len(features) - sequence_length + 1):
            seq = features[i:i + sequence_length]
            label = labels[i + sequence_length - 1]
            
            all_sequences.append(seq)
            all_labels.append(label)
    
    return np.array(all_sequences), np.array(all_labels)


def train_model(epochs=20, batch_size=32, sequence_length=50):
    """Train the anomaly detection model"""
    print("="*60)
    print("TRAINING ANOMALY DETECTION MODEL")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df, val_df, feature_columns = load_data()
    print(f"✓ Loaded train: {len(train_df)} records, val: {len(val_df)} records")
    print(f"✓ Features: {len(feature_columns)}")
    
    # Prepare sequences
    print("\nPreparing sequences...")
    X_train, y_train = prepare_data_by_vehicle(train_df, feature_columns, sequence_length)
    X_val, y_val = prepare_data_by_vehicle(val_df, feature_columns, sequence_length)
    
    print(f"✓ Train sequences: {X_train.shape}")
    print(f"✓ Val sequences: {X_val.shape}")
    print(f"✓ Train anomaly rate: {y_train.mean():.2%}")
    print(f"✓ Val anomaly rate: {y_val.mean():.2%}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = len(feature_columns)
    model = AnomalyDetectionModel(input_size, sequence_length)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("-"*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader)
        val_loss, val_acc = model.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save('src/models/best_anomaly_detector.pth')
    
    print("-"*60)
    print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")
    print("="*60)
    
    return model


if __name__ == '__main__':
    model = train_model(epochs=20, batch_size=32, sequence_length=50)
