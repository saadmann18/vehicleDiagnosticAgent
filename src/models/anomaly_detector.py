"""
Anomaly Detection Model using LSTM Neural Network
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pickle


class LSTMAnomalyDetector(nn.Module):
    """
    LSTM-based anomaly detection model for time-series sensor data
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMAnomalyDetector, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class AnomalyDetectionModel:
    """
    Wrapper class for anomaly detection model with training and inference
    """
    
    def __init__(self, input_size, sequence_length=50, device=None):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = LSTMAnomalyDetector(input_size).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"Initialized Anomaly Detection Model on {self.device}")
    
    def create_sequences(self, data, labels=None):
        """
        Create sequences for LSTM input
        
        Args:
            data: numpy array of shape (n_samples, n_features)
            labels: optional numpy array of labels
            
        Returns:
            Sequences and labels (if provided)
        """
        sequences = []
        seq_labels = []
        
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            if labels is not None:
                # Label is 1 if any point in sequence is anomalous
                label = labels[i + self.sequence_length - 1]
                seq_labels.append(label)
        
        sequences = np.array(sequences)
        
        if labels is not None:
            seq_labels = np.array(seq_labels)
            return sequences, seq_labels
        
        return sequences
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs.squeeze(), batch_y.float())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y.float())
                
                total_loss += loss.item()
                
                preds = (outputs.squeeze() > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        return avg_loss, accuracy
    
    def predict(self, data):
        """
        Predict anomalies for given data
        
        Args:
            data: numpy array of shape (n_samples, n_features)
            
        Returns:
            Anomaly scores and binary predictions
        """
        self.model.eval()
        
        # Create sequences
        sequences = self.create_sequences(data)
        
        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        # Predict
        with torch.no_grad():
            scores = self.model(sequences_tensor).squeeze().cpu().numpy()
        
        # Binary predictions
        predictions = (scores > 0.5).astype(int)
        
        return scores, predictions
    
    def save(self, path):
        """Save model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
        }, path)
        
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.input_size = checkpoint['input_size']
        self.sequence_length = checkpoint['sequence_length']
        
        print(f"✓ Model loaded from {path}")
