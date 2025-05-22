"""
Anomaly Detection Agent - Detects unusual patterns in sensor data
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.anomaly_detector import AnomalyDetectionModel
from typing import Dict, List, Tuple


class AnomalyDetectionAgent:
    """
    Agent responsible for detecting anomalies in vehicle sensor data
    """
    
    def __init__(self, model_path='src/models/best_anomaly_detector.pth', threshold=0.5):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained anomaly detection model"""
        if self.model_path.exists():
            # Get input size from model file
            import torch
            checkpoint = torch.load(self.model_path, map_location='cpu')
            input_size = checkpoint['input_size']
            sequence_length = checkpoint['sequence_length']
            
            self.model = AnomalyDetectionModel(input_size, sequence_length)
            self.model.load(self.model_path)
            print(f"✓ Loaded anomaly detection model from {self.model_path}")
        else:
            print(f"⚠ Model not found at {self.model_path}. Using rule-based detection.")
            self.model = None
    
    def detect_anomalies_ml(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using ML model
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Tuple of (anomaly_scores, anomaly_predictions)
        """
        if self.model is None:
            raise ValueError("ML model not loaded")
        
        scores, predictions = self.model.predict(features)
        return scores, predictions
    
    def detect_anomalies_rules(self, raw_data) -> np.ndarray:
        """
        Detect anomalies using rule-based approach (fallback)
        
        Args:
            raw_data: DataFrame with raw sensor data
            
        Returns:
            Array of anomaly predictions
        """
        anomalies = np.zeros(len(raw_data), dtype=int)
        
        # Rule 1: Engine overheating
        if 'engine_temp' in raw_data.columns:
            anomalies |= (raw_data['engine_temp'] > 2.0).astype(int)  # Normalized threshold
        
        # Rule 2: Low oil pressure
        if 'oil_pressure' in raw_data.columns:
            anomalies |= (raw_data['oil_pressure'] < -1.5).astype(int)
        
        # Rule 3: Battery issues
        if 'battery_voltage' in raw_data.columns:
            anomalies |= (raw_data['battery_voltage'] < -1.0).astype(int)
        
        # Rule 4: High vibration
        if 'vibration_level' in raw_data.columns:
            anomalies |= (raw_data['vibration_level'] > 2.0).astype(int)
        
        # Rule 5: Tire pressure issues
        tire_cols = [col for col in raw_data.columns if 'tire_pressure' in col]
        if tire_cols:
            for col in tire_cols:
                anomalies |= (raw_data[col] < -1.5).astype(int)
        
        return anomalies
    
    def identify_anomalous_sensors(self, raw_data, anomaly_indices: List[int]) -> Dict:
        """
        Identify which sensors are showing anomalous behavior
        
        Args:
            raw_data: DataFrame with raw sensor data
            anomaly_indices: Indices where anomalies were detected
            
        Returns:
            Dictionary mapping sensor names to anomaly information
        """
        if len(anomaly_indices) == 0:
            return {}
        
        anomalous_data = raw_data.iloc[anomaly_indices]
        
        sensor_cols = [col for col in raw_data.columns 
                      if col not in ['vehicle_id', 'timestamp', 'anomaly']]
        
        anomalous_sensors = {}
        
        for col in sensor_cols:
            # Check if this sensor shows unusual values
            overall_mean = raw_data[col].mean()
            overall_std = raw_data[col].std()
            
            anomaly_mean = anomalous_data[col].mean()
            
            # If anomaly mean is more than 2 std away from overall mean
            if abs(anomaly_mean - overall_mean) > 2 * overall_std:
                anomalous_sensors[col] = {
                    'overall_mean': float(overall_mean),
                    'anomaly_mean': float(anomaly_mean),
                    'deviation': float(abs(anomaly_mean - overall_mean) / overall_std),
                    'severity': 'high' if abs(anomaly_mean - overall_mean) > 3 * overall_std else 'medium'
                }
        
        return anomalous_sensors
    
    def calculate_anomaly_score(self, predictions: np.ndarray, scores: np.ndarray = None) -> float:
        """
        Calculate overall anomaly score for the vehicle
        
        Args:
            predictions: Binary anomaly predictions
            scores: Optional continuous anomaly scores
            
        Returns:
            Overall anomaly score (0-1)
        """
        if scores is not None:
            return float(np.mean(scores))
        else:
            return float(np.mean(predictions))
    
    def run(self, prepared_data: Dict) -> Dict:
        """
        Main execution method for the Anomaly Detection Agent
        
        Args:
            prepared_data: Data prepared by Data Ingestion Agent
            
        Returns:
            Dictionary containing anomaly detection results
        """
        print(f"\n{'='*60}")
        print(f"ANOMALY DETECTION AGENT - Vehicle {prepared_data['vehicle_id']}")
        print(f"{'='*60}")
        
        features = prepared_data['features']
        raw_data = prepared_data['raw_data']
        
        # Detect anomalies
        if self.model is not None:
            print("Using ML-based anomaly detection...")
            scores, predictions = self.detect_anomalies_ml(features)
            
            # Pad predictions to match original length
            padded_predictions = np.zeros(len(raw_data), dtype=int)
            padded_predictions[-len(predictions):] = predictions
            
            padded_scores = np.zeros(len(raw_data))
            padded_scores[-len(scores):] = scores
        else:
            print("Using rule-based anomaly detection...")
            padded_predictions = self.detect_anomalies_rules(raw_data)
            padded_scores = padded_predictions.astype(float)
        
        # Find anomaly indices
        anomaly_indices = np.where(padded_predictions == 1)[0].tolist()
        num_anomalies = len(anomaly_indices)
        
        print(f"✓ Detected {num_anomalies} anomalous readings out of {len(raw_data)}")
        print(f"  Anomaly rate: {num_anomalies/len(raw_data):.2%}")
        
        # Calculate overall anomaly score
        overall_score = self.calculate_anomaly_score(padded_predictions, padded_scores)
        print(f"  Overall anomaly score: {overall_score:.3f}")
        
        # Identify anomalous sensors
        anomalous_sensors = {}
        if num_anomalies > 0:
            anomalous_sensors = self.identify_anomalous_sensors(raw_data, anomaly_indices)
            print(f"✓ Identified {len(anomalous_sensors)} sensors with anomalous behavior")
            
            if anomalous_sensors:
                print("  Top anomalous sensors:")
                sorted_sensors = sorted(anomalous_sensors.items(), 
                                      key=lambda x: x[1]['deviation'], 
                                      reverse=True)
                for sensor, info in sorted_sensors[:3]:
                    print(f"    - {sensor}: {info['severity']} severity (deviation: {info['deviation']:.2f}σ)")
        
        # Compare with ground truth if available
        if prepared_data['ground_truth'] is not None:
            ground_truth = prepared_data['ground_truth']
            accuracy = (padded_predictions == ground_truth).mean()
            print(f"  Accuracy vs ground truth: {accuracy:.2%}")
        
        print(f"{'='*60}\n")
        
        result = {
            'vehicle_id': prepared_data['vehicle_id'],
            'anomaly_detected': num_anomalies > 0,
            'num_anomalies': num_anomalies,
            'anomaly_rate': num_anomalies / len(raw_data),
            'overall_score': overall_score,
            'anomaly_indices': anomaly_indices,
            'anomaly_predictions': padded_predictions,
            'anomaly_scores': padded_scores,
            'anomalous_sensors': anomalous_sensors,
            'timestamps': prepared_data['timestamps'],
            'raw_data': raw_data
        }
        
        return result


if __name__ == '__main__':
    # Test the Anomaly Detection Agent
    from data_ingestion_agent import DataIngestionAgent
    
    # Load data
    ingestion_agent = DataIngestionAgent()
    test_df = ingestion_agent.load_test_data()
    test_vehicle_id = test_df['vehicle_id'].iloc[0]
    
    # Prepare data
    prepared_data = ingestion_agent.run(test_vehicle_id, n_readings=200)
    
    # Detect anomalies
    detection_agent = AnomalyDetectionAgent()
    result = detection_agent.run(prepared_data)
    
    print(f"\nAnomaly Detection Summary:")
    print(f"  Anomalies detected: {result['anomaly_detected']}")
    print(f"  Overall score: {result['overall_score']:.3f}")
    print(f"  Anomalous sensors: {len(result['anomalous_sensors'])}")
