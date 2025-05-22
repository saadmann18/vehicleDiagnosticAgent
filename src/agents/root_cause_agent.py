"""
Root Cause Analysis Agent - Identifies the root cause of detected anomalies
"""
import numpy as np
from typing import Dict, List, Tuple


class RootCauseAnalysisAgent:
    """
    Agent responsible for determining the root cause of detected anomalies
    """
    
    def __init__(self):
        # Define fault patterns and their associated root causes
        self.fault_patterns = {
            'engine_overheating': {
                'sensors': ['engine_temp', 'coolant_temp', 'temp_differential'],
                'thresholds': {'engine_temp': 1.5, 'coolant_temp': 1.5, 'temp_differential': 1.0},
                'description': 'Engine temperature exceeds safe operating limits',
                'severity': 'critical',
                'fault_codes': ['P0217', 'P0218', 'P0219']
            },
            'cooling_system_failure': {
                'sensors': ['coolant_temp', 'engine_temp'],
                'thresholds': {'coolant_temp': 2.0, 'engine_temp': 1.8},
                'description': 'Cooling system not maintaining proper temperature',
                'severity': 'critical',
                'fault_codes': ['P0217', 'P0128']
            },
            'oil_pressure_low': {
                'sensors': ['oil_pressure'],
                'thresholds': {'oil_pressure': -1.5},
                'description': 'Oil pressure below safe operating range',
                'severity': 'critical',
                'fault_codes': ['P0520', 'P0521', 'P0522']
            },
            'battery_degradation': {
                'sensors': ['battery_voltage', 'battery_health'],
                'thresholds': {'battery_voltage': -1.0, 'battery_health': -1.0},
                'description': 'Battery voltage or health declining',
                'severity': 'high',
                'fault_codes': ['P0560', 'P0562', 'P0563']
            },
            'tire_pressure_issue': {
                'sensors': ['tire_pressure_fl', 'tire_pressure_fr', 'tire_pressure_rl', 'tire_pressure_rr', 'tire_pressure_imbalance'],
                'thresholds': {'tire_pressure_fl': -1.5, 'tire_pressure_fr': -1.5, 
                              'tire_pressure_rl': -1.5, 'tire_pressure_rr': -1.5,
                              'tire_pressure_imbalance': 1.5},
                'description': 'One or more tires have incorrect pressure',
                'severity': 'medium',
                'fault_codes': ['C1234', 'C1235']
            },
            'excessive_vibration': {
                'sensors': ['vibration_level'],
                'thresholds': {'vibration_level': 2.0},
                'description': 'Abnormal vibration detected',
                'severity': 'high',
                'fault_codes': ['P0300', 'P0301']
            },
            'fuel_system_issue': {
                'sensors': ['fuel_pressure'],
                'thresholds': {'fuel_pressure': -1.5},
                'description': 'Fuel pressure outside normal range',
                'severity': 'high',
                'fault_codes': ['P0087', 'P0088']
            },
            'engine_stress': {
                'sensors': ['engine_stress', 'rpm', 'engine_temp'],
                'thresholds': {'engine_stress': 2.0, 'rpm': 2.0},
                'description': 'Engine operating under excessive stress',
                'severity': 'medium',
                'fault_codes': ['P0101', 'P0102']
            }
        }
    
    def analyze_sensor_patterns(self, anomalous_sensors: Dict, raw_data) -> List[Dict]:
        """
        Analyze anomalous sensor patterns to identify root causes
        
        Args:
            anomalous_sensors: Dictionary of sensors showing anomalous behavior
            raw_data: Raw sensor data DataFrame
            
        Returns:
            List of identified root causes with confidence scores
        """
        identified_causes = []
        
        for fault_name, fault_info in self.fault_patterns.items():
            # Check if any of the fault's sensors are anomalous
            matching_sensors = []
            confidence_scores = []
            
            for sensor in fault_info['sensors']:
                if sensor in anomalous_sensors:
                    matching_sensors.append(sensor)
                    
                    # Calculate confidence based on deviation
                    deviation = anomalous_sensors[sensor]['deviation']
                    confidence = min(deviation / 5.0, 1.0)  # Normalize to 0-1
                    confidence_scores.append(confidence)
                
                # Also check if sensor values exceed thresholds
                elif sensor in raw_data.columns:
                    threshold = fault_info['thresholds'].get(sensor)
                    if threshold is not None:
                        # Check recent values
                        recent_values = raw_data[sensor].tail(20)
                        if threshold > 0:
                            exceeds = (recent_values > threshold).sum() / len(recent_values)
                        else:
                            exceeds = (recent_values < threshold).sum() / len(recent_values)
                        
                        if exceeds > 0.3:  # If 30% of recent values exceed threshold
                            matching_sensors.append(sensor)
                            confidence_scores.append(exceeds)
            
            # If we have matching sensors, this is a potential root cause
            if matching_sensors:
                avg_confidence = np.mean(confidence_scores)
                
                identified_causes.append({
                    'fault_name': fault_name,
                    'description': fault_info['description'],
                    'severity': fault_info['severity'],
                    'confidence': float(avg_confidence),
                    'affected_sensors': matching_sensors,
                    'fault_codes': fault_info['fault_codes'],
                    'num_sensors_affected': len(matching_sensors)
                })
        
        # Sort by confidence
        identified_causes.sort(key=lambda x: x['confidence'], reverse=True)
        
        return identified_causes
    
    def correlate_sensor_failures(self, anomalous_sensors: Dict) -> List[Tuple[str, str, float]]:
        """
        Find correlations between anomalous sensors
        
        Args:
            anomalous_sensors: Dictionary of anomalous sensors
            
        Returns:
            List of correlated sensor pairs with correlation strength
        """
        correlations = []
        
        # Known sensor correlations
        known_correlations = [
            ('engine_temp', 'coolant_temp', 0.9),
            ('engine_temp', 'oil_pressure', -0.7),
            ('rpm', 'engine_temp', 0.6),
            ('battery_voltage', 'battery_health', 0.95),
            ('tire_pressure_fl', 'tire_pressure_fr', 0.8),
            ('tire_pressure_rl', 'tire_pressure_rr', 0.8),
        ]
        
        for sensor1, sensor2, corr_strength in known_correlations:
            if sensor1 in anomalous_sensors and sensor2 in anomalous_sensors:
                correlations.append((sensor1, sensor2, corr_strength))
        
        return correlations
    
    def determine_failure_sequence(self, anomaly_indices: List[int], 
                                   anomalous_sensors: Dict, 
                                   timestamps: np.ndarray) -> Dict:
        """
        Determine the sequence of failures
        
        Args:
            anomaly_indices: Indices where anomalies occurred
            anomalous_sensors: Dictionary of anomalous sensors
            timestamps: Array of timestamps
            
        Returns:
            Dictionary describing failure sequence
        """
        if not anomaly_indices:
            return {'sequence': [], 'duration': 0}
        
        first_anomaly = min(anomaly_indices)
        last_anomaly = max(anomaly_indices)
        duration = last_anomaly - first_anomaly
        
        sequence = {
            'first_anomaly_time': int(timestamps[first_anomaly]),
            'last_anomaly_time': int(timestamps[last_anomaly]),
            'duration': int(duration),
            'progression': 'gradual' if duration > 50 else 'sudden',
            'affected_sensors': list(anomalous_sensors.keys())
        }
        
        return sequence
    
    def run(self, anomaly_result: Dict) -> Dict:
        """
        Main execution method for the Root Cause Analysis Agent
        
        Args:
            anomaly_result: Results from Anomaly Detection Agent
            
        Returns:
            Dictionary containing root cause analysis
        """
        print(f"\n{'='*60}")
        print(f"ROOT CAUSE ANALYSIS AGENT - Vehicle {anomaly_result['vehicle_id']}")
        print(f"{'='*60}")
        
        if not anomaly_result['anomaly_detected']:
            print("✓ No anomalies detected - no root cause analysis needed")
            print(f"{'='*60}\n")
            return {
                'vehicle_id': anomaly_result['vehicle_id'],
                'root_causes': [],
                'correlations': [],
                'failure_sequence': {},
                'analysis_summary': 'No anomalies detected'
            }
        
        anomalous_sensors = anomaly_result['anomalous_sensors']
        raw_data = anomaly_result['raw_data']
        anomaly_indices = anomaly_result['anomaly_indices']
        timestamps = anomaly_result['timestamps']
        
        print(f"Analyzing {len(anomalous_sensors)} anomalous sensors...")
        
        # Identify root causes
        root_causes = self.analyze_sensor_patterns(anomalous_sensors, raw_data)
        print(f"✓ Identified {len(root_causes)} potential root causes")
        
        if root_causes:
            print("\nTop root causes:")
            for i, cause in enumerate(root_causes[:3], 1):
                print(f"  {i}. {cause['fault_name']} ({cause['severity']} severity)")
                print(f"     Confidence: {cause['confidence']:.2%}")
                print(f"     Description: {cause['description']}")
                print(f"     Fault codes: {', '.join(cause['fault_codes'])}")
        
        # Find sensor correlations
        correlations = self.correlate_sensor_failures(anomalous_sensors)
        if correlations:
            print(f"\n✓ Found {len(correlations)} correlated sensor failures")
            for sensor1, sensor2, strength in correlations:
                print(f"    - {sensor1} ↔ {sensor2} (correlation: {strength:.2f})")
        
        # Determine failure sequence
        failure_sequence = self.determine_failure_sequence(
            anomaly_indices, anomalous_sensors, timestamps
        )
        print(f"\n✓ Failure progression: {failure_sequence.get('progression', 'unknown')}")
        print(f"  Duration: {failure_sequence.get('duration', 0)} timesteps")
        
        # Generate analysis summary
        if root_causes:
            primary_cause = root_causes[0]
            summary = (f"Primary issue: {primary_cause['description']} "
                      f"({primary_cause['severity']} severity, "
                      f"{primary_cause['confidence']:.0%} confidence)")
        else:
            summary = "Anomalies detected but root cause unclear"
        
        print(f"\n✓ Analysis summary: {summary}")
        print(f"{'='*60}\n")
        
        result = {
            'vehicle_id': anomaly_result['vehicle_id'],
            'root_causes': root_causes,
            'correlations': correlations,
            'failure_sequence': failure_sequence,
            'analysis_summary': summary,
            'primary_cause': root_causes[0] if root_causes else None
        }
        
        return result


if __name__ == '__main__':
    # Test the Root Cause Analysis Agent
    from data_ingestion_agent import DataIngestionAgent
    from anomaly_detection_agent import AnomalyDetectionAgent
    
    # Load and prepare data
    ingestion_agent = DataIngestionAgent()
    test_df = ingestion_agent.load_test_data()
    
    # Find a vehicle with anomalies
    test_vehicle_id = None
    for vid in test_df['vehicle_id'].unique()[:10]:
        if test_df[test_df['vehicle_id'] == vid]['anomaly'].sum() > 0:
            test_vehicle_id = vid
            break
    
    if test_vehicle_id:
        prepared_data = ingestion_agent.run(test_vehicle_id)
        
        # Detect anomalies
        detection_agent = AnomalyDetectionAgent()
        anomaly_result = detection_agent.run(prepared_data)
        
        # Analyze root cause
        rca_agent = RootCauseAnalysisAgent()
        result = rca_agent.run(anomaly_result)
        
        print(f"\nRoot Cause Analysis Summary:")
        print(f"  Primary cause: {result['primary_cause']['fault_name'] if result['primary_cause'] else 'None'}")
        print(f"  Root causes found: {len(result['root_causes'])}")
