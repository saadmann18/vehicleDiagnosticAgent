"""
Unit tests for individual agents
"""
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from agents.data_ingestion_agent import DataIngestionAgent
from agents.anomaly_detection_agent import AnomalyDetectionAgent
from agents.root_cause_agent import RootCauseAnalysisAgent
from agents.maintenance_recommendation_agent import MaintenanceRecommendationAgent
from agents.report_generation_agent import ReportGenerationAgent


class TestDataIngestionAgent:
    """Test Data Ingestion Agent"""
    
    def test_load_data(self):
        """Test loading test data"""
        agent = DataIngestionAgent()
        df = agent.load_test_data()
        
        assert df is not None
        assert len(df) > 0
        assert 'vehicle_id' in df.columns
        assert 'timestamp' in df.columns
    
    def test_get_vehicle_data(self):
        """Test getting data for specific vehicle"""
        agent = DataIngestionAgent()
        df = agent.load_test_data()
        vehicle_id = df['vehicle_id'].iloc[0]
        
        vehicle_data = agent.get_vehicle_data(vehicle_id)
        
        assert len(vehicle_data) > 0
        assert (vehicle_data['vehicle_id'] == vehicle_id).all()
    
    def test_prepare_for_analysis(self):
        """Test data preparation"""
        agent = DataIngestionAgent()
        df = agent.load_test_data()
        vehicle_id = df['vehicle_id'].iloc[0]
        vehicle_data = agent.get_vehicle_data(vehicle_id)
        
        prepared = agent.prepare_for_analysis(vehicle_data)
        
        assert 'vehicle_id' in prepared
        assert 'features' in prepared
        assert 'timestamps' in prepared
        assert prepared['vehicle_id'] == vehicle_id


class TestAnomalyDetectionAgent:
    """Test Anomaly Detection Agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = AnomalyDetectionAgent()
        assert agent is not None
    
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        ingestion_agent = DataIngestionAgent()
        detection_agent = AnomalyDetectionAgent()
        
        df = ingestion_agent.load_test_data()
        vehicle_id = df['vehicle_id'].iloc[0]
        
        prepared_data = ingestion_agent.run(vehicle_id, n_readings=100)
        result = detection_agent.run(prepared_data)
        
        assert 'vehicle_id' in result
        assert 'anomaly_detected' in result
        assert 'overall_score' in result
        assert 'anomaly_predictions' in result


class TestRootCauseAnalysisAgent:
    """Test Root Cause Analysis Agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = RootCauseAnalysisAgent()
        assert agent is not None
        assert len(agent.fault_patterns) > 0
    
    def test_analyze_no_anomalies(self):
        """Test analysis when no anomalies"""
        agent = RootCauseAnalysisAgent()
        
        anomaly_result = {
            'vehicle_id': 1,
            'anomaly_detected': False,
            'anomalous_sensors': {},
            'raw_data': None,
            'anomaly_indices': [],
            'timestamps': []
        }
        
        result = agent.run(anomaly_result)
        
        assert result['vehicle_id'] == 1
        assert len(result['root_causes']) == 0


class TestMaintenanceRecommendationAgent:
    """Test Maintenance Recommendation Agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = MaintenanceRecommendationAgent()
        assert agent is not None
        assert len(agent.maintenance_actions) > 0
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        agent = MaintenanceRecommendationAgent()
        
        root_causes = [{
            'fault_name': 'engine_overheating',
            'description': 'Test',
            'severity': 'critical',
            'confidence': 0.9,
            'fault_codes': ['P0217']
        }]
        
        recommendations = agent.generate_recommendations(root_causes)
        
        assert len(recommendations) > 0
        assert 'immediate_actions' in recommendations[0]
        assert 'estimated_cost' in recommendations[0]


class TestReportGenerationAgent:
    """Test Report Generation Agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = ReportGenerationAgent()
        assert agent is not None
    
    def test_generate_summary(self):
        """Test summary generation"""
        agent = ReportGenerationAgent()
        
        anomaly_result = {
            'vehicle_id': 1,
            'anomaly_detected': False,
            'num_anomalies': 0,
            'anomaly_rate': 0.0,
            'overall_score': 0.0,
            'anomalous_sensors': {}
        }
        
        root_cause_result = {
            'root_causes': [],
            'primary_cause': None
        }
        
        maintenance_result = {
            'recommendations': [],
            'total_cost': {'cost_range': '$0'}
        }
        
        summary = agent.generate_executive_summary(
            1, anomaly_result, root_cause_result, maintenance_result
        )
        
        assert 'Vehicle 1' in summary
        assert 'normally' in summary.lower()


def test_full_pipeline():
    """Test complete diagnostic pipeline"""
    from orchestrator import VehicleDiagnosticOrchestrator
    
    orchestrator = VehicleDiagnosticOrchestrator()
    
    # Get a test vehicle
    ingestion_agent = DataIngestionAgent()
    df = ingestion_agent.load_test_data()
    vehicle_id = df['vehicle_id'].iloc[0]
    
    # Run diagnostic
    result = orchestrator.diagnose_vehicle(vehicle_id, n_readings=100)
    
    assert result['success'] == True
    assert result['vehicle_id'] == vehicle_id
    assert 'report' in result
    assert 'anomaly_result' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
