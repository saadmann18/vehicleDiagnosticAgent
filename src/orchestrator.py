"""
Multi-Agent Orchestrator using LangGraph
Coordinates the execution of all diagnostic agents
"""
from typing import Dict, TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

from agents.data_ingestion_agent import DataIngestionAgent
from agents.anomaly_detection_agent import AnomalyDetectionAgent
from agents.root_cause_agent import RootCauseAnalysisAgent
from agents.maintenance_recommendation_agent import MaintenanceRecommendationAgent
from agents.report_generation_agent import ReportGenerationAgent


class DiagnosticState(TypedDict):
    """State object passed between agents"""
    vehicle_id: int
    n_readings: int
    prepared_data: Dict
    anomaly_result: Dict
    root_cause_result: Dict
    maintenance_result: Dict
    report_result: Dict
    error: str


class VehicleDiagnosticOrchestrator:
    """
    Orchestrates the multi-agent vehicle diagnostic workflow using LangGraph
    """
    
    def __init__(self):
        self.ingestion_agent = DataIngestionAgent()
        self.anomaly_agent = AnomalyDetectionAgent()
        self.root_cause_agent = RootCauseAnalysisAgent()
        self.maintenance_agent = MaintenanceRecommendationAgent()
        self.report_agent = ReportGenerationAgent()
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the workflow graph
        workflow = StateGraph(DiagnosticState)
        
        # Add nodes for each agent
        workflow.add_node("data_ingestion", self._run_data_ingestion)
        workflow.add_node("anomaly_detection", self._run_anomaly_detection)
        workflow.add_node("root_cause_analysis", self._run_root_cause_analysis)
        workflow.add_node("maintenance_recommendation", self._run_maintenance_recommendation)
        workflow.add_node("report_generation", self._run_report_generation)
        
        # Define the workflow edges (sequential execution)
        workflow.set_entry_point("data_ingestion")
        workflow.add_edge("data_ingestion", "anomaly_detection")
        workflow.add_edge("anomaly_detection", "root_cause_analysis")
        workflow.add_edge("root_cause_analysis", "maintenance_recommendation")
        workflow.add_edge("maintenance_recommendation", "report_generation")
        workflow.add_edge("report_generation", END)
        
        return workflow.compile()
    
    def _run_data_ingestion(self, state: DiagnosticState) -> DiagnosticState:
        """Execute Data Ingestion Agent"""
        try:
            prepared_data = self.ingestion_agent.run(
                state['vehicle_id'], 
                state.get('n_readings')
            )
            state['prepared_data'] = prepared_data
        except Exception as e:
            state['error'] = f"Data Ingestion Error: {str(e)}"
        
        return state
    
    def _run_anomaly_detection(self, state: DiagnosticState) -> DiagnosticState:
        """Execute Anomaly Detection Agent"""
        try:
            if 'error' not in state:
                anomaly_result = self.anomaly_agent.run(state['prepared_data'])
                state['anomaly_result'] = anomaly_result
        except Exception as e:
            state['error'] = f"Anomaly Detection Error: {str(e)}"
        
        return state
    
    def _run_root_cause_analysis(self, state: DiagnosticState) -> DiagnosticState:
        """Execute Root Cause Analysis Agent"""
        try:
            if 'error' not in state:
                root_cause_result = self.root_cause_agent.run(state['anomaly_result'])
                state['root_cause_result'] = root_cause_result
        except Exception as e:
            state['error'] = f"Root Cause Analysis Error: {str(e)}"
        
        return state
    
    def _run_maintenance_recommendation(self, state: DiagnosticState) -> DiagnosticState:
        """Execute Maintenance Recommendation Agent"""
        try:
            if 'error' not in state:
                maintenance_result = self.maintenance_agent.run(state['root_cause_result'])
                state['maintenance_result'] = maintenance_result
        except Exception as e:
            state['error'] = f"Maintenance Recommendation Error: {str(e)}"
        
        return state
    
    def _run_report_generation(self, state: DiagnosticState) -> DiagnosticState:
        """Execute Report Generation Agent"""
        try:
            if 'error' not in state:
                report_result = self.report_agent.run(
                    state['vehicle_id'],
                    state['prepared_data'],
                    state['anomaly_result'],
                    state['root_cause_result'],
                    state['maintenance_result']
                )
                state['report_result'] = report_result
        except Exception as e:
            state['error'] = f"Report Generation Error: {str(e)}"
        
        return state
    
    def diagnose_vehicle(self, vehicle_id: int, n_readings: int = None) -> Dict:
        """
        Run complete diagnostic workflow for a vehicle
        
        Args:
            vehicle_id: ID of the vehicle to diagnose
            n_readings: Optional number of recent readings to analyze
            
        Returns:
            Dictionary containing complete diagnostic results
        """
        print("\n" + "="*60)
        print("VEHICLE DIAGNOSTIC ORCHESTRATOR")
        print("="*60)
        print(f"Starting diagnostic workflow for Vehicle {vehicle_id}")
        print("="*60 + "\n")
        
        # Initialize state
        initial_state = {
            'vehicle_id': vehicle_id,
            'n_readings': n_readings
        }
        
        # Execute workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Check for errors
        if 'error' in final_state:
            print(f"\n❌ Error occurred: {final_state['error']}")
            return {
                'success': False,
                'error': final_state['error'],
                'vehicle_id': vehicle_id
            }
        
        print("\n" + "="*60)
        print("DIAGNOSTIC WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Return comprehensive results
        return {
            'success': True,
            'vehicle_id': vehicle_id,
            'prepared_data': final_state.get('prepared_data'),
            'anomaly_result': final_state.get('anomaly_result'),
            'root_cause_result': final_state.get('root_cause_result'),
            'maintenance_result': final_state.get('maintenance_result'),
            'report': final_state.get('report_result')
        }
    
    def diagnose_multiple_vehicles(self, vehicle_ids: list, n_readings: int = None) -> Dict:
        """
        Run diagnostics for multiple vehicles
        
        Args:
            vehicle_ids: List of vehicle IDs
            n_readings: Optional number of recent readings to analyze
            
        Returns:
            Dictionary mapping vehicle IDs to diagnostic results
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"BATCH DIAGNOSTICS - {len(vehicle_ids)} vehicles")
        print(f"{'='*60}\n")
        
        for i, vehicle_id in enumerate(vehicle_ids, 1):
            print(f"\nProcessing vehicle {i}/{len(vehicle_ids)}: {vehicle_id}")
            results[vehicle_id] = self.diagnose_vehicle(vehicle_id, n_readings)
        
        print(f"\n{'='*60}")
        print(f"BATCH DIAGNOSTICS COMPLETED")
        print(f"{'='*60}")
        
        # Summary statistics
        successful = sum(1 for r in results.values() if r['success'])
        with_anomalies = sum(1 for r in results.values() 
                           if r['success'] and r.get('anomaly_result', {}).get('anomaly_detected'))
        
        print(f"\nSummary:")
        print(f"  Total vehicles: {len(vehicle_ids)}")
        print(f"  Successfully analyzed: {successful}")
        print(f"  Vehicles with anomalies: {with_anomalies}")
        
        return results


def main():
    """Test the orchestrator"""
    orchestrator = VehicleDiagnosticOrchestrator()
    
    # Load test data to get vehicle IDs
    from agents.data_ingestion_agent import DataIngestionAgent
    ingestion_agent = DataIngestionAgent()
    test_df = ingestion_agent.load_test_data()
    
    # Get a vehicle with anomalies
    test_vehicle_id = None
    for vid in test_df['vehicle_id'].unique()[:10]:
        if test_df[test_df['vehicle_id'] == vid]['anomaly'].sum() > 0:
            test_vehicle_id = vid
            break
    
    if test_vehicle_id:
        # Run single vehicle diagnostic
        result = orchestrator.diagnose_vehicle(test_vehicle_id, n_readings=200)
        
        if result['success']:
            print("\n" + "="*60)
            print("DIAGNOSTIC REPORT PREVIEW")
            print("="*60)
            report = result['report']['full_report']
            print(report[:2000] + "\n...\n")
            
            print("\nNatural Language Summary:")
            print("-"*60)
            print(result['report']['natural_language_summary'])


if __name__ == '__main__':
    main()
