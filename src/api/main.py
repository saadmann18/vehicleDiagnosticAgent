"""
FastAPI Backend for Vehicle Diagnostics Agent
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from orchestrator import VehicleDiagnosticOrchestrator
from agents.data_ingestion_agent import DataIngestionAgent

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Diagnostics Agent API",
    description="Multi-agent AI system for predictive vehicle diagnostics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = VehicleDiagnosticOrchestrator()
ingestion_agent = DataIngestionAgent()

# Store for async job results
job_results = {}


# Pydantic models for request/response
class DiagnosticRequest(BaseModel):
    vehicle_id: int = Field(..., description="ID of the vehicle to diagnose")
    n_readings: Optional[int] = Field(None, description="Number of recent readings to analyze")


class DiagnosticResponse(BaseModel):
    success: bool
    vehicle_id: int
    message: str
    anomaly_detected: Optional[bool] = None
    overall_score: Optional[float] = None
    num_anomalies: Optional[int] = None
    primary_cause: Optional[str] = None
    estimated_cost: Optional[str] = None
    report_summary: Optional[str] = None


class BatchDiagnosticRequest(BaseModel):
    vehicle_ids: List[int] = Field(..., description="List of vehicle IDs to diagnose")
    n_readings: Optional[int] = Field(None, description="Number of recent readings to analyze")


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    available_vehicles: int


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Vehicle Diagnostics Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "diagnose": "/diagnose",
            "batch_diagnose": "/batch-diagnose",
            "vehicles": "/vehicles",
            "report": "/report/{vehicle_id}"
        }
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        test_df = ingestion_agent.load_test_data()
        num_vehicles = test_df['vehicle_id'].nunique()
        
        return HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            available_vehicles=num_vehicles
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/vehicles", response_model=Dict)
async def list_vehicles():
    """List available vehicles for diagnosis"""
    try:
        test_df = ingestion_agent.load_test_data()
        vehicle_ids = test_df['vehicle_id'].unique().tolist()
        
        # Get basic stats for each vehicle
        vehicle_info = []
        for vid in vehicle_ids[:20]:  # Limit to first 20 for performance
            vehicle_data = test_df[test_df['vehicle_id'] == vid]
            vehicle_info.append({
                'vehicle_id': int(vid),
                'num_readings': len(vehicle_data),
                'has_anomalies': bool(vehicle_data['anomaly'].sum() > 0),
                'anomaly_count': int(vehicle_data['anomaly'].sum())
            })
        
        return {
            "total_vehicles": len(vehicle_ids),
            "vehicles": vehicle_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list vehicles: {str(e)}")


@app.post("/diagnose", response_model=DiagnosticResponse)
async def diagnose_vehicle(request: DiagnosticRequest):
    """
    Run diagnostic analysis for a single vehicle
    """
    try:
        # Run diagnostic workflow
        result = orchestrator.diagnose_vehicle(
            vehicle_id=request.vehicle_id,
            n_readings=request.n_readings
        )
        
        if not result['success']:
            return DiagnosticResponse(
                success=False,
                vehicle_id=request.vehicle_id,
                message=f"Diagnostic failed: {result.get('error', 'Unknown error')}"
            )
        
        # Extract key information
        anomaly_result = result.get('anomaly_result', {})
        root_cause_result = result.get('root_cause_result', {})
        maintenance_result = result.get('maintenance_result', {})
        report = result.get('report', {})
        
        primary_cause = root_cause_result.get('primary_cause')
        
        return DiagnosticResponse(
            success=True,
            vehicle_id=request.vehicle_id,
            message="Diagnostic completed successfully",
            anomaly_detected=anomaly_result.get('anomaly_detected', False),
            overall_score=anomaly_result.get('overall_score'),
            num_anomalies=anomaly_result.get('num_anomalies'),
            primary_cause=primary_cause['fault_name'] if primary_cause else None,
            estimated_cost=maintenance_result.get('total_cost', {}).get('cost_range'),
            report_summary=report.get('natural_language_summary')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic failed: {str(e)}")


@app.post("/batch-diagnose")
async def batch_diagnose(request: BatchDiagnosticRequest, background_tasks: BackgroundTasks):
    """
    Run diagnostic analysis for multiple vehicles (async)
    """
    try:
        # For simplicity, run synchronously for now
        # In production, this would be handled by a task queue
        results = orchestrator.diagnose_multiple_vehicles(
            vehicle_ids=request.vehicle_ids,
            n_readings=request.n_readings
        )
        
        # Summarize results
        summary = {
            'total_vehicles': len(request.vehicle_ids),
            'successful': sum(1 for r in results.values() if r['success']),
            'with_anomalies': sum(1 for r in results.values() 
                                 if r['success'] and r.get('anomaly_result', {}).get('anomaly_detected')),
            'results': {}
        }
        
        for vid, result in results.items():
            if result['success']:
                anomaly_result = result.get('anomaly_result', {})
                summary['results'][vid] = {
                    'anomaly_detected': anomaly_result.get('anomaly_detected', False),
                    'overall_score': anomaly_result.get('overall_score'),
                    'num_anomalies': anomaly_result.get('num_anomalies')
                }
            else:
                summary['results'][vid] = {
                    'error': result.get('error')
                }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch diagnostic failed: {str(e)}")


@app.get("/report/{vehicle_id}")
async def get_full_report(vehicle_id: int, n_readings: Optional[int] = None):
    """
    Get full diagnostic report for a vehicle
    """
    try:
        # Run diagnostic workflow
        result = orchestrator.diagnose_vehicle(
            vehicle_id=vehicle_id,
            n_readings=n_readings
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        
        report = result.get('report', {})
        
        return {
            'vehicle_id': vehicle_id,
            'report_timestamp': report.get('report_timestamp'),
            'full_report': report.get('full_report'),
            'executive_summary': report.get('executive_summary'),
            'natural_language_summary': report.get('natural_language_summary'),
            'json_report': report.get('json_report')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@app.get("/vehicle/{vehicle_id}/status")
async def get_vehicle_status(vehicle_id: int):
    """
    Get current status of a vehicle without full diagnostic
    """
    try:
        test_df = ingestion_agent.load_test_data()
        vehicle_data = test_df[test_df['vehicle_id'] == vehicle_id]
        
        if len(vehicle_data) == 0:
            raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")
        
        # Get basic statistics
        latest_data = vehicle_data.tail(50)
        sensor_summary = ingestion_agent.get_sensor_summary(latest_data)
        
        return {
            'vehicle_id': vehicle_id,
            'num_readings': len(vehicle_data),
            'latest_timestamp': int(vehicle_data['timestamp'].iloc[-1]),
            'has_anomalies': bool(vehicle_data['anomaly'].sum() > 0),
            'total_anomalies': int(vehicle_data['anomaly'].sum()),
            'sensor_summary': sensor_summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vehicle status: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
