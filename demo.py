#!/usr/bin/env python3
"""
Quick Demo Script for Vehicle Diagnostics Agent
Demonstrates the complete diagnostic workflow
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from orchestrator import VehicleDiagnosticOrchestrator
from agents.data_ingestion_agent import DataIngestionAgent


def main():
    print("\n" + "="*70)
    print("🚗 VEHICLE DIAGNOSTICS AGENT - DEMO")
    print("="*70 + "\n")
    
    # Initialize
    print("Initializing system...")
    orchestrator = VehicleDiagnosticOrchestrator()
    ingestion_agent = DataIngestionAgent()
    
    # Load test data
    print("Loading test data...")
    test_df = ingestion_agent.load_test_data()
    
    # Find vehicles with anomalies
    print("\nFinding vehicles with anomalies...")
    vehicles_with_anomalies = []
    for vid in test_df['vehicle_id'].unique()[:20]:
        vehicle_data = test_df[test_df['vehicle_id'] == vid]
        if vehicle_data['anomaly'].sum() > 0:
            vehicles_with_anomalies.append({
                'id': vid,
                'anomaly_count': int(vehicle_data['anomaly'].sum()),
                'total_readings': len(vehicle_data)
            })
    
    print(f"✓ Found {len(vehicles_with_anomalies)} vehicles with anomalies\n")
    
    # Select a vehicle for demo
    if vehicles_with_anomalies:
        demo_vehicle = vehicles_with_anomalies[0]
        vehicle_id = demo_vehicle['id']
        
        print(f"Demo Vehicle: {vehicle_id}")
        print(f"  - Total readings: {demo_vehicle['total_readings']}")
        print(f"  - Known anomalies: {demo_vehicle['anomaly_count']}")
        print(f"  - Anomaly rate: {demo_vehicle['anomaly_count']/demo_vehicle['total_readings']:.1%}")
        print("\n" + "-"*70 + "\n")
        
        # Run diagnostic
        print(f"Running complete diagnostic workflow for Vehicle {vehicle_id}...\n")
        result = orchestrator.diagnose_vehicle(vehicle_id, n_readings=200)
        
        if result['success']:
            print("\n" + "="*70)
            print("📊 DIAGNOSTIC RESULTS")
            print("="*70 + "\n")
            
            # Anomaly Detection Results
            anomaly_result = result['anomaly_result']
            print("🔍 ANOMALY DETECTION:")
            print(f"  ✓ Anomaly Detected: {'YES ⚠️' if anomaly_result['anomaly_detected'] else 'NO ✅'}")
            print(f"  ✓ Overall Score: {anomaly_result['overall_score']:.3f}")
            print(f"  ✓ Anomalous Readings: {anomaly_result['num_anomalies']}/{len(anomaly_result['anomaly_predictions'])} ({anomaly_result['anomaly_rate']:.1%})")
            print(f"  ✓ Affected Sensors: {len(anomaly_result['anomalous_sensors'])}")
            
            # Root Cause Analysis
            root_cause_result = result['root_cause_result']
            print(f"\n🔬 ROOT CAUSE ANALYSIS:")
            print(f"  ✓ Root Causes Identified: {len(root_cause_result['root_causes'])}")
            
            if root_cause_result['primary_cause']:
                primary = root_cause_result['primary_cause']
                print(f"\n  PRIMARY ISSUE:")
                print(f"    • Fault: {primary['fault_name'].replace('_', ' ').title()}")
                print(f"    • Description: {primary['description']}")
                print(f"    • Severity: {primary['severity'].upper()}")
                print(f"    • Confidence: {primary['confidence']:.0%}")
                print(f"    • Fault Codes: {', '.join(primary['fault_codes'])}")
            
            # Maintenance Recommendations
            maintenance_result = result['maintenance_result']
            print(f"\n🔧 MAINTENANCE RECOMMENDATIONS:")
            print(f"  ✓ Total Items: {len(maintenance_result['recommendations'])}")
            print(f"  ✓ Estimated Cost: {maintenance_result['total_cost']['cost_range']}")
            print(f"  ✓ Immediate Actions: {len(maintenance_result['action_plan']['immediate'])}")
            
            if maintenance_result['top_priority']:
                top = maintenance_result['top_priority']
                print(f"\n  TOP PRIORITY:")
                print(f"    • Urgency: {top['urgency'].upper()}")
                print(f"    • Cost: {top['estimated_cost']}")
                print(f"    • Downtime: {top['estimated_downtime']}")
            
            # Natural Language Summary
            report = result['report']
            print(f"\n📋 SUMMARY FOR VEHICLE OWNER:")
            print("-"*70)
            print(report['natural_language_summary'])
            print("-"*70)
            
            # Save report
            report_file = f"vehicle_{vehicle_id}_report.txt"
            with open(report_file, 'w') as f:
                f.write(report['full_report'])
            print(f"\n✓ Full report saved to: {report_file}")
            
        else:
            print(f"\n❌ Diagnostic failed: {result.get('error')}")
    
    else:
        print("No vehicles with anomalies found in test set.")
        print("Running diagnostic on first available vehicle...")
        
        vehicle_id = test_df['vehicle_id'].iloc[0]
        result = orchestrator.diagnose_vehicle(vehicle_id, n_readings=100)
        
        if result['success']:
            print(f"\n✅ Vehicle {vehicle_id} is healthy!")
            print(result['report']['natural_language_summary'])
    
    print("\n" + "="*70)
    print("DEMO COMPLETED")
    print("="*70)
    print("\nNext steps:")
    print("  • Run Gradio UI: ./run_ui.sh")
    print("  • Run FastAPI: ./run_api.sh")
    print("  • Run tests: pytest tests/ -v")
    print("  • Deploy with Docker: docker-compose up --build")
    print("\n")


if __name__ == '__main__':
    main()
