"""
Report Generation Agent - Generates comprehensive diagnostic reports
"""
from typing import Dict
from datetime import datetime
import json


class ReportGenerationAgent:
    """
    Agent responsible for generating human-readable diagnostic reports
    """
    
    def __init__(self):
        self.report_template = None
    
    def generate_executive_summary(self, vehicle_id: int, 
                                   anomaly_result: Dict,
                                   root_cause_result: Dict,
                                   maintenance_result: Dict) -> str:
        """
        Generate executive summary of the diagnostic report
        
        Args:
            vehicle_id: Vehicle ID
            anomaly_result: Results from anomaly detection
            root_cause_result: Results from root cause analysis
            maintenance_result: Results from maintenance recommendations
            
        Returns:
            Executive summary string
        """
        if not anomaly_result['anomaly_detected']:
            return (f"Vehicle {vehicle_id} is operating normally. "
                   f"No anomalies detected in the analyzed sensor data. "
                   f"No maintenance actions required at this time.")
        
        num_anomalies = anomaly_result['num_anomalies']
        anomaly_rate = anomaly_result['anomaly_rate']
        overall_score = anomaly_result['overall_score']
        
        primary_cause = root_cause_result.get('primary_cause')
        num_recommendations = len(maintenance_result['recommendations'])
        
        summary = f"""
Vehicle {vehicle_id} Diagnostic Summary:

ALERT: Anomalies detected in vehicle sensor data.

Key Findings:
• Anomaly Detection: {num_anomalies} anomalous readings detected ({anomaly_rate:.1%} of analyzed data)
• Overall Anomaly Score: {overall_score:.3f}
• Affected Sensors: {len(anomaly_result['anomalous_sensors'])} sensors showing abnormal behavior
"""
        
        if primary_cause:
            summary += f"""
Primary Issue Identified:
• {primary_cause['description']}
• Severity: {primary_cause['severity'].upper()}
• Confidence: {primary_cause['confidence']:.0%}
• Fault Codes: {', '.join(primary_cause['fault_codes'])}
"""
        
        if num_recommendations > 0:
            top_priority = maintenance_result.get('top_priority')
            total_cost = maintenance_result['total_cost']
            
            summary += f"""
Maintenance Required:
• {num_recommendations} maintenance items identified
• Highest Priority: {top_priority['urgency'].upper()} urgency
• Estimated Cost: {total_cost['cost_range']}
• Immediate Actions: {len(maintenance_result['action_plan']['immediate'])} required
"""
        
        return summary.strip()
    
    def format_anomaly_details(self, anomaly_result: Dict) -> str:
        """Format anomaly detection details"""
        if not anomaly_result['anomaly_detected']:
            return "No anomalies detected."
        
        details = f"""
ANOMALY DETECTION DETAILS
{'='*60}

Overall Statistics:
• Total Readings Analyzed: {len(anomaly_result['anomaly_predictions'])}
• Anomalous Readings: {anomaly_result['num_anomalies']}
• Anomaly Rate: {anomaly_result['anomaly_rate']:.2%}
• Overall Anomaly Score: {anomaly_result['overall_score']:.3f}

Affected Sensors:
"""
        
        anomalous_sensors = anomaly_result['anomalous_sensors']
        sorted_sensors = sorted(anomalous_sensors.items(), 
                               key=lambda x: x[1]['deviation'], 
                               reverse=True)
        
        for sensor, info in sorted_sensors:
            details += f"""
• {sensor.upper()}
  - Severity: {info['severity']}
  - Deviation: {info['deviation']:.2f}σ from normal
  - Normal Mean: {info['overall_mean']:.3f}
  - Anomaly Mean: {info['anomaly_mean']:.3f}
"""
        
        return details.strip()
    
    def format_root_cause_analysis(self, root_cause_result: Dict) -> str:
        """Format root cause analysis details"""
        if not root_cause_result['root_causes']:
            return "No root causes identified."
        
        details = f"""
ROOT CAUSE ANALYSIS
{'='*60}

Analysis Summary:
{root_cause_result['analysis_summary']}

Failure Progression:
• Type: {root_cause_result['failure_sequence'].get('progression', 'unknown').upper()}
• Duration: {root_cause_result['failure_sequence'].get('duration', 0)} timesteps
• First Anomaly: Timestep {root_cause_result['failure_sequence'].get('first_anomaly_time', 'N/A')}
• Last Anomaly: Timestep {root_cause_result['failure_sequence'].get('last_anomaly_time', 'N/A')}

Identified Root Causes:
"""
        
        for i, cause in enumerate(root_cause_result['root_causes'], 1):
            details += f"""
{i}. {cause['fault_name'].upper().replace('_', ' ')}
   Description: {cause['description']}
   Severity: {cause['severity'].upper()}
   Confidence: {cause['confidence']:.0%}
   Fault Codes: {', '.join(cause['fault_codes'])}
   Affected Sensors: {', '.join(cause['affected_sensors'])}
"""
        
        if root_cause_result['correlations']:
            details += "\nCorrelated Sensor Failures:\n"
            for sensor1, sensor2, strength in root_cause_result['correlations']:
                details += f"• {sensor1} ↔ {sensor2} (correlation: {strength:.2f})\n"
        
        return details.strip()
    
    def format_maintenance_recommendations(self, maintenance_result: Dict) -> str:
        """Format maintenance recommendations"""
        if not maintenance_result['recommendations']:
            return "No maintenance required at this time."
        
        details = f"""
MAINTENANCE RECOMMENDATIONS
{'='*60}

Cost Estimate: {maintenance_result['total_cost']['cost_range']}
Total Actions: {maintenance_result['action_plan']['total_actions']}

IMMEDIATE ACTIONS (Perform Now):
"""
        
        for i, action in enumerate(maintenance_result['action_plan']['immediate'], 1):
            details += f"{i}. {action['action']}\n   Related to: {action['related_to'].replace('_', ' ').title()}\n   Urgency: {action['urgency'].upper()}\n\n"
        
        details += "\nSHORT-TERM ACTIONS (Within 1-2 Weeks):\n"
        for i, action in enumerate(maintenance_result['action_plan']['short_term'], 1):
            details += f"{i}. {action['action']}\n   Related to: {action['related_to'].replace('_', ' ').title()}\n\n"
        
        details += "\nLONG-TERM ACTIONS (Preventive Maintenance):\n"
        for i, action in enumerate(maintenance_result['action_plan']['long_term'], 1):
            details += f"{i}. {action['action']}\n   Related to: {action['related_to'].replace('_', ' ').title()}\n\n"
        
        # Add detailed recommendations
        details += "\nDETAILED MAINTENANCE ITEMS:\n"
        for i, rec in enumerate(maintenance_result['recommendations'], 1):
            details += f"""
{i}. {rec['fault_name'].upper().replace('_', ' ')}
   Severity: {rec['severity'].upper()}
   Urgency: {rec['urgency'].upper()}
   Estimated Cost: {rec['estimated_cost']}
   Estimated Downtime: {rec['estimated_downtime']}
   Fault Codes: {', '.join(rec['fault_codes'])}
"""
        
        return details.strip()
    
    def generate_natural_language_summary(self, vehicle_id: int,
                                         anomaly_result: Dict,
                                         root_cause_result: Dict,
                                         maintenance_result: Dict) -> str:
        """Generate natural language summary for non-technical users"""
        if not anomaly_result['anomaly_detected']:
            return (f"Good news! Vehicle {vehicle_id} is running smoothly. "
                   f"Our diagnostic system analyzed all sensor data and found no issues. "
                   f"Continue with regular maintenance schedule.")
        
        primary_cause = root_cause_result.get('primary_cause')
        top_priority = maintenance_result.get('top_priority')
        
        summary = f"Vehicle {vehicle_id} requires attention. "
        
        if primary_cause:
            summary += f"Our analysis detected {primary_cause['description'].lower()}. "
            
            if primary_cause['severity'] == 'critical':
                summary += "This is a critical issue that requires immediate attention. "
            elif primary_cause['severity'] == 'high':
                summary += "This is a high-priority issue that should be addressed soon. "
            else:
                summary += "This issue should be addressed during your next service visit. "
        
        if top_priority:
            summary += f"\n\nWhat you need to do: "
            immediate_actions = maintenance_result['action_plan']['immediate']
            if immediate_actions:
                summary += f"{immediate_actions[0]['action']} "
            
            summary += f"\n\nEstimated repair cost: {maintenance_result['total_cost']['cost_range']}. "
            summary += f"Expected downtime: {top_priority['estimated_downtime']}."
        
        return summary
    
    def generate_json_report(self, vehicle_id: int,
                            prepared_data: Dict,
                            anomaly_result: Dict,
                            root_cause_result: Dict,
                            maintenance_result: Dict) -> Dict:
        """Generate structured JSON report"""
        report = {
            'report_metadata': {
                'vehicle_id': vehicle_id,
                'report_timestamp': datetime.now().isoformat(),
                'report_version': '1.0',
                'analysis_timerange': prepared_data['time_range']
            },
            'anomaly_detection': {
                'anomaly_detected': anomaly_result['anomaly_detected'],
                'num_anomalies': anomaly_result['num_anomalies'],
                'anomaly_rate': anomaly_result['anomaly_rate'],
                'overall_score': anomaly_result['overall_score'],
                'anomalous_sensors': anomaly_result['anomalous_sensors']
            },
            'root_cause_analysis': {
                'root_causes': root_cause_result['root_causes'],
                'primary_cause': root_cause_result.get('primary_cause'),
                'failure_sequence': root_cause_result['failure_sequence'],
                'correlations': root_cause_result['correlations']
            },
            'maintenance_recommendations': {
                'recommendations': maintenance_result['recommendations'],
                'action_plan': maintenance_result['action_plan'],
                'total_cost': maintenance_result['total_cost'],
                'top_priority': maintenance_result.get('top_priority')
            }
        }
        
        return report
    
    def run(self, vehicle_id: int,
            prepared_data: Dict,
            anomaly_result: Dict,
            root_cause_result: Dict,
            maintenance_result: Dict) -> Dict:
        """
        Main execution method for the Report Generation Agent
        
        Args:
            vehicle_id: Vehicle ID
            prepared_data: Data from ingestion agent
            anomaly_result: Results from anomaly detection
            root_cause_result: Results from root cause analysis
            maintenance_result: Results from maintenance recommendations
            
        Returns:
            Dictionary containing complete diagnostic report
        """
        print(f"\n{'='*60}")
        print(f"REPORT GENERATION AGENT - Vehicle {vehicle_id}")
        print(f"{'='*60}")
        
        print("Generating comprehensive diagnostic report...")
        
        # Generate all report sections
        executive_summary = self.generate_executive_summary(
            vehicle_id, anomaly_result, root_cause_result, maintenance_result
        )
        
        anomaly_details = self.format_anomaly_details(anomaly_result)
        root_cause_details = self.format_root_cause_analysis(root_cause_result)
        maintenance_details = self.format_maintenance_recommendations(maintenance_result)
        
        natural_language_summary = self.generate_natural_language_summary(
            vehicle_id, anomaly_result, root_cause_result, maintenance_result
        )
        
        json_report = self.generate_json_report(
            vehicle_id, prepared_data, anomaly_result, root_cause_result, maintenance_result
        )
        
        # Compile full report
        full_report = f"""
{'='*60}
VEHICLE DIAGNOSTIC REPORT
Vehicle ID: {vehicle_id}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

EXECUTIVE SUMMARY
{'='*60}
{executive_summary}

{anomaly_details}

{root_cause_details}

{maintenance_details}

{'='*60}
PLAIN LANGUAGE SUMMARY
{'='*60}
{natural_language_summary}

{'='*60}
END OF REPORT
{'='*60}
"""
        
        print("✓ Generated executive summary")
        print("✓ Generated anomaly detection details")
        print("✓ Generated root cause analysis")
        print("✓ Generated maintenance recommendations")
        print("✓ Generated natural language summary")
        print("✓ Generated JSON report")
        
        print(f"\n✓ Complete diagnostic report generated")
        print(f"{'='*60}\n")
        
        result = {
            'vehicle_id': vehicle_id,
            'full_report': full_report,
            'executive_summary': executive_summary,
            'natural_language_summary': natural_language_summary,
            'json_report': json_report,
            'report_timestamp': datetime.now().isoformat()
        }
        
        return result


if __name__ == '__main__':
    # Test the Report Generation Agent
    from data_ingestion_agent import DataIngestionAgent
    from anomaly_detection_agent import AnomalyDetectionAgent
    from root_cause_agent import RootCauseAnalysisAgent
    from maintenance_recommendation_agent import MaintenanceRecommendationAgent
    
    # Run full pipeline
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
        
        detection_agent = AnomalyDetectionAgent()
        anomaly_result = detection_agent.run(prepared_data)
        
        rca_agent = RootCauseAnalysisAgent()
        rca_result = rca_agent.run(anomaly_result)
        
        maintenance_agent = MaintenanceRecommendationAgent()
        maintenance_result = maintenance_agent.run(rca_result)
        
        # Generate report
        report_agent = ReportGenerationAgent()
        report = report_agent.run(test_vehicle_id, prepared_data, anomaly_result, 
                                 rca_result, maintenance_result)
        
        print("\n" + "="*60)
        print("SAMPLE REPORT OUTPUT")
        print("="*60)
        print(report['full_report'][:1000] + "...")
