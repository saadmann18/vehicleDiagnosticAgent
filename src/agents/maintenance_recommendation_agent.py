"""
Maintenance Recommendation Agent - Provides actionable maintenance recommendations
"""
from typing import Dict, List


class MaintenanceRecommendationAgent:
    """
    Agent responsible for generating maintenance recommendations based on root cause analysis
    """
    
    def __init__(self):
        # Define maintenance actions for each fault type
        self.maintenance_actions = {
            'engine_overheating': {
                'immediate_actions': [
                    'Stop vehicle immediately and allow engine to cool',
                    'Check coolant level and top up if low',
                    'Inspect for coolant leaks'
                ],
                'short_term_actions': [
                    'Replace thermostat if faulty',
                    'Flush and replace coolant',
                    'Check radiator fan operation',
                    'Inspect water pump for proper operation'
                ],
                'long_term_actions': [
                    'Schedule comprehensive cooling system inspection',
                    'Consider radiator replacement if old',
                    'Regular coolant system maintenance every 30,000 miles'
                ],
                'estimated_cost': '$200-$800',
                'urgency': 'critical',
                'downtime': '1-3 days'
            },
            'cooling_system_failure': {
                'immediate_actions': [
                    'Do not operate vehicle',
                    'Tow to service center'
                ],
                'short_term_actions': [
                    'Diagnose cooling system failure',
                    'Replace failed components (radiator, water pump, thermostat)',
                    'Pressure test cooling system'
                ],
                'long_term_actions': [
                    'Monitor coolant levels regularly',
                    'Annual cooling system inspection'
                ],
                'estimated_cost': '$500-$1500',
                'urgency': 'critical',
                'downtime': '2-5 days'
            },
            'oil_pressure_low': {
                'immediate_actions': [
                    'Stop engine immediately',
                    'Check oil level',
                    'Do not restart until issue is resolved'
                ],
                'short_term_actions': [
                    'Add oil if level is low',
                    'Check for oil leaks',
                    'Replace oil pressure sensor if faulty',
                    'Inspect oil pump',
                    'Change oil and filter'
                ],
                'long_term_actions': [
                    'Regular oil changes every 5,000 miles',
                    'Use recommended oil grade',
                    'Monitor oil consumption'
                ],
                'estimated_cost': '$100-$600',
                'urgency': 'critical',
                'downtime': '1-2 days'
            },
            'battery_degradation': {
                'immediate_actions': [
                    'Test battery voltage',
                    'Check battery terminals for corrosion'
                ],
                'short_term_actions': [
                    'Clean battery terminals',
                    'Test alternator output',
                    'Replace battery if failing load test',
                    'Check for parasitic drain'
                ],
                'long_term_actions': [
                    'Replace battery every 3-5 years',
                    'Regular battery maintenance',
                    'Keep terminals clean'
                ],
                'estimated_cost': '$100-$300',
                'urgency': 'high',
                'downtime': '0.5-1 day'
            },
            'tire_pressure_issue': {
                'immediate_actions': [
                    'Check tire pressure on all tires',
                    'Inflate to recommended PSI',
                    'Inspect for punctures or damage'
                ],
                'short_term_actions': [
                    'Repair or replace damaged tire',
                    'Check valve stems',
                    'Inspect for slow leaks',
                    'Rotate tires if needed'
                ],
                'long_term_actions': [
                    'Check tire pressure monthly',
                    'Regular tire rotation every 5,000-7,500 miles',
                    'Replace tires when tread depth is low'
                ],
                'estimated_cost': '$20-$200',
                'urgency': 'medium',
                'downtime': '0.5-1 day'
            },
            'excessive_vibration': {
                'immediate_actions': [
                    'Reduce speed',
                    'Note when vibration occurs (speed, braking, etc.)'
                ],
                'short_term_actions': [
                    'Balance and rotate tires',
                    'Check wheel alignment',
                    'Inspect suspension components',
                    'Check brake rotors for warping',
                    'Inspect engine mounts'
                ],
                'long_term_actions': [
                    'Regular tire balancing',
                    'Annual suspension inspection',
                    'Replace worn suspension components'
                ],
                'estimated_cost': '$100-$500',
                'urgency': 'high',
                'downtime': '1-2 days'
            },
            'fuel_system_issue': {
                'immediate_actions': [
                    'Note any performance issues',
                    'Check for fuel leaks'
                ],
                'short_term_actions': [
                    'Replace fuel filter',
                    'Test fuel pump pressure',
                    'Clean fuel injectors',
                    'Inspect fuel lines'
                ],
                'long_term_actions': [
                    'Use quality fuel',
                    'Replace fuel filter every 30,000 miles',
                    'Add fuel system cleaner periodically'
                ],
                'estimated_cost': '$150-$600',
                'urgency': 'high',
                'downtime': '1-2 days'
            },
            'engine_stress': {
                'immediate_actions': [
                    'Reduce engine load',
                    'Avoid high RPM operation'
                ],
                'short_term_actions': [
                    'Check air filter',
                    'Inspect spark plugs',
                    'Verify proper fuel octane rating',
                    'Check for engine codes'
                ],
                'long_term_actions': [
                    'Regular tune-ups',
                    'Avoid aggressive driving',
                    'Use recommended fuel grade'
                ],
                'estimated_cost': '$100-$400',
                'urgency': 'medium',
                'downtime': '0.5-1 day'
            }
        }
    
    def generate_recommendations(self, root_causes: List[Dict]) -> List[Dict]:
        """
        Generate maintenance recommendations based on root causes
        
        Args:
            root_causes: List of identified root causes
            
        Returns:
            List of maintenance recommendations
        """
        recommendations = []
        
        for cause in root_causes:
            fault_name = cause['fault_name']
            
            if fault_name in self.maintenance_actions:
                actions = self.maintenance_actions[fault_name]
                
                recommendation = {
                    'fault_name': fault_name,
                    'description': cause['description'],
                    'severity': cause['severity'],
                    'confidence': cause['confidence'],
                    'fault_codes': cause['fault_codes'],
                    'immediate_actions': actions['immediate_actions'],
                    'short_term_actions': actions['short_term_actions'],
                    'long_term_actions': actions['long_term_actions'],
                    'estimated_cost': actions['estimated_cost'],
                    'urgency': actions['urgency'],
                    'estimated_downtime': actions['downtime']
                }
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def prioritize_actions(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Prioritize maintenance actions based on urgency and severity
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Prioritized list of actions
        """
        urgency_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        # Sort by urgency and confidence
        prioritized = sorted(
            recommendations,
            key=lambda x: (urgency_order.get(x['urgency'], 4), -x['confidence'])
        )
        
        return prioritized
    
    def calculate_total_cost(self, recommendations: List[Dict]) -> Dict:
        """
        Calculate estimated total maintenance cost
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Dictionary with cost estimates
        """
        total_min = 0
        total_max = 0
        
        for rec in recommendations:
            cost_str = rec['estimated_cost']
            # Parse cost range like "$200-$800"
            costs = cost_str.replace('$', '').split('-')
            if len(costs) == 2:
                total_min += int(costs[0])
                total_max += int(costs[1])
        
        return {
            'min_cost': total_min,
            'max_cost': total_max,
            'cost_range': f'${total_min}-${total_max}'
        }
    
    def generate_action_plan(self, recommendations: List[Dict]) -> Dict:
        """
        Generate a comprehensive action plan
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Dictionary containing action plan
        """
        if not recommendations:
            return {
                'immediate': [],
                'short_term': [],
                'long_term': [],
                'total_actions': 0
            }
        
        immediate = []
        short_term = []
        long_term = []
        
        for rec in recommendations:
            # Add immediate actions
            for action in rec['immediate_actions']:
                immediate.append({
                    'action': action,
                    'related_to': rec['fault_name'],
                    'urgency': rec['urgency']
                })
            
            # Add short-term actions
            for action in rec['short_term_actions']:
                short_term.append({
                    'action': action,
                    'related_to': rec['fault_name'],
                    'estimated_cost': rec['estimated_cost']
                })
            
            # Add long-term actions
            for action in rec['long_term_actions']:
                long_term.append({
                    'action': action,
                    'related_to': rec['fault_name']
                })
        
        return {
            'immediate': immediate,
            'short_term': short_term,
            'long_term': long_term,
            'total_actions': len(immediate) + len(short_term) + len(long_term)
        }
    
    def run(self, root_cause_result: Dict) -> Dict:
        """
        Main execution method for the Maintenance Recommendation Agent
        
        Args:
            root_cause_result: Results from Root Cause Analysis Agent
            
        Returns:
            Dictionary containing maintenance recommendations
        """
        print(f"\n{'='*60}")
        print(f"MAINTENANCE RECOMMENDATION AGENT - Vehicle {root_cause_result['vehicle_id']}")
        print(f"{'='*60}")
        
        root_causes = root_cause_result['root_causes']
        
        if not root_causes:
            print("✓ No maintenance recommendations needed - vehicle is healthy")
            print(f"{'='*60}\n")
            return {
                'vehicle_id': root_cause_result['vehicle_id'],
                'recommendations': [],
                'action_plan': {},
                'total_cost': {'min_cost': 0, 'max_cost': 0, 'cost_range': '$0'},
                'summary': 'No maintenance required'
            }
        
        print(f"Generating recommendations for {len(root_causes)} identified issues...")
        
        # Generate recommendations
        recommendations = self.generate_recommendations(root_causes)
        print(f"✓ Generated {len(recommendations)} maintenance recommendations")
        
        # Prioritize actions
        prioritized_recommendations = self.prioritize_actions(recommendations)
        
        # Calculate total cost
        total_cost = self.calculate_total_cost(recommendations)
        print(f"✓ Estimated total cost: {total_cost['cost_range']}")
        
        # Generate action plan
        action_plan = self.generate_action_plan(prioritized_recommendations)
        print(f"✓ Action plan created:")
        print(f"    - Immediate actions: {len(action_plan['immediate'])}")
        print(f"    - Short-term actions: {len(action_plan['short_term'])}")
        print(f"    - Long-term actions: {len(action_plan['long_term'])}")
        
        # Display top priority recommendation
        if prioritized_recommendations:
            top_rec = prioritized_recommendations[0]
            print(f"\n✓ Top priority: {top_rec['fault_name']}")
            print(f"    Urgency: {top_rec['urgency']}")
            print(f"    Estimated cost: {top_rec['estimated_cost']}")
            print(f"    Downtime: {top_rec['estimated_downtime']}")
            
            if action_plan['immediate']:
                print(f"\n  Immediate actions required:")
                for action in action_plan['immediate'][:3]:
                    print(f"    • {action['action']}")
        
        summary = (f"{len(recommendations)} maintenance items identified. "
                  f"Estimated cost: {total_cost['cost_range']}. "
                  f"Highest priority: {prioritized_recommendations[0]['urgency']} urgency.")
        
        print(f"\n✓ Summary: {summary}")
        print(f"{'='*60}\n")
        
        result = {
            'vehicle_id': root_cause_result['vehicle_id'],
            'recommendations': prioritized_recommendations,
            'action_plan': action_plan,
            'total_cost': total_cost,
            'summary': summary,
            'top_priority': prioritized_recommendations[0] if prioritized_recommendations else None
        }
        
        return result


if __name__ == '__main__':
    # Test the Maintenance Recommendation Agent
    from data_ingestion_agent import DataIngestionAgent
    from anomaly_detection_agent import AnomalyDetectionAgent
    from root_cause_agent import RootCauseAnalysisAgent
    
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
        detection_agent = AnomalyDetectionAgent()
        anomaly_result = detection_agent.run(prepared_data)
        rca_agent = RootCauseAnalysisAgent()
        rca_result = rca_agent.run(anomaly_result)
        
        # Generate recommendations
        maintenance_agent = MaintenanceRecommendationAgent()
        result = maintenance_agent.run(rca_result)
        
        print(f"\nMaintenance Summary:")
        print(f"  Recommendations: {len(result['recommendations'])}")
        print(f"  Total cost: {result['total_cost']['cost_range']}")
