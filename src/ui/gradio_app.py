"""
Gradio UI for Vehicle Diagnostics Agent
"""
import gradio as gr
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from orchestrator import VehicleDiagnosticOrchestrator
from agents.data_ingestion_agent import DataIngestionAgent

# Initialize components
orchestrator = VehicleDiagnosticOrchestrator()
ingestion_agent = DataIngestionAgent()

# Load available vehicles
test_df = ingestion_agent.load_test_data()
available_vehicles = sorted(test_df['vehicle_id'].unique().tolist())


def run_diagnostic(vehicle_id, n_readings):
    """Run diagnostic for a vehicle"""
    try:
        vehicle_id = int(vehicle_id)
        n_readings = int(n_readings) if n_readings else None
        
        # Run diagnostic
        result = orchestrator.diagnose_vehicle(vehicle_id, n_readings)
        
        if not result['success']:
            return f"❌ Error: {result.get('error')}", "", "", None
        
        # Extract results
        anomaly_result = result.get('anomaly_result', {})
        report = result.get('report', {})
        
        # Status summary
        if anomaly_result.get('anomaly_detected'):
            status = f"""
## 🚨 ALERT: Anomalies Detected

**Vehicle ID:** {vehicle_id}  
**Anomaly Score:** {anomaly_result.get('overall_score', 0):.3f}  
**Anomalous Readings:** {anomaly_result.get('num_anomalies', 0)} / {len(anomaly_result.get('anomaly_predictions', []))} ({anomaly_result.get('anomaly_rate', 0):.1%})  
**Status:** ⚠️ Requires Attention
"""
        else:
            status = f"""
## ✅ Vehicle Healthy

**Vehicle ID:** {vehicle_id}  
**Status:** 🟢 All Systems Normal  
**Anomaly Score:** {anomaly_result.get('overall_score', 0):.3f}
"""
        
        # Natural language summary
        nl_summary = report.get('natural_language_summary', 'No summary available')
        
        # Full report
        full_report = report.get('full_report', 'No report available')
        
        # Create visualization
        fig = create_anomaly_visualization(anomaly_result)
        
        return status, nl_summary, full_report, fig
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", None


def create_anomaly_visualization(anomaly_result):
    """Create visualization of anomaly detection results"""
    try:
        timestamps = anomaly_result.get('timestamps', [])
        predictions = anomaly_result.get('anomaly_predictions', [])
        scores = anomaly_result.get('anomaly_scores', [])
        
        if len(timestamps) == 0:
            return None
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add anomaly predictions
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=predictions,
            mode='lines',
            name='Anomaly Detected',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        # Add anomaly scores
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=scores,
            mode='lines',
            name='Anomaly Score',
            line=dict(color='orange', width=1, dash='dot'),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title='Anomaly Detection Over Time',
            xaxis_title='Timestamp',
            yaxis_title='Anomaly Detected (0/1)',
            yaxis2=dict(
                title='Anomaly Score',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
        
    except Exception as e:
        print(f"Visualization error: {e}")
        return None


def get_vehicle_info(vehicle_id):
    """Get basic info about a vehicle"""
    try:
        vehicle_id = int(vehicle_id)
        vehicle_data = test_df[test_df['vehicle_id'] == vehicle_id]
        
        if len(vehicle_data) == 0:
            return "Vehicle not found"
        
        num_readings = len(vehicle_data)
        has_anomalies = vehicle_data['anomaly'].sum() > 0
        num_anomalies = vehicle_data['anomaly'].sum()
        
        info = f"""
### Vehicle Information

**Vehicle ID:** {vehicle_id}  
**Total Readings:** {num_readings}  
**Known Anomalies:** {num_anomalies} ({num_anomalies/num_readings:.1%})  
**Status:** {'⚠️ Has anomalies' if has_anomalies else '✅ Healthy'}
"""
        return info
        
    except Exception as e:
        return f"Error: {str(e)}"


def list_vehicles_with_anomalies():
    """List vehicles that have anomalies"""
    vehicles_with_anomalies = []
    
    for vid in available_vehicles[:50]:  # Limit to first 50
        vehicle_data = test_df[test_df['vehicle_id'] == vid]
        if vehicle_data['anomaly'].sum() > 0:
            vehicles_with_anomalies.append({
                'Vehicle ID': vid,
                'Total Readings': len(vehicle_data),
                'Anomalies': int(vehicle_data['anomaly'].sum()),
                'Anomaly Rate': f"{vehicle_data['anomaly'].sum()/len(vehicle_data):.1%}"
            })
    
    if vehicles_with_anomalies:
        df = pd.DataFrame(vehicles_with_anomalies)
        return df
    else:
        return pd.DataFrame({'Message': ['No vehicles with anomalies found']})


# Create Gradio interface
with gr.Blocks(title="Vehicle Diagnostics Agent") as demo:
    gr.Markdown("""
    # 🚗 Vehicle Diagnostics Agent
    ### Multi-Agent AI System for Predictive Vehicle Diagnostics
    
    This system uses advanced AI agents to analyze vehicle sensor data, detect anomalies, 
    identify root causes, and provide actionable maintenance recommendations.
    """)
    
    with gr.Tab("🔍 Single Vehicle Diagnostic"):
        gr.Markdown("### Analyze a single vehicle")
        
        with gr.Row():
            with gr.Column(scale=1):
                vehicle_id_input = gr.Dropdown(
                    choices=available_vehicles,
                    label="Select Vehicle ID",
                    value=available_vehicles[0] if available_vehicles else None
                )
                n_readings_input = gr.Number(
                    label="Number of Recent Readings (optional)",
                    value=200,
                    precision=0
                )
                
                diagnose_btn = gr.Button("🔬 Run Diagnostic", variant="primary", size="lg")
                
                gr.Markdown("---")
                vehicle_info_output = gr.Markdown(label="Vehicle Info")
                
                # Auto-update vehicle info when selection changes
                vehicle_id_input.change(
                    fn=get_vehicle_info,
                    inputs=[vehicle_id_input],
                    outputs=[vehicle_info_output]
                )
        
            with gr.Column(scale=2):
                status_output = gr.Markdown(label="Diagnostic Status")
                summary_output = gr.Textbox(
                    label="📋 Summary",
                    lines=5,
                    max_lines=10
                )
                
        with gr.Row():
            anomaly_plot = gr.Plot(label="Anomaly Detection Visualization")
        
        with gr.Row():
            full_report_output = gr.Textbox(
                label="📄 Full Diagnostic Report",
                lines=20,
                max_lines=30
            )
        
        diagnose_btn.click(
            fn=run_diagnostic,
            inputs=[vehicle_id_input, n_readings_input],
            outputs=[status_output, summary_output, full_report_output, anomaly_plot]
        )
    
    with gr.Tab("📊 Vehicle Overview"):
        gr.Markdown("### Vehicles with Known Anomalies")
        
        refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
        vehicles_table = gr.Dataframe(
            value=list_vehicles_with_anomalies(),
            label="Vehicles Requiring Attention"
        )
        
        refresh_btn.click(
            fn=list_vehicles_with_anomalies,
            inputs=[],
            outputs=[vehicles_table]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About Vehicle Diagnostics Agent
        
        ### System Architecture
        
        This system employs a multi-agent architecture with the following components:
        
        1. **Data Ingestion Agent** - Loads and prepares vehicle sensor data
        2. **Anomaly Detection Agent** - Uses LSTM neural networks to detect unusual patterns
        3. **Root Cause Analysis Agent** - Identifies the underlying causes of anomalies
        4. **Maintenance Recommendation Agent** - Provides actionable maintenance steps
        5. **Report Generation Agent** - Creates comprehensive diagnostic reports
        
        ### Technology Stack
        
        - **ML Framework:** PyTorch (LSTM-based anomaly detection)
        - **Orchestration:** LangGraph for multi-agent coordination
        - **Backend:** FastAPI for REST API
        - **Frontend:** Gradio for interactive UI
        - **Data Processing:** Pandas, NumPy, Scikit-learn
        
        ### Features
        
        - ✅ Real-time anomaly detection
        - ✅ Root cause analysis with fault code mapping
        - ✅ Maintenance cost estimation
        - ✅ Natural language summaries
        - ✅ Interactive visualizations
        - ✅ Batch processing support
        
        ### Dataset
        
        The system analyzes synthetic vehicle sensor data including:
        - Engine temperature, RPM, speed
        - Battery voltage and health
        - Oil and fuel pressure
        - Tire pressure (all four wheels)
        - Vibration levels
        - And more...
        
        ---
        
        **Version:** 1.0.0  
        **Author:** Vehicle Diagnostics Team  
        **License:** MIT
        """)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
