# 🚗 Vehicle Diagnostics Agent

A production-ready multi-agent AI system for predictive vehicle diagnostics using LangChain, LangGraph, and PyTorch.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Overview

This system employs a sophisticated multi-agent architecture to analyze vehicle sensor data, detect anomalies, identify root causes, and provide actionable maintenance recommendations. It demonstrates production-grade ML pipeline development with agentic AI orchestration.

### Key Features

- ✅ **Real-time Anomaly Detection** - LSTM-based neural network for pattern recognition
- ✅ **Root Cause Analysis** - Rule-based and ML-driven fault identification
- ✅ **Maintenance Recommendations** - Actionable steps with cost estimates
- ✅ **Multi-Agent Orchestration** - LangGraph-powered workflow coordination
- ✅ **REST API** - FastAPI backend for programmatic access
- ✅ **Interactive UI** - Gradio-based dashboard for easy visualization
- ✅ **Comprehensive Reports** - Natural language summaries and detailed diagnostics
- ✅ **Docker Support** - Containerized deployment ready

## 🏗️ Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph Orchestrator                     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Data Ingestion│   │    Anomaly    │   │  Root Cause   │
│     Agent     │──▶│   Detection   │──▶│   Analysis    │
│               │   │     Agent     │   │     Agent     │
└───────────────┘   └───────────────┘   └───────────────┘
                                                │
                    ┌───────────────────────────┘
                    ▼
        ┌───────────────────┐       ┌───────────────────┐
        │  Maintenance      │──────▶│     Report        │
        │ Recommendation    │       │   Generation      │
        │      Agent        │       │      Agent        │
        └───────────────────┘       └───────────────────┘
```

### Technology Stack

- **ML Framework:** PyTorch (LSTM neural networks)
- **Agent Orchestration:** LangChain & LangGraph
- **Backend:** FastAPI
- **Frontend:** Gradio
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Deployment:** Docker & Docker Compose

## 📊 Dataset

The system analyzes vehicle sensor data including:

- Engine temperature, RPM, speed
- Battery voltage and health indicators
- Oil pressure and fuel pressure
- Coolant temperature
- Tire pressure (all four wheels)
- Vibration levels
- Throttle position and brake temperature

**Dataset Statistics:**
- 100 vehicles
- 500 timesteps per vehicle
- 50,000 total sensor readings
- ~30% anomaly rate
- 14 sensor measurements per timestep

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Conda (recommended)
- CUDA-capable GPU (optional, for faster training)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd VehicleDiagnosticsAgent
```

2. **Create conda environment:**
```bash
conda create -n vda python=3.10 -y
conda activate vda
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

4. **Generate dataset and train model:**
```bash
# Generate synthetic vehicle data
python src/utils/download_data.py

# Preprocess data
python src/utils/data_preprocessing.py

# Train anomaly detection model
python src/models/train_anomaly_detector.py
```

### Running the Application

#### Option 1: Gradio UI (Recommended for Demo)

```bash
python src/ui/gradio_app.py
```

Access the UI at: `http://localhost:7860`

#### Option 2: FastAPI Backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API documentation at: `http://localhost:8000/docs`

#### Option 3: Docker Compose (Production)

```bash
docker-compose up --build
```

- API: `http://localhost:8000`
- UI: `http://localhost:7860`

### Testing

Run the test suite:

```bash
pytest tests/ -v
```

## 📖 Usage Examples

### Python API

```python
from src.orchestrator import VehicleDiagnosticOrchestrator

# Initialize orchestrator
orchestrator = VehicleDiagnosticOrchestrator()

# Run diagnostic for a vehicle
result = orchestrator.diagnose_vehicle(vehicle_id=32, n_readings=200)

# Access results
if result['success']:
    print(result['report']['natural_language_summary'])
    print(f"Anomaly Score: {result['anomaly_result']['overall_score']}")
```

### REST API

```bash
# Health check
curl http://localhost:8000/health

# List available vehicles
curl http://localhost:8000/vehicles

# Run diagnostic
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"vehicle_id": 32, "n_readings": 200}'

# Get full report
curl http://localhost:8000/report/32
```

## 🧪 Project Structure

```
VehicleDiagnosticsAgent/
├── data/
│   ├── raw/                    # Raw sensor data
│   └── processed/              # Preprocessed data
├── src/
│   ├── agents/                 # Individual agent implementations
│   │   ├── data_ingestion_agent.py
│   │   ├── anomaly_detection_agent.py
│   │   ├── root_cause_agent.py
│   │   ├── maintenance_recommendation_agent.py
│   │   └── report_generation_agent.py
│   ├── models/                 # ML models
│   │   ├── anomaly_detector.py
│   │   ├── train_anomaly_detector.py
│   │   └── best_anomaly_detector.pth
│   ├── utils/                  # Utility functions
│   │   ├── download_data.py
│   │   └── data_preprocessing.py
│   ├── api/                    # FastAPI backend
│   │   └── main.py
│   ├── ui/                     # Gradio frontend
│   │   └── gradio_app.py
│   └── orchestrator.py         # LangGraph orchestration
├── tests/                      # Unit tests
│   └── test_agents.py
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── README.md                   # This file
└── project.md                  # Project specification
```

## 🔬 Model Performance

### Anomaly Detection Model (LSTM)

- **Architecture:** 2-layer LSTM with 64 hidden units
- **Training:** 20 epochs on 35,000 sequences
- **Validation Accuracy:** 99.53%
- **Best Validation Loss:** 0.0409
- **Sequence Length:** 50 timesteps
- **Features:** 60 engineered features

### Agent Performance

| Agent | Execution Time | Accuracy |
|-------|---------------|----------|
| Data Ingestion | ~0.1s | N/A |
| Anomaly Detection | ~0.5s | 99.5% |
| Root Cause Analysis | ~0.2s | High |
| Maintenance Recommendation | ~0.1s | N/A |
| Report Generation | ~0.1s | N/A |

**Total Pipeline:** ~1 second per vehicle

## 🎓 Key Learnings & Skills Demonstrated

- ✅ Multi-agent AI system design and orchestration
- ✅ Production-grade ML pipeline development
- ✅ Time-series anomaly detection with deep learning
- ✅ LangChain/LangGraph for agent coordination
- ✅ FastAPI backend development
- ✅ Interactive UI with Gradio
- ✅ Docker containerization
- ✅ Data engineering and preprocessing
- ✅ Domain knowledge integration (automotive diagnostics)
- ✅ Testing and validation strategies

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# UI Configuration
UI_HOST=0.0.0.0
UI_PORT=7860

# Model Configuration
MODEL_PATH=src/models/best_anomaly_detector.pth
```

## 📈 Future Enhancements

- [ ] Real-time streaming data support
- [ ] Integration with actual OBD-II devices
- [ ] Advanced visualization dashboard
- [ ] Multi-model ensemble for anomaly detection
- [ ] Predictive maintenance scheduling
- [ ] Integration with vehicle manufacturer APIs
- [ ] Mobile application
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] LLM integration for natural language queries
- [ ] Historical trend analysis

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Vehicle Diagnostics Team** - Initial work

## 🙏 Acknowledgments

- NASA Prognostics Center for dataset inspiration
- LangChain team for agent orchestration framework
- PyTorch team for deep learning framework
- FastAPI and Gradio teams for excellent frameworks

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for the automotive and AI community**
