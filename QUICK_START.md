# 🚀 Quick Start Guide - Vehicle Diagnostics Agent

## ✅ Current Status

**The system is fully operational!** 

- ✅ Conda environment: `vda` (active)
- ✅ Dataset: Generated (50,000 records)
- ✅ Model: Trained (99.53% accuracy)
- ✅ All agents: Implemented and tested
- ✅ Gradio UI: Running at http://localhost:7860
- ✅ Tests: All 12 tests passing

---

## 🎯 Access the System

### Gradio UI (Currently Running)
```
URL: http://localhost:7860
```

The Gradio interface is already running in your cascade terminal!

**Features:**
- 🔍 Single vehicle diagnostics
- 📊 Vehicle overview with anomaly list
- 📋 Full diagnostic reports
- 📈 Interactive visualizations

---

## 🔧 Running Different Components

### 1. Gradio UI (Interactive Dashboard)
```bash
# If not already running:
python src/ui/gradio_app.py

# Or use the quick start script:
./run_ui.sh
```

### 2. FastAPI Backend (REST API)
```bash
# Start the API server:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or use the quick start script:
./run_api.sh
```

**API Endpoints:**
- `http://localhost:8000` - Root
- `http://localhost:8000/docs` - Interactive API documentation
- `http://localhost:8000/health` - Health check
- `http://localhost:8000/vehicles` - List vehicles
- `http://localhost:8000/diagnose` - Run diagnostic

### 3. Python Script (Direct)
```bash
# Run the demo script:
python demo.py

# Or test the orchestrator:
python src/orchestrator.py
```

### 4. Docker (Production Deployment)
```bash
# Build and run with Docker Compose:
docker-compose up --build

# Access:
# - API: http://localhost:8000
# - UI: http://localhost:7860
```

---

## 📝 Quick Examples

### Example 1: Using Gradio UI

1. Open http://localhost:7860 in your browser
2. Go to "Single Vehicle Diagnostic" tab
3. Select a vehicle ID from the dropdown
4. Set number of readings (e.g., 200)
5. Click "Run Diagnostic"
6. View results, visualizations, and full report

### Example 2: Using Python API

```python
from src.orchestrator import VehicleDiagnosticOrchestrator

# Initialize
orchestrator = VehicleDiagnosticOrchestrator()

# Run diagnostic
result = orchestrator.diagnose_vehicle(vehicle_id=32, n_readings=200)

# Access results
if result['success']:
    print(result['report']['natural_language_summary'])
    print(f"Anomaly Score: {result['anomaly_result']['overall_score']}")
```

### Example 3: Using REST API

```bash
# Health check
curl http://localhost:8000/health

# List vehicles
curl http://localhost:8000/vehicles

# Run diagnostic
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"vehicle_id": 32, "n_readings": 200}'

# Get full report
curl http://localhost:8000/report/32
```

---

## 🧪 Testing

```bash
# Run all tests:
pytest tests/ -v

# Run specific test:
pytest tests/test_agents.py::TestDataIngestionAgent -v

# Run with coverage:
pytest tests/ --cov=src --cov-report=html
```

**Current Test Results:**
- ✅ 12/12 tests passing
- ✅ Execution time: ~3.24 seconds
- ✅ 100% success rate

---

## 📊 Sample Vehicles to Try

Based on the test data, here are some interesting vehicles:

**Vehicles with Anomalies:**
- Vehicle 32: High anomaly rate (~75%), cooling system issues
- Vehicle 8: Medium anomaly rate, multiple sensor issues
- Vehicle 15: Low anomaly rate, tire pressure issues

**Healthy Vehicles:**
- Vehicle 1: No anomalies detected
- Vehicle 2: Clean sensor readings
- Vehicle 5: Normal operation

---

## 🎨 Gradio UI Features

### Tab 1: Single Vehicle Diagnostic
- Select vehicle from dropdown
- Set number of readings to analyze
- View real-time diagnostic results
- See anomaly detection visualization
- Read natural language summary
- Access full technical report

### Tab 2: Vehicle Overview
- List all vehicles with anomalies
- See anomaly counts and rates
- Refresh list dynamically

### Tab 3: About
- System architecture
- Technology stack
- Feature list
- Dataset information

---

## 📁 Important Files

### Data Files
- `data/raw/vehicle_sensor_data.csv` - Raw sensor data
- `data/processed/train.csv` - Training data
- `data/processed/test.csv` - Test data
- `data/processed/scaler.pkl` - Feature scaler

### Model Files
- `src/models/best_anomaly_detector.pth` - Trained LSTM model

### Configuration
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker configuration
- `.gitignore` - Git ignore rules

### Documentation
- `README.md` - Comprehensive documentation
- `PROJECT_SUMMARY.md` - Project completion summary
- `QUICK_START.md` - This file

---

## 🔍 Troubleshooting

### Issue: Gradio UI not loading
**Solution:** Check if the UI is already running in another terminal. Only one instance can run on port 7860.

### Issue: Model not found error
**Solution:** Train the model first:
```bash
python src/models/train_anomaly_detector.py
```

### Issue: Data not found error
**Solution:** Generate and preprocess data:
```bash
python src/utils/download_data.py
python src/utils/data_preprocessing.py
```

### Issue: Import errors
**Solution:** Make sure vda conda environment is activated:
```bash
conda activate vda
```

### Issue: Port already in use
**Solution:** Change the port or stop the existing process:
```bash
# For Gradio (default 7860):
python src/ui/gradio_app.py  # Will auto-select next available port

# For FastAPI (default 8000):
uvicorn src.api.main:app --port 8001
```

---

## 🎯 Next Steps

1. **Explore the Gradio UI** - Try diagnosing different vehicles
2. **Test the API** - Use the FastAPI docs at `/docs`
3. **Run the demo** - Execute `python demo.py`
4. **Customize** - Modify agents for your use case
5. **Deploy** - Use Docker for production deployment

---

## 📞 Support

For issues or questions:
- Check `README.md` for detailed documentation
- Review `PROJECT_SUMMARY.md` for project overview
- Examine test files in `tests/` for usage examples

---

## 🎉 Success!

Your Vehicle Diagnostics Agent is fully operational and ready to use!

**Current Status:**
- ✅ System: Running
- ✅ UI: http://localhost:7860
- ✅ Model: Trained (99.53% accuracy)
- ✅ Data: Processed (50,000 records)
- ✅ Tests: Passing (12/12)

**Enjoy your multi-agent AI diagnostic system!** 🚗✨
