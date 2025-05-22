#!/bin/bash
# Quick start script for Vehicle Diagnostics Agent API

echo "=========================================="
echo "Vehicle Diagnostics Agent - FastAPI"
echo "=========================================="
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "vda" ]]; then
    echo "⚠️  Warning: vda conda environment not activated"
    echo "Please run: conda activate vda"
    echo ""
fi

# Check if model exists
if [ ! -f "src/models/best_anomaly_detector.pth" ]; then
    echo "❌ Model not found. Please train the model first:"
    echo "   python src/models/train_anomaly_detector.py"
    exit 1
fi

# Check if data exists
if [ ! -f "data/processed/test.csv" ]; then
    echo "❌ Processed data not found. Please run preprocessing:"
    echo "   python src/utils/data_preprocessing.py"
    exit 1
fi

echo "✅ Starting FastAPI server..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
