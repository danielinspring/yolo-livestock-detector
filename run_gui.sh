#!/bin/bash

# YOLO Combined Model - GUI Launcher
# This script activates the virtual environment and launches the Streamlit GUI

echo "🎯 Starting YOLO Combined Model GUI..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment and run Streamlit
source venv/bin/activate

echo "✓ Virtual environment activated"
echo "✓ Launching Streamlit..."
echo ""
echo "The GUI will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
