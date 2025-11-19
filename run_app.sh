#!/bin/bash
# Script to run the DR-TB Prediction Web Interface

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run app.py

