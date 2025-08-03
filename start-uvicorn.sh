#!/bin/bash
# Fallback start script using uvicorn
set -e

echo "Starting AI Tennis server with uvicorn..."
cd backend

# Create necessary directories
mkdir -p uploads
mkdir -p models

# Set environment
export PYTHONPATH="${PYTHONPATH}:."

# Start with uvicorn
exec uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
