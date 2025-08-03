#!/bin/bash
# Minimal start script - uses python module directly
echo "Starting AI Tennis with minimal configuration..."

# Change to backend directory
cd backend

# Create directories
mkdir -p uploads models

# Debug info
echo "Current directory: $(pwd)"
echo "Python version: $(python --version 2>&1 || python3 --version 2>&1)"
echo "Which python: $(which python || which python3 || echo 'not found')"

# Try to start the server using python module
if command -v python >/dev/null 2>&1; then
    echo "Using python command..."
    exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
elif command -v python3 >/dev/null 2>&1; then
    echo "Using python3 command..."
    exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
else
    echo "No python found, trying direct uvicorn..."
    # Last resort: try to find uvicorn directly
    if command -v uvicorn >/dev/null 2>&1; then
        exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
    else
        echo "ERROR: No python or uvicorn found"
        exit 1
    fi
fi
