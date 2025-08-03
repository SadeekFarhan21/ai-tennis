#!/bin/bash
# Production start script - tries gunicorn, falls back to uvicorn
set -e

echo "Starting AI Tennis production server..."

# Change to backend directory
cd backend

# Create necessary directories
mkdir -p uploads
mkdir -p models

# Set production environment variables
export PYTHONPATH="${PYTHONPATH}:."

# Try gunicorn first
if command -v gunicorn &> /dev/null; then
    echo "Using gunicorn..."
    exec gunicorn app.main:app \
        -w 2 \
        -k uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:$PORT \
        --access-logfile - \
        --error-logfile - \
        --log-level info \
        --timeout 120 \
        --keep-alive 5
else
    echo "gunicorn not found, falling back to uvicorn..."
    # Fallback to uvicorn
    if command -v uvicorn &> /dev/null; then
        echo "Using uvicorn..."
        exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
    else
        echo "Neither gunicorn nor uvicorn found. Trying python -m uvicorn..."
        exec python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
    fi
fi
