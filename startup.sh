#!/bin/bash
# Startup verification script
echo "Starting AI Tennis application..."

# Check if we're in the right directory
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

# Check if gunicorn is available
if ! command -v gunicorn &> /dev/null; then
    echo "Error: gunicorn not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if required directories exist
mkdir -p backend/uploads
mkdir -p backend/models

echo "Pre-flight checks passed. Starting application..."
exec ./start-prod.sh
