#!/bin/bash
# Production start script using Gunicorn
cd backend

# Create necessary directories
mkdir -p uploads
mkdir -p models

# Set production environment variables
export PYTHONPATH="${PYTHONPATH}:."

# Start the application with better logging
exec gunicorn app.main:app \
    -w 2 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:$PORT \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --timeout 120 \
    --keep-alive 5
