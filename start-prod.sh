#!/bin/bash
# Production start script using Gunicorn
cd backend

# Create necessary directories
mkdir -p uploads
mkdir -p models

# Start the application
exec gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
