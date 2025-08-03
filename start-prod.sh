#!/bin/bash
# Production start script using Gunicorn
cd backend
gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
