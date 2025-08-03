#!/bin/bash
set -e

echo "🚀 Starting AI Tennis build process..."

# Update pip
echo "📦 Updating pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Verify key packages are installed
echo "🔍 Verifying installations..."
python3 -c "import uvicorn; print(f'✅ uvicorn: {uvicorn.__version__}')" || python -c "import uvicorn; print(f'✅ uvicorn: {uvicorn.__version__}')" || echo "❌ uvicorn failed"
python3 -c "import fastapi; print(f'✅ fastapi: {fastapi.__version__}')" || python -c "import fastapi; print(f'✅ fastapi: {fastapi.__version__}')" || echo "❌ fastapi failed"
python3 -c "import sqlalchemy; print(f'✅ sqlalchemy: {sqlalchemy.__version__}')" || python -c "import sqlalchemy; print(f'✅ sqlalchemy: {sqlalchemy.__version__}')" || echo "❌ sqlalchemy failed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p backend/uploads
mkdir -p backend/models
mkdir -p backend/logs

echo "✅ Build completed successfully!"
