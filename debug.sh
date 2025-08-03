#!/bin/bash
# Debug script to check environment
echo "=== Environment Debug Info ==="
echo "Python version:"
python --version || python3 --version || echo "No python found"

echo -e "\nPython executable paths:"
which python || echo "python not in PATH"
which python3 || echo "python3 not in PATH"

echo -e "\nPIP version:"
pip --version || pip3 --version || echo "No pip found"

echo -e "\nInstalled packages:"
pip list | grep -E "(gunicorn|uvicorn|fastapi)" || echo "Key packages not found"

echo -e "\nPATH:"
echo $PATH

echo -e "\nPYTHONPATH:"
echo $PYTHONPATH

echo -e "\nCurrent directory:"
pwd

echo -e "\nDirectory contents:"
ls -la

echo -e "\nBackend directory contents:"
ls -la backend/ || echo "Backend directory not found"

echo -e "\nApp directory contents:"
ls -la backend/app/ || echo "App directory not found"

echo "=== End Debug Info ==="
