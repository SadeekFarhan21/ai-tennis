#!/usr/bin/env python3
"""
Python-based startup script for Render deployment
This avoids shell PATH issues by using Python directly
"""
import os
import sys
import subprocess

def main():
    print("Starting AI Tennis application...")
    
    # Change to backend directory
    os.chdir('backend')
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Get port from environment
    port = os.environ.get('PORT', '8000')
    
    # Try gunicorn first, then uvicorn
    try:
        print("Attempting to start with gunicorn...")
        subprocess.run([
            sys.executable, '-m', 'gunicorn',
            'app.main:app',
            '-w', '2',
            '-k', 'uvicorn.workers.UvicornWorker',
            '--bind', f'0.0.0.0:{port}',
            '--access-logfile', '-',
            '--error-logfile', '-',
            '--log-level', 'info'
        ], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Gunicorn failed, falling back to uvicorn...")
        try:
            subprocess.run([
                sys.executable, '-m', 'uvicorn',
                'app.main:app',
                '--host', '0.0.0.0',
                '--port', port
            ], check=True)
        except Exception as e:
            print(f"Failed to start server: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()
