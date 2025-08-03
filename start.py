#!/usr/bin/env python3
"""
Production startup script for AI Tennis application
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_basic_logging():
    """Setup basic logging for startup script"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - STARTUP - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = [
        "backend/uploads",
        "backend/models", 
        "backend/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory created/verified: {directory}")

def get_port():
    """Get port from environment"""
    port = os.getenv("PORT", "8000")
    print(f"🌐 Using port: {port}")
    return port

def get_python_executable():
    """Get the correct Python executable"""
    import shutil
    
    # Try to find python3 first, then python
    for python_cmd in ['python3', 'python']:
        if shutil.which(python_cmd):
            return python_cmd
    
    # If neither found, use sys.executable
    return sys.executable

def start_application():
    """Start the FastAPI application"""
    logger = setup_basic_logging()
    
    python_cmd = get_python_executable()
    logger.info("🚀 Starting AI Tennis Application")
    logger.info(f"🐍 Python executable: {python_cmd}")
    logger.info(f"🐍 Python version: {sys.version}")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'production')}")
    
    # Create directories
    create_directories()
    
    # Get configuration
    port = get_port()
    host = "0.0.0.0"
    
    # Change to backend directory
    os.chdir("backend")
    logger.info(f"📁 Changed to directory: {os.getcwd()}")
    
    # Add current directory to Python path
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Try to import the app first to check for issues
    try:
        logger.info("🔍 Testing app import...")
        import app.main
        logger.info("✅ App import successful")
    except Exception as e:
        logger.error(f"❌ Failed to import app: {e}")
        logger.error(f"Python path: {sys.path}")
        logger.error(f"Current directory contents: {os.listdir('.')}")
        sys.exit(1)
    
    # Start the server
    try:
        logger.info(f"🌟 Starting server on {host}:{port}")
        
        # Use uvicorn directly
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host=host,
            port=int(port),
            log_level="info",
            access_log=True,
            loop="uvloop" if os.name != "nt" else "asyncio"
        )
        
    except ImportError:
        # Fallback to subprocess if uvicorn import fails
        logger.info("📦 Using subprocess to start uvicorn...")
        cmd = [
            python_cmd, "-m", "uvicorn",
            "app.main:app",
            "--host", host,
            "--port", port,
            "--log-level", "info"
        ]
        
        logger.info(f"🚀 Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_application()
