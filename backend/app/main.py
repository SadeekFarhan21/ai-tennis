"""
FastAPI app entry point for AI Tennis application
"""
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.api import upload, result
from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.db.session import engine
from app.models import video

# Setup logging
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

logger.info("Starting AI Tennis application...")
logger.info(f"Environment: {settings.ENVIRONMENT}")
logger.info(f"Debug mode: {settings.DEBUG}")

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)
logger.info("Created necessary directories")

# Create database tables
video.Base.metadata.create_all(bind=engine)
logger.info("Database tables created")

app = FastAPI(
    title="AI Tennis API",
    description="Backend API for AI Tennis video processing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for uploads
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Include API routes
app.include_router(upload.router, prefix="/api")
app.include_router(result.router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/result.html", response_class=HTMLResponse)
async def result_page(request: Request):
    """Serve the results page"""
    return templates.TemplateResponse("result.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        from app.db.session import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        return {
            "message": "AI Tennis API is running",
            "version": "1.0.0",
            "status": "healthy",
            "database": "connected"
        }
    except Exception as e:
        return {
            "message": "AI Tennis API is running",
            "version": "1.0.0", 
            "status": "degraded",
            "database": "error",
            "error": str(e)
        }

@app.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness check"""
    return {"status": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
