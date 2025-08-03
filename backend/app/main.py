"""
FastAPI app entry point for AI Tennis application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import upload, result
from app.core.config import settings
from app.db.session import engine
from app.models import video

# Create database tables
video.Base.metadata.create_all(bind=engine)

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

@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {"message": "AI Tennis API is running", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
