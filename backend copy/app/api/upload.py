"""
Upload endpoint for video files
"""
import uuid
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session

from app.core.celery_app import celery_app
from app.crud.video import create_video, get_video
from app.db.session import get_db
from app.utils.storage import save_uploaded_file
from app.tasks.process_video import process_video_task

router = APIRouter()

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a tennis video for AI analysis
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    
    try:
        # Save file to storage
        file_path = await save_uploaded_file(file, video_id)
        
        # Create video record in database
        video_data = {
            "id": video_id,
            "filename": file.filename,
            "file_path": file_path,
            "status": "uploaded",
            "content_type": file.content_type
        }
        
        video_record = create_video(db, video_data)
        
        # Queue video processing task
        process_video_task.delay(video_id)
        
        return {
            "video_id": video_id,
            "status": "uploaded",
            "message": "Video uploaded successfully and processing started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/upload/status/{video_id}")
async def get_upload_status(video_id: str, db: Session = Depends(get_db)):
    """
    Get the status of an uploaded video
    """
    video = get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {
        "video_id": video_id,
        "status": video.status,
        "filename": video.filename,
        "uploaded_at": video.created_at
    }
