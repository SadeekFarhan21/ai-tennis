"""
Result endpoint for processed video analysis
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.crud.video import get_video
from app.db.session import get_db

router = APIRouter()

@router.get("/result/{video_id}")
async def get_video_result(video_id: str, db: Session = Depends(get_db)):
    """
    Get the analysis results for a processed video
    """
    video = get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {
        "video_id": video_id,
        "status": video.status,
        "filename": video.filename,
        "analysis_results": video.analysis_results,
        "processed_at": video.processed_at,
        "created_at": video.created_at,
        "error_message": video.error_message if video.status == "failed" else None
    }

@router.get("/result/{video_id}/download")
async def download_processed_video(video_id: str, db: Session = Depends(get_db)):
    """
    Download the processed video file
    """
    video = get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Video processing not completed. Current status: {video.status}"
        )
    
    if not video.processed_file_path:
        raise HTTPException(status_code=404, detail="Processed video file not found")
    
    return {
        "download_url": f"/uploads/{video.processed_file_path}",
        "filename": f"processed_{video.filename}"
    }
