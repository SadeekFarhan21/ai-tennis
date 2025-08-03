"""
CRUD operations for Video model
"""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from app.models.video import Video

def create_video(db: Session, video_data: Dict[str, Any]) -> Video:
    """
    Create a new video record in the database
    """
    video = Video(**video_data)
    db.add(video)
    db.commit()
    db.refresh(video)
    return video

def get_video(db: Session, video_id: str) -> Optional[Video]:
    """
    Get a video by ID
    """
    return db.query(Video).filter(Video.id == video_id).first()

def update_video_status(db: Session, video_id: str, status: str, error_message: Optional[str] = None) -> Optional[Video]:
    """
    Update video processing status
    """
    video = get_video(db, video_id)
    if video:
        video.status = status
        video.updated_at = datetime.utcnow()
        
        if status == "completed":
            video.processed_at = datetime.utcnow()
        elif status == "failed" and error_message:
            video.error_message = error_message
            
        db.commit()
        db.refresh(video)
    return video

def update_video_results(db: Session, video_id: str, analysis_results: Dict[str, Any], processed_file_path: Optional[str] = None) -> Optional[Video]:
    """
    Update video with analysis results
    """
    video = get_video(db, video_id)
    if video:
        video.analysis_results = analysis_results
        video.status = "completed"
        video.processed_at = datetime.utcnow()
        video.updated_at = datetime.utcnow()
        
        if processed_file_path:
            video.processed_file_path = processed_file_path
            
        db.commit()
        db.refresh(video)
    return video

def get_videos_by_status(db: Session, status: str, limit: int = 100) -> list[Video]:
    """
    Get videos by status
    """
    return db.query(Video).filter(Video.status == status).limit(limit).all()

def delete_video(db: Session, video_id: str) -> bool:
    """
    Delete a video record
    """
    video = get_video(db, video_id)
    if video:
        db.delete(video)
        db.commit()
        return True
    return False
