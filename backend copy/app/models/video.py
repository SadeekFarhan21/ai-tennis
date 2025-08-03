"""
Video model for storing video metadata and analysis results
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Integer, JSON
from sqlalchemy.sql import func

from app.db.base import Base

class Video(Base):
    """Video model for tracking uploaded and processed videos"""
    
    __tablename__ = "videos"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=True)
    
    # Processing status: uploaded, processing, completed, failed
    status = Column(String, default="uploaded", nullable=False)
    
    # Analysis results stored as JSON
    analysis_results = Column(JSON, nullable=True)
    
    # Processed video file path
    processed_file_path = Column(String, nullable=True)
    
    # Error message if processing failed
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<Video(id={self.id}, filename={self.filename}, status={self.status})>"
