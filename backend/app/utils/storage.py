"""
Storage utilities for handling file uploads and S3 operations
"""
import os
import aiofiles
from typing import Optional
from fastapi import UploadFile, HTTPException

from app.core.config import settings

async def save_uploaded_file(file: UploadFile, video_id: str) -> str:
    """
    Save uploaded file to local storage or S3
    Returns the file path
    """
    if settings.USE_S3:
        return await save_to_s3(file, video_id)
    else:
        return await save_to_local(file, video_id)

async def save_to_local(file: UploadFile, video_id: str) -> str:
    """
    Save file to local uploads directory
    """
    # Ensure uploads directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Generate file path
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    file_path = os.path.join(settings.UPLOAD_DIR, f"{video_id}{file_extension}")
    
    try:
        # Save file asynchronously
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return file_path
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

async def save_to_s3(file: UploadFile, video_id: str) -> str:
    """
    Save file to AWS S3 bucket
    TODO: Implement S3 upload functionality
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        
        # Generate S3 key
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
        s3_key = f"uploads/{video_id}{file_extension}"
        
        # Upload file to S3
        content = await file.read()
        s3_client.put_object(
            Bucket=settings.AWS_S3_BUCKET,
            Key=s3_key,
            Body=content,
            ContentType=file.content_type
        )
        
        return f"s3://{settings.AWS_S3_BUCKET}/{s3_key}"
        
    except ImportError:
        raise HTTPException(status_code=500, detail="boto3 not installed for S3 support")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")

def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return 0

def validate_video_file(file: UploadFile) -> bool:
    """
    Validate uploaded video file
    """
    # Check content type
    if file.content_type not in settings.ALLOWED_VIDEO_TYPES:
        return False
    
    # Check file extension
    if file.filename:
        file_extension = os.path.splitext(file.filename)[1].lower()
        allowed_extensions = ['.mp4', '.avi', '.mov', '.wmv']
        if file_extension not in allowed_extensions:
            return False
    
    return True

def cleanup_file(file_path: str) -> bool:
    """
    Delete a file from local storage
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception as e:
        print(f"Failed to delete file {file_path}: {str(e)}")
    return False
