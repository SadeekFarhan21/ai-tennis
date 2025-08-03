# Render Deployment Configuration for AI Tennis

## Web Service Configuration:
- **Build Command**: `./build.sh`
- **Start Command**: `./start-prod.sh`
- **Environment**: Python 3.11+

## Environment Variables to set in Render Dashboard:
```
DATABASE_URL=sqlite:///./ai_tennis.db
REDIS_URL=<Render Redis URL>
CELERY_BROKER_URL=<Render Redis URL>
CELERY_RESULT_BACKEND=<Render Redis URL>
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=104857600
USE_S3=false
ALLOWED_HOSTS=["*"]
SECRET_KEY=<Generate a secure key>
AI_MODEL_PATH=./models
PROCESSING_TIMEOUT=300
DEBUG=false
ENVIRONMENT=production
PORT=10000
```

## Services needed:
1. **Web Service** (main app)
2. **Redis Service** (for Celery)
3. **Background Worker** (optional - for Celery worker)

## Notes:
- The app will use SQLite for development
- Consider upgrading to PostgreSQL for production
- Redis is required for background task processing
- File uploads will be stored locally (consider S3 for production)
