# AI Tennis Backend

A FastAPI-based backend service for AI-powered tennis video analysis. This application processes uploaded tennis videos using computer vision and machine learning to provide detailed analytics about player performance, rally analysis, and game highlights.

## Features

- **Video Upload**: Support for multiple video formats (MP4, AVI, MOV, WMV)
- **Async Processing**: Background video processing using Celery
- **AI Analysis**: Tennis-specific video analysis (placeholder for ML models)
- **REST API**: Clean RESTful endpoints for video management
- **Storage Options**: Local file storage or AWS S3 integration
- **Real-time Updates**: WebSocket support for processing status
- **Docker Support**: Full containerization with Docker Compose

## Tech Stack

- **Backend Framework**: FastAPI
- **Database**: SQLAlchemy (SQLite/PostgreSQL)
- **Task Queue**: Celery with Redis
- **File Storage**: Local filesystem or AWS S3
- **Containerization**: Docker & Docker Compose
- **AI/ML**: OpenCV, TensorFlow (extensible for custom models)

## Quick Start

### Python Version Compatibility

**Recommended**: Python 3.11 or 3.12 for full compatibility with all dependencies.

**Python 3.13**: Some ML libraries (like TensorFlow) may not be fully compatible yet. Use the minimal requirements for basic functionality:

```bash
# For Python 3.13 or quick testing
pip install -r requirements-minimal.txt

# For full ML features (Python 3.11/3.12 recommended)
pip install -r requirements.txt
```

### Using Docker Compose (Recommended)

1. **Clone and navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Start all services**:
   ```bash
   docker-compose up -d
   ```

3. **Access the application**:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Flower (Celery monitoring): http://localhost:5555

### Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Redis** (required for Celery):
   ```bash
   # Using Docker
   docker run -d -p 6379:6379 redis:7-alpine
   
   # Or install Redis locally
   ```

4. **Start the FastAPI server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Start Celery worker** (in another terminal):
   ```bash
   celery -A app.core.celery_app worker --loglevel=info
   ```

## API Endpoints

### Upload Video
```http
POST /api/upload
Content-Type: multipart/form-data

# Form data:
file: <video_file>
```

**Response**:
```json
{
  "video_id": "uuid-string",
  "status": "uploaded",
  "message": "Video uploaded successfully and processing started"
}
```

### Get Analysis Results
```http
GET /api/result/{video_id}
```

**Response**:
```json
{
  "video_id": "uuid-string",
  "status": "completed",
  "filename": "tennis_match.mp4",
  "analysis_results": {
    "video_duration": 120.5,
    "total_rallies": 15,
    "player_statistics": {
      "player_1": {
        "forehand_shots": 25,
        "backhand_shots": 18,
        "serves": 8,
        "accuracy_percentage": 85.2,
        "average_shot_speed": 95.3
      },
      "player_2": { "..." }
    },
    "rally_analysis": [...],
    "highlights": [...]
  },
  "processed_at": "2024-01-01T12:00:00Z"
}
```

### Download Processed Video
```http
GET /api/result/{video_id}/download
```

### Check Upload Status
```http
GET /api/upload/status/{video_id}
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# Database
DATABASE_URL=sqlite:///./ai_tennis.db

# Redis/Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# File Storage
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=104857600  # 100MB
USE_S3=false

# AWS S3 (if enabled)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_S3_BUCKET=your_bucket

# Security
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=["*"]

# AI Processing
PROCESSING_TIMEOUT=300
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── upload.py        # Upload endpoints
│   │   └── result.py        # Result endpoints
│   ├── core/
│   │   ├── config.py        # Configuration management
│   │   └── celery_app.py    # Celery configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── video.py         # SQLAlchemy models
│   ├── crud/
│   │   ├── __init__.py
│   │   └── video.py         # Database operations
│   ├── db/
│   │   ├── session.py       # Database session
│   │   └── base.py          # Base model class
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── process_video.py # Celery tasks
│   ├── utils/
│   │   ├── __init__.py
│   │   └── storage.py       # File storage utilities
│   └── templates/           # HTML templates
│       ├── index.html
│       └── result.html
├── uploads/                 # Video storage directory
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
flake8 .
```

### Adding New AI Models

1. **Add model files to `app/models/` or separate `models/` directory**
2. **Update processing logic in `app/tasks/process_video.py`**
3. **Add new analysis fields to the Video model if needed**
4. **Update API responses to include new analysis data**

### Database Migrations

For production use with PostgreSQL:

```bash
# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

## Deployment

### Production Considerations

1. **Database**: Use PostgreSQL instead of SQLite
2. **File Storage**: Use S3 for scalability
3. **Security**: 
   - Change default secret keys
   - Configure proper CORS origins
   - Use environment-specific configurations
4. **Monitoring**: 
   - Set up logging
   - Monitor Celery tasks with Flower
   - Health checks for all services

### Docker Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  backend:
    build: .
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/ai_tennis
      - USE_S3=true
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    # ... other production configs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the Celery task logs for processing issues
