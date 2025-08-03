# AI Tennis - Video Analysis Platform

An AI-powered tennis video analysis platform that uses computer vision and machine learning to analyze tennis matches, providing detailed statistics, rally analysis, and performance insights.

## 🎾 Features

- **Video Upload & Processing**: Support for multiple video formats (MP4, AVI, MOV, WMV)
- **AI Analysis**: Automated tennis match analysis including:
  - Shot detection and classification (forehand, backhand, serves)
  - Rally analysis with shot counts and duration
  - Player statistics and performance metrics
  - Speed analysis and accuracy tracking
  - Highlight detection for key moments
- **Real-time Processing**: Background video processing with status updates
- **Modern Web Interface**: Clean, responsive UI for uploading and viewing results
- **REST API**: Complete API for integration with other applications
- **Scalable Architecture**: Docker-based deployment with microservices

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis server
- Docker & Docker Compose (recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ai-tennis.git
   cd ai-tennis
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the backend**:
   ```bash
   cd backend
   cp .env.example .env  # Edit with your configuration
   ```

### Using Docker (Recommended)

1. **Start all services**:
   ```bash
   cd backend
   docker-compose up -d
   ```

2. **Access the application**:
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Task Monitoring: http://localhost:5555

### Manual Setup

1. **Start Redis** (required for background tasks):
   ```bash
   # Using Docker
   docker run -d -p 6379:6379 redis:7-alpine
   ```

2. **Start the backend server**:
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Start the Celery worker** (in another terminal):
   ```bash
   cd backend
   celery -A app.core.celery_app worker --loglevel=info
   ```

## 📁 Project Structure

```
ai-tennis/
├── backend/                 # FastAPI backend application
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Configuration and Celery
│   │   ├── models/         # Database models
│   │   ├── crud/           # Database operations
│   │   ├── db/             # Database setup
│   │   ├── tasks/          # Background tasks
│   │   ├── utils/          # Utility functions
│   │   └── templates/      # HTML templates
│   ├── uploads/            # Video storage
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── requirements.txt        # Main project dependencies
└── README.md
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

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

# AWS S3 (optional)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_S3_BUCKET=your_bucket

# Security
SECRET_KEY=your-secret-key-change-in-production
ALLOWED_HOSTS=["*"]
```

## 📋 API Usage

### Upload Video for Analysis

```bash
curl -X POST "http://localhost:8000/api/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@tennis_match.mp4"
```

### Get Analysis Results

```bash
curl -X GET "http://localhost:8000/api/result/{video_id}" \
     -H "accept: application/json"
```

### Example Response

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
      }
    },
    "rally_analysis": [...],
    "highlights": [...]
  }
}
```

## 🧠 AI Analysis Features

The platform currently provides:

- **Shot Classification**: Automatic detection of forehand, backhand, and serve shots
- **Rally Analysis**: Breakdown of each rally with shot counts and duration
- **Player Statistics**: Performance metrics for each player
- **Speed Analysis**: Ball and racket speed measurements
- **Highlight Detection**: Automatic identification of key moments (aces, winners, etc.)

*Note: Current implementation includes a simulation of AI analysis. Replace with actual computer vision models for production use.*

## 🛠️ Development

### Adding Custom AI Models

1. Place your trained models in the `backend/models/` directory
2. Update the processing logic in `backend/app/tasks/process_video.py`
3. Modify the analysis result structure in the database models

### Running Tests

```bash
cd backend
pytest
```

### Code Formatting

```bash
cd backend
black .
flake8 .
```

## 🚀 Deployment

### Production Setup

1. **Use PostgreSQL** instead of SQLite for the database
2. **Configure S3** for file storage in production
3. **Set up proper security** with environment-specific configurations
4. **Use reverse proxy** (nginx) for serving static files
5. **Set up monitoring** and logging

### Docker Production

```bash
# Update docker-compose.yml for production settings
docker-compose -f docker-compose.prod.yml up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 Check the [API documentation](http://localhost:8000/docs) when running locally
- 🐛 Report bugs by creating an [issue](https://github.com/yourusername/ai-tennis/issues)
- 💬 Join discussions in the project's discussion forum

## 🔮 Future Enhancements

- [ ] Real-time video streaming analysis
- [ ] Advanced player tracking and movement analysis
- [ ] Match strategy recommendations
- [ ] Integration with wearable devices
- [ ] Mobile app for on-court analysis
- [ ] Social sharing and comparison features