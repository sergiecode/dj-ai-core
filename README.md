# DJ AI Core

An AI-powered DJ system that analyzes music tracks and predicts natural transitions between songs.

## Project Structure

```
dj-ai-core/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── models/              # Data models and schemas
│   ├── routers/             # API route handlers
│   ├── services/            # Business logic services
│   └── config.py            # Configuration settings
├── audio/
│   ├── __init__.py
│   ├── analysis.py          # BPM and key detection
│   ├── features.py          # Audio feature extraction
│   └── preprocessing.py     # Audio preprocessing utilities
├── ml/
│   ├── __init__.py
│   ├── models.py            # ML model definitions
│   ├── training.py          # Model training scripts
│   └── prediction.py        # Transition prediction logic
├── tests/
│   ├── __init__.py
│   ├── test_audio.py        # Audio processing tests
│   ├── test_api.py          # API endpoint tests
│   └── test_ml.py           # ML model tests
├── data/                    # Sample audio files and datasets
├── notebooks/               # Jupyter notebooks for experimentation
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## Features

- **Audio Analysis**: BPM detection, key detection, and comprehensive feature extraction
- **Machine Learning**: AI-powered transition prediction between tracks
- **FastAPI Backend**: High-performance REST API for audio processing
- **Extensible Architecture**: Modular design for easy expansion and integration

## Requirements

- Python 3.8+
- Virtual environment (recommended)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd dj-ai-core
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
```

### 5. Run the Server
```bash
# Start the FastAPI development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## Example Usage

### 1. Health Check
```bash
curl http://localhost:8000/
```

### 2. Upload and Analyze Track
```bash
curl -X POST "http://localhost:8000/analyze-track" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your-audio-file.mp3"
```

Response:
```json
{
  "filename": "your-audio-file.mp3",
  "bpm": 128.5,
  "key": "C major",
  "features": {
    "tempo": 128.5,
    "energy": 0.85,
    "danceability": 0.92,
    "valence": 0.73
  }
}
```

### 3. Get Transition Recommendations
```bash
curl -X POST "http://localhost:8000/recommend-transitions" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "current_track_id": "track_123",
       "available_tracks": ["track_456", "track_789"]
     }'
```

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=audio --cov=ml
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Related Projects

This is part of a three-repository ecosystem:

1. **dj-ai-core** (this repo): Backend API and AI processing
2. **dj-ai-frontend**: Web interface for DJ interactions
3. **dj-ai-app**: Mobile application for DJs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## About

Created by **Sergie Code** - Software Engineer and Programming Educator.  
Building AI tools for musicians through YouTube programming tutorials.

- 📸 Instagram: https://www.instagram.com/sergiecode

- 🧑🏼‍💻 LinkedIn: https://www.linkedin.com/in/sergiecode/

- 📽️Youtube: https://www.youtube.com/@SergieCode

- 😺 Github: https://github.com/sergiecode

- 👤 Facebook: https://www.facebook.com/sergiecodeok

- 🎞️ Tiktok: https://www.tiktok.com/@sergiecode

- 🕊️Twitter: https://twitter.com/sergiecode

- 🧵Threads: https://www.threads.net/@sergiecode

## Support

For questions, issues, or feature requests, please open an issue on GitHub or contact through YouTube channel.
