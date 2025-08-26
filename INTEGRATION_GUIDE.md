# DJ AI Core - Backend Integration Guide

**Author**: Sergie Code - Software Engineer & YouTube Programming Educator  
**Purpose**: AI tools for musicians - Complete DJ AI ecosystem  
**Platform**: Windows PowerShell environment

---

## 🎯 Project Overview

This document provides integration instructions for the **DJ AI Core** backend service with the companion repositories:

- **dj-ai-frontend**: Interactive React web interface with waveform visualization
- **dj-ai-app**: Docker Compose orchestrator for unified development environment

---

## 🏗️ Architecture Overview

```
dj-ai-app (Main Orchestrator)
├── dj-ai-core (Backend API) ← THIS REPOSITORY
├── dj-ai-frontend (React Frontend)
└── docker-compose.yml (Service coordination)
```

---

## 📋 DJ AI Core Backend Specifications

### Technology Stack
- **Framework**: FastAPI 0.104.1
- **Audio Processing**: Librosa, Essentia
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Python Version**: 3.12.5
- **Environment**: Windows PowerShell compatible

### Server Configuration
- **Default Port**: 8000
- **Host**: 0.0.0.0 (configurable)
- **Protocol**: HTTP/HTTPS
- **CORS**: Enabled for frontend integration

### API Endpoints

#### Core Endpoints
```http
GET  /                     # Health check and API info
GET  /health              # Detailed service status
GET  /docs                # Interactive API documentation
GET  /supported-formats   # List supported audio formats
```

#### Audio Analysis
```http
POST /analyze-track       # Upload and analyze audio files
Content-Type: multipart/form-data
Body: file (audio file: mp3, wav, flac, m4a)

Response Example:
{
  "track_id": "unique-id",
  "bpm": 128.5,
  "key": "C major",
  "duration": 245.6,
  "features": {
    "spectral_centroid": 2456.7,
    "spectral_rolloff": 8934.2,
    "zero_crossing_rate": 0.14,
    "mfcc": [1.2, -0.8, 0.3, ...],
    "tempo_confidence": 0.92
  }
}
```

#### AI Recommendations
```http
POST /recommend-transitions
Content-Type: application/json
Body: {
  "current_track_id": "track-1",
  "available_tracks": [
    {
      "track_id": "track-2",
      "bpm": 130,
      "key": "A minor",
      "energy": 0.75
    }
  ]
}

Response Example:
{
  "current_track": "track-1",
  "recommendations": [
    {
      "track_id": "track-2",
      "compatibility_score": 0.87,
      "transition_type": "harmonic_mix",
      "suggested_cue_point": 120.5,
      "crossfade_duration": 8.0
    }
  ]
}
```

---

## 🚀 Quick Start Instructions

### 1. Repository Setup
```powershell
# Clone the backend
git clone https://github.com/sergiecode/dj-ai-core.git
cd dj-ai-core

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -m pytest tests/ -v
```

### 2. Development Server
```powershell
# Start the FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Server will be available at:
# http://localhost:8000 (API)
# http://localhost:8000/docs (Interactive docs)
```

### 3. Environment Variables
Create `.env` file in project root:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Audio Processing
MAX_FILE_SIZE=50MB
SUPPORTED_FORMATS=mp3,wav,flac,m4a
SAMPLE_RATE=22050

# ML Models
MODEL_DIR=./ml/models/
ENABLE_GPU=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# CORS (for frontend integration)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
CORS_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_HEADERS=*
```

---

## 🔗 Frontend Integration (dj-ai-frontend)

### Required Frontend Features
1. **Audio File Upload**
   - Drag & drop interface
   - Support for mp3, wav, flac, m4a
   - Progress indicators
   - File validation

2. **Waveform Visualization**
   - Wavesurfer.js integration
   - BPM markers and beat grid
   - Cue point visualization
   - Real-time playback cursor

3. **DJ Controls**
   - Play/pause/stop buttons
   - Crossfader simulation
   - Tempo adjustment controls
   - Loop controls

4. **AI Features Display**
   - BPM and key detection results
   - Transition recommendations
   - Compatibility scores
   - Suggested mix points

### API Integration Examples

#### TypeScript/JavaScript Integration
```typescript
// API client configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Upload and analyze track
const analyzeTrack = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE_URL}/analyze-track`, {
    method: 'POST',
    body: formData,
  });
  
  return response.json();
};

// Get transition recommendations
const getRecommendations = async (currentTrack: string, availableTracks: Track[]) => {
  const response = await fetch(`${API_BASE_URL}/recommend-transitions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      current_track_id: currentTrack,
      available_tracks: availableTracks,
    }),
  });
  
  return response.json();
};
```

#### React Component Integration
```jsx
// Example React hook for backend integration
const useDJAICore = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [trackData, setTrackData] = useState(null);
  
  const analyzeAudioFile = async (file) => {
    setIsAnalyzing(true);
    try {
      const result = await analyzeTrack(file);
      setTrackData(result);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  return { analyzeAudioFile, isAnalyzing, trackData };
};
```

---

## 🐳 Docker Integration (dj-ai-app)

### Backend Dockerfile
```dockerfile
# Suggested Dockerfile for dj-ai-core
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Service Definition
```yaml
# Suggested service definition for docker-compose.yml
services:
  dj-ai-core:
    build: ./dj-ai-core
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - CORS_ORIGINS=http://localhost:3000,http://localhost:5173
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/ml/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - dj-ai-network

  dj-ai-frontend:
    build: ./dj-ai-frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      dj-ai-core:
        condition: service_healthy
    networks:
      - dj-ai-network

networks:
  dj-ai-network:
    driver: bridge
```

---

## 📁 Project Structure Reference

```
dj-ai-core/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI application entry point
├── audio/
│   ├── __init__.py
│   ├── analysis.py          # Audio analysis with Librosa
│   ├── features.py          # Feature extraction
│   └── preprocessing.py     # Audio preprocessing utilities
├── ml/
│   ├── __init__.py
│   ├── models.py           # Neural network models
│   ├── prediction.py       # Transition prediction logic
│   └── training.py         # Training pipeline
├── tests/
│   ├── test_api.py         # API endpoint tests
│   ├── test_audio.py       # Audio processing tests
│   └── test_ml.py          # ML model tests
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── .env                    # Environment configuration
```

---

## 🔧 Development Commands

### Testing
```powershell
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_api.py -v      # API tests
python -m pytest tests/test_audio.py -v    # Audio tests
python -m pytest tests/test_ml.py -v       # ML tests

# Test with coverage
python -m pytest tests/ --cov=app --cov=audio --cov=ml
```

### Development Server
```powershell
# Development mode with auto-reload
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Audio File Testing
```powershell
# Test audio analysis (requires curl or PowerShell Invoke-RestMethod)
curl -X POST "http://localhost:8000/analyze-track" -F "file=@sample.mp3"

# PowerShell alternative
$file = Get-Item "sample.mp3"
$form = @{
    file = $file
}
Invoke-RestMethod -Uri "http://localhost:8000/analyze-track" -Method Post -Form $form
```

---

## 🎵 Audio Format Support

### Supported Input Formats
- **MP3**: Standard compressed audio
- **WAV**: Uncompressed audio (best quality)
- **FLAC**: Lossless compression
- **M4A**: Apple audio format

### Audio Processing Capabilities
- **BPM Detection**: Tempo analysis with confidence scoring
- **Key Detection**: Musical key analysis using chromagram
- **Feature Extraction**: Spectral, rhythmic, and harmonic features
- **Beat Tracking**: Beat onset detection and grid alignment

---

## 🤖 Machine Learning Features

### Transition Prediction
- **Compatibility Scoring**: BPM, key, and energy matching
- **Harmonic Mixing**: Key-based transition recommendations
- **Beat Matching**: Tempo synchronization suggestions
- **Crossfade Points**: Optimal mix timing prediction

### Model Training
- **Synthetic Data Generation**: Training data creation pipeline
- **Feature Engineering**: Audio feature preprocessing
- **Neural Networks**: TensorFlow/Keras model architectures
- **Model Persistence**: Save/load trained models

---

## 🔐 Security & Performance

### File Upload Security
- File size limits (default: 50MB)
- MIME type validation
- Temporary file cleanup
- Malware scanning integration points

### Performance Optimization
- Async audio processing
- Caching of analysis results
- GPU acceleration support (optional)
- Batch processing capabilities

---

## 📚 Educational Integration

### YouTube Content Ideas
1. **"Building an AI DJ with Python"** - Backend development series
2. **"React Audio Visualization"** - Frontend development series
3. **"Docker for Musicians"** - DevOps and deployment series
4. **"Machine Learning for Music"** - AI/ML educational content

### Learning Resources
- Interactive API documentation at `/docs`
- Comprehensive test suite for learning
- Modular architecture for educational breakdown
- Real-world music technology application

---

## 🚨 Important Notes

### Windows-Specific Considerations
- PowerShell command compatibility verified
- Windows path handling in file operations
- Audio codec support on Windows
- Development environment setup for Windows

### Integration Requirements
- **Frontend**: Must handle CORS properly for localhost development
- **Docker**: Audio processing libraries need system dependencies
- **Performance**: Consider file upload timeouts for large audio files
- **Storage**: Plan for audio file storage and cleanup strategies

---

## 📞 Support & Documentation

### API Documentation
- Interactive docs: `http://localhost:8000/docs`
- OpenAPI spec: `http://localhost:8000/openapi.json`
- Health check: `http://localhost:8000/health`

### Development Support
- Comprehensive test suite (56 tests, all passing)
- Type hints throughout codebase
- Detailed logging and error handling
- Production-ready FastAPI configuration

---

**Ready for Integration**: This backend is fully tested, documented, and ready to be integrated with the frontend and orchestrator services. All endpoints are functional and the architecture supports scalable development.

**Creator**: Sergie Code - Empowering musicians through technology education 🎵💻
