<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Setup Checklist

- [x] Verify that the copilot-instructions.md file in the .github directory is created.
- [x] Clarify Project Requirements - Python project for AI-powered DJ system with FastAPI, Librosa, Essentia, ML capabilities
- [x] Scaffold the Project - Complete project structure created manually with all required modules and files
- [x] Customize the Project - Project customized with comprehensive DJ AI system modules including audio analysis, ML models, and FastAPI endpoints  
- [x] Install Required Extensions - No specific extensions required for this Python project
- [x] Compile the Project - Python packages installed successfully including FastAPI, Librosa, and ML dependencies
- [x] Create and Run Task - FastAPI server task created and running successfully on http://0.0.0.0:8000
- [x] Launch the Project - Project launched successfully - FastAPI server running on http://localhost:8000 with interactive docs at /docs
- [x] Ensure Documentation is Complete - README.md file exists with comprehensive documentation and copilot-instructions.md file in the .github directory contains current project information

## Project Overview

DJ AI Core is an AI-powered DJ system that provides:

- Audio analysis (BPM detection, key detection, feature extraction)
- Machine learning models for track transition prediction
- FastAPI REST API for integration with frontend applications
- Comprehensive testing suite
- Modular architecture for easy expansion

## Development Guidelines

- Use Python 3.8+ with virtual environment
- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Document all public APIs
- Use type hints where appropriate
- Keep audio processing efficient and scalable

## Key Components

1. **Audio Module** (`audio/`): Librosa-based audio analysis and feature extraction
2. **ML Module** (`ml/`): TensorFlow/Keras models for transition prediction
3. **API Module** (`app/`): FastAPI web server and endpoints
4. **Tests** (`tests/`): Comprehensive unit and integration tests

## API Endpoints

- `GET /` - Health check
- `POST /analyze-track` - Upload and analyze audio file
- `POST /recommend-transitions` - Get AI-powered transition recommendations
- `GET /supported-formats` - List supported audio formats

## Local Development

```bash
# Start server
python -m uvicorn app.main:app --reload

# Run tests
pytest

# Access API docs
http://localhost:8000/docs
```
