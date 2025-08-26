"""
DJ AI Core - FastAPI Application

Main application entry point for the AI-powered DJ system.
Provides REST API endpoints for audio analysis and ML-based track transitions.

Author: Sergie Code
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
from typing import Dict, Any
import logging

from app.config import settings
from audio.analysis import AudioAnalyzer
from audio.features import FeatureExtractor
from ml.prediction import TransitionPredictor

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DJ AI Core",
    description="AI-powered DJ system for music analysis and intelligent transitions",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
audio_analyzer = AudioAnalyzer()
feature_extractor = FeatureExtractor()
transition_predictor = TransitionPredictor()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "DJ AI Core API is running",
        "status": "healthy",
        "version": "0.1.0",
        "author": "Sergie Code",
        "description": "AI-powered DJ system for intelligent music analysis and transitions"
    }


@app.get("/health")
async def health_check():
    """Detailed health check with system status"""
    return {
        "status": "healthy",
        "services": {
            "audio_analyzer": "operational",
            "feature_extractor": "operational",
            "ml_predictor": "operational"
        },
        "system_info": {
            "python_version": "3.8+",
            "fastapi_version": "0.104.1"
        }
    }


@app.post("/analyze-track")
async def analyze_track(file: UploadFile = File(...)):
    """
    Analyze an audio track for BPM, key, and musical features
    
    Args:
        file: Audio file (mp3, wav, flac, m4a)
        
    Returns:
        Analysis results including BPM, key, and audio features
    """
    try:
        # Validate file format
        allowed_formats = ['mp3', 'wav', 'flac', 'm4a']
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_formats)}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Analyze audio
            logger.info(f"Analyzing track: {file.filename}")
            
            # Get BPM and key
            bpm = audio_analyzer.detect_bpm(temp_file_path)
            key = audio_analyzer.detect_key(temp_file_path)
            
            # Extract features
            features = feature_extractor.extract_features(temp_file_path)
            
            result = {
                "filename": file.filename,
                "bpm": round(bpm, 2),
                "key": key,
                "features": {
                    "tempo": round(bpm, 2),
                    "energy": round(features.get("energy", 0.0), 3),
                    "danceability": round(features.get("danceability", 0.0), 3),
                    "valence": round(features.get("valence", 0.0), 3),
                    "loudness": round(features.get("loudness", 0.0), 3),
                    "spectral_centroid": round(features.get("spectral_centroid", 0.0), 3),
                    "spectral_rolloff": round(features.get("spectral_rolloff", 0.0), 3),
                    "zero_crossing_rate": round(features.get("zero_crossing_rate", 0.0), 6)
                },
                "analysis_status": "completed"
            }
            
            logger.info(f"Analysis completed for {file.filename}")
            return result
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        # Re-raise HTTPException with original status code
        raise
    except Exception as e:
        logger.error(f"Error analyzing track {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/recommend-transitions")
async def recommend_transitions(request: Dict[str, Any]):
    """
    Get AI-powered recommendations for track transitions
    
    Args:
        request: Dictionary with current_track_id and available_tracks list
        
    Returns:
        Ranked list of recommended tracks for seamless transitions
    """
    try:
        current_track_id = request.get("current_track_id")
        available_tracks = request.get("available_tracks", [])
        
        if not current_track_id:
            raise HTTPException(status_code=400, detail="current_track_id is required")
        
        if not available_tracks:
            raise HTTPException(status_code=400, detail="available_tracks list cannot be empty")
        
        logger.info(f"Getting transition recommendations for track: {current_track_id}")
        
        # Get recommendations from ML model
        recommendations = transition_predictor.predict_transitions(
            current_track_id, 
            available_tracks
        )
        
        result = {
            "current_track": current_track_id,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "prediction_status": "completed"
        }
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return result
        
    except HTTPException:
        # Re-raise HTTPException with original status code
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported audio formats"""
    return {
        "formats": ["mp3", "wav", "flac", "m4a"],
        "max_file_size": "50MB",
        "note": "Higher quality formats (wav, flac) provide better analysis results"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
