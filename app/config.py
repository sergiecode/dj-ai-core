"""
Configuration settings for DJ AI Core application

Loads environment variables and provides application configuration.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./dj_ai.db")
    
    # Audio Processing Settings
    MAX_AUDIO_FILE_SIZE: str = os.getenv("MAX_AUDIO_FILE_SIZE", "50MB")
    SUPPORTED_FORMATS: str = os.getenv("SUPPORTED_FORMATS", "mp3,wav,flac,m4a")
    
    # Machine Learning Settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./ml/models/")
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")
    
    # Security Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


# Global settings instance
settings = Settings()
