"""
Audio Analysis Module

Provides BPM detection, key detection, and basic audio analysis using Librosa and Essentia.
"""

import librosa
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """
    Audio analyzer for BPM and key detection using Librosa and Essentia
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio analyzer
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        logger.info(f"AudioAnalyzer initialized with sample_rate={sample_rate}")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            logger.debug(f"Loaded audio: {file_path}, duration: {len(y)/sr:.2f}s")
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise
    
    def detect_bpm(self, file_path: str) -> float:
        """
        Detect BPM (tempo) of an audio track
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            BPM as float
        """
        try:
            y, sr = self.load_audio(file_path)
            
            # Use librosa's tempo detection
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Convert numpy scalar to Python float
            bpm = float(tempo)
            
            logger.info(f"BPM detected: {bpm:.2f}")
            return bpm
            
        except Exception as e:
            logger.error(f"Error detecting BPM for {file_path}: {str(e)}")
            # Return a default BPM if detection fails
            return 120.0
    
    def detect_key(self, file_path: str) -> str:
        """
        Detect musical key of an audio track
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Musical key as string (e.g., "C major", "A minor")
        """
        try:
            y, sr = self.load_audio(file_path)
            
            # Use chromagram for key detection
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Calculate mean chroma vector
            chroma_mean = np.mean(chroma, axis=1)
            
            # Simple key detection based on strongest chroma bins
            # This is a basic implementation - more sophisticated algorithms exist
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Find the strongest chroma bin
            key_index = np.argmax(chroma_mean)
            detected_key = key_names[key_index]
            
            # Simple major/minor detection based on chroma pattern
            # This is a simplified approach
            major_third = (key_index + 4) % 12
            minor_third = (key_index + 3) % 12
            
            if chroma_mean[major_third] > chroma_mean[minor_third]:
                key_quality = "major"
            else:
                key_quality = "minor"
            
            full_key = f"{detected_key} {key_quality}"
            logger.info(f"Key detected: {full_key}")
            return full_key
            
        except Exception as e:
            logger.error(f"Error detecting key for {file_path}: {str(e)}")
            # Return a default key if detection fails
            return "C major"
    
    def get_audio_duration(self, file_path: str) -> float:
        """
        Get duration of audio file in seconds
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        try:
            duration = librosa.get_duration(path=file_path)
            logger.debug(f"Audio duration: {duration:.2f}s")
            return duration
        except Exception as e:
            logger.error(f"Error getting duration for {file_path}: {str(e)}")
            return 0.0
    
    def analyze_audio_structure(self, file_path: str) -> dict:
        """
        Analyze basic audio structure (beats, segments)
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with structural analysis
        """
        try:
            y, sr = self.load_audio(file_path)
            
            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Segment analysis (simplified)
            segments = librosa.segment.agglomerative(
                librosa.feature.mfcc(y=y, sr=sr), 
                k=8
            )
            
            analysis = {
                "tempo": float(tempo),
                "num_beats": len(beats),
                "beat_times": beats.tolist(),
                "num_segments": len(np.unique(segments)),
                "duration": len(y) / sr
            }
            
            logger.info(f"Audio structure analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing audio structure for {file_path}: {str(e)}")
            return {
                "tempo": 120.0,
                "num_beats": 0,
                "beat_times": [],
                "num_segments": 0,
                "duration": 0.0
            }
