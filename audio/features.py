"""
Audio Feature Extraction Module

Extracts comprehensive audio features using Librosa for ML model training and analysis.
"""

import librosa
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract comprehensive audio features for ML analysis
    """
    
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        """
        Initialize the feature extractor
        
        Args:
            sample_rate: Target sample rate for audio processing
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        logger.info(f"FeatureExtractor initialized with sr={sample_rate}, n_mfcc={n_mfcc}")
    
    def load_audio(self, file_path: str) -> tuple:
        """Load audio file using librosa"""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {str(e)}")
            raise
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract spectral features from audio
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of spectral features
        """
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid_mean = np.mean(spectral_centroids)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            return {
                "spectral_centroid": float(spectral_centroid_mean),
                "spectral_rolloff": float(spectral_rolloff_mean),
                "spectral_bandwidth": float(spectral_bandwidth_mean),
                "zero_crossing_rate": float(zcr_mean)
            }
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {str(e)}")
            return {
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "spectral_bandwidth": 0.0,
                "zero_crossing_rate": 0.0
            }
    
    def extract_rhythm_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract rhythm-related features
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of rhythm features
        """
        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Rhythm strength (beat consistency)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            if len(beat_times) > 1:
                beat_intervals = np.diff(beat_times)
                rhythm_strength = 1.0 / (np.std(beat_intervals) + 1e-6)
            else:
                rhythm_strength = 0.0
            
            return {
                "tempo": float(tempo),
                "rhythm_strength": float(rhythm_strength),
                "num_beats": len(beats)
            }
            
        except Exception as e:
            logger.error(f"Error extracting rhythm features: {str(e)}")
            return {
                "tempo": 120.0,
                "rhythm_strength": 0.0,
                "num_beats": 0
            }
    
    def extract_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract harmonic and tonal features
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of harmonic features
        """
        try:
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # Tonnetz (harmonic network) features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            
            return {
                "chroma_mean": chroma_mean.tolist(),
                "chroma_std": chroma_std.tolist(),
                "tonnetz_mean": tonnetz_mean.tolist(),
                "harmonic_stability": float(np.mean(chroma_std))
            }
            
        except Exception as e:
            logger.error(f"Error extracting harmonic features: {str(e)}")
            return {
                "chroma_mean": [0.0] * 12,
                "chroma_std": [0.0] * 12,
                "tonnetz_mean": [0.0] * 6,
                "harmonic_stability": 0.0
            }
    
    def extract_mfcc_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract MFCC (Mel-frequency cepstral coefficients) features
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of MFCC features
        """
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            
            return {
                "mfcc_mean": mfccs_mean.tolist(),
                "mfcc_std": mfccs_std.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {str(e)}")
            return {
                "mfcc_mean": [0.0] * self.n_mfcc,
                "mfcc_std": [0.0] * self.n_mfcc
            }
    
    def calculate_energy_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Calculate energy-related features
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of energy features
        """
        try:
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            energy = np.mean(rms)
            
            # Dynamic range
            dynamic_range = np.max(rms) - np.min(rms)
            
            # Loudness approximation (based on RMS)
            loudness = 20 * np.log10(energy + 1e-6)
            
            # Simple danceability metric (based on energy and rhythm)
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            danceability = min(1.0, (energy * tempo) / 15000)  # Normalized approximation
            
            # Valence approximation (based on spectral features)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            valence = min(1.0, spectral_centroid / 4000)  # Simple approximation
            
            return {
                "energy": float(energy),
                "dynamic_range": float(dynamic_range),
                "loudness": float(loudness),
                "danceability": float(danceability),
                "valence": float(valence)
            }
            
        except Exception as e:
            logger.error(f"Error calculating energy features: {str(e)}")
            return {
                "energy": 0.0,
                "dynamic_range": 0.0,
                "loudness": -60.0,
                "danceability": 0.0,
                "valence": 0.5
            }
    
    def extract_features(self, file_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive audio features from file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        try:
            logger.info(f"Extracting features from: {file_path}")
            
            # Load audio
            y, sr = self.load_audio(file_path)
            
            # Extract different feature groups
            spectral_features = self.extract_spectral_features(y, sr)
            rhythm_features = self.extract_rhythm_features(y, sr)
            harmonic_features = self.extract_harmonic_features(y, sr)
            mfcc_features = self.extract_mfcc_features(y, sr)
            energy_features = self.calculate_energy_features(y, sr)
            
            # Combine all features
            all_features = {
                **spectral_features,
                **rhythm_features,
                **harmonic_features,
                **mfcc_features,
                **energy_features,
                "duration": len(y) / sr,
                "sample_rate": sr
            }
            
            logger.info(f"Feature extraction completed. Extracted {len(all_features)} features.")
            return all_features
            
        except Exception as e:
            logger.error(f"Error extracting features from {file_path}: {str(e)}")
            # Return default features in case of error
            return {
                "energy": 0.0,
                "danceability": 0.0,
                "valence": 0.5,
                "loudness": -60.0,
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "zero_crossing_rate": 0.0,
                "tempo": 120.0,
                "duration": 0.0,
                "sample_rate": self.sample_rate
            }
    
    def extract_features_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract features from multiple files
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for file_path in file_paths:
            try:
                features = self.extract_features(file_path)
                features["file_path"] = file_path
                features_list.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features from {file_path}: {str(e)}")
                continue
        
        logger.info(f"Batch feature extraction completed for {len(features_list)}/{len(file_paths)} files")
        return features_list
