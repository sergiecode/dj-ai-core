"""
Audio Preprocessing Utilities

Common audio preprocessing functions for the DJ AI system.
"""

import librosa
import numpy as np
import soundfile as sf
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Audio preprocessing utilities for normalization, format conversion, and quality enhancement
    """
    
    def __init__(self, target_sr: int = 22050):
        """
        Initialize audio preprocessor
        
        Args:
            target_sr: Target sample rate for processing
        """
        self.target_sr = target_sr
        logger.info(f"AudioPreprocessor initialized with target_sr={target_sr}")
    
    def normalize_audio(self, y: np.ndarray, method: str = "peak") -> np.ndarray:
        """
        Normalize audio signal
        
        Args:
            y: Audio time series
            method: Normalization method ("peak" or "rms")
            
        Returns:
            Normalized audio signal
        """
        try:
            if method == "peak":
                # Peak normalization
                peak = np.max(np.abs(y))
                if peak > 0:
                    y_normalized = y / peak
                else:
                    y_normalized = y
            elif method == "rms":
                # RMS normalization
                rms = np.sqrt(np.mean(y**2))
                if rms > 0:
                    y_normalized = y / rms * 0.1  # Target RMS of 0.1
                else:
                    y_normalized = y
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            logger.debug(f"Audio normalized using {method} method")
            return y_normalized
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {str(e)}")
            return y
    
    def resample_audio(self, y: np.ndarray, orig_sr: int, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Resample audio to target sample rate
        
        Args:
            y: Audio time series
            orig_sr: Original sample rate
            target_sr: Target sample rate (uses self.target_sr if None)
            
        Returns:
            Tuple of (resampled_audio, new_sample_rate)
        """
        try:
            if target_sr is None:
                target_sr = self.target_sr
            
            if orig_sr != target_sr:
                y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
                logger.debug(f"Audio resampled from {orig_sr}Hz to {target_sr}Hz")
                return y_resampled, target_sr
            else:
                return y, orig_sr
                
        except Exception as e:
            logger.error(f"Error resampling audio: {str(e)}")
            return y, orig_sr
    
    def convert_to_mono(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert audio file to mono
        
        Args:
            file_path: Input audio file path
            output_path: Output file path (optional)
            
        Returns:
            Path to mono audio file
        """
        try:
            # Load audio with librosa (automatically converts to mono)
            y, sr = librosa.load(file_path, sr=None, mono=True)
            
            if output_path is None:
                name, ext = os.path.splitext(file_path)
                output_path = f"{name}_mono{ext}"
            
            # Save as mono
            sf.write(output_path, y, sr)
            logger.info(f"Converted to mono: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting to mono: {str(e)}")
            raise
    
    def trim_silence(self, y: np.ndarray, sr: int, top_db: int = 20) -> np.ndarray:
        """
        Remove silence from beginning and end of audio
        
        Args:
            y: Audio time series
            sr: Sample rate
            top_db: Threshold for silence detection
            
        Returns:
            Trimmed audio signal
        """
        try:
            y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
            logger.debug(f"Trimmed silence from audio (top_db={top_db})")
            return y_trimmed
            
        except Exception as e:
            logger.error(f"Error trimming silence: {str(e)}")
            return y
    
    def apply_preemphasis(self, y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """
        Apply preemphasis filter to enhance high frequencies
        
        Args:
            y: Audio time series
            coeff: Preemphasis coefficient
            
        Returns:
            Preemphasized audio signal
        """
        try:
            y_preemph = np.append(y[0], y[1:] - coeff * y[:-1])
            logger.debug(f"Applied preemphasis filter (coeff={coeff})")
            return y_preemph
            
        except Exception as e:
            logger.error(f"Error applying preemphasis: {str(e)}")
            return y
    
    def extract_segment(self, file_path: str, start_time: float, duration: float, output_path: Optional[str] = None) -> str:
        """
        Extract a segment from audio file
        
        Args:
            file_path: Input audio file path
            start_time: Start time in seconds
            duration: Duration in seconds
            output_path: Output file path (optional)
            
        Returns:
            Path to extracted segment
        """
        try:
            # Load audio segment
            y, sr = librosa.load(file_path, sr=None, offset=start_time, duration=duration)
            
            if output_path is None:
                name, ext = os.path.splitext(file_path)
                output_path = f"{name}_segment_{start_time}_{duration}{ext}"
            
            # Save segment
            sf.write(output_path, y, sr)
            logger.info(f"Extracted segment: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting segment: {str(e)}")
            raise
    
    def change_tempo(self, y: np.ndarray, rate: float) -> np.ndarray:
        """
        Change tempo without affecting pitch
        
        Args:
            y: Audio time series
            rate: Tempo change rate (1.0 = no change, >1.0 = faster, <1.0 = slower)
            
        Returns:
            Tempo-modified audio signal
        """
        try:
            y_tempo = librosa.effects.time_stretch(y, rate=rate)
            logger.debug(f"Changed tempo by factor {rate}")
            return y_tempo
            
        except Exception as e:
            logger.error(f"Error changing tempo: {str(e)}")
            return y
    
    def change_pitch(self, y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
        """
        Change pitch without affecting tempo
        
        Args:
            y: Audio time series
            sr: Sample rate
            n_steps: Number of semitones to shift (positive = higher, negative = lower)
            
        Returns:
            Pitch-shifted audio signal
        """
        try:
            y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            logger.debug(f"Changed pitch by {n_steps} semitones")
            return y_pitch
            
        except Exception as e:
            logger.error(f"Error changing pitch: {str(e)}")
            return y
    
    def preprocess_file(self, file_path: str, output_path: Optional[str] = None, 
                       normalize: bool = True, trim_silence: bool = True,
                       mono: bool = True) -> str:
        """
        Complete preprocessing pipeline for audio file
        
        Args:
            file_path: Input audio file path
            output_path: Output file path (optional)
            normalize: Whether to normalize audio
            trim_silence: Whether to trim silence
            mono: Whether to convert to mono
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            logger.info(f"Preprocessing audio file: {file_path}")
            
            # Load audio
            y, sr = librosa.load(file_path, sr=self.target_sr, mono=mono)
            
            # Trim silence
            if trim_silence:
                y = self.trim_silence(y, sr)
            
            # Normalize
            if normalize:
                y = self.normalize_audio(y, method="peak")
            
            # Set output path
            if output_path is None:
                name, ext = os.path.splitext(file_path)
                output_path = f"{name}_preprocessed{ext}"
            
            # Save preprocessed audio
            sf.write(output_path, y, sr)
            
            logger.info(f"Preprocessing completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error preprocessing file {file_path}: {str(e)}")
            raise
