"""
Tests for Audio Processing Module

Unit tests for audio analysis, feature extraction, and preprocessing.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Test imports
from audio.analysis import AudioAnalyzer
from audio.features import FeatureExtractor
from audio.preprocessing import AudioPreprocessor


class TestAudioAnalyzer(unittest.TestCase):
    """Test cases for AudioAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = AudioAnalyzer(sample_rate=22050)
    
    def test_analyzer_initialization(self):
        """Test AudioAnalyzer initialization"""
        self.assertEqual(self.analyzer.sample_rate, 22050)
    
    @patch('librosa.load')
    def test_load_audio(self, mock_load):
        """Test audio loading functionality"""
        # Mock librosa.load return value
        mock_y = np.random.rand(44100)  # 2 seconds at 22050 Hz
        mock_sr = 22050
        mock_load.return_value = (mock_y, mock_sr)
        
        y, sr = self.analyzer.load_audio("dummy_path.mp3")
        
        self.assertEqual(len(y), 44100)
        self.assertEqual(sr, 22050)
        mock_load.assert_called_once()
    
    @patch('librosa.beat.beat_track')
    @patch('audio.analysis.AudioAnalyzer.load_audio')
    def test_detect_bpm(self, mock_load_audio, mock_beat_track):
        """Test BPM detection"""
        # Mock dependencies
        mock_y = np.random.rand(44100)
        mock_sr = 22050
        mock_load_audio.return_value = (mock_y, mock_sr)
        mock_beat_track.return_value = (128.0, np.array([0, 100, 200]))
        
        bpm = self.analyzer.detect_bpm("dummy_path.mp3")
        
        self.assertEqual(bpm, 128.0)
        mock_load_audio.assert_called_once()
        mock_beat_track.assert_called_once()
    
    @patch('audio.analysis.AudioAnalyzer.load_audio')
    def test_detect_bpm_error_handling(self, mock_load_audio):
        """Test BPM detection error handling"""
        mock_load_audio.side_effect = Exception("Audio loading failed")
        
        bpm = self.analyzer.detect_bpm("invalid_path.mp3")
        
        # Should return default BPM on error
        self.assertEqual(bpm, 120.0)
    
    @patch('librosa.feature.chroma_stft')
    @patch('audio.analysis.AudioAnalyzer.load_audio')
    def test_detect_key(self, mock_load_audio, mock_chroma):
        """Test key detection"""
        # Mock dependencies
        mock_y = np.random.rand(44100)
        mock_sr = 22050
        mock_load_audio.return_value = (mock_y, mock_sr)
        
        # Mock chroma features (12-dimensional)
        mock_chroma_data = np.random.rand(12, 100) * 0.3  # Start with low values
        mock_chroma_data[0, :] = 0.9  # Make C the strongest (index 0)
        mock_chroma_data[4, :] = 0.7  # Major third (E) stronger
        mock_chroma_data[3, :] = 0.5  # Minor third (D#) weaker
        mock_chroma.return_value = mock_chroma_data
        
        key = self.analyzer.detect_key("dummy_path.mp3")
        
        self.assertIn("C", key)
        self.assertIn("major", key.lower())
        mock_load_audio.assert_called_once()
        mock_chroma.assert_called_once()
    
    @patch('librosa.get_duration')
    def test_get_audio_duration(self, mock_duration):
        """Test audio duration calculation"""
        mock_duration.return_value = 180.5
        
        duration = self.analyzer.get_audio_duration("dummy_path.mp3")
        
        self.assertEqual(duration, 180.5)
        mock_duration.assert_called_once()


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor(sample_rate=22050, n_mfcc=13)
    
    def test_extractor_initialization(self):
        """Test FeatureExtractor initialization"""
        self.assertEqual(self.extractor.sample_rate, 22050)
        self.assertEqual(self.extractor.n_mfcc, 13)
    
    def test_extract_spectral_features(self):
        """Test spectral feature extraction"""
        # Create dummy audio data
        y = np.random.rand(44100)
        sr = 22050
        
        with patch('librosa.feature.spectral_centroid') as mock_centroid, \
             patch('librosa.feature.spectral_rolloff') as mock_rolloff, \
             patch('librosa.feature.spectral_bandwidth') as mock_bandwidth, \
             patch('librosa.feature.zero_crossing_rate') as mock_zcr:
            
            # Mock return values
            mock_centroid.return_value = np.array([[1500.0] * 100])
            mock_rolloff.return_value = np.array([[3000.0] * 100])
            mock_bandwidth.return_value = np.array([[500.0] * 100])
            mock_zcr.return_value = np.array([[0.05] * 100])
            
            features = self.extractor.extract_spectral_features(y, sr)
            
            # Check that all expected features are present
            expected_features = ['spectral_centroid', 'spectral_rolloff', 
                               'spectral_bandwidth', 'zero_crossing_rate']
            for feature in expected_features:
                self.assertIn(feature, features)
                self.assertIsInstance(features[feature], float)
    
    def test_extract_rhythm_features(self):
        """Test rhythm feature extraction"""
        y = np.random.rand(44100)
        sr = 22050
        
        with patch('librosa.beat.beat_track') as mock_beat_track:
            mock_beat_track.return_value = (128.0, np.array([0, 100, 200, 300]))
            
            features = self.extractor.extract_rhythm_features(y, sr)
            
            expected_features = ['tempo', 'rhythm_strength', 'num_beats']
            for feature in expected_features:
                self.assertIn(feature, features)
                self.assertIsInstance(features[feature], (float, int))
    
    @patch('audio.features.FeatureExtractor.load_audio')
    def test_extract_features_error_handling(self, mock_load_audio):
        """Test feature extraction error handling"""
        mock_load_audio.side_effect = Exception("File not found")
        
        features = self.extractor.extract_features("invalid_path.mp3")
        
        # Should return default features on error
        self.assertIn('energy', features)
        self.assertIn('danceability', features)
        self.assertEqual(features['energy'], 0.0)
    
    def test_extract_features_batch(self):
        """Test batch feature extraction"""
        file_paths = ["file1.mp3", "file2.mp3", "file3.mp3"]
        
        with patch.object(self.extractor, 'extract_features') as mock_extract:
            mock_extract.return_value = {
                'energy': 0.8,
                'tempo': 120.0,
                'danceability': 0.7
            }
            
            features_list = self.extractor.extract_features_batch(file_paths)
            
            self.assertEqual(len(features_list), 3)
            self.assertEqual(mock_extract.call_count, 3)
            
            # Check that file paths are included
            for features in features_list:
                self.assertIn('file_path', features)


class TestAudioPreprocessor(unittest.TestCase):
    """Test cases for AudioPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = AudioPreprocessor(target_sr=22050)
    
    def test_preprocessor_initialization(self):
        """Test AudioPreprocessor initialization"""
        self.assertEqual(self.preprocessor.target_sr, 22050)
    
    def test_normalize_audio_peak(self):
        """Test peak normalization"""
        # Create audio with known peak
        y = np.array([0.5, -0.8, 0.3, -0.2])
        
        y_normalized = self.preprocessor.normalize_audio(y, method="peak")
        
        # Peak should be 1.0 after normalization
        self.assertAlmostEqual(np.max(np.abs(y_normalized)), 1.0, places=5)
    
    def test_normalize_audio_rms(self):
        """Test RMS normalization"""
        y = np.random.rand(1000) * 0.5
        
        y_normalized = self.preprocessor.normalize_audio(y, method="rms")
        
        # Check that RMS is approximately 0.1
        rms = np.sqrt(np.mean(y_normalized**2))
        self.assertAlmostEqual(rms, 0.1, places=1)
    
    def test_resample_audio(self):
        """Test audio resampling"""
        y = np.random.rand(44100)  # 2 seconds at 22050 Hz
        orig_sr = 22050
        target_sr = 16000
        
        with patch('librosa.resample') as mock_resample:
            mock_resample.return_value = np.random.rand(32000)  # 2 seconds at 16000 Hz
            
            y_resampled, new_sr = self.preprocessor.resample_audio(y, orig_sr, target_sr)
            
            self.assertEqual(new_sr, target_sr)
            mock_resample.assert_called_once_with(y, orig_sr=orig_sr, target_sr=target_sr)
    
    def test_trim_silence(self):
        """Test silence trimming"""
        # Create audio with silence at beginning and end
        silence = np.zeros(1000)
        signal = np.random.rand(2000) * 0.5
        y = np.concatenate([silence, signal, silence])
        sr = 22050
        
        with patch('librosa.effects.trim') as mock_trim:
            mock_trim.return_value = (signal, None)
            
            y_trimmed = self.preprocessor.trim_silence(y, sr)
            
            self.assertEqual(len(y_trimmed), len(signal))
            mock_trim.assert_called_once()
    
    def test_apply_preemphasis(self):
        """Test preemphasis filter"""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        coeff = 0.97
        
        y_preemph = self.preprocessor.apply_preemphasis(y, coeff)
        
        # Check that preemphasis was applied correctly
        expected = np.array([1.0, 2.0 - 0.97 * 1.0, 3.0 - 0.97 * 2.0, 4.0 - 0.97 * 3.0])
        np.testing.assert_array_almost_equal(y_preemph, expected, decimal=5)
    
    @patch('librosa.effects.time_stretch')
    def test_change_tempo(self, mock_time_stretch):
        """Test tempo change"""
        y = np.random.rand(44100)
        rate = 1.2  # 20% faster
        
        mock_time_stretch.return_value = np.random.rand(int(44100 / rate))
        
        y_tempo = self.preprocessor.change_tempo(y, rate)
        
        mock_time_stretch.assert_called_once_with(y, rate=rate)
    
    @patch('librosa.effects.pitch_shift')
    def test_change_pitch(self, mock_pitch_shift):
        """Test pitch change"""
        y = np.random.rand(44100)
        sr = 22050
        n_steps = 2  # 2 semitones higher
        
        mock_pitch_shift.return_value = np.random.rand(44100)
        
        y_pitch = self.preprocessor.change_pitch(y, sr, n_steps)
        
        mock_pitch_shift.assert_called_once_with(y, sr=sr, n_steps=n_steps)


if __name__ == '__main__':
    unittest.main()
