"""
ML Training Module

Handles training workflows for different ML models in the DJ AI system.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Dict, Any, Tuple, List, Optional
import joblib
import os

from ml.models import TransitionPredictionModel, MusicFeatureEncoder

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Training pipeline for DJ AI models
    """
    
    def __init__(self, model_save_dir: str = "./ml/models/"):
        """
        Initialize training pipeline
        
        Args:
            model_save_dir: Directory to save trained models
        """
        self.model_save_dir = model_save_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Create model directory if it doesn't exist
        os.makedirs(model_save_dir, exist_ok=True)
        
        logger.info(f"TrainingPipeline initialized with model_save_dir: {model_save_dir}")
    
    def prepare_transition_data(self, tracks_data: List[Dict[str, Any]], 
                              transition_labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for transition prediction training
        
        Args:
            tracks_data: List of track feature dictionaries
            transition_labels: Binary labels for good/bad transitions
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            # Extract relevant features for transition prediction
            feature_keys = [
                'bpm', 'energy', 'danceability', 'valence', 'loudness',
                'spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate'
            ]
            
            features_list = []
            for track in tracks_data:
                track_features = []
                for key in feature_keys:
                    track_features.append(track.get(key, 0.0))
                features_list.append(track_features)
            
            X = np.array(features_list)
            y = np.array(transition_labels)
            
            logger.info(f"Prepared transition data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing transition data: {str(e)}")
            raise
    
    def create_synthetic_transition_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic training data for transition prediction
        This is a placeholder for when real data is not available
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            np.random.seed(42)  # For reproducibility
            
            # Generate synthetic track features
            n_features = 8
            X = np.random.rand(n_samples, n_features)
            
            # Scale features to realistic ranges
            X[:, 0] = X[:, 0] * 80 + 80  # BPM: 80-160
            X[:, 1] = X[:, 1]  # Energy: 0-1
            X[:, 2] = X[:, 2]  # Danceability: 0-1
            X[:, 3] = X[:, 3]  # Valence: 0-1
            X[:, 4] = X[:, 4] * -20 - 40  # Loudness: -60 to -40 dB
            X[:, 5] = X[:, 5] * 2000 + 1000  # Spectral centroid
            X[:, 6] = X[:, 6] * 4000 + 2000  # Spectral rolloff
            X[:, 7] = X[:, 7] * 0.1  # Zero crossing rate
            
            # Generate labels based on simple rules (for demo purposes)
            # Good transitions: similar BPM, complementary energy/valence
            y = np.zeros(n_samples)
            for i in range(0, n_samples, 2):
                if i + 1 < n_samples:
                    bpm_diff = abs(X[i, 0] - X[i+1, 0])
                    energy_complement = abs((X[i, 1] + X[i+1, 1]) - 1.0)
                    
                    # Label as good transition if BPM difference < 10 and energy is complementary
                    if bpm_diff < 10 and energy_complement < 0.3:
                        y[i] = 1
                        y[i+1] = 1
            
            logger.info(f"Created synthetic transition data: {X.shape[0]} samples")
            logger.info(f"Positive transitions: {np.sum(y)} / {len(y)} ({np.mean(y)*100:.1f}%)")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating synthetic data: {str(e)}")
            raise
    
    def train_transition_model(self, X: np.ndarray, y: np.ndarray, 
                             test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the transition prediction model
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Test set proportion
            random_state: Random state for reproducibility
            
        Returns:
            Training results and metrics
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            model = TransitionPredictionModel(input_dim=X.shape[1])
            history = model.train(X_train_scaled, y_train, X_test_scaled, y_test)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_binary = (y_pred > 0.5).astype(int).flatten()
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred_binary)),
                'precision': float(precision_score(y_test, y_pred_binary)),
                'recall': float(recall_score(y_test, y_pred_binary)),
                'f1_score': float(f1_score(y_test, y_pred_binary))
            }
            
            # Save model and scaler
            model_path = os.path.join(self.model_save_dir, "transition_model.h5")
            scaler_path = os.path.join(self.model_save_dir, "transition_scaler.joblib")
            
            model.save_model(model_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Transition model trained successfully")
            logger.info(f"Test Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"Test F1-Score: {metrics['f1_score']:.3f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'history': history,
                'model_path': model_path,
                'scaler_path': scaler_path
            }
            
        except Exception as e:
            logger.error(f"Error training transition model: {str(e)}")
            raise
    
    def train_feature_encoder(self, features_data: List[Dict[str, Any]], 
                            encoding_dim: int = 32) -> Dict[str, Any]:
        """
        Train the music feature autoencoder
        
        Args:
            features_data: List of feature dictionaries
            encoding_dim: Dimension of encoded features
            
        Returns:
            Training results
        """
        try:
            # Prepare feature matrix
            feature_keys = [
                'energy', 'danceability', 'valence', 'loudness',
                'spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate',
                'tempo', 'rhythm_strength'
            ]
            
            features_list = []
            for track in features_data:
                track_features = []
                for key in feature_keys:
                    track_features.append(track.get(key, 0.0))
                features_list.append(track_features)
            
            X = np.array(features_list)
            
            # Split data
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train autoencoder
            encoder = MusicFeatureEncoder(input_dim=X.shape[1], encoding_dim=encoding_dim)
            history = encoder.train_autoencoder(X_train_scaled, X_test_scaled)
            
            # Save encoder and scaler
            encoder_path = os.path.join(self.model_save_dir, "feature_encoder.h5")
            encoder_scaler_path = os.path.join(self.model_save_dir, "encoder_scaler.joblib")
            
            encoder.autoencoder.save(encoder_path)
            joblib.dump(self.scaler, encoder_scaler_path)
            
            logger.info(f"Feature encoder trained successfully")
            
            return {
                'encoder': encoder,
                'history': history,
                'encoder_path': encoder_path,
                'scaler_path': encoder_scaler_path
            }
            
        except Exception as e:
            logger.error(f"Error training feature encoder: {str(e)}")
            raise
    
    def create_training_dataset(self, audio_files: List[str]) -> Dict[str, Any]:
        """
        Create training dataset from audio files
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Training dataset information
        """
        try:
            from audio.features import FeatureExtractor
            
            feature_extractor = FeatureExtractor()
            
            # Extract features from all files
            all_features = []
            for file_path in audio_files:
                try:
                    features = feature_extractor.extract_features(file_path)
                    features['file_path'] = file_path
                    all_features.append(features)
                except Exception as e:
                    logger.warning(f"Failed to extract features from {file_path}: {str(e)}")
                    continue
            
            logger.info(f"Extracted features from {len(all_features)} files")
            
            # Save dataset
            dataset_path = os.path.join(self.model_save_dir, "training_dataset.joblib")
            joblib.dump(all_features, dataset_path)
            
            return {
                'features': all_features,
                'num_files': len(all_features),
                'dataset_path': dataset_path
            }
            
        except Exception as e:
            logger.error(f"Error creating training dataset: {str(e)}")
            raise
    
    def run_full_training_pipeline(self, use_synthetic_data: bool = True) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            use_synthetic_data: Whether to use synthetic data for training
            
        Returns:
            Complete training results
        """
        try:
            logger.info("Starting full training pipeline...")
            
            if use_synthetic_data:
                # Use synthetic data for demonstration
                X, y = self.create_synthetic_transition_data(n_samples=2000)
                transition_results = self.train_transition_model(X, y)
                
                # Create synthetic feature data for autoencoder
                synthetic_features = []
                for i in range(1000):
                    features = {
                        'energy': np.random.rand(),
                        'danceability': np.random.rand(),
                        'valence': np.random.rand(),
                        'loudness': np.random.rand() * -20 - 40,
                        'spectral_centroid': np.random.rand() * 2000 + 1000,
                        'spectral_rolloff': np.random.rand() * 4000 + 2000,
                        'zero_crossing_rate': np.random.rand() * 0.1,
                        'tempo': np.random.rand() * 80 + 80,
                        'rhythm_strength': np.random.rand()
                    }
                    synthetic_features.append(features)
                
                encoder_results = self.train_feature_encoder(synthetic_features)
                
            else:
                # TODO: Implement training with real audio data
                logger.warning("Real data training not implemented yet. Using synthetic data.")
                return self.run_full_training_pipeline(use_synthetic_data=True)
            
            results = {
                'transition_model': transition_results,
                'feature_encoder': encoder_results,
                'status': 'completed'
            }
            
            logger.info("Full training pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise
