"""
Machine Learning Model Definitions

Defines neural network architectures for transition prediction and music analysis.
"""

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class TransitionPredictionModel:
    """
    Neural network model for predicting optimal track transitions
    """
    
    def __init__(self, input_dim: int = 50, hidden_dim: int = 128, output_dim: int = 1):
        """
        Initialize the transition prediction model
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (transition compatibility score)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = None
        logger.info(f"TransitionPredictionModel initialized: {input_dim}->{hidden_dim}->{output_dim}")
    
    def build_model(self) -> keras.Model:
        """
        Build the neural network architecture
        
        Returns:
            Compiled Keras model
        """
        try:
            model = keras.Sequential([
                layers.Input(shape=(self.input_dim,)),
                layers.Dense(self.hidden_dim, activation='relu', name='hidden1'),
                layers.Dropout(0.3),
                layers.Dense(self.hidden_dim // 2, activation='relu', name='hidden2'),
                layers.Dropout(0.2),
                layers.Dense(self.hidden_dim // 4, activation='relu', name='hidden3'),
                layers.Dense(self.output_dim, activation='sigmoid', name='output')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.model = model
            logger.info("Transition prediction model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the transition prediction model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history
        """
        try:
            if self.model is None:
                self.model = self.build_model()
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Define callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=10, 
                    restore_best_weights=True,
                    monitor='val_loss' if validation_data else 'loss'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info(f"Model training completed after {len(history.history['loss'])} epochs")
            return history.history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model
        
        Args:
            X: Input features
            
        Returns:
            Prediction scores
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        try:
            if self.model is None:
                raise ValueError("No model to save. Train model first.")
            
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        try:
            self.model = keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


class MusicFeatureEncoder:
    """
    Autoencoder for learning compact representations of music features
    """
    
    def __init__(self, input_dim: int = 100, encoding_dim: int = 32):
        """
        Initialize the music feature encoder
        
        Args:
            input_dim: Input feature dimension
            encoding_dim: Encoded feature dimension
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        logger.info(f"MusicFeatureEncoder initialized: {input_dim}->{encoding_dim}")
    
    def build_autoencoder(self) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """
        Build encoder, decoder, and autoencoder models
        
        Returns:
            Tuple of (encoder, decoder, autoencoder) models
        """
        try:
            # Encoder
            input_audio = keras.Input(shape=(self.input_dim,))
            encoded = layers.Dense(self.encoding_dim * 2, activation='relu')(input_audio)
            encoded = layers.Dropout(0.2)(encoded)
            encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
            
            self.encoder = keras.Model(input_audio, encoded, name='encoder')
            
            # Decoder
            input_encoded = keras.Input(shape=(self.encoding_dim,))
            decoded = layers.Dense(self.encoding_dim * 2, activation='relu')(input_encoded)
            decoded = layers.Dropout(0.2)(decoded)
            decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)
            
            self.decoder = keras.Model(input_encoded, decoded, name='decoder')
            
            # Autoencoder
            autoencoder_output = self.decoder(self.encoder(input_audio))
            self.autoencoder = keras.Model(input_audio, autoencoder_output, name='autoencoder')
            
            self.autoencoder.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            logger.info("Autoencoder models built successfully")
            return self.encoder, self.decoder, self.autoencoder
            
        except Exception as e:
            logger.error(f"Error building autoencoder: {str(e)}")
            raise
    
    def train_autoencoder(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None,
                         epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the autoencoder
        
        Args:
            X_train: Training data
            X_val: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history
        """
        try:
            if self.autoencoder is None:
                self.build_autoencoder()
            
            validation_data = None
            if X_val is not None:
                validation_data = (X_val, X_val)  # Autoencoder reconstructs input
            
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=15,
                    restore_best_weights=True,
                    monitor='val_loss' if validation_data else 'loss'
                )
            ]
            
            history = self.autoencoder.fit(
                X_train, X_train,  # Autoencoder learns to reconstruct input
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info(f"Autoencoder training completed after {len(history.history['loss'])} epochs")
            return history.history
            
        except Exception as e:
            logger.error(f"Error training autoencoder: {str(e)}")
            raise
    
    def encode_features(self, X: np.ndarray) -> np.ndarray:
        """
        Encode features using the trained encoder
        
        Args:
            X: Input features
            
        Returns:
            Encoded features
        """
        try:
            if self.encoder is None:
                raise ValueError("Encoder not trained. Call train_autoencoder() first.")
            
            encoded = self.encoder.predict(X)
            return encoded
            
        except Exception as e:
            logger.error(f"Error encoding features: {str(e)}")
            raise


class BeatSyncModel:
    """
    Model for beat synchronization and tempo matching
    """
    
    def __init__(self, sequence_length: int = 128):
        """
        Initialize beat sync model
        
        Args:
            sequence_length: Length of beat sequences to process
        """
        self.sequence_length = sequence_length
        self.model = None
        logger.info(f"BeatSyncModel initialized with sequence_length={sequence_length}")
    
    def build_lstm_model(self, input_dim: int = 13) -> keras.Model:
        """
        Build LSTM model for beat prediction
        
        Args:
            input_dim: Input feature dimension per timestep
            
        Returns:
            Compiled LSTM model
        """
        try:
            model = keras.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, input_dim)),
                layers.Dropout(0.3),
                layers.LSTM(32, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')  # Beat presence prediction
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info("LSTM beat sync model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            raise
