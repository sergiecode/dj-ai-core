"""
Transition Prediction Module

Provides AI-powered recommendations for optimal track transitions.
"""

import numpy as np
import logging
from typing import List, Dict, Any
import joblib
import os

logger = logging.getLogger(__name__)


class TransitionPredictor:
    """
    AI-powered system for predicting optimal track transitions
    """
    
    def __init__(self, model_dir: str = "./ml/models/"):
        """
        Initialize the transition predictor
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.transition_model = None
        self.scaler = None
        self.feature_encoder = None
        self.track_database = {}  # Store track features
        
        logger.info(f"TransitionPredictor initialized with model_dir: {model_dir}")
        
        # Try to load existing models
        self._load_models()
    
    def _load_models(self):
        """Load trained models if available"""
        try:
            # Load transition model
            model_path = os.path.join(self.model_dir, "transition_model.h5")
            scaler_path = os.path.join(self.model_dir, "transition_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                from ml.models import TransitionPredictionModel
                self.transition_model = TransitionPredictionModel()
                self.transition_model.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded transition prediction model")
            else:
                logger.warning("Transition model not found. Using fallback methods.")
            
        except Exception as e:
            logger.warning(f"Could not load models: {str(e)}. Using fallback methods.")
    
    def add_track_to_database(self, track_id: str, features: Dict[str, Any]):
        """
        Add a track's features to the internal database
        
        Args:
            track_id: Unique identifier for the track
            features: Dictionary of audio features
        """
        try:
            self.track_database[track_id] = features
            logger.debug(f"Added track {track_id} to database")
            
        except Exception as e:
            logger.error(f"Error adding track to database: {str(e)}")
    
    def calculate_compatibility_score(self, track1_features: Dict[str, Any], 
                                    track2_features: Dict[str, Any]) -> float:
        """
        Calculate compatibility score between two tracks using rule-based approach
        
        Args:
            track1_features: Features of the first track
            track2_features: Features of the second track
            
        Returns:
            Compatibility score between 0 and 1
        """
        try:
            score = 0.0
            weights = {
                'bpm_similarity': 0.3,
                'energy_compatibility': 0.2,
                'key_compatibility': 0.15,
                'valence_similarity': 0.15,
                'loudness_compatibility': 0.1,
                'spectral_similarity': 0.1
            }
            
            # BPM similarity (closer BPMs = higher score)
            bpm1 = track1_features.get('bpm', 120)
            bpm2 = track2_features.get('bpm', 120)
            bpm_diff = abs(bpm1 - bpm2)
            bpm_score = max(0, 1 - (bpm_diff / 20))  # Normalize by 20 BPM difference
            score += weights['bpm_similarity'] * bpm_score
            
            # Energy compatibility (gradual transitions preferred)
            energy1 = track1_features.get('energy', 0.5)
            energy2 = track2_features.get('energy', 0.5)
            energy_diff = abs(energy1 - energy2)
            energy_score = max(0, 1 - energy_diff)
            score += weights['energy_compatibility'] * energy_score
            
            # Key compatibility (simplified - same key or relative keys)
            key1 = track1_features.get('key', 'C major')
            key2 = track2_features.get('key', 'C major')
            key_score = 1.0 if key1 == key2 else 0.5  # Simplified key matching
            score += weights['key_compatibility'] * key_score
            
            # Valence similarity
            valence1 = track1_features.get('valence', 0.5)
            valence2 = track2_features.get('valence', 0.5)
            valence_diff = abs(valence1 - valence2)
            valence_score = max(0, 1 - valence_diff)
            score += weights['valence_similarity'] * valence_score
            
            # Loudness compatibility
            loudness1 = track1_features.get('loudness', -20)
            loudness2 = track2_features.get('loudness', -20)
            loudness_diff = abs(loudness1 - loudness2)
            loudness_score = max(0, 1 - (loudness_diff / 20))  # Normalize by 20dB
            score += weights['loudness_compatibility'] * loudness_score
            
            # Spectral similarity
            spec1 = track1_features.get('spectral_centroid', 1500)
            spec2 = track2_features.get('spectral_centroid', 1500)
            spec_diff = abs(spec1 - spec2)
            spec_score = max(0, 1 - (spec_diff / 2000))  # Normalize by 2000Hz
            score += weights['spectral_similarity'] * spec_score
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating compatibility score: {str(e)}")
            return 0.5  # Return neutral score on error
    
    def predict_transition_ml(self, track1_features: Dict[str, Any], 
                            track2_features: Dict[str, Any]) -> float:
        """
        Use ML model to predict transition quality
        
        Args:
            track1_features: Features of the first track
            track2_features: Features of the second track
            
        Returns:
            ML-predicted transition score
        """
        try:
            if self.transition_model is None or self.scaler is None:
                logger.warning("ML model not available, using rule-based scoring")
                return self.calculate_compatibility_score(track1_features, track2_features)
            
            # Extract features for ML model
            feature_keys = [
                'bpm', 'energy', 'danceability', 'valence', 'loudness',
                'spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate'
            ]
            
            # Combine features from both tracks
            combined_features = []
            for key in feature_keys:
                combined_features.append(track1_features.get(key, 0.0))
                combined_features.append(track2_features.get(key, 0.0))
                # Add difference features
                diff = track1_features.get(key, 0.0) - track2_features.get(key, 0.0)
                combined_features.append(diff)
            
            # Ensure we have the right number of features (pad or truncate if necessary)
            target_features = 24  # 8 features × 3 (track1, track2, diff)
            if len(combined_features) < target_features:
                combined_features.extend([0.0] * (target_features - len(combined_features)))
            elif len(combined_features) > target_features:
                combined_features = combined_features[:target_features]
            
            # Scale features and predict
            X = np.array([combined_features])
            X_scaled = self.scaler.transform(X)
            prediction = self.transition_model.predict(X_scaled)
            
            return float(prediction[0][0])
            
        except Exception as e:
            logger.error(f"Error with ML prediction: {str(e)}")
            return self.calculate_compatibility_score(track1_features, track2_features)
    
    def predict_transitions(self, current_track_id: str, 
                          available_tracks: List[str]) -> List[Dict[str, Any]]:
        """
        Predict and rank transitions for a current track
        
        Args:
            current_track_id: ID of the currently playing track
            available_tracks: List of available track IDs for transition
            
        Returns:
            Ranked list of transition recommendations
        """
        try:
            if current_track_id not in self.track_database:
                logger.warning(f"Current track {current_track_id} not in database")
                # Return default recommendations
                return self._generate_default_recommendations(available_tracks)
            
            current_features = self.track_database[current_track_id]
            recommendations = []
            
            for track_id in available_tracks:
                if track_id == current_track_id:
                    continue  # Skip the same track
                
                if track_id not in self.track_database:
                    logger.warning(f"Track {track_id} not in database, using default features")
                    # Use default features for unknown tracks
                    track_features = {
                        'bpm': 120, 'energy': 0.5, 'danceability': 0.5,
                        'valence': 0.5, 'loudness': -20, 'spectral_centroid': 1500,
                        'spectral_rolloff': 3000, 'zero_crossing_rate': 0.05
                    }
                else:
                    track_features = self.track_database[track_id]
                
                # Calculate compatibility score
                if self.transition_model is not None:
                    score = self.predict_transition_ml(current_features, track_features)
                else:
                    score = self.calculate_compatibility_score(current_features, track_features)
                
                recommendation = {
                    'track_id': track_id,
                    'compatibility_score': round(score, 3),
                    'transition_type': self._classify_transition_type(score),
                    'reasons': self._generate_transition_reasons(current_features, track_features, score)
                }
                
                recommendations.append(recommendation)
            
            # Sort by compatibility score (descending)
            recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)
            
            logger.info(f"Generated {len(recommendations)} transition recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error predicting transitions: {str(e)}")
            return self._generate_default_recommendations(available_tracks)
    
    def _classify_transition_type(self, score: float) -> str:
        """Classify transition type based on score"""
        if score >= 0.8:
            return "Perfect"
        elif score >= 0.6:
            return "Great"
        elif score >= 0.4:
            return "Good"
        elif score >= 0.2:
            return "Okay"
        else:
            return "Challenging"
    
    def _generate_transition_reasons(self, current_features: Dict[str, Any], 
                                   next_features: Dict[str, Any], score: float) -> List[str]:
        """Generate explanations for the transition recommendation"""
        reasons = []
        
        try:
            # BPM analysis
            bpm_current = current_features.get('bpm', 120)
            bpm_next = next_features.get('bpm', 120)
            bpm_diff = abs(bpm_current - bpm_next)
            
            if bpm_diff < 5:
                reasons.append("Very similar BPM - seamless transition")
            elif bpm_diff < 10:
                reasons.append("Compatible BPM - smooth transition")
            elif bpm_diff > 20:
                reasons.append("Large BPM difference - requires skill")
            
            # Energy analysis
            energy_current = current_features.get('energy', 0.5)
            energy_next = next_features.get('energy', 0.5)
            
            if energy_next > energy_current + 0.2:
                reasons.append("Energy increase - builds excitement")
            elif energy_next < energy_current - 0.2:
                reasons.append("Energy decrease - creates calm moment")
            else:
                reasons.append("Maintains energy level")
            
            # Key compatibility
            key_current = current_features.get('key', 'C major')
            key_next = next_features.get('key', 'C major')
            
            if key_current == key_next:
                reasons.append("Same key - harmonically compatible")
            else:
                reasons.append("Different key - adds harmonic interest")
            
            # Overall assessment
            if score >= 0.7:
                reasons.append("Highly recommended transition")
            elif score < 0.3:
                reasons.append("Requires advanced mixing skills")
            
        except Exception as e:
            logger.error(f"Error generating reasons: {str(e)}")
            reasons.append("Analysis based on audio characteristics")
        
        return reasons[:3]  # Return top 3 reasons
    
    def _generate_default_recommendations(self, available_tracks: List[str]) -> List[Dict[str, Any]]:
        """Generate default recommendations when data is unavailable"""
        recommendations = []
        
        for i, track_id in enumerate(available_tracks):
            score = 0.5 + (0.1 * (i % 5)) / 5  # Slightly varied default scores
            recommendation = {
                'track_id': track_id,
                'compatibility_score': round(score, 3),
                'transition_type': "Unknown",
                'reasons': ["Track analysis not available", "Default recommendation"]
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def analyze_dj_set(self, track_sequence: List[str]) -> Dict[str, Any]:
        """
        Analyze a complete DJ set for transition quality
        
        Args:
            track_sequence: Ordered list of track IDs in the set
            
        Returns:
            Analysis of the entire set
        """
        try:
            if len(track_sequence) < 2:
                return {"error": "Need at least 2 tracks for set analysis"}
            
            transitions = []
            total_score = 0.0
            
            for i in range(len(track_sequence) - 1):
                current_track = track_sequence[i]
                next_track = track_sequence[i + 1]
                
                # Get transition score
                recommendations = self.predict_transitions(current_track, [next_track])
                if recommendations:
                    score = recommendations[0]['compatibility_score']
                    transition_type = recommendations[0]['transition_type']
                else:
                    score = 0.5
                    transition_type = "Unknown"
                
                transitions.append({
                    'from_track': current_track,
                    'to_track': next_track,
                    'score': score,
                    'type': transition_type
                })
                
                total_score += score
            
            avg_score = total_score / len(transitions)
            
            analysis = {
                'total_tracks': len(track_sequence),
                'total_transitions': len(transitions),
                'average_transition_score': round(avg_score, 3),
                'set_quality': self._classify_transition_type(avg_score),
                'transitions': transitions,
                'recommendations': self._generate_set_recommendations(transitions)
            }
            
            logger.info(f"Analyzed DJ set: {len(track_sequence)} tracks, avg score: {avg_score:.3f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing DJ set: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_set_recommendations(self, transitions: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving the DJ set"""
        recommendations = []
        
        try:
            scores = [t['score'] for t in transitions]
            avg_score = np.mean(scores)
            min_score = min(scores)
            
            if avg_score >= 0.7:
                recommendations.append("Excellent set flow - great track selection!")
            elif avg_score >= 0.5:
                recommendations.append("Good set with room for improvement")
            else:
                recommendations.append("Consider reorganizing track order for better flow")
            
            if min_score < 0.3:
                recommendations.append("Some challenging transitions detected - review problematic segments")
            
            # Analyze transition patterns
            low_scores = [t for t in transitions if t['score'] < 0.4]
            if len(low_scores) > len(transitions) * 0.3:
                recommendations.append("High number of difficult transitions - consider alternative track choices")
            
        except Exception as e:
            logger.error(f"Error generating set recommendations: {str(e)}")
            recommendations.append("Unable to generate detailed recommendations")
        
        return recommendations
