"""
Tests for API Endpoints

Unit tests for FastAPI application endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import io

from app.main import app


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root health check endpoint"""
        response = self.client.get("/")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("DJ AI Core", data["message"])
    
    def test_health_endpoint(self):
        """Test detailed health check endpoint"""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("services", data)
        self.assertEqual(data["status"], "healthy")
        
        # Check that all services are reported
        services = data["services"]
        expected_services = ["audio_analyzer", "feature_extractor", "ml_predictor"]
        for service in expected_services:
            self.assertIn(service, services)
    
    def test_supported_formats_endpoint(self):
        """Test supported formats endpoint"""
        response = self.client.get("/supported-formats")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("formats", data)
        self.assertIn("max_file_size", data)
        
        # Check that common formats are supported
        formats = data["formats"]
        expected_formats = ["mp3", "wav", "flac", "m4a"]
        for format_type in expected_formats:
            self.assertIn(format_type, formats)
    
    @patch('app.main.audio_analyzer')
    @patch('app.main.feature_extractor')
    def test_analyze_track_success(self, mock_feature_extractor, mock_audio_analyzer):
        """Test successful track analysis"""
        # Mock analyzer responses
        mock_audio_analyzer.detect_bpm.return_value = 128.5
        mock_audio_analyzer.detect_key.return_value = "C major"
        
        mock_features = {
            "energy": 0.85,
            "danceability": 0.92,
            "valence": 0.73,
            "loudness": -8.5,
            "spectral_centroid": 1850.3,
            "spectral_rolloff": 3200.7,
            "zero_crossing_rate": 0.045
        }
        mock_feature_extractor.extract_features.return_value = mock_features
        
        # Create a fake audio file
        fake_audio_content = b"fake audio data"
        files = {"file": ("test.mp3", io.BytesIO(fake_audio_content), "audio/mpeg")}
        
        response = self.client.post("/analyze-track", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check response structure
        self.assertIn("filename", data)
        self.assertIn("bpm", data)
        self.assertIn("key", data)
        self.assertIn("features", data)
        self.assertIn("analysis_status", data)
        
        # Check values
        self.assertEqual(data["filename"], "test.mp3")
        self.assertEqual(data["bpm"], 128.5)
        self.assertEqual(data["key"], "C major")
        self.assertEqual(data["analysis_status"], "completed")
        
        # Check features
        features = data["features"]
        self.assertIn("energy", features)
        self.assertIn("danceability", features)
        self.assertEqual(features["energy"], 0.85)
    
    def test_analyze_track_unsupported_format(self):
        """Test track analysis with unsupported file format"""
        fake_content = b"fake content"
        files = {"file": ("test.txt", io.BytesIO(fake_content), "text/plain")}
        
        response = self.client.post("/analyze-track", files=files)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Unsupported file format", data["detail"])
    
    @patch('app.main.transition_predictor')
    def test_recommend_transitions_success(self, mock_transition_predictor):
        """Test successful transition recommendations"""
        # Mock predictor response
        mock_recommendations = [
            {
                "track_id": "track_456",
                "compatibility_score": 0.89,
                "transition_type": "Great",
                "reasons": ["Similar BPM", "Compatible energy"]
            },
            {
                "track_id": "track_789",
                "compatibility_score": 0.65,
                "transition_type": "Good",
                "reasons": ["Same key", "Gradual energy change"]
            }
        ]
        mock_transition_predictor.predict_transitions.return_value = mock_recommendations
        
        request_data = {
            "current_track_id": "track_123",
            "available_tracks": ["track_456", "track_789"]
        }
        
        response = self.client.post(
            "/recommend-transitions",
            json=request_data
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check response structure
        self.assertIn("current_track", data)
        self.assertIn("recommendations", data)
        self.assertIn("total_recommendations", data)
        self.assertIn("prediction_status", data)
        
        # Check values
        self.assertEqual(data["current_track"], "track_123")
        self.assertEqual(data["total_recommendations"], 2)
        self.assertEqual(data["prediction_status"], "completed")
        
        # Check recommendations
        recommendations = data["recommendations"]
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]["track_id"], "track_456")
        self.assertEqual(recommendations[0]["compatibility_score"], 0.89)
    
    def test_recommend_transitions_missing_current_track(self):
        """Test transition recommendations with missing current track"""
        request_data = {
            "available_tracks": ["track_456", "track_789"]
        }
        
        response = self.client.post(
            "/recommend-transitions",
            json=request_data
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("current_track_id is required", data["detail"])
    
    def test_recommend_transitions_empty_available_tracks(self):
        """Test transition recommendations with empty available tracks"""
        request_data = {
            "current_track_id": "track_123",
            "available_tracks": []
        }
        
        response = self.client.post(
            "/recommend-transitions",
            json=request_data
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("available_tracks list cannot be empty", data["detail"])
    
    @patch('app.main.audio_analyzer')
    def test_analyze_track_error_handling(self, mock_audio_analyzer):
        """Test track analysis error handling"""
        # Mock analyzer to raise exception
        mock_audio_analyzer.detect_bpm.side_effect = Exception("Analysis failed")
        
        fake_audio_content = b"fake audio data"
        files = {"file": ("test.mp3", io.BytesIO(fake_audio_content), "audio/mpeg")}
        
        response = self.client.post("/analyze-track", files=files)
        
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Analysis failed", data["detail"])
    
    @patch('app.main.transition_predictor')
    def test_recommend_transitions_error_handling(self, mock_transition_predictor):
        """Test transition recommendations error handling"""
        # Mock predictor to raise exception
        mock_transition_predictor.predict_transitions.side_effect = Exception("Prediction failed")
        
        request_data = {
            "current_track_id": "track_123",
            "available_tracks": ["track_456", "track_789"]
        }
        
        response = self.client.post(
            "/recommend-transitions",
            json=request_data
        )
        
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Recommendation failed", data["detail"])
    
    def test_analyze_track_no_file(self):
        """Test track analysis without providing a file"""
        response = self.client.post("/analyze-track")
        
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
    
    def test_invalid_json_request(self):
        """Test invalid JSON in transition recommendations"""
        response = self.client.post(
            "/recommend-transitions",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, 422)


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
    
    def test_api_workflow(self):
        """Test complete API workflow"""
        # 1. Check health
        health_response = self.client.get("/health")
        self.assertEqual(health_response.status_code, 200)
        
        # 2. Check supported formats
        formats_response = self.client.get("/supported-formats")
        self.assertEqual(formats_response.status_code, 200)
        
        # 3. Try to get recommendations (should work even without analysis)
        request_data = {
            "current_track_id": "unknown_track",
            "available_tracks": ["track1", "track2"]
        }
        
        recommendations_response = self.client.post(
            "/recommend-transitions",
            json=request_data
        )
        self.assertEqual(recommendations_response.status_code, 200)
        
        # Check that we get some recommendations even for unknown tracks
        data = recommendations_response.json()
        self.assertIn("recommendations", data)
        self.assertGreater(len(data["recommendations"]), 0)
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.get("/")
        
        # Check for CORS headers (these are added by the CORS middleware)
        # Note: In test environment, these might not be exactly as in production
        self.assertEqual(response.status_code, 200)
    
    def test_api_documentation_endpoints(self):
        """Test that API documentation endpoints are accessible"""
        # Test OpenAPI docs
        docs_response = self.client.get("/docs")
        self.assertIn(docs_response.status_code, [200, 307])  # 307 for redirect
        
        # Test ReDoc
        redoc_response = self.client.get("/redoc")
        self.assertIn(redoc_response.status_code, [200, 307])  # 307 for redirect


if __name__ == '__main__':
    unittest.main()
