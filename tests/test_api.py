"""
Test script for the Net Worth Estimator API.
Tests basic functionality without requiring actual image files.
"""

import pytest
import requests
import json
from pathlib import Path
import io
from PIL import Image
import numpy as np

# Test configuration
API_BASE_URL = "http://localhost:8000"

class TestNetWorthEstimatorAPI:
    """Test suite for the Net Worth Estimator API."""
    
    def test_root_endpoint(self):
        """Test the root endpoint returns API information."""
        response = requests.get(f"{API_BASE_URL}/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "Net Worth Estimator API"
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "services" in data
    
    def test_predict_endpoint_with_valid_image(self):
        """Test the predict endpoint with a valid test image."""
        # Create a simple test image
        test_image = self.create_test_image()
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Make request
        files = {"file": ("test_image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(f"{API_BASE_URL}/predict", files=files)
        
        assert response.status_code == 200
        
        data = response.json()
        self.validate_prediction_response(data)
    
    def test_predict_endpoint_with_invalid_file(self):
        """Test the predict endpoint with an invalid file type."""
        # Create a text file instead of an image
        text_content = b"This is not an image file"
        files = {"file": ("test.txt", io.BytesIO(text_content), "text/plain")}
        
        response = requests.post(f"{API_BASE_URL}/predict", files=files)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Invalid file type" in data["detail"]
    
    def test_predict_endpoint_no_file(self):
        """Test the predict endpoint without uploading a file."""
        response = requests.post(f"{API_BASE_URL}/predict")
        assert response.status_code == 422  # Unprocessable Entity (missing required field)
    
    def test_predict_endpoint_large_file(self):
        """Test the predict endpoint with a file that's too large."""
        # Create a large fake image (over 10MB)
        large_image = Image.new('RGB', (5000, 5000), color='red')
        img_bytes = io.BytesIO()
        large_image.save(img_bytes, format='JPEG', quality=100)
        img_bytes.seek(0)
        
        files = {"file": ("large_image.jpg", img_bytes, "image/jpeg")}
        response = requests.post(f"{API_BASE_URL}/predict", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "File too large" in data["detail"]
    
    def create_test_image(self, size=(400, 400)):
        """Create a simple test image with a face-like pattern."""
        # Create a simple image with some patterns that might be detected as a face
        img = Image.new('RGB', size, color='beige')
        pixels = img.load()
        
        # Add some face-like features (simplified)
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Add "eyes" (dark circles)
        for x in range(size[0]):
            for y in range(size[1]):
                # Left eye
                if ((x - center_x + 50) ** 2 + (y - center_y - 30) ** 2) < 400:
                    pixels[x, y] = (50, 50, 50)
                # Right eye
                elif ((x - center_x - 50) ** 2 + (y - center_y - 30) ** 2) < 400:
                    pixels[x, y] = (50, 50, 50)
                # Mouth
                elif ((x - center_x) ** 2 + (y - center_y + 50) ** 2) < 800 and y > center_y + 40:
                    pixels[x, y] = (100, 50, 50)
        
        return img
    
    def validate_prediction_response(self, data):
        """Validate the structure of a prediction response."""
        # Check required fields
        assert "success" in data
        assert data["success"] is True
        
        assert "estimated_net_worth" in data
        assert isinstance(data["estimated_net_worth"], (int, float))
        assert data["estimated_net_worth"] > 0
        
        assert "currency" in data
        assert data["currency"] == "USD"
        
        assert "top_matches" in data
        assert isinstance(data["top_matches"], list)
        assert len(data["top_matches"]) == 3
        
        # Validate each match
        for i, match in enumerate(data["top_matches"]):
            assert "name" in match
            assert "similarity_score" in match
            assert "industry" in match
            assert "rank" in match
            assert match["rank"] == i + 1
            assert 0 <= match["similarity_score"] <= 1
        
        assert "confidence_score" in data
        assert isinstance(data["confidence_score"], (int, float))
        assert 0 <= data["confidence_score"] <= 1
        
        assert "metadata" in data
        assert "filename" in data["metadata"]

def run_manual_tests():
    """Run tests manually without pytest."""
    print("🧪 Starting manual API tests...")
    
    test_suite = TestNetWorthEstimatorAPI()
    
    try:
        print("✅ Testing root endpoint...")
        test_suite.test_root_endpoint()
        print("   Root endpoint test passed!")
        
        print("✅ Testing health endpoint...")
        test_suite.test_health_endpoint()
        print("   Health endpoint test passed!")
        
        print("✅ Testing predict endpoint with valid image...")
        test_suite.test_predict_endpoint_with_valid_image()
        print("   Predict endpoint test passed!")
        
        print("✅ Testing predict endpoint with invalid file...")
        test_suite.test_predict_endpoint_with_invalid_file()
        print("   Invalid file test passed!")
        
        print("✅ Testing predict endpoint without file...")
        test_suite.test_predict_endpoint_no_file()
        print("   No file test passed!")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Can be run directly for manual testing
    success = run_manual_tests()
    exit(0 if success else 1)
