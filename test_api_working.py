#!/usr/bin/env python3
"""
Quick test script to verify the API is working
"""

import requests
import time

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    print("Testing DJ AI Core API...")
    
    try:
        # Test root endpoint
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test health endpoint
        print("\n2. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test supported formats endpoint
        print("\n3. Testing supported formats endpoint...")
        response = requests.get(f"{base_url}/supported-formats")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test docs endpoint availability
        print("\n4. Testing API docs availability...")
        response = requests.get(f"{base_url}/docs")
        print(f"   Status: {response.status_code}")
        print(f"   Docs available: {'Yes' if response.status_code == 200 else 'No'}")
        
        print("\n✅ All basic API tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on http://127.0.0.1:8000")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

if __name__ == "__main__":
    # Give server a moment to start if just launched
    time.sleep(2)
    test_api()
