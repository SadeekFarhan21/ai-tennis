#!/usr/bin/env python3
"""
Comprehensive API testing script for AI Tennis backend
"""
import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8001"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_readiness():
    """Test readiness endpoint"""
    print("🔍 Testing Readiness Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/ready", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_docs():
    """Test documentation endpoints"""
    print("🔍 Testing Documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        print(f"   Docs Status: {response.status_code}")
        
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        print(f"   OpenAPI Schema Status: {response.status_code}")
        schema = response.json()
        print(f"   API Title: {schema.get('info', {}).get('title', 'Unknown')}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_main_interface():
    """Test main web interface"""
    print("🔍 Testing Web Interface...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Content Length: {len(response.text)} characters")
        print(f"   Contains 'AI Tennis': {'AI Tennis' in response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_upload_validation():
    """Test upload endpoint validation"""
    print("🔍 Testing Upload Validation...")
    try:
        # Test with invalid file type
        files = {'file': ('test.txt', 'fake content', 'text/plain')}
        response = requests.post(f"{BASE_URL}/api/upload", files=files, timeout=10)
        print(f"   Invalid file status: {response.status_code}")
        print(f"   Error message: {response.json()}")
        
        # Test without file
        response = requests.post(f"{BASE_URL}/api/upload", timeout=10)
        print(f"   No file status: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_video_endpoints():
    """Test video-related endpoints with fake IDs"""
    print("🔍 Testing Video Endpoints...")
    try:
        fake_id = "fake-video-id"
        
        # Test upload status
        response = requests.get(f"{BASE_URL}/api/upload/status/{fake_id}", timeout=5)
        print(f"   Upload status (fake ID): {response.status_code} - {response.json()}")
        
        # Test result endpoint
        response = requests.get(f"{BASE_URL}/api/result/{fake_id}", timeout=5)
        print(f"   Result (fake ID): {response.status_code} - {response.json()}")
        
        # Test download endpoint
        response = requests.get(f"{BASE_URL}/api/result/{fake_id}/download", timeout=5)
        print(f"   Download (fake ID): {response.status_code} - {response.json()}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_performance():
    """Test API performance"""
    print("🔍 Testing Performance...")
    import time
    try:
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000
        print(f"   Health endpoint response time: {response_time:.2f}ms")
        
        if response_time < 100:
            print("   ✅ Fast response time")
        elif response_time < 500:
            print("   ⚡ Acceptable response time")
        else:
            print("   ⚠️  Slow response time")
            
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Tennis API Test Suite")
    print("=" * 50)
    
    tests = [
        test_health,
        test_readiness,
        test_docs,
        test_main_interface,
        test_upload_validation,
        test_video_endpoints,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("   ✅ PASSED\n")
            else:
                print("   ❌ FAILED\n")
        except Exception as e:
            print(f"   💥 CRASHED: {e}\n")
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working great!")
    elif passed >= total * 0.8:
        print("👍 Most tests passed! Good job!")
    else:
        print("⚠️  Some issues detected. Check the failures above.")
    
    print("\n🔗 Quick Links:")
    print(f"   • API Docs: {BASE_URL}/docs")
    print(f"   • Web Interface: {BASE_URL}/")
    print(f"   • Health Check: {BASE_URL}/health")

if __name__ == "__main__":
    main()
