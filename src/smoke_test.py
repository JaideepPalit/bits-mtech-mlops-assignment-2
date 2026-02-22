import requests
import sys

BASE_URL = "http://34.102.150.120"
IMAGE_PATH = "data/preprocessed/preprocessed_cats_dogs_images/test/Dog/51.jpg"

def run_tests():
    # 1. Health Check
    print(f"Checking {BASE_URL}/health endpoint...")
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=120)
        if health_response.status_code != 200:
            print(f"Health check failed with status {health_response.status_code}")
            sys.exit(1)
        print("Health check passed")
    except Exception as e:
        print(f"Health check failed to connect: {e}")
        sys.exit(1)

    # 2. Predict API Test
    print(f"\nTesting {BASE_URL}/predict endpoint...")
    try:
        with open(IMAGE_PATH, 'rb') as f:
            files = {'file': (IMAGE_PATH, f, 'image/jpeg')}
            predict_response = requests.post(f"{BASE_URL}/predict", files=files, timeout=120)
        
        print(f"Response body: {predict_response.text}")
        print(f"Status code: {predict_response.status_code}")

        if predict_response.status_code != 200:
            print("Predict endpoint failed")
            sys.exit(1)
            
        print("Smoke test passed")
        
    except FileNotFoundError:
        print(f"Error: Test image not found at {IMAGE_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Predict test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()