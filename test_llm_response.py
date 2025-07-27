import requests

# Change this if your Flask app is running elsewhere
BASE_URL = "http://127.0.0.1:5000"

def test_llm_response():
    url = f"{BASE_URL}/llm_test"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("/llm_test response:")
        print(data)
        if data.get("success"):
            print("LLM responded:", data.get("response"))
        else:
            print("LLM error:", data.get("error"))
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_llm_response()
