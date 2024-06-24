import requests

BASE_URL = "http://localhost:8000"


def test_predict():
    url = f"{BASE_URL}/predict"
    sample_input = {
        "texts": "Sample text.",
    }
    response = requests.post(url, json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()

    print(f"Response: {response.text}")


if __name__ == "__main__":
    test_predict()
