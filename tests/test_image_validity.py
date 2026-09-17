from fastapi.testclient import TestClient
from main import app

client = TestClient(app=app)

def test_valid_image():
    with open("tests/assets/valid.png", "rb") as image:
        response = client.post("/api/v1/recommend_product", 
                               files={
                                   "file" : ("valid.png", image, "image/png")
                                   }
        )

    assert response.status_code == 200

def test_invalid_image():
    with open("tests/assets/invalid.txt", "rb") as file:
        response = client.post("/api/v1/recommend_product", 
                               files={
                                   "file" : ("invalid.txt", file, "text/plain")
                                   }
        )

    assert response.status_code == 400