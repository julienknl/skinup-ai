from fastapi.testclient import TestClient
import pytest
from main import app
from utils.image_verifier import is_validate
from PIL import Image
from io import BytesIO

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

def test_valid_image_size():
    with open("tests/assets/normal_size.png", "rb") as file:
        image_bytes = Image.open(BytesIO(file.read())).convert("RGB")

        result, compute_time = is_validate(image_bytes)

    assert result is True
    assert compute_time >= 0

def test_invalid_image_size():
    with open("tests/assets/small.png", "rb") as file:
        image_bytes = Image.open(BytesIO(file.read())).convert("RGB")

    with pytest.raises(ValueError):
        is_validate(image_bytes)

def test_blurry_image():
    with open("tests/assets/small.png", "rb") as file:
            image_bytes = Image.open(BytesIO(file.read())).convert("RGB")
    
    with pytest.raises(ValueError):
        is_validate(image_bytes)

def test_dark_image():
    with open("tests/assets/dark.png", "rb") as file:
        image_bytes = Image.open(BytesIO(file.read())).convert("RGB")

    with pytest.raises(ValueError):
        is_validate(image_bytes)

def test_bright_image():
    with open("tests/assets/bright2.png", "rb") as file:
        image_bytes = Image.open(BytesIO(file.read())).convert("RGB")

    with pytest.raises(ValueError):
        is_validate(image_bytes)
            