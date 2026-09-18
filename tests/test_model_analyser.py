from fastapi.testclient import TestClient
import pytest
from main import app
from model.skin_analyser.model import SkinAnalyser

client = TestClient(app=app)

def test_model_failure():

    def mock_predict_skin(self, image):
        raise RuntimeError("Model crashed")

    monkey_patch = pytest.MonkeyPatch()
    monkey_patch.setattr(SkinAnalyser, "analyse", mock_predict_skin)

    with open("tests/assets/valid.png", "rb") as image:
            response = client.post("/api/v1/recommend_product", 
                                   files={
                                       "file" : ("valid.png", image, "image/png")
                                       }
            )
    
    assert response.status_code == 503

    