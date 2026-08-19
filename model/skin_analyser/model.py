import os
from dotenv import load_dotenv
from transformers import pipeline
load_dotenv()

class SkinAnalyser:
    def __init__(self, type="image-classification", model=os.getenv("SKIN_MODEL")):
        self.classifier = pipeline(type, model=model)
        pass

    def analyse(self, image):
        result = self.classifier(image)
        return result