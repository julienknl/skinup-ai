from transformers import pipeline

def skin_analyser(image):
    classifier = pipeline("image-classification", model="tuphamdf/skincare-detection")
    result = classifier(image)
    return result

if __name__ == "__main__":
    skin_analyser("oily_face.png")