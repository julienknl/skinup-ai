from transformers import pipeline

def predict_skin(image):
    classifier = pipeline("image-classification", model="tuphamdf/skincare-detection")

    result = classifier(image)

    print(result)

if __name__ == "__main__":
    predict_skin("oily_face.png")