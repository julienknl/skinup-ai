from classifier.model import skin_analyser
from utils.image_verifier import is_validate

def main():
    image_path = "testing/tmp_img/normal.png"

    print("Validating clarity of the image...")
    if is_validate(image_path=image_path):
        print("Image clear.. Ready to analyse")
        result = skin_analyser(image=image_path)
        print(f"Analysis result: {result}")

if __name__ == "__main__":
    main()
