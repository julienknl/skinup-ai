from classifier.model import skin_analyser
from utils.image_verifier import is_validate
import json

def main():
    with open("config/config.json", "r") as config:
        CONFIG = json.load(config)
    image_path = "testing/tmp_img/acne3.png"

    print("Validating clarity of the image...")
    if is_validate(image_path=image_path):
        print("Image clear.. Ready to analyse")
        results = skin_analyser(image=image_path)

        skin_condition_high = max(
            results,
            key=lambda result: result["score"]
        )

        skin_condition = skin_condition_high["label"] if skin_condition_high["score"] >= CONFIG.get("skin_condition_identifier_threshold") else "normal"
        print(f"Analysis result: {skin_condition}")

if __name__ == "__main__":
    main()
