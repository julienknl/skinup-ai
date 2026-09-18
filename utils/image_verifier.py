import cv2
import numpy as np
import time
import json

with open("config/config.json", "r") as config:
    CONFIG = json.load(config)

def resize_image(image):
    target_width = CONFIG.get("target_image_resolution")
    height, width = image.shape[:2]
    
    scale = target_width / width
    new_height = int(height * scale)
    
    return cv2.resize(image, (target_width, new_height))

def is_validate(raw_img):

    start_time = time.perf_counter()

    img = cv2.cvtColor(
        np.array(raw_img),
        cv2.COLOR_RGB2BGR
    )

    # 1. Image size verification
    height, width = img.shape[:2]
    if height < 224 and width < 224:
        raise ValueError("Image resolution too small. Please re-upload a high resolution image.")

    # 2. Image visibility
    # Convert the image to grayscale so that changes in pixel intensity can be detected more easily. These changes represent edges in the image and can be used to determine whether the image is dark or very bright.
    resized_image = resize_image(img)
    gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    if _is_blurry(gray_image=gray_img):
        raise ValueError("Image is too blurry.")

    # 3. Image brightness verification
    if gray_img.mean() < 40:
        raise ValueError("Image too dark. Please re-upload a clear picture.")
    elif gray_img.mean() > 200:
        raise ValueError("Image too bright. Please re-upload a clear picture.")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    return True, total_time

def _is_blurry(gray_image, threshold=100):
    blur_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    print(blur_score)
    return blur_score < threshold

# This function is only used for testing
# def __resize(image_path):
#     input_img = cv2.imread(image_path)

#     output_img = cv2.resize(
#         input_img,
#         dsize=None,
#         fx=0.2,
#         fy=0.2,
#         interpolation=cv2.INTER_AREA
#     )

#     cv2.imwrite("testing/tmp_img/small.png", output_img)

if __name__ == "__main__":
    is_validate("testing/tmp_img/normal.png")