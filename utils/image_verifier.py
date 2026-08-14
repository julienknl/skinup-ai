import cv2

def is_validate(image_path):
    img = cv2.imread(image_path)

    # 1. Image size verification
    height, width = img.shape[:2]
    if height < 512 and width < 512:
        raise ValueError("Image resolution too small. Please re-upload a high resolution image.")

    # 2. Image blur verification
    # Convert the image to grayscale so that changes in pixel intensity can be detected more easily. These changes represent edges in the image and can be used to determine whether the image is sharp or blurry.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Image brightness verification
    if gray_img.mean() < 40:
        raise ValueError("Image too dark. Please re-upload a clear picture.")
    elif gray_img.mean() > 200:
        raise ValueError("Image too bright. Please re-upload a clear picture.")
    
    return True

# This function is only used for testing
def __resize(image_path):
    input_img = cv2.imread(image_path)

    output_img = cv2.resize(
        input_img,
        dsize=None,
        fx=0.2,
        fy=0.2,
        interpolation=cv2.INTER_AREA
    )

    cv2.imwrite("testing/tmp_img/small.png", output_img)

if __name__ == "__main__":
    is_validate("testing/tmp_img/normal.png")