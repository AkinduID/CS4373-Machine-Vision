import cv2
import numpy as np

def resize_linear(img, new_width, new_height):
    orig_height, orig_width = img.shape
    resized = np.zeros((new_height, new_width), dtype=np.uint8)

    x_scale = orig_width / new_width
    y_scale = orig_height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x = j * x_scale
            y = i * y_scale

            x0 = int(np.floor(x))
            x1 = min(x0 + 1, orig_width - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, orig_height - 1)

            dx = x - x0
            dy = y - y0

            value = (img[y0, x0] * (1 - dx) * (1 - dy) +
                     img[y0, x1] * dx * (1 - dy) +
                     img[y1, x0] * (1 - dx) * dy +
                     img[y1, x1] * dx * dy)

            resized[i, j] = int(value)

    return resized

# Load and convert to grayscale
image = cv2.imread("im_210113L.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_image.jpg", gray_image)

# Resize to 0.7x
orig_h, orig_w = gray_image.shape
resized_image = resize_linear(gray_image, int(orig_w * 0.7), int(orig_h * 0.7))
cv2.imwrite("resized_image.jpg", resized_image)

# Resize back to original
restored_image = resize_linear(resized_image, orig_w, orig_h)
cv2.imwrite("restored_image.jpg", restored_image)

# Compute MSE
squared_diff = (gray_image.astype(np.float32) - restored_image.astype(np.float32)) ** 2
mse = np.mean(squared_diff)
print("Mean Squared Error (MSE):", mse)
