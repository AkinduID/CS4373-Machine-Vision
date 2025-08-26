import cv2
import numpy as np

def scale_down_image(scale,image):
	return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

def rotate_image_np(image, angle_degrees):
	angle_radians = np.deg2rad(angle_degrees)
	cos_theta = np.cos(angle_radians)
	sin_theta = np.sin(angle_radians)
	h, w = image.shape
	rotated = np.zeros_like(image)
	for y in range(h):
		for x in range(w):
			xr = cos_theta * x - sin_theta * y
			yr = sin_theta * x + cos_theta * y
			x_new = int(round(xr))
			y_new = int(round(yr))
			if 0 <= x_new < w and 0 <= y_new < h:
				rotated[y, x] = image[y_new, x_new]
	return rotated

image = cv2.imread('img.jpg')   
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_array = np.array(gray_image)
scaled_gray_array = scale_down_image(3, gray_array)
rotated_scaled_gray_array = rotate_image_np(scaled_gray_array, 30)
cv2.imwrite('img_gray.jpg', scaled_gray_array)
cv2.imwrite('img_gray_rotated.jpg', rotated_scaled_gray_array)