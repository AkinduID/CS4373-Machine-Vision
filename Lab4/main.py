import cv2 as cv
import numpy as np

def horizontal_convolution(large_img, small_img):
    N, M = large_img.shape
    n, m = small_img.shape
    assert n == N, "Both images must have the same number of rows"

    output_width = M - m + 1
    output = np.zeros((1, output_width))

    for j in range(output_width):
        region = large_img[:, j:j+m]
        output[0, j] = np.sum(region * small_img)
    return output

large_img = cv.imread('img1.jpg')
small_img = cv.imread('img2.jpg')
print(large_img.shape, small_img.shape)

# Convert images to grayscale
large_gray = cv.cvtColor(large_img, cv.COLOR_BGR2GRAY)
small_gray = cv.cvtColor(small_img, cv.COLOR_BGR2GRAY)

large_gray_array = np.array(large_gray)
small_gray_array = np.array(small_gray)

result = horizontal_convolution(large_gray, small_gray)
cv.imwrite('output.jpg', result)
