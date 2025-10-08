import numpy as np
import cv2 as cv

def gaussian_filter(image):
    kernel = np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ], dtype=np.float64)
    kernel = kernel / np.sum(kernel)  # Normalize
    rows, cols = image.shape
    k_half = 2  # For 5x5 kernel
    output = np.zeros_like(image)

    for i in range(k_half, rows - k_half):
        for j in range(k_half, cols - k_half):
            output[i, j] = np.sum(image[i - k_half: i + k_half + 1, j - k_half: j + k_half + 1] * kernel)

    return output[k_half:rows-k_half, k_half:cols-k_half]


def gradient_estimation(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_kernel_size = 3
    rows, cols = image.shape
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Compute gradient using Sobel operators
    half_size = sobel_kernel_size // 2
    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            gradient_x[i, j] = np.sum(window * sobel_x)
            gradient_y[i, j] = np.sum(window * sobel_y)

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    orientation = np.arctan2(gradient_y, gradient_x)

    return magnitude, orientation

def non_maxima_suppression(magnitude, orientation):
    # Apply non-maximum suppression to the gradient magnitude
    suppressed_magnitude = np.copy(magnitude)
    rows, cols = magnitude.shape
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = orientation[i][j]
            q = [0, 0]
            if (-np.pi/8 <= angle < np.pi/8) or (7*np.pi/8 <= angle):
                q[0] = magnitude[i][j+1]
                q[1] = magnitude[i][j-1]
            elif (np.pi/8 <= angle < 3*np.pi/8):
                q[0] = magnitude[i+1][j+1]
                q[1] = magnitude[i-1][j-1]
            elif (3*np.pi/8 <= angle < 5*np.pi/8):
                q[0] = magnitude[i+1][j]
                q[1] = magnitude[i-1][j]
            else:
                q[0] = magnitude[i-1][j+1]
                q[1] = magnitude[i+1][j-1]
            
            if magnitude[i][j] < max(q[0], q[1]):
                suppressed_magnitude[i][j] = 0
    
    return suppressed_magnitude


def double_threshold(magnitude, low_threshold, high_threshold):
    # Apply edge tracking by hysteresis to detect strong and weak edges
    rows, cols = magnitude.shape
    edge_map = np.zeros((rows, cols), dtype=np.uint8)

    strong_edge_i, strong_edge_j = np.where(magnitude >= high_threshold)
    weak_edge_i, weak_edge_j = np.where((magnitude >= low_threshold) & (magnitude < high_threshold))
    
    # mark strong edges as white (255)
    edge_map[strong_edge_i, strong_edge_j] = 255

    # mark weak edges as white if they are connected to strong edges
    for i, j in zip(weak_edge_i, weak_edge_j):
        if (edge_map[i-1:i+2, j-1:j+2] == 255).any():
            edge_map[i, j] = 255
    
    return edge_map

img = cv.imread('img.jpg', cv.IMREAD_GRAYSCALE)
cv.imwrite('0.grayscale.jpg', img)
blurred_img = gaussian_filter(img)
cv.imwrite('1.blurred.jpg', blurred_img)
magnitude, orientation = gradient_estimation(blurred_img)
cv.imwrite('2.magnitude.jpg', magnitude)
nms_magnitude = non_maxima_suppression(magnitude, orientation)
cv.imwrite('3.nms_magnitude.jpg', nms_magnitude)
edges = double_threshold(nms_magnitude, low_threshold=20, high_threshold=60)
cv.imwrite('4.edges.jpg', edges)