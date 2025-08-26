import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1
    return histogram

def cumulative_sum(a):  
    a = iter(a)  
    b = [next(a)]  
    for i in a:    
        b.append(b[-1] + i)
    return np.array(b)

image = cv.imread("im_210113L.jpg")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imwrite("gray_image.jpg", gray_image)
gray_image.flatten()
# Histogram and CDF for original grayscale image
histogram = get_histogram(gray_image, 256)
cdf = cumulative_sum(histogram)
# Normalize CDF
cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
cdf_normalized = cdf_normalized.astype('uint8')
# Map the original gray levels to equalized levels
equalized_image = cdf_normalized[gray_image]
cv.imwrite("equalized_image.jpg", equalized_image)
equalized_image.flatten()
# Histogram and CDF for equalized image
hist_eq = get_histogram(equalized_image, 256)
cdf_eq = cumulative_sum(hist_eq)
cdf_eq_normalized = (cdf_eq - cdf_eq.min()) * 255 / (cdf_eq.max() - cdf_eq.min())

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 8))

# Row 1: Original grayscale image, its histogram, its CDF
axes[0, 0].imshow(gray_image, cmap='gray')
axes[0, 0].set_title("Gray Image")
axes[0, 0].axis("off")

axes[0, 1].hist(gray_image.flatten(), 256, [0,256], color='gray')
axes[0, 1].set_title("Gray Image Histogram")
axes[0, 1].set_xlim([0,256])

axes[0, 2].plot(cdf_normalized, color='blue')
axes[0, 2].set_title("Gray Image CDF")
axes[0, 2].set_xlim([0,256])

# Row 2: Equalized image, its histogram, its CDF
axes[1, 0].imshow(equalized_image, cmap='gray')
axes[1, 0].set_title("Equalized Image")
axes[1, 0].axis("off")

axes[1, 1].hist(equalized_image.flatten(), 256, [0,256], color='gray')
axes[1, 1].set_title("Equalized Histogram")
axes[1, 1].set_xlim([0,256])

axes[1, 2].plot(cdf_eq_normalized, color='blue')
axes[1, 2].set_title("Equalized Image CDF")
axes[1, 2].set_xlim([0,256])

plt.tight_layout()
plt.savefig("histograms_cdfs.png")