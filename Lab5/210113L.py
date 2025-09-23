import cv2 as cv
import numpy as np
import os

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
current_dir = os.path.dirname(os.path.abspath(__file__))
image_files = [f for f in os.listdir(current_dir) if f.lower().endswith(image_extensions)]

images = []
for img_file in image_files:
	img_path = os.path.join(current_dir, img_file)
	img = cv.imread(img_path)
	if img is not None:
		images.append((img_file, img))
		print(f"Opened image: {img_file}")
	else:
		print(f"Failed to open image: {img_file}")
          
def wrappingImage(img, kernelSize: int):
    w = kernelSize // 2
    fetchFirstRows = img[0:w, :]
    fetchLastRows = img[-w:, :]
    imgWrapped = img.copy()
    imgWrapped = np.insert(imgWrapped, 0, fetchLastRows, axis=0)
    imgWrapped = np.append(imgWrapped, fetchFirstRows, axis=0)
    fetchFirstCols = imgWrapped[:, 0:w]
    fetchLastCols = imgWrapped[:, -w:]
    imgWrapped = np.concatenate([fetchLastCols, imgWrapped], axis=1)
    imgWrapped = np.append(imgWrapped, fetchFirstCols, axis=1)
    return imgWrapped

def mean_filter(image, filter_size=3):
    filteredImage = np.zeros(image.shape, dtype=np.int32)
    wrappedImage = wrappingImage(image, filter_size)
    image_h, image_w = image.shape
    w = filter_size // 2
    for i in range(w, image_h - w):
        for j in range(w, image_w - w):
            total = 0
            for m in range(filter_size):
                for n in range(filter_size):
                    total += wrappedImage[i-w+m][j-w+n]
            filteredImage[i-w][j-w] = total // (filter_size * filter_size)
    return filteredImage

def median_filter(image, filter_size=3):
    filteredImage = np.zeros(image.shape, dtype=np.int32)
    image_h, image_w = image.shape
    wrappedImage = wrappingImage(image, filter_size)
    w = filter_size // 2
    for i in range(w, image_h - w):
        for j in range(w, image_w - w):
            overlapImg = wrappedImage[i-w : i+w+1, j-w : j+w+1]
            filteredImage[i][j] = np.median(overlapImg)
    return filteredImage

def midpoint_filter(image, filter_size=3):
    filteredImage = np.zeros(image.shape, dtype=np.int32)
    image_h, image_w = image.shape
    wrappedImage = wrappingImage(image, filter_size)
    w = filter_size // 2
    for i in range(w, image_h - w):
        for j in range(w, image_w - w):
            overlapImg = wrappedImage[i-w : i+w+1, j-w : j+w+1]
            overlapImg = overlapImg.astype('int32')
            maxVal = np.max(overlapImg)
            minVal = np.min(overlapImg)
            filteredImage[i][j] = (maxVal + minVal) // 2
    return filteredImage

output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

for i in range(len(images)):
    img_name, img = images[i]
    print(f"Processing image: {img_name}")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    print("Applying Mean Filter")
    mean_filtered = mean_filter(img, filter_size=3)
    cv.imwrite(os.path.join(output_dir, f'mean_filtered_{img_name}'), mean_filtered)

    print("Applying Median Filter")
    median_filtered = median_filter(img, filter_size=3)
    cv.imwrite(os.path.join(output_dir, f'median_filtered_{img_name}'), median_filtered)

    print("Applying Midpoint Filter")
    midpoint_filtered = midpoint_filter(img, filter_size=3)
    cv.imwrite(os.path.join(output_dir, f'midpoint_filtered_{img_name}'), midpoint_filtered)