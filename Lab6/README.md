## Assignment 6

Write a python script to implement the Canny edge detection algorithm.  You may use OpenCV or similar library only for loading and saving an image.  All other functions should be in your native code.

### Explanation

The Canny edge detector is a multi-step process for identifying edges in images:

**Step 0: Load and Convert Image**

Load the original image and convert it to grayscale to simplify processing.
<div class="image-container">
  <img src="img.jpg" width="300" alt="Source Image" />
    <img src="1.blurred.jpg" width="300" alt="Source Image" />
</div>

**Step 1: Noise Reduction**

Apply a Gaussian blur to the grayscale image to reduce noise and avoid false edge detection.
<div class="image-container">
  <img src="1.blurred.jpg" width="300" alt="Source Image" />
</div>

**Step 2: Gradient Calculation**

Use the Sobel operator to compute intensity gradients in both horizontal and vertical directions. This highlights regions with rapid intensity change, indicating possible edges.
<div class="image-container">
  <img src="2.magnitude.jpg" width="300" alt="Source Image" />
</div>

**Step 3: Non-Maximum Suppression**

Thin out the edges by keeping only the pixels with the highest gradient magnitude in the direction of the edge. This produces sharp, single-pixel-wide edges.
<div class="image-container">
  <img src="3.nms_magnitude.jpg" width="300" alt="Source Image" />
</div>

**Step 4: Edge Tracking by Hysteresis**

Apply two thresholds (high and low) to classify strong and weak edges. Strong edges are kept, and weak edges are included only if they are connected to strong edges, ensuring continuous edge lines.
<div class="image-container">
  <img src="edges.jpg" width="300" alt="Source Image" />
</div>
