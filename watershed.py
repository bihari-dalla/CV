#10B. Segmentation using watershed transform
# Watershed Image Segmentation

# Install required libraries
!pip install opencv-python matplotlib -q
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
# ============================================================
# 1. Upload Image
# ============================================================
print("Upload an image for segmentation:")
uploaded = files.upload()
if not uploaded:
    raise Exception("No image was uploaded.")
filename = next(iter(uploaded))
# Read image
image = cv2.imread(filename)
if image is None:
    raise Exception("Unable to read the uploaded image.")
# Convert BGR to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#Convert from RGB to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 3. Apply Thresholding
_, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# 4. Remove Noise Using Morphological Opening
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel,iterations=2)
# 5. Find Sure Background
sure_background = cv2.dilate(opening,kernel,iterations=3)
# 6. Distance Transform
distance_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# 7. Find Sure Foreground
# Change 0.5 if necessary:
# Lower value -> more foreground
# Higher value -> less foreground
_, sure_foreground = cv2.threshold(distance_transform,0.5 * distance_transform.max(),255,0)
sure_foreground = np.uint8(sure_foreground)
# 8. Find Unknown Region
unknown = cv2.subtract(sure_background,sure_foreground)
# 9. Create Markers
num_labels, markers = cv2.connectedComponents(sure_foreground)
# Add 1 so that the background is not 0
markers = markers + 1
# Mark unknown region as 0
markers[unknown == 255] = 0
# 10. Apply Watershed Transform
markers = cv2.watershed(image,markers)
# 11. Create Segmentation Result
result = image_rgb.copy()
# Watershed boundaries are represented by -1
result[markers == -1] = [255, 0, 0]
# 12. Create Colored Segmentation
segmented = np.zeros_like(image_rgb)
# Generate random colors for each segment
np.random.seed(42)
colors = np.random.randint(0,255,size=(num_labels + 1, 3),dtype=np.uint8)
for label in range(2, markers.max() + 1):
    segmented[markers == label] = colors[label]
# 13. Display Results
plt.figure(figsize=(16, 10))
plt.subplot(2, 4, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.subplot(2, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")
plt.subplot(2, 4, 3)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")
plt.subplot(2, 4, 4)
plt.imshow(opening, cmap="gray")
plt.title("Morphological Opening")
plt.axis("off")
plt.subplot(2, 4, 5)
plt.imshow(distance_transform, cmap="jet")
plt.title("Distance Transform")
plt.axis("off")
plt.subplot(2, 4, 6)
plt.imshow(sure_foreground, cmap="gray")
plt.title("Sure Foreground")
plt.axis("off")
plt.subplot(2, 4, 7)
plt.imshow(markers, cmap="nipy_spectral")
plt.title("Watershed Markers")
plt.axis("off")
plt.subplot(2, 4, 8)
plt.imshow(result)
plt.title("Watershed Segmentation")
plt.axis("off")
plt.tight_layout()
plt.show()
# ============================================================
# 14. Display Colored Segmentation
# ============================================================
plt.figure(figsize=(8, 6))
plt.imshow(segmented)
plt.title("Colored Watershed Segmentation")
plt.axis("off")
plt.show()
# ============================================================
# 15. Save Segmented Image
# ============================================================
output_filename = "watershed_segmented.png"
# Convert RGB to BGR before saving with OpenCV
segmented_bgr = cv2.cvtColor(segmented,cv2.COLOR_RGB2BGR)
cv2.imwrite(output_filename,segmented_bgr)
print("Segmentation completed successfully!")
print("Output saved as:", output_filename)


# ============================================================
# 16. Download Result
# ============================================================

files.download(output_filename)
