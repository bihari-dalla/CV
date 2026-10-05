#Program for Image Filtering
# 3A. Low pass filter
# Use: Image filtering — Average, Gaussian, Median, Sobel and Laplacian

!pip install opencv-python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
uploaded = files.upload()
# Read image
image = cv2.imread(next(iter(uploaded)))
# Convert BGR to RGB for displaying with matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Average Filter
average = cv2.blur(image, (5,5))
# Gaussian / Weighted Average Filter
gaussian = cv2.GaussianBlur(image, (5,5), 0)
# Median Filter
median = cv2.medianBlur(image, 5)
# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Sobel X and Y
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
# Combine Sobel X and Y
sobel = cv2.magnitude(sobelx, sobely)
# Laplacian Filter
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
# Display all results
plt.figure(figsize=(18,10))

plt.subplot(2,3,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(average)
plt.title("Average Filter")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(gaussian)
plt.title("Weighted Average (Gaussian)")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(median)
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel Operator")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Operator")
plt.axis("off")
plt.tight_layout()
plt.show()

-------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------

# PRACTICAL 3B
# Use: Non-linear filtering using Median Filter

!pip install opencv-python

import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
uploaded = files.upload()
# Read image
image = cv2.imread(next(iter(uploaded)))
# Convert BGR to RGB for display
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Apply Median Filter
median = cv2.medianBlur(image, 7)
# Display results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Median Filtered Image")
plt.axis("off")
plt.tight_layout()
plt.show()
