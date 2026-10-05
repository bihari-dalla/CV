#4. Edge detection with gradient and convolution of an Image
# Use: Edge detection using Gradient (Sobel) and Convolution

!pip install opencv-python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
uploaded = files.upload()
# Read image
image = cv2.imread(next(iter(uploaded)))
# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Sobel X
sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

# Sobel Y
sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)
# Calculate gradient magnitude
gradient = cv2.magnitude(sobel_x, sobel_y)

# Define convolution kernel
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Apply convolution
convolution = cv2.filter2D(
    gray,
    -1,
    kernel
)

# Display results
plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(gradient, cmap="gray")
plt.title("Gradient (Sobel)")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(convolution, cmap="gray")
plt.title("Convolution Edge Detection")
plt.axis("off")
plt.tight_layout()
plt.show()
