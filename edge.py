#8. Determination of edge detection using operators.
# Determination of Edge Detection Using Operators

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
uploaded = files.upload()

# Get the uploaded filename
filename = list(uploaded.keys())[0]

# Read image
image = cv2.imread(filename)

if image is None:
    print("Error: Image not found. Check the file path.")
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel Operator
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(sobel)

    # Prewitt Operator
    prewitt_x_kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    prewitt_y_kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ], dtype=np.float32)

    prewitt_x = cv2.filter2D(gray, cv2.CV_32F, prewitt_x_kernel)
    prewitt_y = cv2.filter2D(gray, cv2.CV_32F, prewitt_y_kernel)
    prewitt = cv2.magnitude(prewitt_x, prewitt_y)
    prewitt = cv2.convertScaleAbs(prewitt)

    # Laplacian Operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Display results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(sobel, cmap="gray")
    plt.title("Sobel Operator")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(prewitt, cmap="gray")
    plt.title("Prewitt Operator")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(laplacian, cmap="gray")
    plt.title("Laplacian Operator")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
