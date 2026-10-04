"""
1B. To Flip the image around the vertical and horizontal line.
"""

from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from google.colab import files

uploaded = files.upload()
img = Image.open(next(iter(uploaded)))

horizontal_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
vertical_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
both_flips = horizontal_flip.transpose(Image.FLIP_TOP_BOTTOM)

plt.figure(figsize=(20, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

# Horizontal
plt.subplot(1, 4, 2)
plt.imshow(horizontal_flip)
plt.title("Horizontal Flip")
plt.axis("off")

# Vertical
plt.subplot(1, 4, 3)
plt.imshow(vertical_flip)
plt.title("Vertical Flip")
plt.axis("off")

# Both
plt.subplot(1, 4, 4)
plt.imshow(both_flips)
plt.title("Horizontal + Vertical")
plt.axis("off")

plt.tight_layout()
plt.show()
