#brightness
#1A. Program to change the Brightness of Image.

!pip install opencv-python-headless

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
from IPython.display import display
from PIL import Image
import io

uploaded = files.upload()

for file_name in uploaded.keys():
    image = cv2.imdecode(
        np.frombuffer(uploaded[file_name], np.uint8),
        cv2.IMREAD_COLOR
    )

def change_brightness(image, brightness_value):
    brightness_matrix = np.ones(image.shape, dtype='uint8') * abs(brightness_value)

    if brightness_value > 0:
        bright_image = cv2.add(image, brightness_matrix)
    else:
        bright_image = cv2.subtract(image, brightness_matrix)

    return bright_image

brightness_value = 60
bright_image = change_brightness(image, brightness_value)

print("Original Image:")
cv2_imshow(image)

print(f"Brightness Adjusted Image (Brightness value: {brightness_value})")
cv2_imshow(bright_image)


-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------

#flipping
#1B. To Flip the image around the vertical and horizontal line.

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

--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------

#color components
#1C.Display the color components of the image.

!pip install opencv-python-headless matplotlib

import cv2
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np

uploaded = files.upload()

for file_name in uploaded.keys():

    image = cv2.imread(file_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    R, G, B = cv2.split(image_rgb)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(R, cmap="Reds")
    plt.title("Red Channel")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(G, cmap="Greens")
    plt.title("Green Channel")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(B, cmap="Blues")
    plt.title("Blue Channel")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------

#grayscale
#1D. Display of gray scale images.

from google.colab import files
from google.colab.patches import cv2_imshow
import cv2
import numpy as np

uploaded = files.upload()

image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)

if image is None:
    print("Error loading image")
else:
    cv2_imshow(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

-----------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------

#negative
#1.E To find the negative of an image.

import cv2
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np

uploaded = files.upload()

image_path = next(iter(uploaded))
img = cv2.imread(image_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

negative = 255 - img

negative_rgb = cv2.cvtColor(
    negative,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(negative_rgb)
plt.title("Negative Image")
plt.axis("off")

plt.show()
