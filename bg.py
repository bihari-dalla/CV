#6. Program to estimate and subtract the background of an image.
#6 - estimate background

import cv2
import matplotlib.pyplot as plt
from google.colab import files

#upload image
uploaded = files.upload()
image_name = list(uploaded.keys())[0]
#Read Image
img = cv2.imread(image_name)
#Convert into RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#estimate Background using Gaussaian Blur
background = cv2.GaussianBlur(rgb, (21,21), 0)
#background subtraction
foreground = cv2.absdiff(background, rgb)

#Display Images
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(background)
plt.title('Estimated Background')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Foreground')
plt.imshow(foreground)
plt.axis("off")
plt.tight_layout()
plt.show()
