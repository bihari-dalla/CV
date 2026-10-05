#7. Program to convert color image to gray and hsv.
#gray hsv 7th

import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upoad image
uploaded = files.upload()
# Get the uploaded filename
filename = list(uploaded.keys())[0]
# Read image
image = cv2.imread(filename)

#check whether the image was read successfully
if image is None:
  print("Error:  Image not found. Check the file path.")
else:
  #convert BGR image to Grayscale
  gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

  #convert BGR image to HSV
  hsv_image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
  #display original, grayscale and hsv images
  plt.figure(figsize=(12,4))
  plt.subplot(1,3,1)
  plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
  plt.title("Original Color Image")
  plt.axis("off")

  plt.subplot(1,3,2)
  plt.imshow(gray_image,cmap="gray")
  plt.title("Grayscale Image")
  plt.axis("off")

  plt.subplot(1,3,3)
  plt.imshow(cv2.cvtColor(hsv_image,cv2.COLOR_HSV2RGB))
  plt.title("HSV Image")
  plt.axis("off")

  plt.tight_layout()
  plt.show()l
