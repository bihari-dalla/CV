# threshold gray
# 5A. Program to find threshold of grayscale image.
import cv2
import matplotlib.pyplot as plt
from google.colab import files
#upload image
uploaded=files.upload()
image_name =list(uploaded.keys())[0]
#read image
img=cv2.imread(image_name)
#convert into RGB
rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#convert image to grayscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#APPLY THRESHOLD
threshold_value=127
ret, binary=cv2.threshold(gray,threshold_value,255,cv2.THRESH_BINARY)

#display images
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(gray,cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(binary,cmap='gray')
plt.title('THRESHOLD IMAGE')
plt.axis('off')
plt.show()

--------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------

#5B. Program to find threshold of RGB image.
# threshold rgb
import cv2
import matplotlib.pyplot as plt
from google.colab import files

#upload image
uploaded = files.upload()
image_name = list(uploaded.keys())[0]
#read image
img = cv2.imread(image_name)
#convert into RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#Split channels
R,G,B = cv2.split(rgb)
threshold = 127
#Threshold  each channels
_, R1 = cv2.threshold(R, threshold, 255, cv2.THRESH_BINARY)
_, G1 = cv2.threshold(G, threshold, 255, cv2.THRESH_BINARY)
_, B1 = cv2.threshold(B, threshold, 255, cv2.THRESH_BINARY)

#Merge channels
threshold_rgb = cv2.merge((R1, G1, B1))

#Display Result
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(rgb)
plt.subplot(1,2,2)
plt.title('Threshold Image')
plt.imshow(threshold_rgb)
plt.show()
