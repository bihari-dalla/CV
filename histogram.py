# 2. Using histogram for image quality analysis
#basic histogram
# 2A. Calculate the Histogram of a given image.

# STEP 1: IMPORT LIBRARIES
from google.colab import files
import cv2
import matplotlib.pyplot as plt
# STEP 2: UPLOAD IMAGE
uploaded = files.upload()
# STEP 3: GET FILENAME
filename = list(uploaded.keys())[0]
# STEP 4: READ ORIGINAL IMAGE
img = cv2.imread(filename)
# STEP 5: CONVERT IMAGE TO GRAYSCALE
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# STEP 6: DISPLAY GRAYSCALE IMAGE
plt.figure(figsize=(5,5))
plt.imshow(gray, cmap='gray')
plt.title('GRAYSCALE IMAGE')
plt.axis('off')
plt.show()
# STEP 7: CALCULATE HISTOGRAM
hist = cv2.calcHist([gray], [0], None, [256], [0,256])
# STEP 8: PLOT HISTOGRAM
plt.figure(figsize=(8,5))
plt.plot(hist)
plt.title("HISTOGRAM OF THE IMAGE")
plt.xlabel("PIXEL INTENSITY")
plt.ylabel("NUMBER OF PIXELS")
plt.grid()
plt.show()

------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------

# Equalization
# 2B. Histogram Equalization.
#import required libraries
import cv2
import matplotlib.pyplot as plt
from google.colab import files
#uploaded images
uploaded=files.upload()
#get uploaded file name
image_path=list(uploaded.keys())[0]
#read image in grayscale
image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
#APPLY HISTOGRAM EQUALIZATION
equalized=cv2.equalizeHist(image)
#DISPLAY original and equalized images
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(image,cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(equalized,cmap='gray')
plt.title('Equalized Image')
plt.axis('off')
plt.show()
#plot histograms
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(image.ravel(),bins=256,range=[0,256],color='blue')
plt.title("Histogram of Original Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(equalized.ravel(),bins=256,range=[0,256],color='green')
plt.title("Histogram of Equalized Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------

#2.C histogram  stretchiing

#import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

#upload image
uploaded=files.upload()
#get uploaded file name
image_path=list(uploaded.keys())[0]
#read image no grayscale
image =cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
#find minimum and maximum pixel values
min_val=np.min(image)
max_val=np.max(image)
#perform histogram steching
stretched=((image-min_val)/(max_val-min_val)*255).astype(np.uint8)
#display images
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(image,cmap='gray')
plt.title('Orignal image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(stretched,cmap='gray')
plt.title('Histogram stretched image')
plt.axis('off')
plt.show()
#plot histogram
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(image.ravel(), bins=256, range=[0,256], color='blue')
plt.title(' HISTOGRAM OF ORGINAL IMAGE')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')


plt.subplot(1,2,2)
plt.hist(stretched.ravel(), bins=256, range=[0,256], color='green')
plt.title(' HISTOGRAM OF EQUALIZED IMAGE')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')
plt.show()
