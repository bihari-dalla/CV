#9 A. Display of colour images.
# OR
#9B. Conversion between colour spaces.

#step1: import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image

#step2:upload an image
print("Please upload colour image")
uploaded=files.upload()

#get the uploaded file name
image_name=list(uploaded.keys())[0]
print("\nImage uploaded successfully: ", image_name)

#step3: read the image
image_bgr=cv2.imread(image_name)

#check whether image was loaded successfully
if image_bgr is None:
  print("Error: Could not read the image")
else:
  print("Image loaded successfully")

#step 4:convert BGR TO RGB
image_rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)

#step 5:Display the original colour image
plt.figure(figsize=(8,6))
plt.imshow(image_rgb)
plt.title("Original Colour Image(RGB)")
plt.axis("off")
plt.show()

#STEP 6: DISPLAY IMAGE INFORMATION
print("\nImage Information")
print("--------------------------")
print("Image Name:  ",image_name)
print("Image Width: ",image_rgb.shape[1],"pixels")
print("Image Height: ",image_rgb.shape[0],"pixels")
print("Number of Channels: ",image_rgb.shape[2])
print("Image Data Type: ",image_rgb.dtype)

#step 7: RGB TO GRAYSCALE
gray_image=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
#STEP 8: BGR TO HSV
hsv_image=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
#step 9:BGR TO LAB
lab_image=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2LAB)
#step10: BGR TO YCrCb
ycrcb_image=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2YCrCb)

#step11: convert HSV BACK TO RGB
hsv_to_rgb=cv2.cvtColor(hsv_image,cv2.COLOR_HSV2RGB)
#step12: convert LAB BACK TO RGB
lab_to_rgb=cv2.cvtColor(lab_image,cv2.COLOR_LAB2RGB)
#step13: convert YCrCb Back to RGB
ycrcb_to_rgb=cv2.cvtColor(ycrcb_image,cv2.COLOR_YCrCb2RGB)

#display all colour spaces
plt.figure(figsize=(16,12))

#original RGB
plt.subplot(2,3,1)
plt.imshow(image_rgb)
plt.title("RGB")
plt.axis("off")

#grayscale
plt.subplot(2,3,2)
plt.imshow(gray_image,cmap="gray")
plt.title("Grayscale")
plt.axis("off")

#HSV
plt.subplot(2,3,3)
plt.imshow(hsv_to_rgb)
plt.title("HSV")
plt.axis("off")


#convert lab to RGB only for proper visualization
plt.subplot(2,3,4)
plt.imshow(lab_to_rgb)
plt.title("LAB")
plt.axis("off")

#YCrCb
plt.subplot(2,3,5)
plt.imshow(ycrcb_to_rgb)
plt.title("YCrCb")
plt.axis("off")
plt.show()

#original BGR displayed after conversion
plt.subplot(2,3,6)
plt.imshow(cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB))
plt.title("ORIGINAL IMAGE")
plt.axis("off")

plt.tight_layout()
plt.show()

#step 15: display RBG CHANNELS
R, G, B = cv2.split(image_rgb)
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.imshow(R, cmap='Reds')
plt.title("Red Channel")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(G, cmap='Greens')
plt.title("Green Channel")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(B, cmap='Blues')
plt.title("Blue Channel")
plt.axis('off')

plt.tight_layout()
plt.show()

#step 16: display HSV CHANNELS
H, S, V=cv2.split(hsv_image)
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.imshow(H, cmap='gray')
plt.title("Hue Channel")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(S, cmap='gray')
plt.title("Saturation Channel")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(V, cmap='gray')
plt.title("Value Channel")
plt.axis('off')

plt.tight_layout()
plt.show()

#step 17: SAVE CONVERTED IMAGES
cv2.imwrite("original_image.jpg", image_bgr)
cv2.imwrite("grayscale_image.jpg",gray_image)
cv2.imwrite("hsv_image.jpg",hsv_image)
cv2.imwrite("lab_image.jpg",lab_image)
cv2.imwrite("ycrcb_image.jpg",ycrcb_image)
print("\nConverted Images saved successfully!")
#step 18: download images
print("\nYOU CAN DOWNLOAD CONVERTED IMAGES FROM Colab.")
files.download("grayscale_image.jpg")
