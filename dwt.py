#10A. DWT of images

!pip install PyWavelets -q
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from google.colab import files
uploaded = files.upload()
filename = next(iter(uploaded))
img = cv2.imread(filename)
if img is None:
  raise ValueError("Unable to read the uploaded iamge.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

#Part A: discrete wavelet transform (DWT) of image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8, 6))
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

#4. apply 2D DWT
coeffs2= pywt.dwt2(gray, 'haar')
#extract four sub-brands
LL, (LH, HL, HH) = coeffs2
print("DWT Components:")
print("LL shape:", LL.shape)
print("LH shape:", LH.shape)
print("HL shape:", HL.shape)
print("HH shape:", HH.shape)

#5. normalize DWT components
def normalize_image(image):
  return cv2.normalize(np.abs(image),None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
LL_vis = normalize_image(LL)
LH_vis = normalize_image(LH)
HL_vis = normalize_image(HL)
HH_vis = normalize_image(HH)


#6. display DWT Components
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(LL_vis, cmap='gray')
plt.title("LL - Horizontal Details")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(LH_vis, cmap='gray')
plt.title("LH Component")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(HL_vis, cmap='gray')
plt.title("HL - Vertical Details")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(HH_vis, cmap='gray')
plt.title("HH - Diagonal Details")
plt.axis('off')

plt.tight_layout()
plt.show()
