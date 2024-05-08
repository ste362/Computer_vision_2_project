import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt

img = Image.open('TU-Graz/test/image_351.png')


W, H = img.size
cW = W // 2
imgA = img.crop((0, 0, cW, H))
imgB = img.crop((cW, 0, W, H))

np_img = np.array(imgA)


# to grayscale
imgGray = cv.cvtColor(np_img, cv.COLOR_BGR2GRAY)

# guassian blur
imgBlur = cv.GaussianBlur(imgGray, (13, 13), 0)

# canny edge detection
imgCanny = cv.Canny(imgBlur, 50, 50)

#blurred_img = cv.blur(np_img, ksize=(5, 5))
#med_val = np.median(blurred_img)
#lower = int(max(0, 0.7 * med_val))
#upper = int(min(255, 1.3 * med_val))
#edges = cv.Canny(image=blurred_img, threshold1=lower, threshold2=upper)

plt.subplot(121), plt.imshow(imgA, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(imgCanny, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
