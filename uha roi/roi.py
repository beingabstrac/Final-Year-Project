import cv2
import numpy as np
from PIL import Image

# Read image
im = cv2.imread('image.jpg')
    
# Select ROI
r = cv2.selectROI(im)
    
# Crop image
imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.imwrite('image_crop.jpg', imCrop)

# Display cropped image
cv2.imshow("Image Crop", imCrop)
cv2.waitKey(0)

# Drag rectangle from top left to bottom right
fromCenter = False
r = cv2.selectROI(im, fromCenter)

# Mean
im = Image.open('image_crop.jpg')
m = np.array(im).mean(axis=(0,1))
np.savetxt('mean.txt', m, delimiter=",")

