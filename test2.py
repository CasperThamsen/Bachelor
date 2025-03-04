import cv2
import numpy as np

# Load image
img = cv2.imread('/root/workspace/bachelor/nFoldMark/4.JPG', cv2.IMREAD_COLOR)
height, width = img.shape[:2]

# Orientation obtained from Fourier transform analysis
orientation = -22.28743197
# Compute the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width//2, height//2), -orientation, 1.0)

# Rotate the image to correct the orientation
corrected_img = cv2.warpAffine(img, rotation_matrix, (width, height))

# Display the result
cv2.imshow('Corrected Image', corrected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()