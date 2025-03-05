import cv2
import numpy as np

# Load image
img = cv2.imread('/root/workspace/bachelor/nFoldMark/4r.JPG', cv2.IMREAD_COLOR)
height, width = img.shape[:2]

# Orientation obtained from Fourier transform analysis (in radians)
orientation_radians = 0.782229  # Example value in radians

# Convert orientation from radians to degrees
orientation_degrees = np.degrees(orientation_radians)

# Compute the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), -orientation_degrees, 1.0)

# Rotate the image to correct the orientation
corrected_img = cv2.warpAffine(img, rotation_matrix, (width, height))

# Display the result
cv2.imshow('Corrected Image', corrected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()