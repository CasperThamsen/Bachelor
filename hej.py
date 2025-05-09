import numpy as np



calibration_data = np.load('phone_calibration.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']

print(mtx)
print(dist)
