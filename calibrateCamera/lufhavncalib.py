import numpy as np
mtx = np.array([[1644.04096932, 0, 552.02394704],
                  [0, 1648.24363287, 955.90516159],
                  [0, 0, 1]])
dist = np.array([[2.56298042e-01, -1.31667269e+00, 2.62578829e-03, 9.57626151e-04, 2.37193971e+00]])
np.savez('phone_calibration.npz', mtx=mtx, dist=dist)