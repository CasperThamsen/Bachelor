import matplotlib.pyplot as plt
import numpy as np
file_name = r"C:\Users\Thamsn\OneDrive - Syddansk Universitet\Skrivebord\optitrack\5markerrotation2.csv"
val = np.loadtxt(file_name, delimiter=',')
fig, ax = plt.subplots()
ax.plot(val[:,0], val[:,4], '.r')
ax.plot(val[:,0], val[:,5], '.b')
ax.plot(val[:,0], val[:,6], '.g')
ax.plot(val[:,0], val[:,1], '.y')
ax.plot(val[:,0], val[:,2], '.c')
ax.plot(val[:,0], val[:,3], '.m')
plt.show()
#         #cv2.imshow("img", img)


