import matplotlib.pyplot as plt
import numpy as np
file_name = r"C:\Users\caspe\Workspace\Bachelor\airporttestfiles\5markerrotation.csv"
val = np.loadtxt(file_name, delimiter=',')
fig, ax = plt.subplots()
ax.plot(val[:,0], val[:,1], '.y')
ax.plot(val[:,0], val[:,2], '.c')
ax.plot(val[:,0], val[:,3], '.m')
ax.plot(val[:,0], val[:,4], '.r')
ax.plot(val[:,0], val[:,5], '.b')
ax.plot(val[:,0], val[:,6], '.g')
plt.show()



