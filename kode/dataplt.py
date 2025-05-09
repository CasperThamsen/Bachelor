import matplotlib.pyplot as plt
import numpy as np
file_name = r"csvfiles\experiment_002pose.csv"
val_raw = np.loadtxt(file_name, delimiter=',')
fig, ax = plt.subplots()
val = val_raw[val_raw[:,7] == 0]
ax.plot(val[:,0], val[:,1], '.r')
ax.plot(val[:,0], val[:,2], '.y')
ax.plot(val[:,0], val[:,3], '.b')
ax.plot(val[:,0], val[:,4], '.r')
ax.plot(val[:,0], val[:,5], '.y')
ax.plot(val[:,0], val[:,6], '.b')
plt.savefig(r"datapictures\experiment_001.png")
plt.show()




