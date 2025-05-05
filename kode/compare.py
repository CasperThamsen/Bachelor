import matplotlib.pyplot as plt
import numpy as np

file_name1 = r"C:\Users\caspe\Workspace\Bachelor\airporttestfiles\5markerrotation2.csv"
val1 = np.loadtxt(file_name1, delimiter=',')
file_name2 = r"C:\Users\caspe\Workspace\Bachelor\5markerrotation2pose.csv"
val2 = np.loadtxt(file_name2, delimiter=',')

val2_del = val2[val2[:,7] == 0]
framrate = 30
unix = 1744193149.2328787
fig, ax = plt.subplots()
ax.plot(val2_del[:,0]/30, val2_del[:,1], '.r',markersize=5)
ax.plot(val2_del[:,0]/30, val2_del[:,2], '.y',markersize=5)
ax.plot(val2_del[:,0]/30, val2_del[:,3], '.b',markersize=5)
ax.plot(val1[:,0]-unix, val1[:,1], '.r',markersize=1)
ax.plot(val1[:,0]-unix, val1[:,2], '.y',markersize=1)
ax.plot(val1[:,0]-unix, val1[:,3], '.b',markersize=1)
# plt.savefig(r"C:\Users\caspe\Workspace\Bachelor\datapictures\whaaaat.png")
plt.show()
