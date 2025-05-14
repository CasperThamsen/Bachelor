import matplotlib.pyplot as plt
import numpy as np
import cv2

file_name1 = r"airporttestfiles\experiment_005output.csv"
val1 = np.loadtxt(file_name1, delimiter=',',skiprows=7)
file_name2 = r"csvfiles\experiment_005pose.csv"
val2 = np.loadtxt(file_name2, delimiter=',')

val2_del = val2[val2[:,7] == 0]

fig, ax = plt.subplots()
scale = 8
offset = 873
# pose
ax.plot(val2_del[:,0] * scale + offset, val2_del[:,1]+20, '.r',   markersize=2)
ax.plot(val2_del[:,0] * scale + offset, val2_del[:,2]+20, '.g',markersize=2)
ax.plot(val2_del[:,0] * scale + offset, val2_del[:,3]+20, '.b',markersize=2)
ax.plot(val2_del[:,0] * scale + offset, val2_del[:,4] / 180*np.pi, '.m')
ax.plot(val2_del[:,0] * scale + offset, val2_del[:,5]/ 180*np.pi, '.y')
ax.plot(val2_del[:,0] * scale + offset, val2_del[:,6]/ 180*np.pi, '.c')




#opti

ax.plot(val1[:,0], val1[:,4], '-m')
ax.plot(val1[:,0], val1[:,5], '-y')
ax.plot(val1[:,0], val1[:,6], '-c')
ax.plot(val1[:,0], val1[:,1]+20, '-r',markersize=1)
ax.plot(val1[:,0], val1[:,2]+20, '-g',markersize=1)
ax.plot(val1[:,0], val1[:,3]+20, '-b',markersize=1)
# plt.savefig(r"C:\Users\caspe\Workspace\Bachelor\datapictures\whaaaat.png")

#x,y,z (1,2,3) passer med x,y,z(11,12,13)


plt.show()


