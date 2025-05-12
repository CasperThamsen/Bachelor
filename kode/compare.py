import matplotlib.pyplot as plt
import numpy as np
import cv2

file_name1 = r"airporttestfiles\experiment_005output.csv"
val1 = np.loadtxt(file_name1, delimiter=',',skiprows=7)
file_name2 = r"csvfiles\experiment_005pose.csv"
val2 = np.loadtxt(file_name2, delimiter=',')

val2_del = val2[val2[:,7] == 0]

fig, ax = plt.subplots()
# pose
ax.plot(val2_del[:,0]/30+3, val2_del[:,1], '.r',   markersize=2)
ax.plot(val2_del[:,0]/30+3, val2_del[:,2], '.y',markersize=2)
ax.plot(val2_del[:,0]/30+3, val2_del[:,3], '.b',markersize=2)

# ax.plot(val2_del[:,0]/30+3, pose_euler_angles[:,0], '.m')
# ax.plot(val2_del[:,0]/30+3, pose_euler_angles[:,1], '.g')
# ax.plot(val2_del[:,0]/30+3, pose_euler_angles[:,2], '.c')




#opti

# ax.plot(val1[:,0]/240, val1[:,8], '-m')
# ax.plot(val1[:,0]/240, val1[:,9], '-c')
# ax.plot(val1[:,0]/240, val1[:,10], '-g')
ax.plot(val1[:,0]/240, val1[:,1], '-r',markersize=1)
ax.plot(val1[:,0]/240, val1[:,2], '-y',markersize=1)
ax.plot(val1[:,0]/240, val1[:,3], '-b',markersize=1)
# plt.savefig(r"C:\Users\caspe\Workspace\Bachelor\datapictures\whaaaat.png")

#x,y,z (1,2,3) passer med x,y,z(11,12,13)


plt.show()


