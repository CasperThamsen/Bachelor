import matplotlib.pyplot as plt
import numpy as np
import cv2
import rotationFix as RF

file_name1 = r"airporttestfiles\experiment_005output.csv"
val1 = np.loadtxt(file_name1, delimiter=',',skiprows=7)
file_name2 = r"csvfiles\experiment_005pose.csv"
val2 = np.loadtxt(file_name2, delimiter=',')

val2_del = val2[val2[:,7] == 10]

# Convert rvecs (columns 4, 5, 6) in val2_del and val1 to euler angles using rotationfix.py
rvecs_pose = val2_del[:, 4:7]
eulers_pose = np.array([RF.rvec_to_euler(rvec) for rvec in rvecs_pose])
val2_del[:, 4:7] = eulers_pose

rvecs_opti = val1[:, 4:7]
eulers_opti = np.array([RF.rvec_to_euler(rvec) for rvec in rvecs_opti])
val1[:, 4:7] = eulers_opti


fig, ax = plt.subplots()
scale = 8
offset = 873
# pose
ax.plot(val2_del[:,0] * scale + offset, val2_del[:,1], '.r',   markersize=2)
ax.plot(val2_del[:,0] * scale + offset, val2_del[:,2], '.g',markersize=2)
ax.plot(val2_del[:,0] * scale + offset, val2_del[:,3], '.b',markersize=2)
# ax.plot(val2_del[:,0] * scale + offset, val2_del[:,4]/180*np.pi, 'm')
# ax.plot(val2_del[:,0] * scale + offset, val2_del[:,5]/180*np.pi, 'y')
# ax.plot(val2_del[:,0] * scale + offset, val2_del[:,6]/180*np.pi, 'c')




#opti

# ax.plot(val1[:,0], val1[:,4]/180*np.pi, '-m')
# ax.plot(val1[:,0], val1[:,5]/180*np.pi, '-y')
# ax.plot(val1[:,0], val1[:, 6] / 180 * np.pi, '-c')
ax.plot(val1[:,0], val1[:,1], '-r',markersize=1)
ax.plot(val1[:,0], val1[:,2], '-g',markersize=1)
ax.plot(val1[:,0], val1[:,3], '-b',markersize=1)
# plt.savefig(r"C:\Users\caspe\Workspace\Bachelor\datapictures\whaaaat.png")

plt.savefig("N-fold0_transformed.png")
#x,y,z (1,2,3) passer med x,y,z(11,12,13)


plt.show()


