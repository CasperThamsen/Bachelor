import matplotlib.pyplot as plt
import numpy as np
import cv2
import rotationFix as RF
from dataset_configs import datasets

selected = "experiment_005"  
cfg = datasets[selected]
scale = cfg["hz"] / 30
offset = cfg["offset"]
duration_of_video = cfg["duration_of_video"]
file_name1 = f"airporttestfiles\\" + selected + "output.csv"
file_name2 = f"csvfiles\\" + selected + "pose_transformed.csv"


# file_name1 = r"airporttestfiles\experiment_005output.csv"
#+offset, max_rows=duration_of_video*scale
val1 = np.loadtxt(file_name1, delimiter=',',skiprows=7)  # OptiTrack
# file_name2 = r"csvfiles\experiment_005pose_transformed.csv"
val2 = np.loadtxt(file_name2, delimiter=',')
val2_del = val2[val2[:,7] == 10]

offset += 100

# Convert rvecs (columns 4, 5, 6) in val2_del and val1 to euler angles using rotationfix.py
# rvecs_pose = val2_del[:, 4:7]
# eulers_pose = np.array([RF.rvec_to_euler(rvec) for rvec in rvecs_pose])
# val2_del[:, 4:7] = eulers_pose
# print("Pose data euler angles:", eulers_pose)

# rvecs_opti = val1[:, 4:7]
# eulers_opti = np.array([RF.rvec_to_euler(rvec) for rvec in rvecs_opti])
# val1[:, 4:7] = eulers_opti


fig, ax = plt.subplots()
# pose
# ax.plot(val2_del[:,0] * scale + offset, val2_del[:,1], '.r',   markersize=2, label='pose rx')
# ax.plot(val2_del[:,0] * scale + offset, val2_del[:,2], '.g',markersize=2, label='pose ry')
# ax.plot(val2_del[:,0] * scale + offset, val2_del[:,3], '.b',markersize=2, label='pose rz')
# ax.plot(val2_del[:,0] * scale + offset, -np.unwrap(val2_del[:,4])/180*np.pi-np.pi, 'm', label='pose rx')
# ax.plot(val2_del[:,0] * scale + offset, val2_del[:,5]/180*np.pi+np.pi/2, 'y', label='pose ry')
# ax.plot(val2_del[:,0] * scale + offset, val2_del[:,6]/180*np.pi-np.pi/2, 'c', label='pose rz')





#opti

# ax.plot(val1[:,0], val1[:,4]/180*np.pi, '.m', label='opti rx')
# ax.plot(val1[:,0], val1[:,5]/180*np.pi, '.c',label='opti rz')
# ax.plot(val1[:,0], val1[:, 6]/180*np.pi, '.y',label='opti ry')
# ax.plot(val1[:,0], val1[:,1], '-r',markersize=1, label='opti x')
# ax.plot(val1[:,0], val1[:,2], '-g',markersize=1, label='opti y')
# ax.plot(val1[:,0], val1[:,3], '-b',markersize=1, label='opti z')
# plt.savefig(r"C:\Users\caspe\Workspace\Bachelor\datapictures\whaaaat.png")
plt.legend()
# plt.savefig("Aruco0_e2.png")
#x,y,z (1,2,3) passer med x,y,z(11,12,13)


plt.show()


