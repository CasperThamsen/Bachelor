import matplotlib.pyplot as plt
import numpy as np
import cv2

file_name1 = r"airporttestfiles\experiment_005.csv"
val1 = np.loadtxt(file_name1, delimiter=',',skiprows=7)
file_name2 = r"csvfiles\experiment_005pose.csv"
val2 = np.loadtxt(file_name2, delimiter=',')

val2_del = val2[val2[:,7] == 3]

def rvec_to_euler(rvec):
    """
    Convert a rotation vector (rvec) to Euler angles (roll, pitch, yaw).
    """
    # Convert rvec to a rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Extract Euler angles from the rotation matrix
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])



pose_euler_angles = []
for rvec in val2_del[:, 4:7]:  # Extract rvec from columns 4, 5, 6
    rvec = rvec.reshape(3, 1)  # Reshape rvec to 3x1 for cv2.Rodrigues
    euler_angles = rvec_to_euler(rvec)  # Convert to Euler angles
    pose_euler_angles.append(euler_angles)
    print(f"rvec: {rvec.flatten()}, Euler angles: {euler_angles}")

pose_euler_angles = np.array(pose_euler_angles)



fig, ax = plt.subplots()
# pose
ax.plot(val2_del[:,0]/30+3, val2_del[:,1]+20+3, 'r',   markersize=2)
ax.plot(val2_del[:,0]/30+3, val2_del[:,2]+30+1.5, 'b',markersize=2)
ax.plot(val2_del[:,0]/30+3, val2_del[:,3]+40-3, 'y',markersize=2)

# ax.plot(val2_del[:,0]/30+3, pose_euler_angles[:,0], '.m')
# ax.plot(val2_del[:,0]/30+3, pose_euler_angles[:,1], '.g')
# ax.plot(val2_del[:,0]/30+3, pose_euler_angles[:,2], '.c')




#opti

# ax.plot(val1[:,0]/240, val1[:,8], '-m')
# ax.plot(val1[:,0]/240, val1[:,9], '-c')
# ax.plot(val1[:,0]/240, val1[:,10]-90, '-g')
ax.plot(val1[:,0]/240, val1[:,11]+20, '-r',markersize=1)
ax.plot(val1[:,0]/240, val1[:,12]+30, '-b',markersize=1)
ax.plot(val1[:,0]/240, val1[:,13]+40, '-y',markersize=1)
# plt.savefig(r"C:\Users\caspe\Workspace\Bachelor\datapictures\whaaaat.png")

#x,y,z (1,2,3) passer med x,y,z(11,12,13)


plt.show()


