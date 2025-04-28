import matplotlib.pyplot as plt
import numpy as np

rot1optiLoc = r"C:\Users\caspe\Workspace\Bachelor\airporttestfiles\5markerrotationoutput.csv"
rot1poseLoc = r"C:\Users\caspe\Workspace\Bachelor\5markerrotationpose.csv"
rot1opti = np.loadtxt(rot1optiLoc, delimiter=',')
rot1pose = np.loadtxt(rot1poseLoc, delimiter=',')


pose_start_time = 63
opti_start_time = 1744193042.0
duration_of_video = 1450

fig, ax = plt.subplots()
ax.plot(rot1opti[:,0], rot1opti[:,1], '.y')
ax.plot(rot1opti[:,0], rot1opti[:,2], '.c')
ax.plot(rot1opti[:,0], rot1opti[:,3], '.m')
ax.plot(rot1opti[:,0], rot1opti[:,4], '.r')
ax.plot(rot1opti[:,0], rot1opti[:,5], '.b')
ax.plot(rot1opti[:,0], rot1opti[:,6], '.g')
plt.show()


i = 0
index = pose_start_time
last_time = None
for i in range(len(rot1opti)):
    current_time = rot1opti[i][0]
    if current_time >= opti_start_time:
        if last_time is not None and current_time > last_time + 0.33:
            missing_frames = int((current_time - last_time) / 0.33)
            index += missing_frames 
            print(f"Missing frames: {missing_frames} at time {index}")
        rot1opti[i][0] = index
        index += 1
        last_time = current_time
        if index > 1450:
            break

#write a new file consisting only of the rows with a frame number between 0 and 1450
rot1opti = rot1opti[rot1opti[:,0] <= 1450]
rot1opti = rot1opti[rot1opti[:,0] >= 63]
rot1opti = rot1opti[rot1opti[:,0].argsort()]
#do the same for rot1pose
rot1pose = rot1pose[rot1pose[:,0] <= 1450]
rot1pose = rot1pose[rot1pose[:,0] >= 63]
rot1pose = rot1pose[rot1pose[:,0].argsort()]

rot1opticsv = np.savetxt("thisisatest.csv", rot1opti, delimiter=",",fmt='%.7f')
rot1posecsv = np.savetxt("thisisatestpose.csv", rot1pose, delimiter=",",fmt='%.7f')


