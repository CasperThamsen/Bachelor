import matplotlib.pyplot as plt
import numpy as np

rot1optiLoc = r"C:\Users\caspe\Workspace\Bachelor\airporttestfiles\5markerrotationoutput.csv"
rot1poseLoc = r"C:\Users\caspe\Workspace\Bachelor\5markerrotationpose.csv"
rot1opti = np.loadtxt(rot1optiLoc, delimiter=',')
rot1pose = np.loadtxt(rot1poseLoc, delimiter=',')


pose_start_time = 63
opti_start_time = 1744193042.0
duration_of_video = 1450

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

rot1opticsv = np.savetxt("optitrackdata.csv", rot1opti, delimiter=",",fmt='%.7f')
rot1posecsv = np.savetxt("posedata.csv", rot1pose, delimiter=",",fmt='%.7f')
rot1optiLoc2 = "optitrackdata.csv"
rot1poseLoc2 = "posedata.csv"

optival = np.loadtxt(rot1optiLoc2, delimiter=',')
poseval = np.loadtxt(rot1poseLoc2, delimiter=',')





# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# #compare the data sets visually (maybe matching?)
# #for x,y,z
# ax1.plot(optival[:,0], optival[:,1], '.r')
# ax1.plot(optival[:,0], optival[:,2], '.y')
# ax1.plot(optival[:,0], optival[:,3], '.b')
# ax1.plot(optival[:,0], optival[:,4], '.c')
# ax1.plot(optival[:,0], optival[:,5], '.m')
# ax1.plot(optival[:,0], optival[:,6], '.g')
# ax1.set_title("Optitrack data")
# ax2.plot(poseval[:,0], poseval[:,1], '.r')
# ax2.plot(poseval[:,0], poseval[:,2], '.y')
# ax2.plot(poseval[:,0], poseval[:,3], '.b')
# ax2.plot(poseval[:,0], poseval[:,4], '.c')
# ax2.plot(poseval[:,0], poseval[:,5], '.m')
# ax2.plot(poseval[:,0], poseval[:,6], '.g')
# ax2.set_title("Pose data")
# ax1.set_xlabel("Frame number")
# ax1.set_ylabel("Position (m)")
# ax2.set_xlabel("Frame number")
# ax2.set_ylabel("Position (m)")
# plt.show()



