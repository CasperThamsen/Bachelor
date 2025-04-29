import matplotlib.pyplot as plt
import numpy as np
import csv



#Rotation1
# rot1optiLoc = r"C:\Users\caspe\Workspace\Bachelor\airporttestfiles\5markerrotationoutput.csv"
# rot1poseLoc = r"C:\Users\caspe\Workspace\Bachelor\5markerrotationpose.csv"
# Data_name = "5markerrotation1"
# pose_start_time = 65
# opti_start_time = 1744193042.0
# duration_of_video = 1450


# #Rotation2
# rot1optiLoc = r"C:\Users\caspe\Workspace\Bachelor\airporttestfiles\5markerrotation2output.csv"
# rot1poseLoc = r"C:\Users\caspe\Workspace\Bachelor\5markerrotation2pose.csv"
# Data_name = "5markerrotation2"
# opti_start_time = 1744193157.0
# pose_start_time = 66
# duration_of_video = 910



rot1opti = np.loadtxt(rot1optiLoc, delimiter=',')
rot1pose = np.loadtxt(rot1poseLoc, delimiter=',')
hz = 30/100
best_shift = None
best_opti_shifted = None
best_error = float('inf')

for shift in range(int(-30/hz), int(30/hz)):
    shifted_opti_start_time = opti_start_time + shift * hz
    opti_shifted = rot1opti.copy()
    index = pose_start_time
    last_time = None
    total_time_missing = 0
    for i in range(len(opti_shifted)):
        current_time = opti_shifted[i][0]
        if current_time >= shifted_opti_start_time:
            if last_time is not None and current_time > last_time + hz:
                missing_frames = int((current_time - last_time) / hz)
                index += missing_frames 
                total_time_missing += missing_frames
                # print(f"Missing frames: {missing_frames} at time {index}, In seconds: {missing_frames*hz}")
            opti_shifted[i][0] = index
            index += 1
            last_time = current_time
            if index > duration_of_video:
                break


    opti_shifted = opti_shifted[(opti_shifted[:,0] >= pose_start_time) & (opti_shifted[:,0] <= duration_of_video)]
    opti_shifted = opti_shifted[opti_shifted[:,0].argsort()]
    common_frames = np.intersect1d(opti_shifted[:,0], rot1pose[:,0])

    pose_pos = []
    opti_pos = []
    for frame in common_frames:
        pose_rows = rot1pose[rot1pose[:,0] == frame, 1:4]
        opti_rows= opti_shifted[opti_shifted[:,0] == frame, 1:4]
        opti_duplicate = np.repeat(opti_rows, len(pose_rows), axis=0)
        pose_pos.append(pose_rows)
        opti_pos.append(opti_duplicate)
    pose_pos = np.vstack(pose_pos)
    opti_pos = np.vstack(opti_pos)

    #RMS error
    RMSE = np.sqrt(np.mean((opti_pos - pose_pos)**2))
    if RMSE < best_error:
        best_error = RMSE
        best_shift = shift
        best_opti_shifted = opti_shifted.copy()
        print(f"Best shift {best_shift*hz}, RMSE: {RMSE}, shift: {shift}")
        print(total_time_missing)

# print(f"Best shift: {best_shift*hz}")
# print(f"Best RMSE: {best_error}")

# np.savetxt("markerrotation2optitrack.csv", opti_shifted, delimiter=",", fmt='%.7f')
# np.savetxt("markerrotation2pose.csv", rot1pose, delimiter=",", fmt='%.7f')
new_data = [best_shift * hz, best_error, opti_start_time + best_shift*hz, Data_name]
with open("best_shifts.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(new_data)