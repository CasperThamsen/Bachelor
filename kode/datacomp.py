import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv



#Rotation1
# rot1optiLoc = r"C:\Users\caspe\Workspace\Bachelor\airporttestfiles\5markerrotationoutput.csv"
# rot1poseLoc = r"C:\Users\caspe\Workspace\Bachelor\5markerrotationpose.csv"
# Data_name = "rotation1"
# save_name = "rotation1shifted.csv"
# pose_start_time = 65
# opti_start_time = 1744193042.0
# duration_of_video = 1450


# #Rotation2
rot1optiLoc = r"C:\Users\caspe\Workspace\Bachelor\airporttestfiles\5markerrotation2output.csv"
rot1poseLoc = r"C:\Users\caspe\Workspace\Bachelor\5markerrotation2pose.csv"
Data_name = "rotation2"
save_name = "rotation2shifted.csv"
opti_start_time = 1744193157.0
pose_start_time = 66
duration_of_video = 910



rot1opti = np.loadtxt(rot1optiLoc, delimiter=',')
rot1pose = np.loadtxt(rot1poseLoc, delimiter=',')
hz = 30/100
best_shift = None
best_opti_shifted = None
best_error = float('inf')

for shift in range(int(-60/hz), int(30/hz)):
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
    pose_R = []
    opti_R = []
    for frame in common_frames:
        pose_tvec = rot1pose[rot1pose[:,0] == frame, 1:4]
        opti_tvec= opti_shifted[opti_shifted[:,0] == frame, 1:4]
        opti_tvec_duplicate = np.repeat(opti_tvec, len(pose_tvec), axis=0)
        pose_pos.append(pose_tvec)
        opti_pos.append(opti_tvec_duplicate)
        pose_rvec = rot1pose[rot1pose[:,0] == frame, 4:7]
        opti_rvec = opti_shifted[opti_shifted[:,0] == frame, 4:7]
        opti_rvec_duplicate = np.repeat(opti_rvec, len(pose_rvec), axis=0)
        pose_R.append(pose_rvec)
        opti_R.append(opti_rvec_duplicate)
    pose_pos = np.vstack(pose_pos)
    opti_pos = np.vstack(opti_pos)
    pose_R = np.vstack(pose_R)
    opti_R = np.vstack(opti_R)

    pose_filtered = rot1pose[rot1pose[:,0] >= pose_start_time]
    #RMS error
    RMSE = np.sqrt(np.mean((opti_pos - pose_pos)**2))
    if RMSE < best_error:
        best_error = RMSE
        best_shift = shift
        best_opti_shifted = opti_shifted.copy()
        opti_shifted = np.unique(opti_shifted,axis=0)
        np.savetxt(save_name.replace(".csv", "opti.csv"), opti_shifted, delimiter=",", fmt='%.7f')
        np.savetxt(save_name, pose_filtered, delimiter=",", fmt='%.7f')


new_data = [best_shift * hz, best_error, opti_start_time + best_shift*hz, Data_name]
with open("best_shifts.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(new_data)