import matplotlib.pyplot as plt
import cv2
import numpy as np
import icecream as ic
import csv
from dataset_configs import datasets


# Choose the dataset to run:
selected = "rotation2"
cfg = datasets[selected]

# Then access like this:
rot1optiLoc = cfg["rot1optiLoc"]
rot1poseLoc = cfg["rot1poseLoc"]
Data_name = cfg["Data_name"]
save_name = cfg["save_name"]
pose_start_time = cfg["pose_start_time"]
opti_start_time = cfg["opti_start_time"]
duration_of_video = cfg["duration_of_video"]



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
    common_frames = np.intersect1d(opti_shifted[:,0], rot1pose[:,0])

    pose_pos = []
    opti_pos = []
    pose_rotations = []
    opti_rotations = []
    for frame in common_frames:
        pose_tvec = rot1pose[rot1pose[:,0] == frame, 1:7]
        opti_tvec= opti_shifted[opti_shifted[:,0] == frame, 1:7]
        opti_tvec_duplicate = np.repeat(opti_tvec, len(pose_tvec), axis=0)
        pose_pos.append(pose_tvec[:,0:3])
        opti_pos.append(opti_tvec_duplicate[:,0:3])
        pose_rot = pose_tvec[:,3:6]
        opti_rot = opti_tvec_duplicate[:,3:6]
        for i in range(len(pose_rot)):
                R_pose, _ = cv2.Rodrigues(pose_rot[i])
                R_opti, _ = cv2.Rodrigues(opti_rot[i])

                pose_rotations.append(R_pose)
                opti_rotations.append(R_opti)
    pose_pos = np.vstack(pose_pos)
    opti_pos = np.vstack(opti_pos)


    angular_differences = []
    for R_pose, R_opti in zip(pose_rotations, opti_rotations):
        R_relative = R_pose.T @ R_opti
        relative_rotation_vector, _ = cv2.Rodrigues(R_relative)
        angular_difference = np.linalg.norm(relative_rotation_vector)
        angular_differences.append(angular_difference)
    mean_angular_difference = np.mean(angular_differences)






    pose_filtered = rot1pose[rot1pose[:,0] >= pose_start_time]
    #RMS error
    RMSE = np.sqrt(np.mean((opti_pos - pose_pos)**2))
    combined_error = RMSE + mean_angular_difference
    if combined_error < best_error:
        best_error = combined_error
        best_shift = shift
        best_opti_shifted = opti_shifted.copy()
        opti_shifted = np.unique(opti_shifted,axis=0)
        np.savetxt(save_name.replace(".csv", "opti.csv"), opti_shifted, delimiter=",", fmt='%.7f')
        np.savetxt(save_name, pose_filtered, delimiter=",", fmt='%.7f')
        print(f"Best shift: {best_shift}, Error: {best_error}, Shifted start time: {shifted_opti_start_time}")
        


new_data = [best_shift * hz, best_error, opti_start_time + best_shift*hz, Data_name]
with open("best_shifts.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(new_data)