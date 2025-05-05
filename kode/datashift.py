import matplotlib.pyplot as plt
import cv2
import numpy as np
import icecream as ic
import csv
from dataset_configs import datasets

""" This script is used to find the best shift for the opti data to match the pose data.
    It will iterate through a range of shifts and calculate the error between the shifted opti data and the pose data."""

# Choose the dataset to run:
selected = "rotation2"
cfg = datasets[selected]

rot1optiLoc = cfg["rot1optiLoc"]
rot1poseLoc = cfg["rot1poseLoc"]
Data_name = cfg["Data_name"]
save_name = cfg["save_name"]
pose_start_time = cfg["pose_start_time"]
opti_start_time = cfg["opti_start_time"]
duration_of_video = cfg["duration_of_video"]
save_location = r"csvfiles\\"
save = save_location + save_name



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
        R_relative = R_opti.T @ R_pose
        relative_rotation_vector, _ = cv2.Rodrigues(R_relative)
        angular_difference = np.linalg.norm(relative_rotation_vector)
        angular_differences.append(angular_difference)
    mean_angular_difference = np.mean(angular_differences)



    pose_filtered = rot1pose[rot1pose[:,0] >= pose_start_time]
    #RMS error
    RMSE = np.sqrt(np.mean((opti_pos - pose_pos)**2))
    combined_error = RMSE + mean_angular_difference
    if RMSE < best_error:
        best_error = RMSE
        best_shift = shift
        best_opti_shifted = opti_shifted.copy()
        opti_shifted = np.unique(opti_shifted,axis=0)
        #save best shifted data
        np.savetxt(save.replace(".csv", "pose.csv"), pose_filtered, delimiter=",", fmt='%.7f')
        np.savetxt(save.replace(".csv", "opti.csv"), opti_shifted, delimiter=",", fmt='%.7f')
        print(f"Best shift: {best_shift}, Error: {best_error}, Shifted start time: {shifted_opti_start_time}")
        # find homogenous transformation matrix to align coordinate systems
        #taking mean on rvec is not valid. Fix. (enten brug fra bedste fitting pose, eller brug anden metode)
        best_rvec = np.mean(pose_rot, axis=0)  # gennemsnit af pose rotation for nu... (FORKERT METODE)
        best_tvec = np.mean(pose_pos - opti_pos, axis=0) 
        R,_, = cv2.Rodrigues(best_rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = best_tvec
        print("Homogeneous transformation matrix:")
        print(T)

#doesnt work yet, should be made as a new file to avoid having to find best shift every time?
#note to self, best shifted dataset is saved in "name"shiftedpose/opti.csv
#Fix variable names to be more descriptive, should be pose transformed into opti
#OptiPose≈T⋅Pose
transformed_data = []
for pos, pose_r in zip(pose_pos,pose_rotations):
    R_cur, _ = cv2.Rodrigues(pose_r)
    T_cur = np.eye(4)
    T_cur[:3, :3] = R_cur
    T_cur[:3, 3] = pos
    T_new = T @ T_cur
    rvec_new, _ = cv2.Rodrigues(T_new[:3, :3])
    tvec_new = T_new[:3, 3]
    pose_in_opti = np.hstack((tvec_new, rvec_new.flatten()))
    transformed_data.append(pose_in_opti)
np.savetxt(save.replace("shifted.csv", "pose_transformed.csv"), transformed_data, delimiter=",", fmt='%.7f')



new_data = [best_shift * hz, best_error, opti_start_time + best_shift*hz, Data_name]
with open(save_location + "best_shifts.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(new_data)