import numpy as np
import csv
from dataset_configs import datasets
import time
import cv2
import rotationFix as rotFix

# Choose the dataset to run:
# "rotation1", "rotation2", "1marker", "1markerfar", "5markervid1", "5markervid2",
# "experiment_001", "experiment_002", "experiment_003", "experiment_005", "experiment_006"
def main():
    selected = "experiment_005"  
    cfg = datasets[selected]

    optiLoc = f"csvfiles\\" + selected + "shiftedopti.csv"
    poseLoc = cfg["rot1poseLoc"]
    Data_name = cfg["Data_name"]
    save_name = cfg["save_name"]
    pose_start_time = cfg["pose_start_time"]
    opti_start_time = cfg["opti_start_time"]
    duration_of_video = cfg["duration_of_video"]
    hz = cfg["hz"]
    save_location = r"csvfiles\\"
    save = save_location + save_name

    optiData = np.loadtxt(optiLoc, delimiter=',', skiprows=1)
    poseData = np.loadtxt(poseLoc, delimiter=',')

    transform_data_aruco(optiData, poseData, save)


def transform_data_aruco(optiData, poseData, save):
    # should transform the pose data to the opti data for the best shift based on aruco marker 0
    transformed_data_aruco = []
    #take mean if marker id == 0
    id0_indices = np.where(poseData[:, 7] == 0)[0]
    pose_id0 = poseData[id0_indices]
    aruco0_pose_tvec = pose_id0[:, 1:4]
    aruco0_pose_rvec = pose_id0[:, 4:7]

    opti_id0 = optiData[id0_indices]
    aruco0_opti_tvec = opti_id0[:, 1:4]
    aruco0_opti_rvec = opti_id0[:, 4:7]
    # Compare the rotations (rvecs) between pose and opti for marker id 0
    rot_diffs = []
    rot_diffs = []
    poseData_transform = poseData.copy()
    for idx, (pose_row, opti_row) in enumerate(zip(pose_id0, opti_id0)):
        pose_rvec = pose_row[4:7]
        opti_rvec = opti_row[4:7]
        R_pose, _ = cv2.Rodrigues(pose_rvec)
        R_opti, _ = cv2.Rodrigues(opti_rvec)
        R_rel = R_opti @ R_pose.T
        rvec_rel, _ = cv2.Rodrigues(R_rel)
        angle_diff = rotFix.rvec_to_euler(rvec_rel)
        rot_diffs.append(angle_diff)
    rot_diffs = np.array(rot_diffs)
    mean_rot_diff = np.mean(rot_diffs, axis=0)
    print(f"Mean rotation difference (rad) for marker id 0 (x, y, z): {mean_rot_diff}")
    # Apply mean rotation difference to all pose rvecs in one loop
    R_mean_diff, _ = cv2.Rodrigues(mean_rot_diff)
    for i in range(poseData_transform.shape[0]):
        pose_rvec = poseData_transform[i, 4:7]
        R_pose, _ = cv2.Rodrigues(pose_rvec)
        R_new = R_mean_diff @ R_pose
        rvec_new, _ = cv2.Rodrigues(R_new)
        poseData_transform[i, 4:7] = rvec_new.flatten()
    
    transform_tvec = np.mean(aruco0_opti_tvec - aruco0_pose_tvec, axis=0)
    poseData_transform = poseData.copy()
    for i in range(poseData_transform.shape[0]):
        poseData_transform[i, 1:4] += transform_tvec
    np.savetxt(save.replace("shifted.csv", "pose_transformed.csv"), poseData_transform, delimiter=",", fmt='%.7f')


    


main()
