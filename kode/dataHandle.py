import numpy as np
import csv
from dataset_configs import datasets
import time
import cv2
import rotationFix as RF

# Choose the dataset to run:
# "rotation1", "rotation2", "1marker", "1markerfar", "5markervid1", "5markervid2",
# "experiment_001", "experiment_002", "experiment_003", "experiment_005", "experiment_006"
def main():
    selected = "experiment_005"  
    cfg = datasets[selected]
    optiLoc = f"csvfiles\\" + selected + "shiftedopti.csv"
    poseLoc = cfg["rot1poseLoc"]
    save_name = cfg["save_name"]
    save_location = r"csvfiles\\"
    save = save_location + save_name
    optiData = np.loadtxt(optiLoc, delimiter=',', skiprows=1)
    poseData = np.loadtxt(poseLoc, delimiter=',')
    transform_data_Nfold(optiData, poseData, save)


def transform_data_Nfold(optiData, poseData, save):
    # should transform the pose data to the opti data for the best shift based on aruco marker 0
    transformed_data_aruco = []
    #take mean if marker id == 0
    id0_indices = np.where(poseData[:, 7] == 10)[0]
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
    # Convert all pose rvecs to Euler angles (in degrees)
    # pose_euler_deg = []
    # for i in range(poseData_transform.shape[0]):
    #     rvec = poseData_transform[i, 4:7]
    #     euler_rad = rotFix.rvec_to_euler(rvec)
    # pose_euler_deg = np.array(euler_rad)  # Convert radians to degrees
    # # Optionally, replace the rvec columns with Euler angles in poseData_transform
    # poseData_transform[:, 4:7] = pose_euler_deg

    rvecs_pose = poseData[:, 4:7]
    eulers_pose_rad = np.array([RF.rvec_to_euler(rvec) for rvec in rvecs_pose])
    eulers_pose_deg = np.degrees(eulers_pose_rad)
    poseData_transform[:, 4:7] = eulers_pose_deg
    
    transform_tvec = np.mean(aruco0_opti_tvec - aruco0_pose_tvec, axis=0)
    for i in range(poseData_transform.shape[0]):
        poseData_transform[i, 1:4] += transform_tvec
    np.savetxt(save.replace("shifted.csv", "pose_transformed.csv"), poseData_transform, delimiter=",", fmt='%.7f')


    


main()
