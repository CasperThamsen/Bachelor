import numpy as np
import csv
from dataset_configs import datasets
import time
import cv2

""" This script is used to find the best shift for the opti data to match the pose data.
    It will iterate through a range of shifts and calculate the RMSE between the shifted opti data and the pose data.
    this file is for handling experiment files which are all from the seconds visit to the airport
    They are formatted frame,time(seconds),rx,rz,ry,x,y,z,rx,rz,ry,x,y,z. Rewritten to frame,x,y,z,rx,ry,rz in handledata2.py
    The most important difference is the change of time format from unix to frames, seconds."""

# Choose the dataset to run:
# "rotation1", "rotation2", "1marker", "1markerfar", "5markervid1", "5markervid2",
# "experiment_001", "experiment_002", "experiment_003", "experiment_005", "experiment_006"
def main():
    selected = "experiment_005"  
    cfg = datasets[selected]

    rot1optiLoc = cfg["rot1optiLoc"]
    rot1poseLoc = cfg["rot1poseLoc"]
    Data_name = cfg["Data_name"]
    save_name = cfg["save_name"]
    pose_start_time = cfg["pose_start_time"]
    opti_start_time = cfg["opti_start_time"]
    duration_of_video = cfg["duration_of_video"]
    hz = cfg["hz"]
    save_location = r"csvfiles\\"
    save = save_location + save_name



    rot1opti = np.loadtxt(rot1optiLoc, delimiter=',')
    rot1pose = np.loadtxt(rot1poseLoc, delimiter=',')
    best_shift = None
    best_opti_shifted = None
    best_error = float('inf')
    best_frameset = None
    fps = 30 # phone fps

    #2 methods are possible due to the framerate missmatch
    # 1. divide the opti data by 8, thereby making it 30hz and equal in size to the pose data
    # 2. keep the opti data at 240hz, and compare every 8 values to the pose data at the corresponding time

    # will start with 1 and see results. 
    # currently takes every 8th element, maybe shift the "scale" by +1 for every possibility to match 8 frames per 1 frame

    scale = hz / fps
    num_frames = int(len(rot1opti)/8)
    intscale = int(scale)
    start_time = time.time()


    for frameset in range(intscale): 
        print(f"Processing frameset: {frameset}")
        opti_downsampled = rot1opti[frameset::8].copy() # Downsample optitrack frame rate from 240 hz to 30 to match phone fps
        shift_start_time = time.time()
        for shift in range(num_frames): #num_Frames
            shifted_opti_start_time = shift
            opti_shifted = opti_downsampled.copy()
            index = pose_start_time
            
            if index == 0:
                index = 1
            last_time = None
            total_time_missing = 0
            for i in range(len(opti_shifted)):
                current_time = opti_shifted[i][0]
                if current_time >= shifted_opti_start_time:
                    if last_time is not None and current_time > last_time + scale:
                        missing_frames = int((current_time - last_time) / scale)
                        index += missing_frames 
                        total_time_missing += missing_frames
                        print(f"Missing frames: {missing_frames} at time {index}, total time missing: {total_time_missing}")
                    opti_shifted[i][0] = index
                    index += 1
                    last_time = current_time
                    if index > duration_of_video:
                        break
                else:
                    opti_shifted[i][0] = np.inf #set data before start time to inf to remove in next line (opti_shifted)
            shift_end_time = time.time()

            opti_shifted = opti_shifted[(opti_shifted[:,0] >= pose_start_time) & (opti_shifted[:,0] <= duration_of_video)]
            common_frames = np.intersect1d(opti_shifted[:,0], rot1pose[:,0]) #Ensures both datasets have data in frame
            
            # commented rotation out for now.
            pose_pos = []
            opti_pos = []
            id_list = []
            frame_list = []
            pose_rotations = []
            opti_duplicate = []
            #save common frames to a new file
            frame_start_time = time.time()
            for frame in common_frames:           
                pose_data = rot1pose[rot1pose[:,0] == frame, 0:8]
                opti_data = opti_shifted[opti_shifted[:,0] == frame, 0:8]
                opti_duplicate_tvec = np.repeat(opti_data, len(pose_data), axis=0)
                opti_duplicate.append(opti_duplicate_tvec)
                frame_list.append(pose_data[:,0])
                id_list.append(pose_data[:,7])
                pose_pos.append(pose_data[:,1:4])
                opti_pos.append(opti_duplicate_tvec[:,1:4])
                pose_rotations.append(pose_data[:,4:7])
            frame_end_time = time.time()

            pose_pos = np.vstack(pose_pos)
            opti_pos = np.vstack(opti_pos)
            pose_rotations = np.vstack(pose_rotations)
            frame_list = np.concatenate(frame_list)
            id_list = np.concatenate(id_list)
            

            #RMS error
            RMSE = np.sqrt(np.mean((opti_pos - pose_pos)**2))
            # combined_error = RMSE + mean_angular_difference
            if RMSE < best_error:
                best_error = RMSE
                best_shift = shift
                best_opti_shifted = opti_duplicate.copy()
                best_frameset = frameset
            end_time = time.time()
            running_time = end_time - start_time
            print(f"\rRunning total time: {running_time:.2f} seconds", end='', flush=True)
        print(f"\nTime taken for shift: {shift_end_time - shift_start_time} seconds")
        print(f"Time taken for frame: {frame_end_time - frame_start_time} seconds")
        
                

        #doesnt work yet, should be made as a new file to avoid having to find best shift every time?
        #note to self, best shifted dataset is saved in "name"shiftedpose/opti.csv
        #Fix variable names to be more descriptive, should be pose transformed into opti
        #OptiPose≈T⋅Pose
    



    new_data = [best_shift * scale, best_error, best_shift,best_frameset ,Data_name]
    with open(save_location + "best_shifts.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_data)


    np.savetxt(save.replace(".csv", "opti.csv"), np.vstack(best_opti_shifted), delimiter=",", fmt='%.7f', header=f"shifted by {best_shift}")
    print(f"Best shift: {best_shift}, Best RMSE: {best_error}, Best frameset: {best_frameset}")
    print(f"Time taken for all shifts: {end_time - start_time} seconds")


main()