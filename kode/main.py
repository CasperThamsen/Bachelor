import cv2
import MarkerTracker as MarkerTracker
import numpy as np
from icecream import ic
import cv2.aruco as aruco
import csv
import time

def main():
    start_time = time.time()
    # Load calibration data
    calibration_data = np.load('phone_calibration.npz')
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']
    validation_ratios = np.load('validation_ratios.npz')
    ratios = validation_ratios['ratios']

    #pose related variables
    marker_length = 0.154
    obj_points = np.array([
        [-marker_length / 2, marker_length / 2, 0],  # Top-left corner
        [marker_length / 2, marker_length / 2, 0],   # Top-right corner
        [marker_length / 2, -marker_length / 2, 0],  # Bottom-right corner
        [-marker_length / 2, -marker_length / 2, 0]  # Bottom-left corner
    ], dtype=np.float32)

    #Aruco
    detector_params = aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    detector = aruco.ArucoDetector(dictionary, detector_params)
    
    save_name = 'experiment_005'
    csv_file_name = r"csvfiles\\" + save_name+'pose.csv'
    cap = cv2.VideoCapture(r"videos\unprocessed\\" + save_name + ".mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    out = cv2.VideoWriter(r"videos\processed\\" +save_name+'pose.mp4', 
                        cv2.VideoWriter_fourcc(*'XVID'), 
                        30.0, 
                        frame_size)
    mt = MarkerTracker.MarkerTracker(order = 4, #number of shaded regions
                                kernel_size=20,
                                scale_factor=1)
    mt.track_marker_with_missing_black_leg = False
    mt.expected_ratios = ratios
    with open(csv_file_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
    ref_intensity = []

    

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            ic("Failed to capture image")
            break
        #unitycoin clean bob lesson1
        #Initialize MarkerTracker functionality
        mt.locate_marker_init(frame = img[:,:,1])
        poses, number_of_markers, reference_intensity = mt.detect_multiple_markers(frame=img[:,:,1])
        distance_between_markers = mt.distances_between_markers(poses,number_of_markers)
        marker_pairs = mt.detect_marker_pairs(poses,distance_between_markers)
        mt.numerate_markers(marker_pairs)
        marker_corners_n = mt.marker_corners(marker_pairs)
        
        ref_intensity.append(reference_intensity)

        #marker differentiation
        marker_corners = []
        marker_corners.extend(marker_corners_n)
        #Aruco
        marker_corners_aruco, marker_ids, rejected_candidates = detector.detectMarkers(img)
        marker_corners.extend(marker_corners_aruco)

        #Draw on canvas
        img_copy=img.copy()
        # for pair in marker_pairs:
        #     sorted_pair = sorted(pair, key=lambda pose: pose.number)  # Sort markers by their number
        #     for i in range(len(sorted_pair)-1):
        #         cv2.line(img_copy, (int(sorted_pair[i].x), int(sorted_pair[i].y)), (int(sorted_pair[(i+1)].x), int(sorted_pair[(i+1)].y)), (0, 255, 0), 4)
        #     for i in range(len(sorted_pair)):
        #         cv2.putText(img_copy, str(sorted_pair[i].number), (int(sorted_pair[i].x), int(sorted_pair[i].y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        for pose in poses:
            if pose.number != 1:
                cv2.circle(img_copy, (int(pose.x), int(pose.y)), 5, (0, 0, 255), -1)
            elif pose.number == 1:
                cv2.circle(img_copy, (int(pose.x), int(pose.y)), 5, (255, 0, 0), -1)

        if marker_pairs is not None or marker_ids is not None:
            rvecs = []            
            tvecs = []
            tvec_marker_frame = [] 
            text_offset = 0
            aruco.drawDetectedMarkers(img_copy, marker_corners_n)
            aruco.drawDetectedMarkers(img_copy, marker_corners_aruco, marker_ids)

            for i, marker_corner in enumerate(marker_corners):
                img_points = np.array(marker_corner[0], dtype=np.float32)
                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    #transform the translation vector to the marker coordinate system
                    R, _ = cv2.Rodrigues(rvec)
                    cam_in_mkr = -R.T @ tvec
                    tvec_marker_frame.append(cam_in_mkr)
                    # Draw axis for each detected marker
                    cv2.drawFrameAxes(img_copy, mtx, dist, rvec, tvec, marker_length*1.5,2)

                    # Display the translation vector (tvec) on the image
                    if i < len(marker_corners_n):
                        marker_type = "n-fold"
                    else:
                        marker_type = "Aruco"

                    cv2.putText(img_copy, f"{marker_type} tvec: {cam_in_mkr}, rvec: {rvec.flatten()}", (10, 30 + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    text_offset += 20
        #Write csv file
        closest_aruco_ids = []  # Store closest ArUco IDs for each n-fold marker
        for i, tvec_nfold in enumerate(tvecs[:len(marker_corners_n)]):  # Iterate over n-fold markers
            closest_aruco_id = None
            min_distance = float('inf')
            for j, tvec_aruco in enumerate(tvecs[len(marker_corners_n):]):  # Iterate over ArUco markers
                distance = np.linalg.norm(np.array(tvec_nfold) - np.array(tvec_aruco))  # Euclidean distance
                if distance < min_distance:
                    min_distance = distance
                    closest_aruco_id = marker_ids[j]  # Get the ArUco marker ID
            closest_aruco_ids.append(closest_aruco_id)
        


        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        with open(csv_file_name, "a",  newline="") as csvfile:
            writer = csv.writer(csvfile)
            for i, (tvec, rvec) in enumerate(zip(tvec_marker_frame, rvecs)):
                if i < len(marker_corners_n):
                    try:
                        marker_type = closest_aruco_ids[i] + 10
                        if marker_type is not None:
                            if marker_type == 10:
                                if "marker_type_10_counter" not in globals():
                                    global marker_type_10_counter
                                    marker_type_10_counter = 0
                                marker_type_10_counter += 1
                    except :
                        marker_type = 100
                else:   
                    marker_type = marker_ids[i-len(marker_corners_n)]
                    if marker_type == 0:
                        if "marker_type_0_counter" not in globals():
                            global marker_type_0_counter
                            marker_type_0_counter = 0
                        marker_type_0_counter += 1
                writer.writerow([frame_number,*tvec.flatten(),*rvec.flatten(), int(marker_type)])
        

        running_time = time.time() - start_time
        print(f"\rRunning total time: {running_time:.2f}, Frame number: {frame_number}", end='', flush=True)
        out.write(img_copy)
        cv2.namedWindow("sorted_pairs_test", cv2.WINDOW_NORMAL)
        # screen_width = 1280
        # screen_height = 720
        # cv2.resizeWindow("sorted_pairs_test", screen_width, screen_height)
        cv2.imshow("sorted_pairs_test", img_copy)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    if "marker_type_10_counter" in globals():
            print(f"marker_type_10_counter: {marker_type_10_counter}")
    if "marker_type_0_counter" in globals():
            print(f"marker_type_0_counter: {marker_type_0_counter}")
    print(f"Reference intensity: {np.mean(ref_intensity)}")
    
    #------------------------------------------------------------

    

main()