import cv2
import MarkerTracker as MarkerTracker
import numpy as np
from icecream import ic

def main():
    # Load calibration data
    calibration_data = np.load('calibration_data.npz')
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    out = cv2.VideoWriter('livetest2.mp4', 
                        cv2.VideoWriter_fourcc(*'XVID'), 
                        20.0, 
                        frame_size)
    mt = MarkerTracker.MarkerTracker(order = 4, #number of shaded regions
                                kernel_size=25,
                                scale_factor=100)
    mt.track_marker_with_missing_black_leg = False
    mt.expected_ratios = [1.000000, 1.438993, 1.735489, 2.020540, 2.144925, 2.096221, 2.999138, 2.999155, 2.096223, 2.144939]

    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            ic("Failed to capture image")
            break
        #unitycoin clean bob lesson1
        #Initialize MarkerTracker functionality
        mt.locate_marker_init(frame = img[:,:,1])
        poses, number_of_markers = mt.detect_multiple_markers(frame=img[:,:,1])
        distance_between_markers = mt.distances_between_markers(poses,number_of_markers)
        marker_pairs = mt.detect_marker_pairs(poses,distance_between_markers)
        mt.numerate_markers()
        corners = mt.marker_cornors(marker_pairs)




        #Draw on canvas
        img_pairs_copy=img.copy()
        for pair in marker_pairs:
            sorted_pair = sorted(pair, key=lambda pose: pose.number)  # Sort markers by their number
            for i in range(len(sorted_pair)-1):
                cv2.line(img_pairs_copy, (int(sorted_pair[i].x), int(sorted_pair[i].y)), (int(sorted_pair[(i+1)].x), int(sorted_pair[(i+1)].y)), (0, 255, 0), 4)
            for i in range(len(sorted_pair)):
                cv2.putText(img_pairs_copy, str(sorted_pair[i].number), (int(sorted_pair[i].x), int(sorted_pair[i].y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        for pose in poses:
            if pose not in corners:
                cv2.circle(img_pairs_copy, (int(pose.x), int(pose.y)), 5, (0, 0, 255), -1)
        for _, corner in enumerate(corners):
            cv2.circle(img_pairs_copy, (int(corners[0].x), int(corners[0].y)), 5, (255, 0, 0), -1)

        out.write(img_pairs_copy)
        cv2.namedWindow("sorted_pairs_test", cv2.WINDOW_NORMAL)
        screen_width = 1280
        screen_height = 720
        cv2.resizeWindow("sorted_pairs_test", screen_width, screen_height)
        cv2.imshow("sorted_pairs_test", img_pairs_copy)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    #------------------------------------------------------------
    

main()