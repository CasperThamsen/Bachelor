import cv2
import MarkerTracker as MarkerTracker
import numpy as np
from icecream import ic
import time

def main():
    cap = cv2.VideoCapture('/root/workspace/bachelor/nFoldMarkers/irl/vid.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    out = cv2.VideoWriter('summethodtested.mp4', 
                        cv2.VideoWriter_fourcc(*'XVID'), 
                        20.0, 
                        frame_size)
    while cap.isOpened():
        time_start = time.time()
        # img = cv2.imread('/root/workspace/bachelor/nFoldMarkers/irl/medium.JPG')
        ret, img = cap.read()
        if not ret:
            ic("Failed to capture image")
            break
        # blur = np.random.normal(0.5,0.1,img.shape)
        # img = (img * blur).astype(np.uint8)
        #unitycoin clean bob lesson1
        mt = MarkerTracker.MarkerTracker(order = 4, #number of shaded regions
                                        kernel_size=30,
                                        scale_factor=100)
        mt.track_marker_with_missing_black_leg = False

        mt.locate_marker_init(img[:,:,1])
        poses, number_of_markers = mt.detect_multiple_markers(frame = img[:,:,1])
        distance_between_markers = mt.distance_between_markers(poses,number_of_markers)
        marker_pairs = mt.detect_marker_pairs(poses,distance_between_markers)
        # marker_pairs,number_of_pairs = mt.detect_marker_pair(poses,marker_combinations) #sum method
        mt.numerate_markers_distance(marker_pairs)

    
        # IC TESTS---------------------------------------------------
        # ic("distance",poses)
        # ic(distance_between_markers)
        # ic(marker_pairs)
        # ic(number_of_pairs)

        #warning if quality of a marker is low.
        # for pose in poses:
        #     if(pose.quality < 0.5):
        #         ic("Pose quality is low", pose.quality)
        
        #opencv video capture


        #------------------------------------------------------------
        #This section tests the detect_marker_pair function
        img_pairs_copy=img.copy()
        for pair in marker_pairs:
            sorted_pair = sorted(pair, key=lambda pose: pose.number)  # Sort markers by their number
            for i in range(len(sorted_pair)-1):
                cv2.line(img_pairs_copy, (int(sorted_pair[i].x), int(sorted_pair[i].y)), (int(sorted_pair[(i+1)].x), int(sorted_pair[(i+1)].y)), (0, 255, 0), 4)
            for i in range(len(sorted_pair)):
                cv2.putText(img_pairs_copy, str(sorted_pair[i].number), (int(sorted_pair[i].x), int(sorted_pair[i].y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        for pose in poses:
            cv2.circle(img_pairs_copy, (int(pose.x), int(pose.y)), 5, (0, 0, 255), -1)


        out.write(img_pairs_copy)
        cv2.namedWindow("sorted_pairs_test", cv2.WINDOW_NORMAL)
        screen_width = 1280
        screen_height = 720
        cv2.resizeWindow("sorted_pairs_test", screen_width, screen_height)
        time_end = time.time()
        ic("Time used: main", time_end-time_start)
        cv2.imshow("sorted_pairs_test", img_pairs_copy)
        if number_of_markers < 20:
            if cv2.waitKey(0) == ord('q'):
                break
        if number_of_markers == 20:
            if cv2.waitKey(1) == ord('q'):
                break

        if number_of_markers > 20:
            if cv2.waitKey(0) == ord('q'):
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    #------------------------------------------------------------

main()