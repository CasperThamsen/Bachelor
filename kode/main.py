import cv2
import MarkerTracker as MarkerTracker
import numpy as np
from icecream import ic
import time

def main():
    time_start = time.time()
    img = cv2.imread('/root/workspace/bachelor/bachelor/nFoldMarkers/pc/4.JPG')
    # blur = np.random.normal(0.5,0.1,img.shape)
    # img = (img * blur).astype(np.uint8)
    #unitycoin clean bob lesson1
    mt = MarkerTracker.MarkerTracker(order = 4, #number of shaded regions
                                     kernel_size=30,
                                     scale_factor=100)
    mt.track_marker_with_missing_black_leg = False

    #pose = mt.locate_marker(img[:, :, 1])
    poses, number_of_markers = mt.detect_multiple_markers(frame = img[:,:,1])
    marker_combinations = mt.generate_pair_combinations(number_of_markers)
    marker_pairs,number_of_pairs = mt.detect_marker_pair(poses,marker_combinations)
    mt.numerate_markers_distance(marker_pairs)



    
    # IC TESTS---------------------------------------------------
    # ic("distance",poses)
    # ic(distance_between_markers)
    ic(marker_pairs)
    ic(number_of_pairs)

    #warning if quality of a marker is low.
    for pose in poses:
        if(pose.quality < 0.5):
            ic("Pose quality is low", pose.quality)
    
    #opencv video capture


    #------------------------------------------------------------
    #This section tests the detect_marker_pair function
    img_pairs_copy=img.copy()
    for pair in marker_pairs:
        sorted_pair = sorted(pair, key=lambda pose: pose.number)  # Sort markers by their number
        for i in range(len(sorted_pair)-1):
            cv2.line(img_pairs_copy, (int(sorted_pair[i].x), int(sorted_pair[i].y)), (int(sorted_pair[(i+1)].x), int(sorted_pair[(i+1)].y)), (0, 255, 0), 2)
            cv2.putText(img_pairs_copy, str(sorted_pair[i].number), (int(sorted_pair[i].x), int(sorted_pair[i].y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.namedWindow("sorted_pairs_test", cv2.WINDOW_NORMAL)
    screen_width = 960
    screen_height = 640
    cv2.resizeWindow("sorted_pairs_test", screen_width, screen_height)
    time_end = time.time()
    ic("Time used: main", time_end-time_start)
    cv2.imshow("sorted_pairs_test", img_pairs_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #------------------------------------------------------------


main()