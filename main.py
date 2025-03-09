import cv2
import MarkerTracker
import numpy as np
from icecream import ic

def main():
    img = cv2.imread('/root/workspace/bachelor/nFoldMark/4x5o4h.JPG')
    # cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    # cv2.imshow("input", img[:, :, 1])
    blur = np.random.normal(0.5,0.1,img.shape)
    img = (img * blur).astype(np.uint8)
    #unitycoin clean bob lesson1
    mt = MarkerTracker.MarkerTracker(order = 4, #number of shaded regions
                                     kernel_size=30,   #130
                                     scale_factor=100)
    mt.track_marker_with_missing_black_leg = False

    #pose = mt.locate_marker(img[:, :, 1])
    poses, number_of_markers = mt.detect_multiple_markers(frame = img[:,:,1])
    # mt.numerate_markers_distance(poses,number_of_markers,all_info)
    marker_combinations = mt.generate_pair_combinations(number_of_markers)
    marker_pairs,number_of_pairs = mt.detect_marker_pair(poses,marker_combinations)


    
    # IC TESTS---------------------------------------------------
    # ic("distance",poses)
    # ic(distance_between_markers)
    # ic(marker_pairs)
    
    #warning if quality of a marker is low.
    for pose in poses:
        if(pose.quality < 0.5):
            ic("Pose quality is low", pose.quality)
    
    #opencv video capture

    ic(marker_pairs)
    #------------------------------------------------------------
    #This section tests the detect_marker_pair function
    img_pairs_copy=img.copy()
    mt.numerate_markers_orientation(marker_pairs)
    for pair in marker_pairs:
        sorted_pair = sorted(pair, key=lambda pose: pose.number)  # Sort markers by their number
        for i in range(len(sorted_pair)):
            start_pose = sorted_pair[i]
            end_pose = sorted_pair[(i + 1) % len(sorted_pair)]  # Wrap around to the first marker
            cv2.line(img_pairs_copy, (int(start_pose.x), int(start_pose.y)), (int(end_pose.x), int(end_pose.y)), (0, 255, 0), 2)
            cv2.putText(img_pairs_copy, str(start_pose.number), (int(start_pose.x), int(start_pose.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.namedWindow("sorted_pairs_test", cv2.WINDOW_NORMAL)
    screen_width = 1280
    screen_height = 800
    cv2.resizeWindow("sorted_pairs_test", screen_width, screen_height)
    cv2.imshow("sorted_pairs_test", img_pairs_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #------------------------------------------------------------



    while True:
        key = cv2.waitKey(0)  
        if key == ord('q'):   
            break





main()