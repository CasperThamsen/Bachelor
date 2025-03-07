import cv2
import MarkerTracker
import numpy as np
from icecream import ic

def main():
    img = cv2.imread('/root/workspace/bachelor/nFoldMark/4o4hr.JPG')
    # cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    # cv2.imshow("input", img[:, :, 1])
    blur = np.random.normal(0.5,0.1,img.shape)
    img = (img * blur).astype(np.uint8)
    #unitycoin clean bob lesson1
    mt = MarkerTracker.MarkerTracker(order = 4, #number of shaded regions
                                     kernel_size=30,   #130
                                     scale_factor=100)
    mt.track_marker_with_missing_black_leg = False
    pose = mt.locate_marker(img[:, :, 1])

    poses, number_of_markers = mt.detect_multiple_markers(frame = img[:,:,1])
    summed_distances,distance_between_markers, middle_marker = mt.distances_between_markers(poses,number_of_markers)
    mt.numerate_markers_distance(poses,number_of_markers,summed_distances)
    test = mt.detect_marker_pair(poses,number_of_markers)

    
    # IC TESTS---------------------------------------------------
    # ic("distance",poses)
    ic(distance_between_markers)
    ic(summed_distances)
    ic(test)
    
    #warning if quality of a marker is low.
    for pose in poses:
        if(pose.quality < 0.5):
            ic("Pose quality is low", pose.quality)
    
    #opencv video capture

    #For distance
    sorted_poses = sorted(poses, key = lambda pose: pose.number)
    #Draws lines
    for i in range(len(sorted_poses)-1):
        current_pose = sorted_poses[i]
        next_pose = sorted_poses[i+1]
        cv2.line(img, (int(current_pose.x), int(current_pose.y)), (int(next_pose.x), int(next_pose.y)), (255, 255, 0), 2)
    #Draws numbers
    for pose in poses:
        for k in range (4):
            # cv2.circle(img, (int(pose.x + r * np.cos(pose.theta+k*np.pi/2)), int(pose.y + r * np.sin(pose.theta+k*np.pi/2))), 25, (0, 0, 255), 1)
            cv2.putText(img, str(pose.number), (int(pose.x), int(pose.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


    #code for orientation 
    mt.numerate_markers_orientation(poses,number_of_markers)
    # ic(poses)
    for pose in poses:
        for k in range (4): 
            cv2.putText(img, str(pose.number), (int(pose.x)-25, int(pose.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.namedWindow("Green = theta, teal = distance", cv2.WINDOW_NORMAL)
    screen_width = 1280
    screen_height = 800
    cv2.resizeWindow("Green = theta, teal = distance", screen_width, screen_height)
    cv2.imshow("Green = theta, teal = distance", img)


    while True:
        key = cv2.waitKey(0)  
        if key == ord('q'):   
            break





main()