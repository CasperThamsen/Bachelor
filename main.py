import cv2
import MarkerTracker
import numpy as np
from icecream import ic

def main():
    img = cv2.imread('/root/workspace/bachelor/nFoldMark/5o4hr.JPG')
    # cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    # cv2.imshow("input", img[:, :, 1])
    blur = np.random.normal(0.5,0.1,img.shape)
    # ic(img)
    img = (img * blur).astype(np.uint8)
    # ic(img)
    #unitycoin clean bob lesson1
    mt = MarkerTracker.MarkerTracker(order = 4, #number of shaded regions
                                     kernel_size=30,   #130
                                     scale_factor=100)
    mt.track_marker_with_missing_black_leg = False
    pose = mt.locate_marker(img[:, :, 1])

    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.imshow("output", mt.frame_sum_squared * 0.001)    #5(0.1), 4(0.01)(0.1) 7(0.1)               Hvilken er bedst?, punktet synes skarpere med værdi 2, ikke stor forskel i quality
    #ic(mt.frame_sum_squared)

    #Tuple unpacking
    #ic(markers) #orientation, quality has been put into function to handle printing values for multiple markers
    poses, number_of_markers = mt.detect_multiple_markers(frame = img[:,:,1])
    summed_distances = mt.distances_between_markers(poses,number_of_markers)
    test = mt.numerate_markers(poses,number_of_markers,summed_distances)

    
    # IC TESTS---------------------------------------------------
    ic("distance",poses)
    
    #opencv video capture
    r = 100 #Hvor langt væk cirkerne skal være fra midten af markøren
    prev_pose = None
    for pose in poses:

        # cv2.circle(img, (int(pose.x), int(pose.y)), 25, (0, 0, 255), 1)
        for k in range (4):
            # cv2.circle(img, (int(pose.x + r * np.cos(pose.theta+k*np.pi/2)), int(pose.y + r * np.sin(pose.theta+k*np.pi/2))), 25, (0, 0, 255), 1)
            cv2.putText(img, str(pose.number), (int(pose.x), int(pose.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    poses = mt.numerate_markers_orientation(poses,number_of_markers)
    ic(poses)
    for pose in poses:
        for k in range (4): 
            cv2.putText(img, str(pose.number), (int(pose.x)-25, int(pose.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Green = theta, teal = distance", img)


    while True:
        key = cv2.waitKey(0)  
        if key == ord('q'):   
            break





main()