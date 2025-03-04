import cv2
import MarkerTracker
import numpy as np
from icecream import ic

def main():
    img = cv2.imread('/root/workspace/bachelor/nFoldMark/5o4.JPG')
    # cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    # cv2.imshow("input", img[:, :, 1])
    blur = np.random.normal(0.5,0.1,img.shape)
    # ic(img)
    img = (img * blur).astype(np.uint8)
    # ic(img)
    
    mt = MarkerTracker.MarkerTracker(order = 4, #number of shaded regions
                                     kernel_size=120,   #130
                                     scale_factor=100)
    mt.track_marker_with_missing_black_leg = False
    pose = mt.locate_marker(img[:, :, 1])

    ic(pose)
    
   
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.imshow("output", mt.frame_sum_squared * 0.001)    #5(0.1), 4(0.01)(0.1) 7(0.1)               Hvilken er bedst?, punktet synes skarpere med værdi 2, ikke stor forskel i quality
    #ic(mt.frame_sum_squared)

    #Tuple unpacking
    #ic(markers) #orientation, quality has been put into function to handle printing values for multiple markers
    orient, number_of_markers, quality_q, location = mt.detect_multiple_markers(frame=img[:, :, 1])
    #distance_between_markers, summed_distances, middle_marker = mt.numerate_markers(frame=img[:, :, 1])

    r = 125
    
    # IC TESTS---------------------------------------------------
    #Multiple markers
    ic(number_of_markers,np.rad2deg(orient),quality_q,location)
    #ic(distance_between_markers, summed_distances)
    #location of middle marker
    # ic(location[middle_marker])
    # ic(middle_marker)
    # ic(quality_q)



    # Single marker test
    ic(mt.quality)
    # ic(mt.pose)
    # ic(np.rad2deg(mt.orientation))

    #print("markers: ", mt.detect_multiple_markers(frame=img[:, :, 1]))


    for loc,ori in zip(location,orient):

        cv2.circle(img, (int(loc[0]), int(loc[1])), 25, (0, 0, 255), 1)
        for k in range (4):
            cv2.circle(img, (int(loc[0] + r * np.cos(ori+k*np.pi/2)), int(loc[1] + r * np.sin(ori+k*np.pi/2))), 25, (0, 0, 255), 1)
    cv2.imshow("output", img)
    while True:
        key = cv2.waitKey(0)  # Wait indefinitely for a keypress
        if key == ord('q'):   # Check if the key pressed is 'q'
            break
    # while True:
    #     key = cv2.waitKey(0)  # Wait indefinitely for a keypress
    #     if key == ord('q'):   # Check if the key pressed is 'q'
    #         break





main()