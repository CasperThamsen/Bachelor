import cv2
import MarkerTracker as MarkerTracker
import numpy as np
from icecream import ic
import time

#videostream imports
import socket
import pickle
import struct



def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('172.17.192.1', 5000))  # IP address of the Windows machine streaming the webcam

    mt = MarkerTracker.MarkerTracker(order=4,  # number of shaded regions
                                      kernel_size=30,
                                      scale_factor=100)
    mt.track_marker_with_missing_black_leg = False
    frame_size = (1280, 720)
    out = cv2.VideoWriter('livetestwnoice.mp4', 
                          cv2.VideoWriter_fourcc(*'XVID'), 
                          20.0, 
                          frame_size)

    while True:
        # Receive the size of the message (frame size)
        data = client_socket.recv(4)
        if not data:
            print("No data received, closing connection.")
            break
        
        # Unpack the message size (4 bytes for 32-bit integer)
        message_size = struct.unpack("I", data)[0]
        
        # Receive the actual frame data
        data = b""
        while len(data) < message_size:
            packet = client_socket.recv(message_size - len(data))
            if not packet:
                break
            data += packet
        # Deserialize the frame data (Pickle)
        img = pickle.loads(data)

        mt.locate_marker_init(frame=img[:,:,1])
        poses, number_of_markers = mt.detect_multiple_markers(frame=img[:,:,1])
        distance_between_markers = mt.distances_between_markers(poses, number_of_markers)
        marker_pairs = mt.detect_marker_pairs(poses, distance_between_markers)
        mt.numerate_markers_distance(marker_pairs)

        img_pairs_copy = img.copy()
        for pair in marker_pairs:
            sorted_pair = sorted(pair, key=lambda pose: pose.number)
            for i in range(len(sorted_pair) - 1):
                cv2.line(img_pairs_copy, (int(sorted_pair[i].x), int(sorted_pair[i].y)),(int(sorted_pair[(i + 1)].x), int(sorted_pair[(i + 1)].y)), (0, 255, 0), 4)
            for i in range(len(sorted_pair)):
                cv2.putText(img_pairs_copy, str(sorted_pair[i].number), (int(sorted_pair[i].x), int(sorted_pair[i].y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        for pose in poses:
            cv2.circle(img_pairs_copy, (int(pose.x), int(pose.y)), 5, (0, 0, 255), -1)

        out.write(img_pairs_copy)
        cv2.namedWindow("sorted_pairs_test", cv2.WINDOW_NORMAL)
        screen_width = 1280
        screen_height = 720
        cv2.resizeWindow("sorted_pairs_test", screen_width, screen_height)
        cv2.imshow("sorted_pairs_test", img_pairs_copy)

        if cv2.waitKey(1) == ord('q'):
            break

    client_socket.close()
    out.release()
    cv2.destroyAllWindows()

main()