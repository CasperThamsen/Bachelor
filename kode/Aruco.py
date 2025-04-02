import cv2
import icecream as ic
import cv2.aruco as aruco
import numpy as np

def main():
    detector_params = aruco.DetectorParameters()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = aruco.ArucoDetector(dictionary, detector_params)
    calibration_data = np.load('calibration_data.npz')
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']

    
    marker_length = 0.094
    obj_points = np.array([
        [-marker_length / 2, marker_length / 2, 0],  # Top-left corner
        [marker_length / 2, marker_length / 2, 0],   # Top-right corner
        [marker_length / 2, -marker_length / 2, 0],  # Bottom-right corner
        [-marker_length / 2, -marker_length / 2, 0]  # Bottom-left corner
    ], dtype=np.float32)

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            ic("Failed to capture image")
            break


        marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(img)

        # Draw detected markers on the image
        if marker_ids is not None:
            aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
            rvecs = []            
            tvecs = []
            text_offset = 0

            for marker_corner in marker_corners:
                img_points = np.array(marker_corner[0], dtype=np.float32)
                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    # Draw axis for each detected marker
                    axis_length = marker_length / 2
                    axis_points = np.array([ 
                        [axis_length, 0, 0],    # X-axis
                        [0, axis_length, 0],    # Y-axis
                        [0, 0, axis_length]     # Z-axis
                    ], dtype=np.float32)

                    cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, marker_length*1.5,2)

                    # Display the translation vector (tvec) on the image
                    cv2.putText(img, f"tvec: {tvec.flatten()}", (10, 30 + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    text_offset += 20
        cv2.imshow("Detected Markers", img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


main()