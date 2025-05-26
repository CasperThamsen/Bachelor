import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2

#this file is for handling experiment files which are all from the first visit to the airport
#They are formatted differently from the second visit.
#time(unix),X,Y,Z,X,Y,Z,X,Y,Z,X,Y,Z


file_name = "experiment_004"

with open(f"airporttestfiles/{file_name}.csv", "r") as opti_file:
    reader = csv.reader(opti_file)
    header = next(reader)  # Skip the header row
    data = [row for row in reader]
    start_time = float(data[0][0])
    for row in data:
        row[0] = int(round((float(row[0]) - start_time) * 30))
    # Remove duplicate frames based on frame number (row[0])
    unique_data = []
    seen_frames = set()
    for row in data:
        if row[0] not in seen_frames:
            unique_data.append(row)
            seen_frames.add(row[0])
    data = unique_data




with open(f"airporttestfiles/{file_name}"+"output.csv", 'w',newline='') as opti_output:
    writer = csv.writer(opti_output)
    for row in data:
        time = float(row[0])
        x1, y1, z1 = map(float, row[1:4])
        x2,y2,z2 = map(float, row[7:10])
        dx = x2 - x1
        dy = z2 - z1
        dz = y2 - y1        


        rx1,ry1,rz1 = map(float, row[4:7])
        rx2,ry2,rz2 = map(float, row[10:13])
        # board_rotation = np.array([rx2, rz2,ry2])
        # phone_rotation = np.array([rx1, rz1,ry1 ])
        #Skulle gerne konvertere til rotation matrix, sammenligne dem og konvertere til vektor igen
        # R_board, _ = cv2.Rodrigues(board_rotation)
        # R_phone, _ = cv2.Rodrigues(phone_rotation)
        # R_relative = R_board.T @ R_phone.T
        # relative_rotation_vector, _ = cv2.Rodrigues(R_relative)
        # r = relative_rotation_vector.flatten()
        writer.writerow([time, dx, dy, dz, rx1, ry1,rz1])




    