import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import rotationFix as RF


#this file is for handling experiment files which are all from the seconds visit to the airport
#They are formatted differently from the first visit.
#Frame,Time (Seconds),RX,RZ,RY,X,Y,Z,RX,RZ,RY,X,Y,Z

file_name = "experiment_005"

with open(f"airporttestfiles/{file_name}.csv", "r") as opti_file:
    reader = csv.reader(opti_file)
    header = next(reader)  # Skip the header row
    for _ in range(6):
        next(reader)
    data = [row for row in reader]




with open(f"airporttestfiles/{file_name}"+"output.csv", 'w',newline='') as opti_output:
    writer = csv.writer(opti_output)
    for row in data:
        time = float(row[0])
        x1, y1, z1 = map(float, row[5:8])
        x2, y2, z2 = map(float, row[11:14])
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1


        rx1,rz1,ry1 = map(float, row[2:5])
        rx2,rz2,ry2 = map(float, row[8:11])
        R_phone = RF.euler_to_rotation_matrix(rx2, ry2, rz2)
        R_board = RF.euler_to_rotation_matrix(rx1, ry1, rz1)
        #convert euler to rotation matrix
        
        R_relative = R_board.T @ R_phone
        relative_rotation_vector, _ = cv2.Rodrigues(R_relative)
        r = relative_rotation_vector.flatten()

        R = RF.rvec_to_euler(relative_rotation_vector)
        writer.writerow([time, dx, dy, dz, *r])




    