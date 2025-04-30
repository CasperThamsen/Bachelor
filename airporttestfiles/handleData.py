import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2

with open("airporttestfiles/5markerrotation2.csv", "r") as opti_file:
    reader = csv.reader(opti_file)
    header = next(reader)  # Skip the header row
    data = [row for row in reader]




with open("airporttestfiles/5markerrotation2output.csv", 'w',newline='') as opti_output:
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
        board_rotation = np.array([rx1, rz1,ry1])
        phone_rotation = np.array([rx2, rz2,ry2 ])
        R_board, _ = cv2.Rodrigues(board_rotation)
        R_phone, _ = cv2.Rodrigues(phone_rotation)
        R_relative = np.dot(R_phone, R_board.T)
        relative_rotation_vector, _ = cv2.Rodrigues(R_relative)
        r = relative_rotation_vector.flatten()
        writer.writerow([time, dx, dy, dz, r[0], r[1], r[2]])




    