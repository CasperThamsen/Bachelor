import csv

with open("airporttestfiles/5markerrotation.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row
    data = [row for row in reader]

with open("airporttestfiles/5markerrotationoutput.csv", 'w',newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["t,dx","dy","dz","rx","ry","rz"])
    time_start = float(data[0][0])
    for row in data:
        time = float(row[0])-time_start
        x1, y1, z1 = map(float, row[1:4])
        x2,y2,z2 = map(float, row[7:10])
        rx1,ry1,rz1 = map(float, row[4:7])
        rx2,ry2,rz2 = map(float, row[10:13])
        dx = x2 - x1
        dy = z2 - z1
        dz = y2 - y1
        rx = rx2 - rx1
        ry = rz2 - rz1
        rz = ry2 - ry1
        writer.writerow([time,dx,dy,dz,rx,ry,rz])



    