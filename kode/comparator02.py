import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

def main():
    # Fra optitrack
    # epoch, x, y, z, rx, ry, rz
    filename1 = 'input/experiment_001.csv'
    data1 = np.loadtxt(filename1, 
                       delimiter=",", 
                       skiprows=7)
    #data1 = data1[0:10000, :]
    ic(data1[0:3, :])
    ic(data1.shape)

    filename2 = 'experiment_001_aruco_markers.csv'
    data2 = np.loadtxt(filename2, 
                       delimiter=",")
    data2 = data2[data2[:, 7] == 0]


    framerate = 30
    plt.plot(data1[:, 0] - data1[0, 0], 
             3 - data1[:, 8] / 180 * np.pi, '-c')
    plt.plot(data1[:, 0] - data1[0, 0], 
             data1[:, 9] / 180 * np.pi, '-m')
    plt.plot(data1[:, 0] - data1[0, 0], 
             -3*np.pi + data1[:, 10] / 180 * np.pi, '-y')
    plt.plot(data1[:, 0] - data1[0, 0], 
             data1[:, 11] - 9, '-r')
    plt.plot(data1[:, 0] - data1[0, 0], 
             data1[:, 12], '-g')
    plt.plot(data1[:, 0] - data1[0, 0], 
             data1[:, 13] + 10, '-b')

    scale = 8
    offset = 770
    plt.plot((data2[:, 0] - data2[0, 0]) * scale + offset, 
             data2[:, 1] - 6, '.r')
    plt.plot((data2[:, 0] - data2[0, 0]) * scale + offset, 
             data2[:, 2] + 1, '.g')
    plt.plot((data2[:, 0] - data2[0, 0]) * scale + offset, 
             data2[:, 3] + 7.5, '.b')
    plt.plot((data2[:, 0] - data2[0, 0]) * scale + offset, 
             data2[:, 4] + 2*np.pi * (data2[:, 4] < 1.5), '.c')
    plt.plot((data2[:, 0] - data2[0, 0]) * scale + offset, 
             -2.5*np.pi + data2[:, 5], '.y')
    plt.plot((data2[:, 0] - data2[0, 0]) * scale + offset, 
             data2[:, 6] + 7, '.m')

    plt.show()

    

main()
