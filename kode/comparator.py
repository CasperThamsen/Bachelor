import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

def main():
    # Fra optitrack
    # epoch, x, y, z, rx, ry, rz
    filename1 = 'input/experiment_002.csv'
    data1 = np.loadtxt(filename1, 
                       delimiter=",", 
                       skiprows=7)
    #data1 = data1[0:10000, :]
    ic(data1[0:3, :])
    ic(data1.shape)

    framerate = 30
    plt.plot(data1[:, 0] - data1[0, 0], 
             data1[:, 8] / 180 * np.pi, '-r')
    plt.plot(data1[:, 0] - data1[0, 0], 
             data1[:, 9] / 180 * np.pi, '-g')
    plt.plot(data1[:, 0] - data1[0, 0], 
             data1[:, 10] / 180 * np.pi, '-b')
    plt.plot(data1[:, 0] - data1[0, 0], 
             data1[:, 11], '.r')
    plt.plot(data1[:, 0] - data1[0, 0], 
             data1[:, 12], '.g')
    plt.plot(data1[:, 0] - data1[0, 0], 
             data1[:, 13], '.b')
    plt.show()

    

main()
