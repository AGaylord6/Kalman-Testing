'''
main.py
Author: Andrew Gaylord

uses fake models to simulate and compare different kalman filters

'''

import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

from irishsat_ukf.PySOL.wmm import *
from irishsat_ukf.simulator import *
from irishsat_ukf.UKF_algorithm import *
from irishsat_ukf.hfunc import *
from filter import *

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import signal

'''

https://cs.adelaide.edu.au/~ianr/Teaching/Estimation/LectureNotes2.pdf
https://github.com/FrancoisCarouge/Kalman
https://www.researchgate.net/post/How_can_I_validate_the_Kalman_Filter_result
https://stats.stackexchange.com/questions/40466/how-can-i-debug-and-check-the-consistency-of-a-kalman-filter
file:///C:/Users/andre/Downloads/WikibookonKalmanFilter.pdf

notes:
EOMs are messed up. changed rw_config back to gamma and such
    change I_trans from non-zero?

must ensure quaternion is normalized

issues:
    how is B_true connected to our starting state + initial eoms progagation
        need to check other B_trues and rotations about other axis (like y)
    
    was is orientation of simulation?? what are we enforcing with starting quaternion??

TODO:
    statistical tests
    effiency test/graphs (read article)
    plotting comparisons between filter/unfiltered
    figure out what to plot for 3D graphs (time dependent?)

optional:
    more comprehensive plotting: wrappers, options, make part of Filter class
    flesh out 3D graphs more: colors, many at once (ideal, data, filtered)
    switch to euler angles instead of xyz?
    generate fake imu data using matlab functionality??

'''


def signal_handler(sig, frame):
    '''
    signal_handler

    closes all pyplot tabs
    '''

    plt.close('all')


def plot_multiple_lines(data, labels, title, x, y):
    """Plots multiple lines on the same graph.

    Args:
        data: A list of lists of data points.
        labels: A list of labels for each line.
        title: title for graph
    """

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot each line.
    for i, line in enumerate(data):
        ax.plot(line, label=labels[i])

    # Add a legend
    ax.legend()

    plt.title(title)

    move_figure(fig, x, y)

    # Show the plot
    # plt.show()


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def plot3DVectors(vectors, plotSegment):
    # plotSegment: 3 digit number 'nmi'
    #       split plot into n by m plots and picks ith one
    #       ex: 121 splits into 1 row, 2 columns and picks the first (the left half)

    bounds = [-1.5, 1.5]

    result = np.array([np.concatenate(([0, 0, 0], vectors[0]))])
    # print(result)

    for i in range (1, len(vectors)):
        v = vectors[i]

        result = np.append(result, np.array([np.concatenate(([0, 0, 0], v))]), axis=0)


    # original = np.concatenate(([0, 0, 0], vectors[0]))
    # rotated = np.concatenate(([0, 0, 0], vectors[1]))
    # rotated2 = np.concatenate(([0, 0, 0], vectors[2]))
    # soa = np.array([original, rotated])
    # X, Y, Z, U, V, W = zip(*soa)
    # print(X, Y, Z, U, V, W)

    fig = plt.figure()
    ax = fig.add_subplot(plotSegment, projection='3d')
    # ax.quiver(X, Y, Z, U, V, W, cmap=cm.coolwarm)
    # print(np.array([original, rotated, rotated2]))
    # ax.quiver(*[x for x in zip(*np.array([original, rotated, rotated2]))])
    ax.quiver(*[x for x in zip(*result)])

    ax.set_xlim([bounds[0], bounds[1]])
    ax.set_ylim([bounds[0], bounds[1]])
    ax.set_zlim([bounds[0], bounds[1]])

    # plt.show()


def plotData3D(data, numVectors, plotSegment):
    # print numVector elements of data, divided evenly

    section = int(len(data) / (2*numVectors))

    result = np.array([data[0][:3]])

    for i in range(1, numVectors):
        result = np.append(result, np.array([data[i*section][:3]]), axis=0)
    
    print(result)

    # plot3DVectors(np.array([ukf.B_true, data[50][:3], data[100][:3], data[150][:3]]), 121)
    plot3DVectors(result, 111)


def plot_xyz(data, title, x, y):
     # given a numpy 2D list (where every element contains x, y, z), plot them on graph

    newData = data.transpose()

    if len(data[0]) == 4:
        plot_multiple_lines(newData, ["a", "b", "c", "d"], title, x, y)
    else:
        plot_multiple_lines(newData, ["x", "y", "z"], title, x, y)


def plotState_xyz(data):
    # plots our 7 dimesnional state on 2 graphs: quaternion and angular velocity

    quaternions = np.array([data[0][:4]])
    velocities = np.array([data[0][4:]])
    for i in range(1, len(data)):
        quaternions = np.append(quaternions, np.array([data[i][:4]]), axis=0)
        velocities = np.append(velocities, np.array([data[i][4:]]), axis=0)

    plot_xyz(velocities, "Angular Velocity", 50, 50)
    plot_xyz(quaternions, "Quaternion", 0, 0)


def plotData_xyz(data):

    magData = np.array([data[0][:3]])
    gyroData = np.array([data[0][3:]])
    for i in range(1, len(data)):
        magData = np.append(magData, np.array([data[i][:3]]), axis=0)
        gyroData = np.append(gyroData, np.array([data[i][3:]]), axis=0)

    plot_xyz(gyroData, "Gyroscope", 1050, 50)
    plot_xyz(magData, "Magnetometer", 1000, 0)


if __name__ == "__main__":

    # set up signal handler to shut down pyplot tabs
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))

    ukf = Filter(150, 0.1, 7, 6, 0, 0, np.array([-1, 0, 0]), np.array([0, 0, 0]), UKF)

    # set process noise
    ukf.ukf_setQ(.01, 10)

    # set measurement noise
    ukf.ukf_setR(.25, .5)


    ideal_reaction_speeds = ukf.generateSpeeds(1300, -1300, ukf.n, 100, np.array([0, 0, 1]))
    # print(ideal_reaction_speeds[:20])

    ideal = ukf.propagate(ideal_reaction_speeds)
    # print("state: ", ideal[:10])

    magNoises = np.random.normal(0, .001, (ukf.n, 3))
    gyroNoises = np.random.normal(0, .001, (ukf.n, 3))

    data = ukf.generateData(ideal, magNoises, gyroNoises, 0)

    filtered = ukf.simulate(data, ideal_reaction_speeds)

    # # plot3DVectors(np.array([ukf.B_true, data[50][:3], data[100][:3], data[150][:3]]), 121)
    # plot3DVectors(result, 111)

    # plotData3D(data, 5, 111)

    # ideal_xyz = [np.matmul(quaternion_rotation_matrix(x), np.array([1, 0, 0])) for x in ideal]
    # plotData3D(ideal_xyz, 3, 111)

    plotState_xyz(ideal)
    plotData_xyz(data)

    # only show plot at end so they all show up

    ukf.visualizeResults(ideal)
    plt.show()
    # ukf.visualizeResults(filtered)




