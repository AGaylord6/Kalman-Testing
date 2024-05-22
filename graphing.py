''''
graphing.py
Author: Andrew Gaylord

contains the graphing functionality for kalman data visualization

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


def plot_multiple_lines(data, labels, title, x, y):
    """Plots multiple lines on the same graph.

    Args:
        data: A list of lists of data points.
        labels: A list of labels for each line.
        title: title for graph
        x, y: pixel location on screen
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
    """Move figure's upper left corner to pixel (x, y) from stackoverflow"""
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


def plotState_xyz(data, ideal):
    # plots our 7 dimesnional state on 2 graphs: quaternion and angular velocity
    # ideal: true if top left, false if bottom middle

    quaternions = np.array([data[0][:4]])
    velocities = np.array([data[0][4:]])
    for i in range(1, len(data)):
        quaternions = np.append(quaternions, np.array([data[i][:4]]), axis=0)
        velocities = np.append(velocities, np.array([data[i][4:]]), axis=0)

    if ideal:
        plot_xyz(velocities, "Angular Velocity", 50, 0)
        plot_xyz(quaternions, "Ideal Quaternion", 0, 0)
    else:
        plot_xyz(velocities, "Angular Velocity", 575, 370)
        plot_xyz(quaternions, "Filtered Quaternion", 525, 370)


def plotData_xyz(data):

    magData = np.array([data[0][:3]])
    gyroData = np.array([data[0][3:]])
    for i in range(1, len(data)):
        magData = np.append(magData, np.array([data[i][:3]]), axis=0)
        gyroData = np.append(gyroData, np.array([data[i][3:]]), axis=0)

    plot_xyz(gyroData, "Gyroscope", 1100, 0)
    plot_xyz(magData, "Magnetometer", 1050, 0)