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
from mpl_toolkits.mplot3d import Axes3D

'''

https://cs.adelaide.edu.au/~ianr/Teaching/Estimation/LectureNotes2.pdf
https://github.com/FrancoisCarouge/Kalman
https://www.researchgate.net/post/How_can_I_validate_the_Kalman_Filter_result
https://stats.stackexchange.com/questions/40466/how-can-i-debug-and-check-the-consistency-of-a-kalman-filter
file:///C:/Users/andre/Downloads/WikibookonKalmanFilter.pdf


EOMs are messed up. changed rw_config back to gamma and such
    change I_trans from non-zero?

must ensure quaternion is normalized

how is B_true connected to our starting state + initial eoms progagation

generate fake imu data using matlab functionality??

flesh out 3D graphs more: colors, many at once (ideal, data, filtered)

'''

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

    plt.show()


def plotData(data, numVectors, plotSegment):
    # print numVector elements of data, divided evenly

    section = int(len(data) / (2*numVectors))

    result = np.array([data[0][:3]])

    for i in range(1, numVectors):
        result = np.append(result, np.array([data[i*section][:3]]), axis=0)
    
    print(result)

    # plot3DVectors(np.array([ukf.B_true, data[50][:3], data[100][:3], data[150][:3]]), 121)
    plot3DVectors(result, 111)

def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


if __name__ == "__main__":
    

    ukf = Filter(180, 0.1, 7, 6, 0, 0, np.array([-1, 0, 0]), np.array([0, 0, 0]), UKF)

    # set process noise
    ukf.ukf_setQ(.01, 10)

    # set measurement noise
    ukf.ukf_setR(.1, .1)


    ideal_reaction_speeds = ukf.generateSpeeds(900, -900, ukf.n/2, 100, np.array([0, 0, 1]))
    # print(ideal_reaction_speeds[:20])

    ideal = ukf.propagate(ideal_reaction_speeds)
    # print("state: ", ideal[:10])

    magNoises = np.random.normal(0, .001, (ukf.n, 3))
    gyroNoises = np.random.normal(0, .001, (ukf.n, 3))

    data = ukf.generateData(ideal, magNoises, gyroNoises, 0)

    # # plot3DVectors(np.array([ukf.B_true, data[50][:3], data[100][:3], data[150][:3]]), 121)
    # plot3DVectors(result, 111)

    # plotData(data, 5, 111)

    # ideal_xyz = [np.matmul(quaternion_rotation_matrix(x), np.array([1, 0, 0])) for x in ideal]
    # plotData(ideal_xyz, 3, 111)

    filtered = ukf.simulate(data, ideal_reaction_speeds)


    # ukf.plotResults(ideal)
    ukf.plotResults(filtered)




