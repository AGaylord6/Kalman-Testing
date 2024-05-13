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
wrong orientation after hfuncing??
how is B_true connected to our starting state + initial eoms progagation

do we trust our process or measurements more?
    should be similar?

generate fake imu data using matlab functionality??

'''

def plot3D(vectors, plotSegment):
    # plotSegment: 3 digit number 'nmi'
    #       split plot into n by m plots and picks ith one
    #       ex: 121 splits into 1 row, 2 columns and picks the first (the left half)

    bounds = [-2, 2]

    result = np.array([np.concatenate(([0, 0, 0], vectors[0]))])
    print(result)

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

if __name__ == "__main__":
    

    ukf = Filter(200, 0.1, 7, 6, 0, 0, np.array([1, 0, 0]), np.array([0, 0, 0]), UKF)

    # set process noise
    ukf.ukf_setQ(.005, 10)

    # set measurement noise
    ukf.ukf_setR(.2, .02)

    ideal_reaction_speeds = ukf.generateSpeeds(1000, -1000, ukf.n/2, 100, np.array([0, 0, 1]))
    # print(ideal_reaction_speeds[:20])

    ideal = ukf.propagate(ideal_reaction_speeds)
    # print("state: ", ideal[:10])

    magNoises = np.random.normal(0, .01, (ukf.n, 3))
    gyroNoises = np.random.normal(0, .001, (ukf.n, 3))

    data = ukf.generateData(ideal, magNoises, gyroNoises, 0)
    # print("data: ", data[:20])

    # number of vectors to plot
    numVectors = 5

    plot3D(np.array([ukf.B_true, data[50][:3], data[100][:3], data[150][:3]]), 121)


    filtered = ukf.simulate(data, ideal_reaction_speeds)


    # ukf.plotResults(ideal)
    # ukf.plotResults(filtered)




