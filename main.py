'''
main.py
Author: Andrew Gaylord

Main file for Kalman-Testing repo

Sets up fake models to simulate CubeSat and compare results of different kalman filters
Utilizes graphing.py for vizualizations of state and statistical tests
Uses the Filter class from filter.py to represent a state estimation model for a certain kalman filter

'''


import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

from irishsat_ukf.PySOL.wmm import *
from irishsat_ukf.simulator import *
from irishsat_ukf.UKF_algorithm import *
from irishsat_ukf.hfunc import *
from filter import *
from graphing import *
from tests import *

import matplotlib.pyplot as plt
import signal

'''

resources used:
https://cs.adelaide.edu.au/~ianr/Teaching/Estimation/LectureNotes2.pdf
https://github.com/FrancoisCarouge/Kalman
https://www.researchgate.net/post/How_can_I_validate_the_Kalman_Filter_result
https://stats.stackexchange.com/questions/40466/how-can-i-debug-and-check-the-consistency-of-a-kalman-filter
file:///C:/Users/andre/Downloads/WikibookonKalmanFilter.pdf

notes:
    must ensure quaternion is normalized when output by EOMs
    added innovation and innovation covariance to UKF_algorithm

TODO:
    statistical tests, effiency test/graphs + speed testing (read article)
    rewrite filter class to hold entire data sets
    rewrite visualization/plotting functions for coherency and wrap in filter class (plot all 3 on 1 graph with different line types) (see graphing.py)
    which method is correct for normalized innovation covariance? (and which CI?) (see tests.py)
        should bound be added to measurement, 0, or average?
    print output to file??

optional:
    more comprehensive plotting: wrappers, options
    flesh out 3D graphs more: colors, many at once (ideal, data, filtered)
    generate fake imu data using matlab functionality??
    change innovation plots so that upper and lower bounds are same color

'''


def signal_handler(sig, frame):
    '''
    closes all pyplot tabs when CTRL+C is entered
    '''

    plt.close('all')


if __name__ == "__main__":

    # set up signal handler to shut down pyplot tabs
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))

    # create unscented kalman filter object
    # parameters: # of steps to simulate, timestep, state space dimension, measurement space dimension, measurement noise, process noise, true magnetic field, starting reaction wheel speed, filter type
    # ukf = Filter(350, 0.1, 7, 6, 0, 0, np.array([1, 0, 0]), np.array([0, 0, 0]), UKF)

    # Our North West Up true magnetic field in stenson remick should be: 19.42900375, 1.74830615, 49.13746833 [micro Teslas]
    ukf = Filter(100, 0.1, 7, 6, 0, 0, np.array([19, 1.7, 49]), np.array([0, 0, 0]), UKF)

    # set process noise
    # parameters: noise magnitude, k (see Estimation II article by Ian Reed)
    ukf.ukf_setQ(.001, 10)
    # good for test #2 with df = 600
    ukf.ukf_setQ(.00025, 10)


    # set measurement noise
    # parameters: magnetometer noise, gyroscope noise
    ukf.ukf_setR(.001, .01)
    # good for test #2 with df = 600
    ukf.ukf_setR(.00025, .001)


    # create array of reaction wheel speed at each time step
    # parameters: max speed, min speed, number of steps to flip speed after, step, bitset of which wheels to activate
    ideal_reaction_speeds = ukf.generateSpeeds(3000, -3000, ukf.n, 100, np.array([0, 1, 0]))
    # print(ideal_reaction_speeds[:20])

    # find ideal state of cubesat through physics equations of motion
    ideal = ukf.propagate(ideal_reaction_speeds)
    # print("state: ", ideal[:10])

    # set sensor noises
    magNoises = np.random.normal(0, .035, (ukf.n, 3))
    gyroNoises = np.random.normal(0, .01, (ukf.n, 3))

    # 1 = plots + visualizer, 0 = visualizer only, 2 = none
    plot = 2

    # generate data reading for each step 
    data = ukf.generateData(ideal, magNoises, gyroNoises, 0)

    # run our data through the specified kalman function (ukf)
    filtered = ukf.simulate(data, ideal_reaction_speeds)

    # print("ideal: ", ideal[:3])
    # print("data: ", data[:3])
    # print("filtered: ", filtered[:3])


    # plotInnovations(ukf.innovations, ukf.innovationCovs)
    # plotInnovationSquared(ukf.innovations, ukf.innovationCovs)
    plotAutocorrelation(ukf.innovations)


    if plot == 1:
        ukf.visualizeResults(filtered)

        plotState_xyz(ideal, True)
        plotData_xyz(data)
        plotState_xyz(filtered, False)


    elif plot == 0:
        # ukf.visualizeResults(ideal)
        ukf.visualizeResults(filtered)


    # print(innovationTest(ukf.innovations, ukf.innovationCovs, ukf.dim_mes))

    # print(autocorrelation2D(ukf.innovations)[0])


    # only show plot at end so they all show up
    plt.show()

        
    # # plot3DVectors(np.array([ukf.B_true, data[50][:3], data[100][:3], data[150][:3]]), 121)
    # plot3DVectors(result, 111)
    # plotData3D(data, 5, 111)
    # ideal_xyz = [np.matmul(quaternion_rotation_matrix(x), np.array([1, 0, 0])) for x in ideal]
    # plotData3D(ideal_xyz, 3, 111)
