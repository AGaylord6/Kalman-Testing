'''
main.py
Author: Andrew Gaylord

Main file for Kalman-Testing rep
Sets up fake models to simulate CubeSat and compare results of different kalman filters

'''

import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

from irishsat_ukf.PySOL.wmm import *
from irishsat_ukf.simulator import *
from irishsat_ukf.UKF_algorithm import *
from irishsat_ukf.hfunc import *
from filter import *
from graphing import *

import matplotlib.pyplot as plt
import signal

'''

https://cs.adelaide.edu.au/~ianr/Teaching/Estimation/LectureNotes2.pdf
https://github.com/FrancoisCarouge/Kalman
https://www.researchgate.net/post/How_can_I_validate_the_Kalman_Filter_result
https://stats.stackexchange.com/questions/40466/how-can-i-debug-and-check-the-consistency-of-a-kalman-filter
file:///C:/Users/andre/Downloads/WikibookonKalmanFilter.pdf

one of the ways to check Kalman filters performance is to check for error covariance matrix P 
to be converging. If it converges to + or - standard deviation of the estimated value, 
it can be considered as a stable point. 
calculate square of difference between estimated and real
You can verify that the estimated state converges to the actual state. 
The error covariance, P, must decrease.

innovation (V) or residual: difference between a measurement and its prediction at time k
    measures new info provided by adding another measurement in estimation
    can be used to validate a measurement before adding to observation sequence

    innovation tests: test that innovation has zero mean and white with cov S_k
        1) check that it is consistent with its cov/within bounds--checks filter consistency
        2) chi square test for unbiased 
        3) whiteness (autocorrelation) test

notes:
EOMs are messed up. changed rw_config back to gamma and such
    change I_trans from non-zero?

must ensure quaternion is normalized

issues:
    how is B_true connected to our starting state + initial eoms progagation
        need to check other B_trues and rotations about other axis (like y)
    
    what is orientation of simulation?? what are we enforcing with starting quaternion??

TODO:
    statistical tests
    effiency test/graphs (read article)
    plotting comparisons between filter/unfiltered
    figure out what to plot for 3D graphs (time dependent?)

optional:
    more comprehensive plotting: wrappers, options, make part of Filter class
    flesh out 3D graphs more: colors, many at once (ideal, data, filtered)
    generate fake imu data using matlab functionality??

'''


def signal_handler(sig, frame):
    '''
    signal_handler

    closes all pyplot tabs
    '''

    plt.close('all')


if __name__ == "__main__":

    # set up signal handler to shut down pyplot tabs
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))

    # create unscented kalman filter object
    # steps to simulate, timestep, state space dimension, measurement space dimension, measurement noise, process noise, true magnetic field, starting reaction wheel speed, filter type
    ukf = Filter(350, 0.1, 7, 6, 0, 0, np.array([1, 0, 0]), np.array([0, 0, 0]), UKF)

    # Our North West Up true magnetic field in stenson remick should be: 19.42900375, 1.74830615, 49.13746833 [micro Teslas]
    ukf = Filter(350, 0.1, 7, 6, 0, 0, np.array([19, 1.7, 49]), np.array([0, 0, 0]), UKF)

    # set process noise
    # noise magnitude, k (see Estimation II article by Ian Reed)
    ukf.ukf_setQ(.01, 10)

    # set measurement noise
    # magnetometer noise, gyroscope noise
    ukf.ukf_setR(.25, .5)
    ukf.ukf_setR(.01, .01)

    # create array of reaction wheel speed at each time step
    # max speed, min speed, number of steps to flip speed after, step, bitset of which wheels to activate
    ideal_reaction_speeds = ukf.generateSpeeds(1300, -1300, ukf.n/2, 100, np.array([0, 0, 1]))
    # ideal_reaction_speeds = ukf.generateSpeeds(0,0, ukf.n, 100, np.array([0, 0, 1]))
    # print(ideal_reaction_speeds[:20])

    # find ideal state of cubesat through physics equations of motion
    ideal = ukf.propagate(ideal_reaction_speeds)
    # print("state: ", ideal[:10])

    # set sensor noises
    magNoises = np.random.normal(0, .001, (ukf.n, 3))
    gyroNoises = np.random.normal(0, .001, (ukf.n, 3))

    plot = 1

    # generate data reading for each step 
    data = ukf.generateData(ideal, magNoises, gyroNoises, 0)

    # run our data through the specified kalman function (ukf)
    filtered = ukf.simulate(data, ideal_reaction_speeds)

    # # plot3DVectors(np.array([ukf.B_true, data[50][:3], data[100][:3], data[150][:3]]), 121)
    # plot3DVectors(result, 111)

    # plotData3D(data, 5, 111)

    # ideal_xyz = [np.matmul(quaternion_rotation_matrix(x), np.array([1, 0, 0])) for x in ideal]
    # plotData3D(ideal_xyz, 3, 111)

    if plot == 1:
        ukf.visualizeResults(filtered)

        plotState_xyz(ideal, True)
        plotData_xyz(data)
        plotState_xyz(filtered, False)

        # only show plot at end so they all show up
        plt.show()

    else:
        # ukf.visualizeResults(ideal)
        ukf.visualizeResults(filtered)







