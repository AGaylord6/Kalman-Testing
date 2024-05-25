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

import matplotlib.pyplot as plt
import signal

'''

resources used:
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
    must ensure quaternion is normalized when output by EOMs
    added innovation and innovation covariance to UKF_algorithm

TODO:
    statistical tests
    effiency test/graphs + speed testing (read article)
    figure out what to plot for 3D graphs (time dependent?)
    rewrite filter class to hold entire data sets
    rewrite visualization functions for coherency and wrap in filter class

optional:
    more comprehensive plotting: wrappers, options
    flesh out 3D graphs more: colors, many at once (ideal, data, filtered)
    generate fake imu data using matlab functionality??

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
    ukf = Filter(350, 0.1, 7, 6, 0, 0, np.array([19, 1.7, 49]), np.array([0, 0, 0]), UKF)

    # set process noise
    # parameters: noise magnitude, k (see Estimation II article by Ian Reed)
    ukf.ukf_setQ(.01, 10)

    # set measurement noise
    # parameters: magnetometer noise, gyroscope noise
    ukf.ukf_setR(.01, .01)

    # create array of reaction wheel speed at each time step
    # parameters: max speed, min speed, number of steps to flip speed after, step, bitset of which wheels to activate
    ideal_reaction_speeds = ukf.generateSpeeds(1300, -1300, ukf.n/2, 100, np.array([0, 1, 0]))
    # print(ideal_reaction_speeds[:20])

    # find ideal state of cubesat through physics equations of motion
    ideal = ukf.propagate(ideal_reaction_speeds)
    # print("state: ", ideal[:10])

    # set sensor noises
    magNoises = np.random.normal(0, .05, (ukf.n, 3))
    gyroNoises = np.random.normal(0, .001, (ukf.n, 3))

    # 1 = plots + visualizer, 0 = visualizer only, 2 = none
    plot = 1

    # generate data reading for each step 
    data = ukf.generateData(ideal, magNoises, gyroNoises, 0)

    # run our data through the specified kalman function (ukf)
    filtered = ukf.simulate(data, ideal_reaction_speeds)

    # print("ideal: ", ideal[:3])
    # print("data: ", data[:3])
    # print("filtered: ", filtered[:3])

    # find magnitudes of innovation arrays
    innovationMags = np.array([np.linalg.norm(x) for x in ukf.innovations])

    # to get standard deviation, take sqrt of diagonal
    # divide by number of observations to get standard error of mean
    # get magnitude afterwards
    innovationCovMags = np.array([(np.linalg.norm(y)/ ukf.dim_mes) for y in np.array([np.sqrt(np.diag(x)) for x in ukf.innovationCovs])])
    # print(innovationCovMags[:3])

    # find upper and lower bounds of 2 * standard deviation
    upper = innovationMags + 2 * innovationCovMags
    lower = innovationMags - 2 * innovationCovMags

    # plot to check whether innovation is centered on 0 and 95% of measurements are consistent with standard deviation
    plot_multiple_lines(np.array([innovationMags, upper, lower]), ["innovation magnitude", "upper sd", "lower sd"], "innovation", 300, 200)

    if plot == 1:
        ukf.visualizeResults(filtered)

        plotState_xyz(ideal, True)
        plotData_xyz(data)
        plotState_xyz(filtered, False)


    elif plot == 0:
        # ukf.visualizeResults(ideal)
        ukf.visualizeResults(filtered)

    # only show plot at end so they all show up
    plt.show()

        
    # # plot3DVectors(np.array([ukf.B_true, data[50][:3], data[100][:3], data[150][:3]]), 121)
    # plot3DVectors(result, 111)
    # plotData3D(data, 5, 111)
    # ideal_xyz = [np.matmul(quaternion_rotation_matrix(x), np.array([1, 0, 0])) for x in ideal]
    # plotData3D(ideal_xyz, 3, 111)

