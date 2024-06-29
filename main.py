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
from saving import *

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
    user must pip install fpdf

TODO:
    speed testing + other correctness tests?
    rewrite visualization/plotting functions for coherency and wrap in Filter class (plot all 3 on 1 graph with different line types) (see graphing.py)
    which method is correct for normalized innovation covariance (test #2)? (and which CI?) (see tests.py)
        should interval bound be added to measurement, 0, or average?

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
    ukf = Filter(200, 0.1, 7, 6, 0, 0, np.array([19, 1.7, 49]), np.array([0, 0, 0]), UKF)
    
    # clear output directory from last simulation
    clearDir(outputDir)

    # set process noise
    # parameters: noise magnitude, k (see Estimation II article by Ian Reed)
    ukf.ukf_setQ(.001, 10)
    # good for vn100 noises
    ukf.ukf_setQ(.00001, 10)
    # good for test 2
    ukf.ukf_setQ(.00001, 10)



    # set measurement noise
    # parameters: magnetometer noise, gyroscope noise
    ukf.ukf_setR(.001, .01)
    # good for vn100 noises
    ukf.ukf_setR(.001, .01)
    # good for test 2
    ukf.ukf_setR(.00025, .0025)


    # create array of reaction wheel speed at each time step
    # parameters: max speed, min speed, number of steps to flip speed after, step, bitset of which wheels to activate
    ukf.generateSpeeds(3000, -3000, ukf.n, 100, np.array([0, 1, 0]))
    # print(ukf.reaction_speeds[:20])

    # find ideal state of cubesat through physics equations of motion
    ukf.propagate()
    # print("state: ", ukf.ideal_states[:10])

    # set sensor noises
    # noise sd = noise density * sqrt(sampling rate)
    # vn100 imu sampling rate from user manual = 200 Hz
    
    # magNoises = np.random.normal(0, .035, (ukf.n, 3))
    # mag noise density from vn100 website = 140 uGauss /sqrt(Hz)
    magSD = (140 * 10e-6) * np.sqrt(200)
    magNoises = np.random.normal(0, magSD, (ukf.n, 3))

    # gyroNoises = np.random.normal(0, .01, (ukf.n, 3))
    # gyro noise density from vn100 website = 0.0035 /s /sqrt(Hz)
    gyroSD = 0.0035 * np.sqrt(200)
    gyroNoises = np.random.normal(0, gyroSD, (ukf.n, 3))


    # 0 = only create pdf output, 1 = also show 3D animation visualization
    visualize = 0

    # generate data reading for each step 
    ukf.generateData(magNoises, gyroNoises, 0)

    # run our data through the specified kalman function (ukf)
    ukf.simulate()

    # print("ukf.ideal_states: ", ukf.ideal_states[:3])
    # print("ukf.data: ", ukf.data[:3])
    # print("filtered: ", ukf.filtered_states[:3])

    plotState_xyz(ukf.ideal_states, True)
    plotData_xyz(ukf.data)
    plotState_xyz(ukf.filtered_states, False)

    # test 1, 2, 3 respectively (see tests.py)
    plotInnovations(ukf.innovations, ukf.innovationCovs)
    sum = plotInnovationSquared(ukf.innovations, ukf.innovationCovs)
    plotAutocorrelation(ukf.innovations)


    if visualize == 1:
        # ukf.visualizeResults(ukf.ideal_states)
        ukf.visualizeResults(ukf.filtered_states)


    # print(innovationTest(ukf.innovations, ukf.innovationCovs, ukf.dim_mes))

    # print(autocorrelation2D(ukf.innovations)[0])

    outputFile = "output.pdf"

    # savePNGs(outputDir)

    # outputDir is global variable declared in saving.py
    savePDF(outputFile, outputDir, ukf, sum)

    openFile(outputFile)
    
    # only show plot at end so they all show up
    plt.show()

        
    # # plot3DVectors(np.array([ukf.B_true, ukf.data[50][:3], ukf.data[100][:3], ukf.data[150][:3]]), 121)
    # plot3DVectors(result, 111)
    # plotData3D(ukf.data, 5, 111)
    # ideal_xyz = [np.matmul(quaternion_rotation_matrix(x), np.array([1, 0, 0])) for x in ukf.ideal_states]
    # plotData3D(ideal_xyz, 3, 111)
