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

PySOL tells us the B field, ECI, ECEF, LLA


Still must figure out ideal state based on previous state and EOMs

Can get fake sensor mag data by rotating B field by ideal quaternion

Fake gyro data by adding noise to ideal angular velocity

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
    # Our North West Up true magnetic field in stenson remick should be: 19.42900375, 1.74830615, 49.13746833 [micro Teslas]
    ukf = Filter(200,                       # number of steps to simulate
                 0.1,                       # timestep
                 7, 6,                      # state space dimension, measurement space dimension
                 0, 0,                      # measurement noise, process noise (overwritten later)
                 np.array([19, 1.7, 49]),   # true magnetic field
                 np.array([0, 0, 0]),       # starting reaction wheel speed
                 True,                      # whether we know ideal state or we are using actual sensor data
                 UKF)                       # filter type
    
    # clear output directory from last simulation
    clearDir(outputDir)

    # text file with data values
    dataFile = "data.txt"

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


    # if we aren't using real sensor data, we need to make up reaction wheel speeds, find ideal state, and generate fake data
    if ukf.ideal_known:

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

        # generate data reading for each step 
        ukf.generateData(magNoises, gyroNoises, 0)
    
    else:
        # load sensor data from file
        # this populates ukf.data and ukf.reaction_speeds
        ukf.loadData(dataFile)


    # TODO: create loop for propogation, data creation, simulation
    # add B field graphs to save or just show


    ukf.plotData()
    # plots filtered states (and ideal states if ideal_konwn = True)
    ukf.plotStates()

    sum = ukf.runTests()

    # 0 = only create pdf output, 1 = also show 3D animation visualization
    visualize = 0

    if visualize == 1:
        # ukf.visualizeResults(ukf.ideal_states)
        ukf.visualizeResults(ukf.filtered_states)
    else:

        outputFile = "output.pdf"

        ukf.saveFile(outputFile, sum)
    
    # only show plot at end so they all show up
    plt.show()
