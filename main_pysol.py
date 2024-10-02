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


class params:
    # Importing motor parameters - Maxon DCX 8 M (9 volts)
    Rwa = 3.54      # Ohms, winding resistance at ambient temperature
    Lw = 0.424e-3  # Henry
    Kt = 8.82e-3   # Torque constant Nm/A
    Kv = Kt    # Voltage constant V*s/rad
    Jm = 5.1*(1e-7)   # Kg m^2
    bm = 3.61e-6   # [N·m·s/rad] Viscous friction
    Rha = 16.5      # K/W
    Rwh = 2.66     # K/W
    Cwa = 2.31/Rwh     # Thermal Capacitance
    Cha = 162/Rha      # Thermal Capacitance
    alpha_Cu = 0.00393 # copper's temperature coefficient [1/K]
    # Moments of Inertia [g cm^2]- from CAD of CubeSat test bed
    Ixx = 46535.388 
    Ixy = 257.834 
    Ixz = 536.12
    Iyx = 257.834 
    Iyy = 47934.771 
    Iyz = -710.058
    Izx = 546.12 
    Izy = -710.058 
    Izz = 23138.181

    # Moment of Inertia Tensor of full 2U cubesat [kg m^2]
    J_B = (1e-7)*np.array([[Ixx, Ixy, Ixz],
                    [Iyx, Iyy, Iyz],
                    [Izx, Izy, Izz]])
    
    J_B_inv = np.linalg.inv(J_B)

    # Moments of Inertia of rxn wheels [g cm^2] - measured
    Iw1 = (1/2)*38*1.8**2 # I_disc = 1/2 * M * R^2
    Iw2 = Iw1 
    Iw3 = Iw1 
    Iw4 = Iw1

    # Moment of inertia tensor of rxn wheels [kg m^2]
    J_w = (1e-7)*np.array([[Iw1, 0, 0, 0],
                    [0, Iw2, 0, 0],
                    [0, 0, Iw3, 0],
                    [0, 0, 0, Iw4]])

    # External torques (later this can be from magnetorquers)
    L_B = np.array([0, 0, 0])

    # Transformation matrix for NASA config given in Fundamentals pg
    # 153-154
    W = np.array([[1, 0, 0, 1/np.sqrt(3)],
            [0, 1, 0, 1/np.sqrt(3)],
            [0, 0, 1, 1/np.sqrt(3)]])

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
                 np.array([0, 0, 0, 0]),    # starting reaction wheel speed
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

        # ukf.generateSpeeds(1000, -1000, ukf.n/2, 100, np.array([0, 1, 0, 0]))

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

    # set first data point
    ukf.generateData_step(0, magNoises[0], gyroNoises[0])

    target = np.array([1, 0, 0, 0])

    for i in range(1, ukf.n):

        # get ideal next state based on current state and reaction wheel speeds of this step
        ideal = ukf.propagate_step(i)
        # game_visualize(np.array([ukf.propagate_step(i)]), i-1)
        
        # create mag data using transform and orbit data
        # create gyro data by adding noise to ideal
        ukf.generateData_step(i, magNoises[i], gyroNoises[i])

        # filter our data and get next state
        # also run through our controls to get pwm => voltage => current => speed of reaction wheels
        # ukf.filtered_states[i] = ukf.ideal_states[i]
        filtered = ukf.simulate_step(i, params, target)
        game_visualize(np.array([filtered]), i-1)



    # TODO: impliment PySol and print B field 

    # ukf.plotData()
    # plots filtered states (and ideal states if ideal_known = True)
    # ukf.plotStates()

    # sum = ukf.runTests()

    # 0 = only create pdf output, 1 = show 3D animation visualization
    visualize = 1

    if visualize == 1:
        # ukf.visualizeResults(ukf.ideal_states)
        # ukf.visualizeResults(ukf.filtered_states)
        pass
    else:

        outputFile = "output.pdf"

        ukf.saveFile(outputFile, sum)
    
    # only show plot at end so they all show up
    plt.show()



def current_to_speed(current, external_torque, reaction_speed):
    '''
    calculates hypothetical speed of reaction wheels 
    taken from Patrick's code (CubeSat_Model_discrete.py)
    uses the params object
    @params:
        current: current of reaction wheels
        external_torque: external torque
        reaction_speed: current speed of reaction wheels
    
    @returns:
        next speeds of 4 wheels in rad/s
    '''

    omega_w_dot = (params.Kt*current + external_torque - params.bm*reaction_speed)/params.Jm

    return omega_w_dot
