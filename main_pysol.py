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
from params import *

import matplotlib.pyplot as plt
import signal


'''

PySOL tells us the B field, ECI, ECEF, LLA

https://kieranwynn.github.io/pyquaternion/#normalisation
https://csimn.com/CSI_pages/PID.html

'''


def signal_handler(sig, frame):
    '''
    closes all pyplot tabs when CTRL+C is entered
    '''
    plt.close('all')
    plt.clf()
    plt.cla()
    plt.close()

def run_filter_sim(filter, magNoises, gyroNoises):
    '''
    Generates ideal states and sensor data, allowing us to benchmark our kalman filter against simulated "truth". 
    Can also be run with pre-existing sensor data (ideal_known = False)

    @params:
        filter: kalman Filter object to run simulation with
        magNoises: magnetometer noise at each step
        gyroNoises: gyroscope noise at each step
    '''

    # text file with data values
    dataFile = DATA_FILE

    # if we aren't using real data, we need to make up reaction wheel speeds, find ideal state, and generate fake data
    if not filter.ideal_known:
        # this populates ukf.data and ukf.reaction_speeds
        filter.loadData(dataFile)
    else:
        # decide how we want our reaction wheels to spin at each time step
        # parameters: max speed, min speed, number of steps to flip speed after, step, bitset of which wheels to activate
        filter.generateSpeeds(400, -400, filter.n, 40, np.array([0, 1, 0, 0]))

        # find ideal state of cubesat through physics equations of motion
        filter.propagate()

        # generate data reading for each step 
        filter.generateData(magNoises, gyroNoises, 0)


    # run our data through the specified kalman function
    filter.simulate()

    # if true, run statistical tests outlined in Estimation II by Ian Reed
    # these tests allow us to see how well our filter is performing
    runTests = RUN_STATISTICAL_TESTS
    sum = 0
    if runTests:
        sum = filter.runTests()

    # plot our results and create pdf output + 3D visualization
    plot_and_viz_results(filter, sum=sum)


def run_controls_sim(filter, magNoises, gyroNoises):
    '''
    Combines motor dynamics and PID controller to orient towards a target
    Propogates our state step by step, as we want to dynamically change our "ideal" state based on our control output

    @params:
        filter: Filter object to run simulation with
        magNoises: magnetometer noise at each step
        gyroNoises: gyroscope noise at each step
    '''

    # generate data for first step so we can start at i = 1
    filter.generateData_step(0, magNoises[0], gyroNoises[0])

    # Initialize PID controller
    kp = KP   # Proportional gain
    # kp = MAX_PWM * 7e-8
    # close to kp allows for narrowing in on target, but not too close
    # smaller = oscillating more frequently, larger = overshooting more
    ki = KI     # Integral gain
    # ki = MAX_PWM * 5e-9
    # if this is too high, it overrotates
    kd = KD  # Derivative gain
    # kd = MAX_PWM * 1e-8
    pid = PIDController(kp, ki, kd, filter.dt)

    # define our target orientation and whether we want to reverse it halfway through
    # TODO: x axis is bugged (or just different moments of inertia). Wants to go sideways
    target = normalize(TARGET)
    flip = False

    for i in range(1, filter.n):

        # get ideal next state based on current state and reaction wheel speeds of this step
        # NOTE: this "ideal" state is not super based on truth because it is not generated beforehand. 
        #       it basically follows what our filter does, so it is not a good representation of the truth
        ideal = filter.propagate_step(i)
        
        # create fake magnetometer data by rotating B field by ideal quaternion, and gyro by adding noise to angular velocity
        filter.generateData_step(i, magNoises[i], gyroNoises[i])

        # filter our data and get next state
        # also run through our controls to get pwm => voltage => current => speed of reaction wheels
        filtered = filter.simulate_step(i, target, pid)
        # game_visualize(np.array([filtered]), i-1)

        if i > filter.n / 2 and flip == True:
            target = normalize(QUAT_INITIAL)

    # plot our results and create pdf output + 3D visualization
    plot_and_viz_results(filter, controller=pid, target=target)


def plot_and_viz_results(filter, controller=None, target=np.array([1, 0, 0, 0]), sum=0):
    '''
    Plots out filter states, data, and reaction wheel speeds, and creates pdf output + 3D visualization
    Allows us to visualize results of our filter/controls sim
    Based upon RESULT variable in params.py

    @params:
        filter: Filter object to plot and visualize results of
        controller: PIDController object (for controls sim)
        target: target quaternion (for controls sim)
        sum: sum of statistical tests if they were run
    '''
    # plot mag and gyro data
    filter.plotData()
    # plots filtered states (and ideal states if ideal_known = True)
    filter.plotStates()
    # plot reaction wheel speeds
    filter.plotWheelInfo()

    # 0 = only create pdf output, 1 = show 3D animation visualization, 2 = both, 3 = none
    visualize = RESULT

    if visualize == 1:
        filter.visualizeResults(filter.filtered_states)

    elif visualize == 0:

        filter.saveFile(OUTPUT_FILE, controller, target, sum, RUN_STATISTICAL_TESTS)

    elif visualize == 2:

        filter.saveFile(OUTPUT_FILE, controller, target, sum, RUN_STATISTICAL_TESTS)

        filter.visualizeResults(filter.filtered_states)

    # only show plot at end so they all show up
    plt.show()


if __name__ == "__main__":

    # TODO: impliment PySol and print B field (and globe?)
    # TODO: move quat_multiply and others from hfunc (or rename to orientaiton helper)
    # TODO: clean up params.py

    # set up signal handler to shut down pyplot tabs
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))

    tf = TF
    # time step should be small to combat instability of euler's method
    dt = DT
    n = int(tf/dt)

    # create unscented kalman filter object
    # Our North West Up true magnetic field in stenson remick should be: 19.42900375, 1.74830615, 49.13746833 [micro Teslas]
    ukf = Filter(n,                         # number of steps to simulate
                 dt,                        # timestep
                 7, 6,                      # state space dimension, measurement space dimension
                 0, 0,                      # measurement noise, process noise (overwritten later)
                 np.array([19, 1.7, 49]),   # true magnetic field
                 RW_INITIAL,                # starting reaction wheel speed
                 IDEAL_KNOWN,               # whether we know ideal state or we are using actual sensor data
                 UKF)                       # filter type
    
    # clear output directory from last simulation
    clearDir(outputDir)

    # # text file with data values
    # dataFile = DATA_FILE

    # set process noise
    # parameters: noise magnitude, k (see Estimation II article by Ian Reed)
    ukf.ukf_setQ(PROCESS_NOISE_MAG, PROCESS_NOISE_K)

    # set measurement noise
    # parameters: magnetometer noise, gyroscope noise
    ukf.ukf_setR(MEASUREMENT_MAGNETOMETER_NOISE, MEASUREMENT_GYROSCOPE_NOISE)

    # set sensor noises
    magSD = SENSOR_MAGNETOMETER_SD
    magNoises = np.random.normal(0, magSD, (ukf.n, 3))
    
    gyroSD = SENSOR_GYROSCOPE_SD
    gyroNoises = np.random.normal(0, gyroSD, (ukf.n, 3))

    run_filter_sim(ukf, magNoises, gyroNoises)
    # run_controls_sim(ukf, magNoises, gyroNoises)

