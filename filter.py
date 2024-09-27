'''
filter.py
Author: Andrew Gaylord

contains filter class for an arbitrary kalman filter
object contains system info, initialized values, state values, and filter specifications
class functions allow for easy initialization, propagation, data generation, simulation, and visualization

'''


from irishsat_ukf.PySOL.wmm import *
from irishsat_ukf.simulator import *
from irishsat_ukf.UKF_algorithm import *
from irishsat_ukf.hfunc import *
import time
from graphing import *
from tests import *
from saving import *


class Filter():
    def __init__ (self, n, dt, dim, dim_mes, r_mag, q_mag, B_true, reaction_speeds, ideal_known, kalmanMethod):
        # number of steps to simulate
        self.n = n
        # timestep between steps
        self.dt = dt
        # dimension of state and measurement space
        self.dim = dim
        self.dim_mes = dim_mes

        # measurement noise
        self.R = np.diag([r_mag] * dim_mes)
        # process noise
        self.Q = np.diag([q_mag] * dim)

        # starting state (default is standard quaternion and no angular velocity)
        self.state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # enforce normalized quaternion
        self.state[:4] = normalize(self.state[:4])
        # starting covariance (overrid by ukf_setQ)
        self.cov = np.identity(dim) * 5e-7

        # 2D array of n innovations and covariances (populated by filter.simulate)
        self.innovations = np.zeros((n, dim_mes))
        self.innovationCovs = np.zeros((n, dim_mes, dim_mes))

        # true magnetic field for simulation
        self.B_true = B_true

        # 1x3 array of current reaction wheel speeds
        self.curr_reaction_speeds = reaction_speeds
        # reaction wheel speed of last time step
        self.last_reaction_speeds = np.zeros(3)

        # reaction wheel speeds for all n steps
        self.reaction_speeds = np.zeros((n, 3))

        # data values for all n steps
        self.data = np.zeros((n, dim_mes))

        # ideal states from EOMs for all n steps
        self.ideal_states = np.zeros((n, dim))

        # indicates whether we know our ideal states or not (i.e. if we are simulating or not)
        self.ideal_known = ideal_known

        # kalman filtered states for all n steps
        self.filtered_states = np.zeros((n, dim))

        # covariance of system for all n steps
        self.covs = np.zeros((n, dim, dim))

        # what kalman filter to apply to this system
        self.kalmanMethod = kalmanMethod

        # filter times for each step
        self.times = np.zeros(n)


    def ukf_setR(self, magNoise, gyroNoise):
        '''
        set measurement noise R (dim_mes x dim_mes)

        @params:
             magNoise: noise for magnetometer
             gyroNoise: noise for gyroscope   
        '''

        self.R = np.array([[magNoise, 0, 0, 0, 0, 0],
                 [0, magNoise, 0, 0, 0, 0],
                 [0, 0, magNoise, 0, 0, 0],
                 [0, 0, 0, gyroNoise, 0, 0],
                 [0, 0, 0, 0, gyroNoise, 0],
                 [0, 0, 0, 0, 0, gyroNoise]])


    def ukf_setQ(self, noiseMagnitude, R = 10):
        '''
        set process noise Q (dim x dim) and update initial covariance
        Q is based on dt (according to research) and initial cov = Q * R according to Estimation II by Ian Reed

        @params:
            noiseMagnitude: magnitude of Q
            R: parameter for initial covariance (10 is optimal)
        '''

        self.Q = np.array([[self.dt, 3*self.dt/4, self.dt/2, self.dt/4, 0, 0, 0],
                [3*self.dt/4, self.dt, 3*self.dt/4, self.dt/2, 0, 0, 0],
                [self.dt/2, 3*self.dt/4, self.dt, 3*self.dt/4, 0, 0, 0],
                [self.dt/4, self.dt/2, 3*self.dt/4, self.dt, 0, 0, 0],
                [0, 0, 0, 0, self.dt, 2*self.dt/3, self.dt/3],
                [0, 0, 0, 0, 2*self.dt/3, self.dt, 2*self.dt/3],
                [0, 0, 0, 0, self.dt/3, 2*self.dt/3, self.dt]
        ])
        self.Q = self.Q * noiseMagnitude

        # update starting cov guess
        self.cov = R * self.Q
    

    def generateSpeeds(self, max, min, flipSteps, step, indices):
        '''
        generates ideal/actual reaction wheel speeds for n steps
        goes to max for flipSteps and then decreases by step until min is reached
        populates self.reaction_speeds

        @params:
            max, min: max and min speeds
            flipSteps: how many stepts until speed is reversed
            step: how much to change speed by for each time step
            indices: bitset of sorts to signify which axis you want movement about (which reaction wheels to activate)
                speed on x and z would equal [1, 0, 1]
        '''

        # start with 0 speed on all axices
        ideal_reaction_speeds = [self.curr_reaction_speeds]
        thing = 0

        for a in range(self.n):
            # increase/decrease by step if max/min is not reached
            # also check if inflection point (flipSteps) has been reached
            if (a < flipSteps and thing < max):
                thing += step
            elif thing > min and a > flipSteps:
                thing -= step
            
            result = np.array([thing, thing, thing])
            # multiply by bitset to only get speed on proper axis
            result = indices * result
            ideal_reaction_speeds.append(result)

        
        # store in filter object
        self.reaction_speeds = np.array(ideal_reaction_speeds[:self.n])
        
        return np.array(ideal_reaction_speeds[:self.n])


    def propagate(self):
        '''
        generates ideal/actual states of cubesat for n time steps
        uses starting state and reaction wheel speeds at each step to progate through our EOMs (equations of motion)

        these equations give rough idea of how our satellite would respond to these conditions at each time step
        from this physics-based ideal state, we can generate fake data to pass through our filter

        '''

        # initialize propogator object with inital quaternion and angular velocity
        # propagator = AttitudePropagator(q_init=self.state[:4], w_init=self.curr_reaction_speeds)
        # t0 = 0
        # tf = self.n * self.dt
        # # use attitude propagator to find actual ideal quaternion for n steps
        # states = propagator.propagate_states(t0, tf, self.n)

        # intertia constants of cubesat from juwan
        I_body = I_body_sat * 1e-7
        I_spin = 5.1e-7
        # I_trans = 5.1e-7
        I_trans = 0
        # intialize EOMs using intertia measurements of cubeSat
        EOMS = TEST1EOMS(I_body, I_spin, I_trans)

        currState = self.state

        # make array of all states
        states = np.array([currState])

        self.curr_reaction_speeds = np.zeros(3)

        for i in range(self.n):

            # store speed from last step
            self.last_reaction_speeds = self.curr_reaction_speeds
            self.curr_reaction_speeds = self.reaction_speeds[i]

            # calculate reaction wheel acceleration
            alpha = (self.curr_reaction_speeds - self.last_reaction_speeds) / self.dt
            
            # progate through our EOMs
            # params: current quaternion, angular velocity, reaction wheel speed, tau(?), reaction wheel acceleration, time step
            currState = EOMS.eoms(currState[:4], currState[4:], self.curr_reaction_speeds, 0, alpha, self.dt)

            states = np.append(states, np.array([currState]), axis=0)
        
        # remove duplicate first element
        states = states[1:]
        
        self.ideal_states = states
        return states
    

    def propagate_step(self, i):

        # get updated inertia constants and make global

        # use the previous state and wheel speeds to calculate the next ideal state

        return 0


    def generateData(self, magNoises, gyroNoises, hallNoises):
        '''
        generates fake data array (n x dim_mes)
        adds noise to the ideal states to mimic what our sensors would be giving us

        @params:
            magNoises: gaussian noise for magnetometer (n x 3)
            gyroNoises: gaussian noise for gyroscope (n x 3)
            hallNoises: guassian hall sensor noise to be added to our reaction wheel speeds (n x 3)
        
        '''

        # calculate sensor b field for every time step (see h func for more info on state to measurement space conversion)
        # rotation matrix(q) * true B field + noise
        # first value, then all the otheres
        B_sens = np.array([np.matmul(quaternion_rotation_matrix(self.ideal_states[0]), self.B_true)])
        for a in range(1, self.n):
            B_sens = np.append(B_sens, np.array([np.matmul(quaternion_rotation_matrix(self.ideal_states[a]), self.B_true)]), axis=0)
            # print("{}: {}".format(a, np.matmul(quaternion_rotation_matrix(self.ideal_states[a]), self.B_true)))
        
        # add noise
        B_sens += magNoises

        # create sensor data matrix of magnetomer reading and angular velocity
        data = np.zeros((self.n, self.dim_mes))
        for a in range(self.n):
            data[a][0] = B_sens[a][0]
            data[a][1] = B_sens[a][1]
            data[a][2] = B_sens[a][2]
            # add gyro noise to ideal angular velocity
            data[a][3] = self.ideal_states[a][4] + gyroNoises[a][0]
            data[a][4] = self.ideal_states[a][5] + gyroNoises[a][1]
            data[a][5] = self.ideal_states[a][6] + gyroNoises[a][2]

        self.data = data
        return data
    

    def generateData_step(self, i):

        # use current B field of earth to transform ideal state to measurement space + add noise

        # get ideal speed + add noise

        # update data array

        return 0
    

    def loadData(self, fileName):
        '''
        alternate to sumulate and generateData. used when ideal_known = False
        populates self.data with sensor data from file
        populates self.reaction_speeds with reaction wheel speeds from file

        @params:
            fileName: name of file to load data from
        '''
        # data is in the format a, b, c, x, y, z, e, f, g
        # a, b, c are magnetic field in state space readings, x, y, z are angular velocity, e, f, g are reaction wheel speeds
        # each line is a new time step
        # read in file line by line and store data and reaction wheel speeds in self.data and self.reaction_speeds
        data = []
        speeds = []
        with open(fileName, 'r') as file:
            for line in file:
                data.append(np.array([float(x) for x in line.split(",")[:6]]))
                speeds.append(np.array([float(x) for x in line.split(",")[6:]]))
        
        self.data = np.array(data)
        self.reaction_speeds = np.array(speeds)
        return data


    def simulate(self):
        '''
        simulates the state estimation process for n time steps
        runs the specified kalman filter upon the the object's initial state and data/reaction wheel speeds for each time step
            uses self.reaction_speeds: reaction wheel speed for each time step (n x 3) and self.data: data reading for each time step (n x dim_mes)

        stores 2D array of estimated states (quaternions, angular velocity) in self.filter_states, covariances in self.covs, and innovation values and covariances in self.innovations/self.innovationCovs
        also stores time taken for each estimation in self.times
        
        '''

        states = []
        self.curr_reaction_speeds = np.zeros(3)
        
        # run each of n steps through the filter
        for i in range(self.n):
            # store old reaction wheel speed
            self.old_reaction_speeds = self.curr_reaction_speeds
            self.curr_reaction_speeds = self.reaction_speeds[i]
            
            start = time.time()
            # propagate current state through kalman filter and store estimated state and innovation
            self.state, self.cov, self.innovations[i], self.innovationCovs[i] = self.kalmanMethod(self.state, self.cov, self.Q, self.R, self.B_true, self.curr_reaction_speeds, self.old_reaction_speeds, self.data[i])
            end = time.time()

            # store time taken for each step
            self.times[i] = end - start

            states.append(self.state)
            self.covs[i] = self.cov

        self.filtered_states = states
        return states
    

    def simulate_step(self, i, target):

        # run last state, reaction wheel speed, and data through filter

        # run state through our controls to get pwms

        # convert pwms to reaction wheel speeds and update next speeds

        # time step


        return 0


    def plotData(self):
        '''
        plots the magnetometer (magData.png) and gyroscope data (magData.png) found in self.data
        '''
        plotData_xyz(self.data)


    def plotStates(self):
        '''
        plots the filtered states (filteredQuaternion.png, filteredVelocity.png) found in self.filtered_states
        also plots ideal states (idealQuaternion.png, idealVelocity.png) found in self.ideal_states if self.ideal_known = True
        '''
        if self.ideal_known:
            plotState_xyz(self.ideal_states, self.ideal_known)
        plotState_xyz(self.filtered_states, False)

    
    def runTests(self):
        '''
        runs 3 statistical tests on filter results according to Estimation II by Ian Reed:
            1. innovation test
            2. innovation squared test
            3. autocorrelation test
        
        creates approriate plots, prints info to command line, and returns the sum of innovations squared
        '''
        # test 1, 2, 3 respectively (see tests.py)
        plotInnovations(self.innovations, self.innovationCovs)
        sum = plotInnovationSquared(self.innovations, self.innovationCovs)
        plotAutocorrelation(self.innovations)
        return sum
    

    def saveFile(self, fileName, sum):
        '''
        takes all saved pngs and compiles a pdf with the given fileName
        uses the formating funciton found within saving.py
        stores in outputDir global variable declared in saving.py and opens completed file
        '''

        # savePNGs(outputDir)

        savePDF(fileName, outputDir, self, sum)

        openFile(fileName)


    def visualizeResults(self, states):
        # TODO: rewrite functions that visualize different data sets: ideal, filtered, data
        #   with plotting, cubesat, etc

        # or visualize 3 things: raw, filtered, ideal

        game_visualize(np.array(states), 0)
    
