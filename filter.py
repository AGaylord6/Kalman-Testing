'''
filter.py
Author: Andrew Gaylord

contains filter class for an arbitrary kalman filter
object contains system info, initialized values, state values, and filter specifications
class functions allow for easy initialization, propagation, data generation, simulation, and visualization

TODO: 
    add testing functionality
    rewrite class attributes to store state/data/reaction wheel speeds for all n steps

'''


from irishsat_ukf.PySOL.wmm import *
from irishsat_ukf.simulator import *
from irishsat_ukf.UKF_algorithm import *
from irishsat_ukf.hfunc import *


class Filter():
    def __init__ (self, n, dt, dim, dim_mes, r_mag, q_mag, B_true, reaction_speeds, kalmanMethod):
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

        # 2D array of n innovations and covariances
        self.innovations = np.zeros((n, dim_mes))
        self.innovationCovs = np.zeros((n, dim_mes, dim_mes))

        # true magnetic field for simulation
        self.B_true = B_true

        # 1x3 array of reaction wheel speeds
        self.reaction_speeds = reaction_speeds
        # reaction wheel speed of last time step
        self.old_reaction_speeds = np.zeros(3)

        # what kalman filter to apply to this system
        self.kalmanMethod = kalmanMethod


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


        @params:
            max, min: max and min speeds
            flipSteps: how many stepts until speed is reversed
            step: how much to change speed by for each time step
            indices: bitset of sorts to signify which axis you want movement about (which reaction wheels to activate)
                speed on x and z would equal [1, 0, 1]
        '''

        # start with 0 speed on all axices
        ideal_reaction_speeds = [np.array([0, 0, 0])]
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
            
        return np.array(ideal_reaction_speeds[:self.n])


    def propagate(self, reaction_speeds):
        '''
        generates ideal/actual states of cubesat for n time steps
        uses starting state and reaction wheel speeds to progate through our EOMs (equations of motion)

        these equations give rough idea of how our satellite would respond to these conditions at each time step
        from this physics-based ideal state, we can generate fake data to pass through our filter

        '''

        # initialize propogator object with inital quaternion and angular velocity
        # propagator = AttitudePropagator(q_init=self.state[:4], w_init=self.reaction_speeds)
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

        self.reaction_speeds = np.zeros(3)

        for i in range(self.n):

            # store speed from last step
            self.old_reaction_speeds = self.reaction_speeds
            self.reaction_speeds = reaction_speeds[i]

            # calculate reaction wheel acceleration
            alpha = (self.reaction_speeds - self.old_reaction_speeds) / self.dt
            
            # progate through our EOMs
            # params: current quaternion, angular velocity, reaction wheel speed, tau(?), reaction wheel acceleration, time step
            currState = EOMS.eoms(currState[:4], currState[4:], self.reaction_speeds, 0, alpha, self.dt)

            states = np.append(states, np.array([currState]), axis=0)
        
        # remove duplicate first element
        states = states[1:]
        
        return states


    def generateData(self, states, magNoises, gyroNoises, hallNoises):
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
        B_sens = np.array([np.matmul(quaternion_rotation_matrix(states[0]), self.B_true)])
        for a in range(1, self.n):
            B_sens = np.append(B_sens, np.array([np.matmul(quaternion_rotation_matrix(states[a]), self.B_true)]), axis=0)
            # print("{}: {}".format(a, np.matmul(quaternion_rotation_matrix(states[a]), self.B_true)))
        
        # add noise
        B_sens += magNoises

        # create sensor data matrix of magnetomer reading and angular velocity
        data = np.zeros((self.n, self.dim_mes))
        for a in range(self.n):
            data[a][0] = B_sens[a][0]
            data[a][1] = B_sens[a][1]
            data[a][2] = B_sens[a][2]
            # add gyro noise to ideal angular velocity
            data[a][3] = states[a][4] + gyroNoises[a][0]
            data[a][4] = states[a][5] + gyroNoises[a][1]
            data[a][5] = states[a][6] + gyroNoises[a][2]

        return data


    def simulate(self, data, reaction_speeds):
        '''
        simulates the state estimation process for n time steps
        runs the specified kalman filter upon the the object's initial state and passed data/reaction wheel speeds
        returns 2D array of estimated states (quaternions, angular velocity) and innovation values and covariances

        @params:
            data: data reading for each time step (n x dim_mes)
            reaction_speeds: reaction wheel speed for each time step (n x 3)
        
        '''

        states = []
        self.reaction_speeds = np.zeros(3)
        
        # run each of n steps through the filter
        for i in range(self.n):
            # store old reaction wheel speed
            self.old_reaction_speeds = self.reaction_speeds
            self.reaction_speeds = reaction_speeds[i]
            
            # propagate current state through kalman filter and store estimated state and innovation
            self.state, self.cov, innovation, innovationCov = self.kalmanMethod(self.state, self.cov, self.Q, self.R, self.B_true, self.reaction_speeds, self.old_reaction_speeds, data[i])

            self.innovations[i] = innovation
            self.innovationCovs[i] = innovationCov
            states.append(self.state)

        return states


    def visualizeResults(self, states):
        # TODO: rewrite functions that visualize different data sets: ideal, filtered, data
        #   with plotting, cubesat, etc

        # or visualize 3 things: raw, filtered, ideal

        game_visualize(np.array(states), 0)

    #TODO: function for innovations
