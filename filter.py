'''
filter.py
Author: Andrew Gaylord

contains filter class for an arbitrary kalman filter
allows for easy initializiation, progagation, and testing

'''


from irishsat_ukf.PySOL.wmm import *
from irishsat_ukf.simulator import *
from irishsat_ukf.UKF_algorithm import *
from irishsat_ukf.hfunc import *

class Filter():
    def __init__ (self, n, dt, dim, dim_mes, r_mag, q_mag, B_true, reaction_speeds, kalmanMethod):
        self.n = n
        self.dt = dt
        self.dim = dim
        self.dim_mes = dim_mes

        # measurement noise
        self.R = np.diag([r_mag] * dim_mes)
        # process noise
        self.Q = np.diag([q_mag] * dim)

        self.state = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.state[:4] = normalize(self.state[:4])
        # print("starting state: ", self.state)
        self.cov = np.identity(dim) * 5e-7

        self.B_true = B_true

        self.reaction_speeds = reaction_speeds

        self.old_reaction_speeds = np.zeros(3)
        # self.old_reaction_speeds = reaction_speeds

        self.kalmanMethod = kalmanMethod


    def ukf_setR(self, magNoise, gyroNoise):
        # r: measurement noise (m x m)
        self.R = np.array([[magNoise, 0, 0, 0, 0, 0],
                 [0, magNoise, 0, 0, 0, 0],
                 [0, 0, magNoise, 0, 0, 0],
                 [0, 0, 0, gyroNoise, 0, 0],
                 [0, 0, 0, 0, gyroNoise, 0],
                 [0, 0, 0, 0, 0, gyroNoise]])


    def ukf_setQ(self, noiseMagnitude, R):
        # q: process noise (n x n)
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
    
    def generateSpeeds(self, max, min, steps, step, indices):
        # indices = bitset of sorts to signify which axis you want movement about
        # speed on x and z would equal [1, 0, 1]

        ideal_reaction_speeds = [np.array([0, 0, 0])]
        thing = 0

        for a in range(self.n):
            if (a < steps and thing < max):
                thing += step
            elif thing > min and a > steps:
                thing -= step
            
            result = np.array([thing, thing, thing])
            # multiply by bitset to get only proper axis
            result = indices * result
            ideal_reaction_speeds.append(result)

        return np.array(ideal_reaction_speeds[:self.n])


    def propagate(self, reaction_speeds):
        # given initial state + correct wheel speeds, return ideal states

        # initialize propogator object with inital quaternion and angular velocity
        # propagator = AttitudePropagator(q_init=self.state[:4], w_init=self.reaction_speeds)
        # t0 = 0
        # tf = self.n * self.dt
        # # use attitude propagator to find actual ideal quaternion for n steps
        # states = propagator.propagate_states(t0, tf, self.n)

        # intertia constants from juwan
        I_body = np.array([[46535.388, 257.834, 536.12],
                [257.834, 47934.771, -710.058],
                [536.12, -710.058, 23138.181]])
        I_body = I_body * 1e-7
        I_spin = 5.1e-7
        I_trans = 5.1e-7
        # intialize 1D EOMs using intertia measurements of cubeSat
        EOMS = TEST1EOMS(I_body, I_spin, I_trans)

        currState = self.state
        states = np.array([currState])
        self.reaction_speeds = np.zeros(3)

        # propagate states using our equations of motion
        for i in range(self.n):
            self.old_reaction_speeds = self.reaction_speeds
            self.reaction_speeds = reaction_speeds[i]
            alpha = (self.reaction_speeds - self.old_reaction_speeds) / self.dt
            
            currState = EOMS.eoms(currState[:4], currState[4:], self.reaction_speeds, 0, alpha, self.dt)

            states = np.append(states, np.array([currState]), axis=0)
        
        # remove duplicate first element
        states = states[1:]
        
        return states

    def generateData(self, states, magNoises, gyroNoises, hallNoise):
        # given ideal states, generate data readings for each

        # calculate sensor b field for every time step
        # rotation matrix(q) * true B field + noise
        # first value, then all the otheres
        B_sens = np.array([np.matmul(quaternion_rotation_matrix(states[0]), self.B_true)])
        for a in range(1, self.n):
            B_sens = np.append(B_sens, np.array([np.matmul(quaternion_rotation_matrix(states[a]), self.B_true)]), axis=0)
            # print("{}: {}".format(a, np.matmul(quaternion_rotation_matrix(states[a]), self.B_true)))
        
        B_sens += magNoises

        # create sensor data matrix of magnetomer reading and angular velocity
        data = np.zeros((self.n, self.dim_mes))
        for a in range(self.n):
            data[a][0] = B_sens[a][0]
            data[a][1] = B_sens[a][1]
            data[a][2] = B_sens[a][2]
            data[a][3] = states[a][4] + gyroNoises[a][0]
            data[a][4] = states[a][5] + gyroNoises[a][1]
            data[a][5] = states[a][6] + gyroNoises[a][2]

        return data

    def simulate(self, data, reaction_speeds):
        # run given kalman filter, return set of states

        states = []
        self.reaction_speeds = np.zeros(3)
        for i in range(self.n):
            
            self.old_reaction_speeds = self.reaction_speeds
            self.reaction_speeds = reaction_speeds[i]

            self.state, self.cov = self.kalmanMethod(self.state, self.cov, self.Q, self.R, self.B_true, self.reaction_speeds, self.old_reaction_speeds, data[i])
            states.append(self.state)

        return states

    def visualizeResults(self, states):
        # plot residuals
        # or visualize 3 things: raw, filtered, ideal

        game_visualize(np.array(states), 0)
