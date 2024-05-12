'''
main.py
Author: Andrew Gaylord

uses fake models to simulate and compare different kalman filters

'''

import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

from irishsat_ukf.PySOL.wmm import *
from irishsat_ukf.simulator import *
from irishsat_ukf.UKF_algorithm import *
from irishsat_ukf.hfunc import *

'''

https://cs.adelaide.edu.au/~ianr/Teaching/Estimation/LectureNotes2.pdf
https://github.com/FrancoisCarouge/Kalman
https://www.researchgate.net/post/How_can_I_validate_the_Kalman_Filter_result




'''


class Filter():
    def __init__ (self, n, dt, dim, dim_mes, r_mag, q_mag, B_true, kalmanMethod):
        self.n = n
        self.dt = dt
        self.dim = dim
        self.dim_mes = dim_mes

        # measurement noise
        self.R = np.diag([r_mag] * dim_mes)
        # process noise
        self.Q = np.diag([q_mag] * dim)

        self.state = np.array([1, 0, 0, 0, 0, 0, 0])
        self.cov = np.identity(dim) * 5e-4

        self.B_true = B_true

        # self.reaction_speeds = np.zeros(3)
        self.reaction_speeds = np.array([1, 0, 0])

        self.old_reaction_speeds = np.zeros(3)

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

    def propagate(self):
        # given initial state + correct wheel speeds, return ideal states

        # TODO: need to get angular velocity from reaction wheel speeds

        # initialize propogator object with inital quaternion and angular velocity
        propagator = AttitudePropagator(q_init=self.state[:4], w_init=self.reaction_speeds)
        
        t0 = 0
        tf = self.n * self.dt
        
        # use attitude propagator to find actual ideal quaternion for n steps
        states = propagator.propagate_states(t0, tf, self.n)
        
        return states

    def generateData(self, states, magNoise, gyroNoise, hallNoise):
        # given ideal states, generate data readings for each

        # calculate sensor b field for every time step
        # rotation matrix(q) * true B field + noise
        # first value, then all the otheres
        B_sens = np.array([np.matmul(quaternion_rotation_matrix(states[0]), self.B_true)])
        for a in range(1, self.n):
            B_sens = np.append(B_sens, np.array([np.matmul(quaternion_rotation_matrix(states[a]), self.B_true)]), axis=0)

        # create sensor data matrix of magnetomer reading and angular velocity
        # data = [[0] * self.dim_mes] * self.n
        data = np.zeros((self.n, self.dim_mes))
        for a in range(self.n):
            data[a][0] = B_sens[a][0]
            data[a][1] = B_sens[a][1]
            data[a][2] = B_sens[a][2]
            data[a][3] = self.reaction_speeds[0]
            data[a][4] = self.reaction_speeds[1]
            data[a][5] = self.reaction_speeds[2]

        return data

    def simulate(self, data):
        # run given kalman filter, return set of states
        states = []
        for i in range(self.n):
            self.state, self.cov = self.kalmanMethod(self.state, self.cov, self.Q, self.R, self.B_true, self.reaction_speeds, self.old_reaction_speeds, data[i])
            states.append(self.state)
        
        return states

    def plotResults(self, states):
        # plot residuals
        # or visualize 3 things: raw, filtered, ideal

        game_visualize(np.array(states), 0)


if __name__ == "__main__":

    ukf = Filter(1000, 0.1, 7, 6, 0, 0, np.array([0, 0, 1]), UKF)

    ukf.ukf_setQ(.1, 10)

    ukf.ukf_setR(.1, .1)


    ideal = ukf.propagate()
    
    print("state: ", ideal[:10])

    data = ukf.generateData(ideal, 0, 0, 0)

    print("data: ", data[:10])

    filtered = ukf.simulate(data)

    ukf.plotResults(filtered)




