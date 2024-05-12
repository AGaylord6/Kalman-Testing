'''
main.py
Author: Andrew Gaylord

uses fake models to simulate and compare different kalman filters

'''

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

        self.state = np.array([1, 0, 0, 0])
        self.cov = np.identity(dim) * 5e-4

        self.B_true = B_true

        self.reaction_speeds = np.zeros(3)
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
        q = q * noiseMagnitude

        # update starting cov guess
        self.cov = R * self.Q

    def propagate(self):
        # given initial state + correct wheel speeds, return ideal states

        # need to get angular velocity from reaction wheel speeds

        # initialize propogator object with inital quaternion and angular velocity
        propagator = AttitudePropagator(q_init=initQ, w_init=w)
        
        # use attitude propagator to find actual ideal quaternion for n steps
        states = propagator.propagate_states(t0, tf, n)
        
        
        # need to add small sensor noise
        pass

    def generateData(self, magNoise, gyroNoise, hallNoise):
        # given ideal states, generate data readings for each

        # create sensor data matrix of magnetomer reading and angular velocity
        data = [0] * dimMes
        data[0] = B_sens[i][0]
        data[1] = B_sens[i][1]
        data[2] = B_sens[i][2]
        data[3] = w[0]
        data[4] = w[1]
        data[5] = w[2]

        # calculate sensor b field for every time step
        # rotation matrix(q) * true B field + noise
        # first value, then all the otheres
        B_sens = np.array([np.matmul(hfunc.quaternion_rotation_matrix(states[0]), B_true)])
        for a in range(1, n):
            B_sens = np.append(B_sens, np.array([np.matmul(hfunc.quaternion_rotation_matrix(states[a]), B_true)]), axis=0)

        pass

    def simulate():
        # run given kalman filter, return set of states
        start, cov = UKF_algorithm.UKF(start, cov, q, r, list(B_true), reaction_speeds, old_reaction_speeds, data)
        pass

    def plotResults():
        # plot residuals
        # or visualize 3 things: raw, filtered, ideal

        # game_visualize(np.array(results), 0)

        pass

if __name__ == "__main__":

    


