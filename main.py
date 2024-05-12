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


EOMs are messed up. changed rw_config back to gamma and such

'''


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

        self.state = np.array([1, 0, 0, 0, 0, 0, 0])
        self.cov = np.identity(dim) * 5e-4

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
        I_trans = 0
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
        
        return states

    def generateData(self, states, magNoises, gyroNoises, hallNoise):
        # given ideal states, generate data readings for each

        # calculate sensor b field for every time step
        # rotation matrix(q) * true B field + noise
        # first value, then all the otheres
        B_sens = np.array([np.matmul(quaternion_rotation_matrix(states[0]), self.B_true)])
        for a in range(1, self.n):
            B_sens = np.append(B_sens, np.array([np.matmul(quaternion_rotation_matrix(states[a]), self.B_true)]), axis=0)
        
        B_sens += magNoises

        # create sensor data matrix of magnetomer reading and angular velocity
        data = np.zeros((self.n, self.dim_mes))
        for a in range(self.n):
            data[a][0] = B_sens[a][0]
            data[a][1] = B_sens[a][1]
            data[a][2] = B_sens[a][2]
            data[a][3] = states[a][3] + gyroNoises[a]
            data[a][4] = states[a][4] + gyroNoises[a]
            data[a][5] = states[a][5] + gyroNoises[a]

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

    def plotResults(self, states):
        # plot residuals
        # or visualize 3 things: raw, filtered, ideal

        game_visualize(np.array(states), 0)


if __name__ == "__main__":
    

    ukf = Filter(500, 0.1, 7, 6, 0, 0, np.array([0, 0, 0]), np.array([0, 0, 0]), UKF)

    ukf.ukf_setQ(.1, 10)

    ukf.ukf_setR(.2, .2)

    # TODO: define ideal reaction wheel speed at each iteration?
    max = 100
    min = -100
    thing = 0
    ideal_reaction_speeds = [np.array([0, 0, 0])]
    for a in range(ukf.n):
        if (a < 150 and thing < max):
            thing += 10
        elif thing > min:
            thing -= 10
        
        ideal_reaction_speeds.append(np.array([0, 0, thing]))
        # ideal_reaction_speeds = np.append(ideal_reaction_speeds, np.array([0, 0, thing]), axis=0)
    ideal_reaction_speeds = np.array(ideal_reaction_speeds)

    print(ideal_reaction_speeds[:20])

    ideal = ukf.propagate(ideal_reaction_speeds)
    
    print("state: ", ideal[:10])

    magNoises = np.random.normal(0, .01, (ukf.n, 3))
    gyroNoises = np.random.normal(0, .01, ukf.n)
    data = ukf.generateData(ideal, magNoises, gyroNoises, 0)

    print("data: ", data[:10])


    filtered = ukf.simulate(data, ideal_reaction_speeds)

    # ukf.plotResults(ideal)
    ukf.plotResults(filtered)




